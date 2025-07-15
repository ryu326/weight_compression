import torch
import math
# import utils
from lib import utils
import os
from lib.algo import quip
from lib.algo import code_optimize
from lib.algo import optimize_qmap
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm
import copy
from torch.utils.data import TensorDataset, DataLoader
import sys
sys.path.append('/workspace/Weight_compression')
import wandb
from NWC.loss import *
import glog


def compress_linear(W, H, comp_model, Qlevel, args, device='cpu'):
    
    W = W.to(device)
    H = H.to(device)
    
    comp_model = comp_model.to(device)
    comp_model.scale = comp_model.scale.to(device)
    comp_model.shift = comp_model.shift.to(device)
    
    SU, SV, scaleWH = None, None, None
    if args.incoh_mode != 'none':
        Lhr, H, W, SU, SV, scaleWH = quip.incoherence_preprocess(H, W, args)
    if args.scaleH: ## scaleh2 --scaleh 랑 --row_normalize 하는게 scaleh2
        assert args.row_normalize == True
        diagH = torch.diag(H)
        diagH = torch.clamp(diagH, min=1e-8)
        scaleH = diagH.sqrt()
        W = W * scaleH[None, :]
        H = H / scaleH[None, :]
        H = H / scaleH[:, None]
    if args.scaleHinv:
        assert args.row_normalize == True
        Lhr = torch.linalg.cholesky(H)
        H_inv = torch.cholesky_inverse(Lhr)
        diagH_inv = torch.diag(H_inv)
        scaleH = 1/diagH_inv
        scaleH = torch.clamp(scaleH, min=1e-8)
        scaleH = scaleH.sqrt()
        W = W * scaleH[None, :]
        H = H / scaleH[None, :]
        H = H / scaleH[:, None]
    if args.layer_normalize:
        comp_model.scale = W.std().to(device)
        comp_model.shift = W.mean().to(device)
    col_std = None
    row_std = None
    if args.row_normalize:
        comp_model.scale = torch.tensor(1).to(device)
        comp_model.shift = torch.tensor(0).to(device)
        # row_std = W.std(dim=1, keepdim=True).to(torch.float16) # (B, m, 1)
        row_std = W.std(dim=1, keepdim=True) # (B, m, 1)
    if args.col_normalize:
        comp_model.scale = torch.tensor(1).to(device)
        comp_model.shift = torch.tensor(0).to(device)
        col_std = W.std(dim=0, keepdim=True).to(torch.float16)  # (B, 1, n)
    if args.scale_cond:
        assert args.row_normalize
        col_std = (W/row_std).std(dim=0, keepdim=True)  # (B, 1, n)
        if comp_model.config.uniform_scale_max is not None:
            # comp_model.config.uniform_scale_max = 1 ## for test
            glog.info(f'== clamp col_std {comp_model.config.uniform_scale_max} ==')
            glog.info(f'{col_std.mean()} {col_std.min()} {col_std.max()}')
            col_std = torch.clamp(col_std, max = comp_model.config.uniform_scale_max)
        if args.scale_cond_test is not None:
            col_std = torch.full_like(col_std, args.scale_cond_test)
            glog.info(f'{col_std.mean()} {col_std.min()} {col_std.max()}')
            
            
    if args.scale_std is not None:
        print(f"Scale scale *{args.scale_std}")
        comp_model.scale = args.scale_std * comp_model.scale
        print('scale:', comp_model.scale)        

    if args.ql == True:
        assert Qlevel == None
        if args.Q == 4:
            top = np.array([0.1, 1, 10])
            qlevels = [3, 2, 1]
            in_norm = torch.diag(H)
            topk = (top * len(in_norm)/100).astype(int)
            Qlevel = torch.zeros_like(in_norm, dtype=torch.int32)
            _, topk_indices = torch.topk(in_norm, k=topk.sum())
            start = 0
            for count, value in zip(topk , qlevels):
                indices = topk_indices[start:start + count]
                Qlevel[indices] = value
                start += count
        elif args.Q == 2:
            # top = np.array([0.1])
            top = np.array([args.ql_search_r])
            qlevels = [args.ql_search_value] if comp_model.Q == 4 else [1]
            in_norm = torch.diag(H)
            topk = (top * len(in_norm)/100).astype(int)
            Qlevel = torch.zeros_like(in_norm, dtype=torch.int32)
            _, topk_indices = torch.topk(in_norm, k=topk.sum())
            start = 0    
            for count, value in zip(topk , qlevels):
                indices = topk_indices[start:start + count]
                Qlevel[indices] = value
                start += count
        # unique_vals, counts = torch.unique(Qlevel, return_counts=True)
        # print(unique_vals)  # tensor([1, 2, 3, 4, 5])
        # print(counts)       # tensor([1, 2, 1, 3, 1])
        
    if args.ql_invH == True:
        assert Qlevel == None
        assert comp_model.Q == 4
        Lhr = torch.linalg.cholesky(H)
        H_inv = torch.cholesky_inverse(Lhr)
        top = np.array([0.1, 1, 10])
        qlevels = [3, 2, 1]
        diag = torch.diag(H_inv)
        topk = (top * len(diag)/100).astype(int)
        Qlevel = torch.zeros_like(diag, dtype=torch.int32)
        _, topk_indices = torch.topk(diag, k=topk.sum(), largest=False)
        start = 0    
        for count, value in zip(topk , qlevels):
            indices = topk_indices[start:start + count]
            Qlevel[indices] = value
            start += count
        
    lstats = None
    if args.layerwise_cdt == True:
        Wstats = describe_distribution(W)
        stat_keys = ["mean", "median", "std", "range", "iqr", "skewness", "kurtosis"]
        lstats = torch.tensor([Wstats[key] for key in stat_keys]).to(device)
        
    if args.ql_tuned:
        if args.layer_name == 'v':
            Qlevel = torch.full_like(Qlevel, 3)
        if args.layer_name == 'o':
            Qlevel = torch.max(Qlevel, torch.tensor(1))    
        if args.layer_idx == 0:
            Qlevel = torch.max(Qlevel, torch.tensor(1))

    if args.ql_search:
        # ql_search_layer_idx = list(map(int, args.ql_search_layer_idx.split(',')))
        # ql_search_layer_name = args.ql_search_layer_name.split(',')
        # assert args.ql
        # if args.layer_name in ql_search_layer_name and args.layer_idx in ql_search_layer_idx:
        #     Qlevel = torch.full_like(Qlevel, args.ql_search_value)    
        Qlevel = torch.full((W.shape[1],), args.ql_search_value, dtype=torch.int32)    


    Qlevel = Qlevel.to(device) if Qlevel is not None else None
    
    Qmap = None
    if args.qmap_uniform is not None:
        Qmap = torch.full((W.shape[0]//comp_model.input_size, W.shape[1]), args.qmap_uniform, device=device)
    if args.qmap_hessian:
        in_mag = torch.diag(H)
        in_mag = in_mag - in_mag.min() + 1e-8  # 0 이상으로 이동
        alpha = args.qmap_alpha
        in_mag = in_mag.pow(alpha)
        x_min = in_mag.min()
        x_max = in_mag.max()
        Qmap = (in_mag - x_min) / (x_max - x_min)
        Qmap =  Qmap.reshape(1, W.shape[1]).expand(W.shape[0]//comp_model.input_size, W.shape[1]).to(device)
    if args.qmap_hessian_ql:
        top = np.array([0.1, 1, 10])
        qlevels = [1, 0.66, 0.33]
        in_norm = torch.diag(H)
        topk = (top * len(in_norm)/100).astype(int)
        Qmap = torch.zeros_like(in_norm, dtype=torch.torch.float)
        _, topk_indices = torch.topk(in_norm, k=topk.sum())
        start = 0    
        for count, value in zip(topk , qlevels):
            indices = topk_indices[start:start + count]
            Qmap[indices] = value
            start += count            
        Qmap =  Qmap.reshape(1, W.shape[1]).expand(W.shape[0]//comp_model.input_size, W.shape[1]).to(device= device, dtype=torch.float)
        
    # if args.qmap_optim:
    #     with torch.enable_grad():
    #         Qmap = optimize_qmap.optimize_qmap(W, H, comp_model, args, Qmap = Qmap).to(device) ## (m//16, n)
    # if args.rnorm_optim:
    #     with torch.enable_grad():
    #         row_std = optimize_qmap.optimize_rnorm(W, H, comp_model, args, row_norm = row_std, qlevel = Qlevel).to(device) ## (m//16, n)
    #         W = W / row_std
            
    if args.qmap_optim or args.rnorm_optim or args.cnorm_optim:
        Qmap, row_std, col_std = optimize_qmap.optimize_qmap_rnorm(W, H, comp_model, args, qlevel = Qlevel, row_norm=row_std, col_norm=col_std, qmap = Qmap)
        Qmap = Qmap.to(device) if Qmap is not None else None
        row_std = row_std.to(device) if row_std is not None else None
        col_std = col_std.to(device) if col_std is not None else None   

    ft_result = None
    optimize_out = None

    res = comp_W(W, H, comp_model, args, qlevel = Qlevel, row_norm=row_std, col_norm=col_std, qmap = Qmap)
    # col_std = res['col_norm'] 
    # row_std = res['row_norm']
    
    if args.incoh_mode != 'none':
        res['W_hat'] = quip.incoherence_process(res['W_hat'], SU, SV, scaleWH, args)
        res['W_hat_init'] = quip.incoherence_process(res['W_hat_init'], SU, SV, scaleWH, args) if res['W_hat_init'] is not None else None
        res['W_hat_sga'] = quip.incoherence_process(res['W_hat_sga'], SU, SV, scaleWH, args) if res['W_hat_sga'] is not None else None
        res['W_hat_round'] = quip.incoherence_process(res['W_hat_round'], SU, SV, scaleWH, args) if res['W_hat_round'] is not None else None
    
    if args.scaleH or args.scaleHinv:
        res['W_hat'] = res['W_hat'] / scaleH[None, :]
    
    if args.col_normalize:
        # for key in ['W_hat', 'W_hat_init', 'W_hat_sga', 'W_hat_round']:
        #     if res.get(key) is not None:
        #         res[key] = res[key] * col_std
        for key in ['bpp_loss_sum', 'bpp_loss_sum_init', 'bpp_loss_sum_sga', 'bpp_loss_sum_round', 'bpp_sum']:
            if res.get(key) is not None:
                res[key] += col_std.numel() * 16

    if args.row_normalize:
        # for key in ['W_hat', 'W_hat_init', 'W_hat_sga', 'W_hat_round']:
        #     if res.get(key) is not None:
        #         res[key] = res[key] * row_std
        for key in ['bpp_loss_sum', 'bpp_loss_sum_init', 'bpp_loss_sum_sga', 'bpp_loss_sum_round', 'bpp_sum']:
            if res.get(key) is not None:
                res[key] += row_std.numel() * 16
                
    if args.ql or args.ql_invH:
        for key in ['bpp_loss_sum', 'bpp_loss_sum_init', 'bpp_loss_sum_sga', 'bpp_loss_sum_round', 'bpp_sum']:
            if res.get(key) is not None:
                res[key] += W.shape[1] * math.ceil(math.log2(args.Q))
    
    # for key in ['W_hat', 'W_hat_init', 'W_hat_sga', 'W_hat_round']:
    #     if res.get(key) is not None:
    #         res[key] = res[key].cpu()

    utils.clean()
    return res, SU, SV, scaleWH, ft_result, optimize_out
    
def comp_W(W, H, model, args, **kwargs):
    ## col으로 W 자르기
    
    bs = min(W.shape[1], 4096*4096 // W.shape[0]) if args.comp_batch_size == -1 else args.comp_batch_size
    (m, n) = W.shape
    W_hat = torch.zeros_like(W)
    num_pixels = 0
    bpp_loss_sum = 0
    bpp_sum = 0
    codes = []
    std = W.std()

    if args.code_optim:
        W_hat_init = torch.zeros_like(W)
        num_pixels_init = 0
        bpp_loss_sum_init = 0
        bpp_sum_init = 0
    if args.code_optim_test:
        W_hat_sga = torch.zeros_like(W)
        num_pixels_sga = 0
        bpp_loss_sum_sga = 0
        W_hat_round = torch.zeros_like(W)
        num_pixels_round = 0
        bpp_loss_sum_round = 0
    
    row_norm = kwargs.get('row_norm', None)  # (m, 1)
    col_norm = kwargs.get('col_norm', None)  # (1, n)
    
    qlevel = kwargs.get('qlevel', None)
    qlevel = qlevel.reshape(W.shape[1], ) if qlevel is not None else None
    
    qmap = kwargs.get('qmap', None)  ## (m//16, n)
    # if qmap is not None and qmap.shape != (m//model.input_size, n):
    #     qmap = qmap.reshape(1, n).expand(m//model.input_size, n)
        
    if args.ldlq:
        bs = 128 if args.comp_batch_size == -1 else args.comp_batch_size
        # assert args.direction == 'col'
        L, D = block_LDL(H, bs)
        # L, D = block_LDL(H, 128)
        assert n % bs == 0
    
    q_size = torch.ones(W.shape[1],).to(W.device) if args.optim_qs else None

    for i,e in enumerate(range(n, 0, -bs)):
        s = max(0, e - bs)
        if args.ldlq:
            w = W[:, s:e] + (W[:, e:] - W_hat[:, e:]) @ L[e:, s:e]
        else:
            w = W[:, s:e]        
        
        ql = qlevel[s:e] if qlevel is not None else None
        r_norm = row_norm[:, :] if row_norm is not None else None
        c_norm = col_norm[:, s:e] if col_norm is not None else None
        qs = q_size[s:e] if q_size is not None else None
        # qm = qmap[:, s:e].to(w.device) if qmap is not None else None
        qm = qmap[:, s:e] if qmap is not None else None
        
        # x_hat, n_pixels, bpp_loss_, out, out_enc, nbits = model_foward(w.clone().T, model, args, ql = ql.clone(), qmap = qmap)
        # x_hat, n_pixels, bpp_loss_, out, out_enc, nbits = model_foward(w.clone().T, model, args, ql = ql, qm = qm)
        x_hat, n_pixels, bpp_loss_, out, out_enc, nbits = model_foward_one_batch(w.clone(), model, args, ql = ql, qm = qm, rnorm = r_norm, cnorm = c_norm)
        
        if args.code_optim:
            bpp_sum_init += nbits
            W_hat_init[:, s:e] = x_hat.T
            num_pixels_init += n_pixels
            bpp_loss_sum_init += bpp_loss_.item()

            best_y, w_hat_sga, bpp_loss_sga, n_pixels_sga, best_rnorm, best_cnorm, best_qs = code_optimize(w.clone().T, model, out, args, ql = ql.clone(), std = std, mode = 'sga', batch_idx = i, rnorm = r_norm, cnorm = c_norm, qs = qs)
            x_hat, n_pixels, bpp_loss_, out, out_enc, nbits = model_foward(None, model, args, ql = ql.clone(), y_in = best_y, mode='round', shape = w.T.shape, qs = best_qs)
            # x_hat_test, n_pixels_test, bpp_loss_test, out_test, out_enc_test, nbits_test = model_foward(None, model, args, ql = ql.clone(), y_in = best_y, mode='round', shape = w.T.shape)

        codes.append(out_enc)
        bpp_sum += nbits
        W_hat[:, s:e] = x_hat
        num_pixels += n_pixels
        bpp_loss_sum += bpp_loss_.item()
        # if args.optim_norm:
        #     col_norm[:, s:e] = best_cnorm.T
        #     row_norm[s:e, :] = best_rnorm.T
        if args.optim_qs:
            q_size[s:e] = best_qs

        if args.code_optim_test:
            W_hat_sga[:, s:e] = w_hat_sga.T
            num_pixels_sga += n_pixels_sga
            bpp_loss_sum_sga += bpp_loss_sga

            x_hat, n_pixels, bpp_loss_, out, out_enc, nbits = model_foward(w.clone().T, model, args, ql = ql.clone(), mode='round')
            W_hat_round[:, s:e] = x_hat.T
            num_pixels_round += n_pixels
            bpp_loss_sum_round += bpp_loss_.item()
    
    
    if args.code_optim_test:
        assert num_pixels_round == num_pixels_sga
        assert num_pixels_round == num_pixels
        assert num_pixels_init == num_pixels
    
    return {'W_hat': W_hat,
            'bpp_loss_sum': bpp_loss_sum,
            'bpp_loss': bpp_loss_sum / num_pixels,
            'num_pixels': num_pixels,
            'bpp_sum': bpp_sum,
            'bpp': bpp_sum / num_pixels,
            'codes': codes,
            'W_hat_init': W_hat_init if args.code_optim else None,
            'bpp_loss_sum_init': bpp_loss_sum_init if args.code_optim else None, 
            'bpp_loss_init': bpp_loss_sum_init / num_pixels_init if args.code_optim else None,
            'bpp_init': bpp_sum_init / num_pixels_init if args.code_optim else None,
            'W_hat_sga': W_hat_sga if args.code_optim_test else None,
            'bpp_loss_sum_sga': bpp_loss_sum_sga if args.code_optim_test else None, 
            'bpp_loss_sga': bpp_loss_sum_sga / num_pixels_sga if args.code_optim_test else None,
            'W_hat_round': W_hat_round if args.code_optim_test else None,
            'bpp_loss_sum_round': bpp_loss_sum_round if args.code_optim_test else None,
            'bpp_loss_round': bpp_loss_sum_round / num_pixels_round if args.code_optim_test else None,
            'col_norm': col_norm,
            'row_norm': row_norm
            }   

def model_foward_one_batch(w, model, args, **kwargs):
    y_in = kwargs.get('y_in', None)
    mode = kwargs.get('mode', 'init')
    qs = kwargs.get('qs', None)  # scalar??
    qm = kwargs.get('qm', None)  # (m//16, n)
    ql = kwargs.get('ql', None)  # (n, )
    rnorm = kwargs.get('rnorm', None)  # (m, 1)
    cnorm = kwargs.get('cnorm', None)  # (1, n)
    
    (m, n) = w.shape # (m, n), T안하고 들어오는 걸로 가정
    blks = model.input_size
    assert (m if args.direction == 'col' else n) % blks == 0
    
    w = w / rnorm if rnorm is not None else w
    w = w / cnorm if cnorm is not None and not args.scale_cond else w
    
    if ql is not None:
        ql = ql.reshape(1, n).expand(m//blks, n)
    elif args.ql_search_value is not None:
        ql = torch.full((m//blks, n), args.ql_search_value, dtype=torch.int32, device=w.device)
    
    transpose = args.direction == 'col'

    w = w.T if (w is not None and transpose) else w
    ql = ql.T if (ql is not None and transpose) else ql
    qm = qm.T if (qm is not None and transpose) else qm

    data = {}
    w = w.reshape(1, -1, blks)
    data['weight_block'] = w
    
    if ql is not None:
        data['q_level'] = ql.reshape(1, w.shape[1])
    if qm is not None:
        data['qmap'] = qm.reshape(1, w.shape[1])
    if hasattr(model, 'pe') and model.pe:
        wtype_mapping = {'q': 0, 'k': 1, 'v': 2, 'o': 3, 'gate': 4, 'up': 5, 'down': 6}
        depth = args.layer_idx
        ltype = wtype_mapping[args.layer_name]
        data['depth'] = torch.full((1, 1), depth, dtype=torch.long).to(w.device)
        data['ltype'] = torch.full((1, 1), ltype, dtype=torch.long).to(w.device)
    if args.scale_cond:
        scale_cond = cnorm.repeat(m, 1)
        assert scale_cond.shape == (m, n)
        scale_cond = scale_cond.reshape(1, -1, blks)
        data['scale_cond'] = scale_cond
        
    num_pixels = m*n
    bpp_loss_sum = torch.tensor(0)
    nbits = 0
    out_enc = None
    out = None
    
    if args.use_codes:
        out_enc = model.compress(data)
        out_dec = model.decompress(out_enc)
        w_hat = out_dec['x_hat']
        for s in out_enc["strings"]:
            nbits += len(s[0]) * 8.0
    else:
        if hasattr(model, 'sga'):
            out = model(data, mode = mode, y_in = y_in, qs = qs)
        else:
            out = model(data)
        w_hat = out['x_hat']
            
        if isinstance(out["likelihoods"], dict):
            bpp_loss_sum = sum(
                (torch.log(likelihoods).sum() / -math.log(2))
                for likelihoods in out["likelihoods"].values()
            )
        else :
            bpp_loss_sum = (torch.log(out["likelihoods"]).sum() / -math.log(2))
            
    if args.direction == 'col':
        w_hat = w_hat.reshape(n, m).transpose(0, 1).contiguous()
    else:
        w_hat = w_hat.reshape(m, n)

    if cnorm is not None and not args.scale_cond:
        w_hat = w_hat * cnorm
    if rnorm is not None:
        w_hat = w_hat * rnorm
    
    if args.use_codes:
        del out_dec['x_hat']
    else:
        del out['x_hat']
    torch.cuda.empty_cache()
    
    return w_hat, num_pixels, bpp_loss_sum, out, out_enc, nbits

def code_optimize(w, comp_model, init_out, args, **kwargs):
    ql = kwargs.get('ql', None).reshape(w.shape[0], 1)
    std = kwargs.get('std', None)
    # mode = kwargs.get('mode', 'sga')
    batch_idx = kwargs.get('batch_idx', -1)
    ori_shape = w.shape
    w = w.reshape(w.shape[0], -1, comp_model.input_size)
    qs = kwargs.get('qs', None)
    
    rnorm = kwargs.get('rnorm', None) # (1, m) 
    cnorm = kwargs.get('cnorm', None) # (n, 1)
    
    if args.optim_norm:
        assert rnorm.dim() == 2 and cnorm.dim() == 2
        assert rnorm.shape == (1, w.shape[1]) and cnorm.shape == (w.shape[0], 1)
        assert cnorm == None or cnorm.dim() == 2 
        assert cnorm == None or cnorm.shape == (w.shape[0], 1)
    
    wandb.init(project=f"NWC_code_optim", name=f"{'_'.join(args.save_path.split('/')[-2:])}_{args.layer_idx}_{args.layer_name}_batch{batch_idx}", config=vars(args))

    loss_fn =  get_loss_fn(args, std=std, device = w.device)

    with torch.no_grad():
        data = {'weight_block': w, 'q_level': ql}
        init_loss = loss_fn(data, init_out)

    y = init_out['y'].clone()
    y = nn.Parameter(y, requires_grad=True)
    
    tune_params = [y]
    if rnorm is not None and args.optim_norm:
        rnorm = nn.Parameter(rnorm, requires_grad=True)
        tune_params.append(rnorm)
    if cnorm is not None and args.optim_norm:
        cnorm = nn.Parameter(cnorm, requires_grad=True)
        tune_params.append(cnorm)
    if args.optim_qs:
        qs = nn.Parameter(qs, requires_grad=True)
        tune_params.append(qs)
    optimizer = optim.Adam(tune_params, lr=args.code_optim_lr)

    # comp_model.train()
    for param in comp_model.parameters():
        param.requires_grad = False

    best_loss = init_loss['loss'].item()
    best_loss_bpp = init_loss['bpp_loss'].item()
    best_loss_recon = init_loss['recon_loss'].item()
    best_y = y.detach().clone()
    best_rnorm = rnorm.detach().clone() if rnorm is not None else None
    best_cnorm = cnorm.detach().clone() if cnorm is not None else None
    best_w_hat = init_out['x_hat'].detach().clone()
    best_qs = qs.detach().clone() if qs is not None else None
    
    with torch.enable_grad():
        for it in range(args.code_optim_it):
            optimizer.zero_grad()
            # qs = torch.clamp(qs, min=0.1)
            qs.data.clamp_(min=0.1) if qs is not None else None
            data = {'weight_block': None, 'q_level': ql}
            out = comp_model(data, mode='sga', y_in = y, it = it, tot_it = args.code_optim_it, qs = qs)
            data = {'weight_block': w, 'q_level': ql}
            # loss = loss_fn(data, out, rnorm = rnorm, cnorm = cnorm)
            loss = loss_fn(data, out)
            loss['loss'].backward()
            optimizer.step()
            
            if loss['loss'].item() < best_loss:
                best_loss = loss['loss'].item()
                best_loss_bpp = loss['bpp_loss'].item()
                best_loss_recon = loss['recon_loss'].item()
                best_y = y.detach().clone()
                best_w_hat = out['x_hat'].detach().clone()
                best_rnorm = rnorm.detach().clone() if rnorm is not None else None
                best_cnorm = cnorm.detach().clone() if cnorm is not None else None
                best_qs = qs.detach().clone() if qs is not None else None
            wandb.log({
                "loss": loss['loss'].item(),
                "bpp_loss": loss['bpp_loss'].item(),
                "recon_loss": loss['recon_loss'].item(),
                "it": it,
                "best_loss": best_loss,
                "best_loss_bpp": best_loss_bpp,
                "best_loss_recon": best_loss_recon,
                "init_loss": init_loss['loss'].item(),
                "init_bpp_loss": init_loss['bpp_loss'].item(),
                "init_recon_loss": init_loss['recon_loss'].item(),
                "best_qs_mean": best_qs.mean().item() if qs is not None else None,
            })
            
    glog.info(f'y == best_y :: {torch.equal(y, best_y)}')
    glog.info(f'y == init_y :: {torch.equal(y, init_out["y"])}')
    wandb.finish()

    if isinstance(out["likelihoods"], dict):
        bpp_loss = sum(
            (torch.log(likelihoods).sum() / -math.log(2))
            for likelihoods in out["likelihoods"].values()
        ).item()
    else :
        bpp_loss = (torch.log(out["likelihoods"]).sum() / -math.log(2)).item()
    num_pixels = ori_shape[0] * ori_shape[1]

    return best_y, best_w_hat.reshape(ori_shape), bpp_loss, num_pixels, best_rnorm, best_cnorm, best_qs
    # return y, out['x_hat'].reshape(ori_shape)


def block_LDL(H, b, check_nan=True):
    n = H.shape[0]
    assert (n % b == 0)
    m = n // b
    try:
        L = torch.linalg.cholesky(H)
    except:
        return None
    DL = torch.diagonal(L.reshape(m, b, m, b), dim1=0, dim2=2).permute(2, 0, 1)
    D = (DL @ DL.permute(0, 2, 1)).cpu()
    DL = torch.linalg.inv(DL)
    L = L.view(n, m, b)
    for i in range(m):
        L[:, i, :] = L[:, i, :] @ DL[i, :, :]

    if check_nan and L.isnan().any():
        return None

    L = L.reshape(n, n)
    return (L, D.to(DL.device))

def describe_distribution(x):
    assert isinstance(x, torch.Tensor), "Input must be a PyTorch tensor"
    x = x.flatten().float()
    n = x.numel()
    
    # 중심 경향
    mean = x.mean()
    median = x.median()

    # 산포도
    std_dev = x.std(unbiased=False)
    value_range = x.max() - x.min()
    q1 = x.kthvalue(int(0.25 * n + 1)).values
    q3 = x.kthvalue(int(0.75 * n + 1)).values
    iqr = q3 - q1

    # 모양
    skewness = ((x - mean)**3).mean() / (std_dev**3)
    kurtosis = ((x - mean)**4).mean() / (std_dev**4) - 3  # Fisher's definition

    del x
    return {
        "mean": mean.item(),
        "median": median.item(),
        "std": std_dev.item(),
        "range": value_range.item(),
        "iqr": iqr.item(),
        "skewness": skewness.item(),
        "kurtosis": kurtosis.item()
    }
