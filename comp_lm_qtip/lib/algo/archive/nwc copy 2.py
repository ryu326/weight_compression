import torch
import math
# import utils
from lib import utils
import os
from lib.algo import quip
from lib.algo import code_optimize
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
    
    if args.layerwise_scale:
        comp_model.scale = W.std().to(device)
        comp_model.shift = W.mean().to(device)
    
    col_std = None
    row_std = None
    if args.row_normalize and args.col_normalize:
        comp_model.scale = torch.tensor(1).to(device)
        comp_model.shift = torch.tensor(0).to(device)
        row_std = W.std(dim=1, keepdim=True).to(torch.float16)
        W = W / row_std
        col_std = W.std(dim=0, keepdim=True).to(torch.float16)
        W = W / col_std
    elif args.row_normalize:
        comp_model.scale = torch.tensor(1).to(device)
        comp_model.shift = torch.tensor(0).to(device)
        row_std = W.std(dim=1, keepdim=True).to(torch.float16)
        W = W / row_std
    elif args.col_normalize:
        comp_model.scale = torch.tensor(1).to(device)
        comp_model.shift = torch.tensor(0).to(device)
        col_std = W.std(dim=0, keepdim=True).to(torch.float16)
        W = W / col_std
    
    SU, SV, scaleWH = None, None, None
    if args.incoh_mode != 'none':
        Lhr, H, W, SU, SV, scaleWH = quip.incoherence_preprocess(H, W, args)

    if args.ql == True:
        assert Qlevel == None
        assert comp_model.Q == 4
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
        ql_search_layer_idx = list(map(int, args.ql_search_layer_idx.split(',')))
        ql_search_layer_name = args.ql_search_layer_name.split(',')
        assert args.ql
        if args.layer_name in ql_search_layer_name and args.layer_idx in ql_search_layer_idx:
            Qlevel = torch.full_like(Qlevel, args.ql_search_value)    

    Qlevel = Qlevel.to(device) if Qlevel is not None else None
    
    ft_result = None
    optimize_out = None

    res = comp_W(W, H, comp_model, args, qlevel = Qlevel, row_norm=row_std, col_norm=col_std)   
    
    if args.incoh_mode != 'none':
        res['W_hat'] = quip.incoherence_process(res['W_hat'], SU, SV, scaleWH, args)
        res['W_hat_init'] = quip.incoherence_process(res['W_hat_init'], SU, SV, scaleWH, args) if res['W_hat_init'] is not None else None
        res['W_hat_sga'] = quip.incoherence_process(res['W_hat_sga'], SU, SV, scaleWH, args) if res['W_hat_sga'] is not None else None
        res['W_hat_round'] = quip.incoherence_process(res['W_hat_round'], SU, SV, scaleWH, args) if res['W_hat_round'] is not None else None
    if args.row_normalize and args.col_normalize:
        res['W_hat'] = res['W_hat'] * col_std  # col 먼저 복원
        res['W_hat'] = res['W_hat'] * row_std  # 그 다음 row 복원
        res['W_hat_init'] = res['W_hat_init'] * col_std if res['W_hat_init'] is not None else None
        res['W_hat_init'] = res['W_hat_init'] * row_std if res['W_hat_init'] is not None else None
        res['W_hat_sga'] = res['W_hat_sga'] * col_std if res['W_hat_sga'] is not None else None
        res['W_hat_sga'] = res['W_hat_sga'] * row_std if res['W_hat_sga'] is not None else None
        res['W_hat_round'] = res['W_hat_round'] * col_std if res['W_hat_round'] is not None else None
        res['W_hat_round'] = res['W_hat_round'] * row_std if res['W_hat_round'] is not None else None
        
        res['bpp_loss_sum'] = res['bpp_loss_sum'] + (row_std.numel() + col_std.numel()) * 16
        res['bpp_loss_sum_init'] = res['bpp_loss_sum_init'] + (row_std.numel() + col_std.numel()) * 16 if res['bpp_loss_sum_init'] is not None else None
        res['bpp_loss_sum_sga'] = res['bpp_loss_sum_sga'] + (row_std.numel() + col_std.numel()) * 16 if res['bpp_loss_sum_sga'] is not None else None   
        res['bpp_loss_sum_round'] = res['bpp_loss_sum_round'] + (row_std.numel() + col_std.numel()) * 16 if res['bpp_loss_sum_round'] is not None else None
        
    elif args.row_normalize:
        res['W_hat'] = res['W_hat'] * row_std
        res['W_hat_init'] = res['W_hat_init'] * row_std if res['W_hat_init'] is not None else None
        res['W_hat_sga'] = res['W_hat_sga'] * row_std if res['W_hat_sga'] is not None else None
        res['W_hat_round'] = res['W_hat_round'] * row_std if res['W_hat_round'] is not None else None
        
        res['bpp_loss_sum'] = res['bpp_loss_sum'] + (row_std.numel()) * 16
        res['bpp_loss_sum_init'] = res['bpp_loss_sum_init'] + (row_std.numel()) * 16 if res['bpp_loss_sum_init'] is not None else None
        res['bpp_loss_sum_sga'] = res['bpp_loss_sum_sga'] + (row_std.numel()) * 16 if res['bpp_loss_sum_sga'] is not None else None   
        res['bpp_loss_sum_round'] = res['bpp_loss_sum_round'] + (row_std.numel()) * 16 if res['bpp_loss_sum_round'] is not None else None
        
    elif args.col_normalize:
        res['W_hat'] = res['W_hat'] * col_std
        res['W_hat_init'] = res['W_hat_init'] * col_std if res['W_hat_init'] is not None else None
        res['W_hat_sga'] = res['W_hat_sga'] * col_std if res['W_hat_sga'] is not None else None
        res['W_hat_round'] = res['W_hat_round'] * col_std if res['W_hat_round'] is not None else None
        
        res['bpp_loss_sum'] = res['bpp_loss_sum'] + (col_std.numel()) * 16
        res['bpp_loss_sum_init'] = res['bpp_loss_sum_init'] + (col_std.numel()) * 16 if res['bpp_loss_sum_init'] is not None else None
        res['bpp_loss_sum_sga'] = res['bpp_loss_sum_sga'] + (col_std.numel()) * 16 if res['bpp_loss_sum_sga'] is not None else None   
        res['bpp_loss_sum_round'] = res['bpp_loss_sum_round'] + (col_std.numel()) * 16 if res['bpp_loss_sum_round'] is not None else None
        
    res['W_hat'] =  res['W_hat'].cpu()
    res['W_hat_init'] = res['W_hat_init'].cpu() if res['W_hat_init'] is not None else None
    res['W_hat_sga'] = res['W_hat_sga'].cpu() if res['W_hat_sga'] is not None else None
    res['W_hat_round'] = res['W_hat_round'].cpu() if res['W_hat_round'] is not None else None

    utils.clean()
    return res, SU, SV, scaleWH, ft_result, optimize_out
    
def comp_W(W, H, model, args, **kwargs):
    ## col으로 가정
    
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
    
    qlevel = kwargs.get('qlevel', None)
    qlevel = qlevel.reshape(W.shape[1], ) if qlevel is not None else None
    
    if args.ldlq:
        bs = 128 if args.comp_batch_size == -1 else args.comp_batch_size
        assert args.direction == 'col'
        L, D = block_LDL(H, bs)
        # L, D = block_LDL(H, 128)
        assert n % bs == 0

        
    if args.direction == 'row':
        W = W.T
        W_hat = W_hat.T
        (m, n) = W.shape


    for i,e in enumerate(range(n, 0, -bs)):
        s = max(0, e - bs)
        if args.ldlq:
            w = W[:, s:e] + (W[:, e:] - W_hat[:, e:]) @ L[e:, s:e]
        else:
            w = W[:, s:e]        
        
        ql = qlevel[s:e] if qlevel is not None else None

        x_hat, n_pixels, bpp_loss_, out, out_enc, nbits = model_foward(w.clone().T, model, args, ql = ql.clone())

        if args.code_optim:
            bpp_sum_init += nbits
            W_hat_init[:, s:e] = x_hat.T
            num_pixels_init += n_pixels
            bpp_loss_sum_init += bpp_loss_

            best_y, w_hat_sga, bpp_loss_sga, n_pixels_sga = code_optimize(w.clone().T, model, out, args, ql = ql.clone(), std = std, mode = 'sga', batch_idx = i)
            x_hat, n_pixels, bpp_loss_, out, out_enc, nbits = model_foward(None, model, args, ql = ql.clone(), y_in = best_y, mode='round', shape = w.T.shape)
            # x_hat_test, n_pixels_test, bpp_loss_test, out_test, out_enc_test, nbits_test = model_foward(None, model, args, ql = ql.clone(), y_in = best_y, mode='round', shape = w.T.shape)

        codes.append(out_enc)
        bpp_sum += nbits
        W_hat[:, s:e] = x_hat.T
        num_pixels += n_pixels
        bpp_loss_sum += bpp_loss_

        if args.code_optim_test:
            W_hat_sga[:, s:e] = w_hat_sga.T
            num_pixels_sga += n_pixels_sga
            bpp_loss_sum_sga += bpp_loss_sga

            x_hat, n_pixels, bpp_loss_, out, out_enc, nbits = model_foward(w.clone().T, model, args, ql = ql.clone(), mode='round')
            W_hat_round[:, s:e] = x_hat.T
            num_pixels_round += n_pixels
            bpp_loss_sum_round += bpp_loss_
    
    if args.direction == 'row':
        W_hat = W_hat.T
        if args.code_optim:
            W_hat_init = W_hat_init.T
        if args.code_optim_test:
            W_hat_sga = W_hat_sga.T
            W_hat_round = W_hat_round.T
    
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
            }   

def model_foward(w, model, args, **kwargs):
    y_in = kwargs.get('y_in', None)
    mode = kwargs.get('mode', 'init')
    ori_shape = w.shape if w is not None else kwargs.get('shape', None)
    w = w.reshape(w.shape[0], -1, model.input_size) if w is not None else None
    data = {}
    data['weight_block'] = w
    
    ql = kwargs.get('ql', None)
    # glog.info(f'{w == None} {mode} {ori_shape} {ql.shape}')

    if ql is not None:
        data['q_level'] = ql.reshape(ori_shape[0], 1)
        
    lstats = kwargs.get('lstats', None)
    if lstats is not None:
        data['l_cdt'] = lstats.unsqueeze(0).repeat(ori_shape[0], 1)
        
    num_pixels = ori_shape[0] * ori_shape[1]
    bpp_loss = 0
    nbits = 0
    out_enc = None
    out = None
    
    if args.use_codes:
        out_enc = model.compress(data)
        out_dec = model.decompress(out_enc)
        w_hat = out_dec['x_hat'].reshape(ori_shape)
        for s in out_enc["strings"]:
            nbits += len(s[0]) * 8.0

    else:
        if hasattr(model, 'sga'):
        # if True:
            out = model(data, mode = mode, y_in = y_in)
        else:
            out = model(data)
        w_hat = out['x_hat'].reshape(ori_shape)
        
        if isinstance(out["likelihoods"], dict):
            bpp_loss = sum(
                (torch.log(likelihoods).sum() / -math.log(2))
                for likelihoods in out["likelihoods"].values()
            ).item()
        else :
            bpp_loss = (torch.log(out["likelihoods"]).sum() / -math.log(2)).item()
    
    return w_hat, num_pixels, bpp_loss, out, out_enc, nbits

def code_optimize(w, comp_model, init_out, args, **kwargs):
    ql = kwargs.get('ql', None).reshape(w.shape[0], 1)
    std = kwargs.get('std', None)
    # mode = kwargs.get('mode', 'sga')
    batch_idx = kwargs.get('batch_idx', -1)
    ori_shape = w.shape
    w = w.reshape(w.shape[0], -1, comp_model.input_size)
    
    row_norm = kwargs.get('row_norm', None)
    row_norm = row_norm.reshape(ori_shape[0], 1) if row_norm is not None else None
    col_norm = kwargs.get('col_norm', None)
    col_norm = col_norm.reshape(1, ori_shape[1]) if col_norm is not None else None
    
    wandb.init(project=f"NWC_code_optim", name=f"{'_'.join(args.save_path.split('/')[-2:])}_{args.layer_idx}_{args.layer_name}_batch{batch_idx}", config=vars(args))

    loss_fn =  get_loss_fn(args, std=std, device = w.device)

    with torch.no_grad():
        data = {'weight_block': w, 'q_level': ql}
        init_loss = loss_fn(data, init_out)

    y = init_out['y'].clone()
    y = nn.Parameter(y, requires_grad=True)
    
    # comp_model.train()
    for param in comp_model.parameters():
        param.requires_grad = False

    best_loss = init_loss['loss'].item()
    best_loss_bpp = init_loss['bpp_loss'].item()
    best_loss_recon = init_loss['recon_loss'].item()
    best_y = y.detach().clone()
    # best_w_hat = init_out['x_hat'].detach().clone()

    tune_params = [y]
    if row_norm is not None:
        tune_params.append(row_norm)
    if col_norm is not None:
        tune_params.append(col_norm)
    optimizer = optim.Adam(tune_params, lr=args.code_optim_lr)

    with torch.enable_grad():
        for it in range(args.code_optim_it):
            optimizer.zero_grad()
            
            data = {'weight_block': None, 'q_level': ql}
            out = comp_model(data, mode='sga', y_in = y, it = it, tot_it = args.code_optim_it)
            data = {'weight_block': w, 'q_level': ql}
            loss = loss_fn(data, out)
            loss['loss'].backward()
            optimizer.step()
            
            if loss['loss'].item() < best_loss:
                best_loss = loss['loss'].item()
                best_loss_bpp = loss['bpp_loss'].item()
                best_loss_recon = loss['recon_loss'].item()
                best_y = y.detach().clone()
                # best_w_hat = out['x_hat'].detach().clone()

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

    return best_y, out['x_hat'].reshape(ori_shape), bpp_loss, num_pixels
    # return y, out['x_hat'].reshape(ori_shape)


def encode_latent(W, model, args, **kwargs):
    if args.direction == 'col':
        W = W.T

    latent = torch.zeros_like(W)

    qlevel = kwargs.get('qlevel', None)
    qlevel = qlevel.reshape(W.shape[0], ) if qlevel is not None else None
    
    bs = args.comp_batch_size
    for start_idx in range(0, W.shape[0], bs):
        end_idx = min(start_idx + bs, W.shape[0])
        batch = W[start_idx:end_idx]
        
        ql = qlevel[start_idx:end_idx] if qlevel is not None else None

        ori_shape = batch.shape
        batch = batch.reshape(ori_shape[0], -1, model.input_size)
        
        data = {}
        data['weight_block'] = batch
        if ql is not None:
            data['q_level'] = ql.reshape(ori_shape[0], 1)
        out = model.forward_encoder(data)
        latent[start_idx:end_idx] = out['y'].reshape(ori_shape)

    if args.direction == 'col':
        latent = latent.T
    
    return latent

# def comp_W(W, model, args, **kwargs):
# def encode_latent(W, model, args, **kwargs):
#     if args.direction == 'col':
#         W = W.T

#     latent = torch.zeros_like(W)

#     qlevel = kwargs.get('qlevel', None)
#     qlevel = qlevel.reshape(W.shape[0], ) if qlevel is not None else None
    
#     bs = args.comp_batch_size
#     for start_idx in range(0, W.shape[0], bs):
#         end_idx = min(start_idx + bs, W.shape[0])
#         batch = W[start_idx:end_idx]
        
#         ql = qlevel[start_idx:end_idx] if qlevel is not None else None

#         ori_shape = batch.shape
#         batch = batch.reshape(ori_shape[0], -1, model.input_size)
        
#         data = {}
#         data['weight_block'] = batch
#         if ql is not None:
#             data['q_level'] = ql.reshape(ori_shape[0], 1)
#         out = model.forward_encoder(data)
#         latent[start_idx:end_idx] = out['y'].reshape(ori_shape)

#     if args.direction == 'col':
#         latent = latent.T
    
#     return latent


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


#     # if args.gptq:
#     #     out = comp_W_gptq(W, comp_model, args.direction, args.comp_batch_size, ql, H, device, args)     
# def comp_W_gptq(W, model, direction, bs, q_level, H, device, args):
    
#     assert direction == 'col'
#     if q_level is not None:
#         q_level = q_level.reshape(-1, )
    
#     W = W.to(device)
#     Losses = torch.zeros_like(W, device=W.device)
#     Q = torch.zeros_like(W, device=W.device)
#     assert torch.isfinite(H).all()
    
#     H = torch.linalg.cholesky(H)
#     H = torch.cholesky_inverse(H)
#     H = torch.linalg.cholesky(H, upper=True)
#     Hinv = H
#     assert torch.isfinite(H).all()
    
#     rows = W.shape[0]
#     columns = W.shape[1]
    
#     num_pixels = 0
#     bpp_loss = 0
#     bpp = 0

#     for i1 in range(0, columns, bs):
#         i2 = min(i1 + bs, columns)
#         count = i2 - i1
        
#         ql = None
#         if q_level is not None:
#             ql = q_level[i1:i2]

#         W1 = W[:, i1:i2].clone()
#         Q1 = torch.zeros_like(W1, device=W1.device)
#         Err1 = torch.zeros_like(W1, device=W1.device)
#         Losses1 = torch.zeros_like(W1, device=W1.device)
#         Hinv1 = Hinv[i1:i2, i1:i2]

#         for i in range(count):
#             w = W1[:, i]
#             d = Hinv1[i, i]
            
#             assert w.size(-1) == rows
#             if args.bundle:
#                 w_reshape = w.reshape(1, -1, model.input_size)  # (row, col) --> (row, -1, inputsize)
#             else :
#                 w_reshape = w.reshape(-1, model.input_size)

#             ql_ = None if ql is None else ql[i]
#             x_hat, n_pixels, bpp_loss_ = model_foward(w_reshape, model, ql)
            
#             q = x_hat.flatten()
#             num_pixels += n_pixels
#             bpp_loss += bpp_loss_

#             Q1[:, i] = q
#             Losses1[:, i] = (w - q)**2 / d**2

#             err1 = (w - q) / d
#             assert torch.isfinite(err1).all()
#             W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
#             Err1[:, i] = err1

#         Q[:, i1:i2] = Q1
#         Losses[:, i1:i2] = Losses1 / 2

#         W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

#     return {'W_hat': Q.cpu(),
#             'bpp_loss_sum': bpp_loss,
#             'num_pixels': num_pixels,
#             'bpp': bpp}


# def hessian_proxy_loss(W, W_hat, H):
#     diff = W_hat - W
#     H = H.float()
#     trace_H = H.trace()
#     if trace_H > 0:
#         H = H / trace_H * H.shape[0] / W.numel()
#     loss = torch.trace(diff @ H @ diff.T)     # scalar
#     return loss

# class RateDistortionLoss(nn.Module):
#     def __init__(self, std, Hr, lmbda):
#         super().__init__()
#         self.mse = nn.MSELoss()
#         self.lmbda = lmbda
#         self.std = std

#         # def clip_outliers_quantile_global(tensor, lower_q=0.03, upper_q=0.97):
#         #     lower = torch.quantile(tensor, lower_q)
#         #     upper = torch.quantile(tensor, upper_q)
#         #     return torch.clip(tensor, min=lower.item(), max=upper.item())
        
#         # self.Hr = clip_outliers_quantile_global(Hr)
#         self.Hr = Hr

#     def forward(self, ori_w, output):        
#         out = {}
#         num_pixels = output["x"].numel()
#         w_hat = output["x_hat"].reshape(ori_w.shape)
#         # H = self.Hr[start_idx:end_idx][start_idx:end_idx]
        
#         out["mse_loss"] = self.mse(ori_w,  w_hat) / self.std**2
#         out["adaptive_loss"] = hessian_proxy_loss(ori_w, w_hat, self.Hr) / self.std**2

#         if isinstance(output["likelihoods"], dict):
#             out["bpp_loss"] = sum(
#                 (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
#                 for likelihoods in output["likelihoods"].values()
#             )
#         else :
#             out["bpp_loss"] = (torch.log(output["likelihoods"]).sum() / (-math.log(2) * num_pixels))


#         # out["loss"] = self.lmbda * out["adaptive_loss"] + out["bpp_loss"]
#         out["loss"] = self.lmbda * out["mse_loss"] + out["bpp_loss"]
#         # out["loss"] = self.lmbda * (out["mse_loss"]+ out["adaptive_loss"]) /2 + out["bpp_loss"]
        
#         return out

# def configure_optimizers(net, args, other_parms):
#     """Separate parameters for the main optimizer and the auxiliary optimizer.
#     Return two optimizers"""

#     parameters = {n for n, p in net.named_parameters() if ".quantiles" not in n and p.requires_grad}
#     aux_parameters = {n for n, p in net.named_parameters() if ".quantiles" in n and p.requires_grad}

#     # print(aux_parameters)  # {'module.entropy_bottleneck_z.quantiles'}

#     params_dict = dict(net.named_parameters())

#     optimizer = optim.Adam(
#         list((params_dict[n] for n in sorted(parameters))) + other_parms,
#         lr=args.ft_comp_learning_rate,
#     )
#     aux_optimizer = optim.Adam(
#         (params_dict[n] for n in sorted(aux_parameters)),
#         lr=args.ft_comp_aux_learning_rate,
#     )
#     return optimizer, aux_optimizer

# def fine_tune_comp_model_v3(W, HR, comp_model, args, **kwargs):
    
#     ft_result = defaultdict(list)
#     ft_result['best_loss_epoch'] = []

#     qlevel = kwargs.get('qlevel', None)
#     # start test
#     mse_fn = nn.MSELoss()
#     out = comp_W(W, comp_model, args, qlevel = qlevel)    
    
#     W_hat = out['W_hat']
#     base_err = torch.trace((W - W_hat) @ HR @ ((W - W_hat).T)).item()
#     trWHW = torch.trace(W @ HR @ W.T).item()
#     base_proxy_err =  base_err / trWHW
#     base_bpp_loss = out['bpp_loss_sum']/out['num_pixels']
#     base_mse = mse_fn(W, W_hat).item()
#     # print(f'Before LayerFT {args.layer_idx}_{args.layer_name} | proxy err {base_proxy_err.item()} err {base_err.item()} tr(WHW.T) {trWHW.item()}')    
#     # print(f"bpp_loss :{base_bpp_loss:.3f}")
#     del out
    
#     ft_result['base_proxy_err'] = base_proxy_err
#     ft_result['base_err'] = base_err
#     ft_result['trWHW'] = trWHW
#     ft_result['base_bpp_loss'] = base_bpp_loss
#     ft_result['base_mse'] = base_mse
#     # print(ft_result['base_bpp_loss'])
#     # print(type(ft_result['base_bpp_loss']))
    
#     new_comp_model = NWC_without_encoder(comp_model.input_size,
#                                             comp_model.dim_encoder,
#                                             comp_model.n_resblock,
#                                             comp_model.input_size,
#                                             comp_model.scale,
#                                             comp_model.shift
#                                             )
#     new_comp_model.load_state_dict(comp_model.state_dict())
#     new_comp_model.to(W.device)
#     latent  = encode_latent(W, new_comp_model, args, qlevel = qlevel)
#     code_latent = nn.Parameter(latent, requires_grad=True)
#     # code_latent = nn.Parameter(torch.zeros(W.shape, device=device), requires_grad=True)
    
#     # args.direction = 'row'
#     assert args.direction == 'row'
#     comp_model = new_comp_model
#     comp_model.train()
    
#     for param in comp_model.parameters():
#         param.requires_grad = True
#     for param in comp_model.g_s.parameters():
#         param.requires_grad = args.ft_train_dec    

#     loss_fn = RateDistortionLoss(std=comp_model.scale.mean(), Hr=HR, lmbda=args.ft_comp_lmbda)

#     bs = 4096*1024 // W.shape[1]
#     step = 0
#     optimizer, aux_optimizer = configure_optimizers(comp_model, args, [code_latent])
    
#     best_loss = float("inf")
#     best_state_dict = copy.deepcopy(comp_model.state_dict())
#     best_code = code_latent.detach().clone()
#     best_loss_epoch = 0
    
#     dataset = TensorDataset(W, code_latent)
#     loader = DataLoader(dataset, batch_size=bs, shuffle=False, drop_last=False)
    
#     total_samples = W.shape[0]
    
#     wandb.init(project="NWC_layerwise_ft3", name=f"{args.layer_idx}_{args.layer_name}_trdec{args.ft_train_dec}", config=vars(args))
#     with torch.enable_grad():
#         # with tqdm(range(args.ft_comp_ep), desc=f"{args.layer_idx}_{args.layer_name}_{W.shape}_bs{bs}") as pbar:\
#             # for epoch in pbar: 
#         for epoch in range(10000):
#             if step >= args.ft_comp_steps:
#                 break     
           
#             num_pixels = 0
#             bpp_loss_total = 0
#             adaptive_loss_total = 0
#             loss_total = 0
#             mse_total = 0
#             W_hat = torch.zeros_like(W)

#             start_idx = 0
#             for w_batch, code_batch in loader:
#                 optimizer.zero_grad()
#                 aux_optimizer.zero_grad()                                      
#                 x_hat, n_pixels, bpp_loss_, out, out_enc, bpp = model_foward(
#                     code_batch,
#                     comp_model,
#                     args
#                 )

#                 num_pixels += n_pixels
#                 bpp_loss_total += bpp_loss_
            
#                 loss = loss_fn(w_batch, out)
#                 loss['loss'].backward()
                
#                 optimizer.step()
#                 try:
#                     aux_loss = comp_model.aux_loss()
#                 except:
#                     aux_loss = comp_model.module.aux_loss()
                    
#                 aux_loss.backward()
#                 aux_optimizer.step()                
                
#                 ft_result['loss'].append(loss['loss'].item())
#                 ft_result['adaptive_loss'].append(loss['adaptive_loss'].item())
#                 ft_result['bpp_loss'].append(loss['bpp_loss'].item())
#                 ft_result['mse_loss'].append(loss['mse_loss'].item())
#                 ft_result['step'].append(step)
                
#                 batch_size = w_batch.shape[0]                        
#                 mse_total += loss['mse_loss'].item() * batch_size  # ← 배치 크기 반영
#                 adaptive_loss_total += loss['adaptive_loss'].item() * batch_size  # ← 배치 크기 반영
#                 loss_total += loss['loss'].item() * batch_size  
                
#                 step += 1
#                 end_idx = min(start_idx + bs, W.shape[0])
#                 W_hat[start_idx:end_idx] = x_hat.detach()
#                 start_idx += batch_size

#             bpp_loss_epoch = bpp_loss_total / num_pixels
#             adaptive_loss_per_epoch = adaptive_loss_total / total_samples
#             mse_loss_per_epoch = mse_total / total_samples
#             loss_per_epoch = loss_total / total_samples

#             if loss_per_epoch < best_loss:
#                 best_loss = loss_per_epoch
#                 best_state_dict = copy.deepcopy(comp_model.state_dict())
#                 best_code = code_latent.detach().clone()
#                 best_loss_epoch = epoch

#             # pbar.set_postfix(
#             #     epoch=epoch,
#             #     loss=f"{loss_per_epoch:.4f}",
#             #     adaptive=f"{adaptive_loss_per_epoch:.4f}",
#             #     bpp=f"{bpp_loss_epoch:.4f}",
#             #     mse=f"{mse_loss_per_epoch:.4f}",
#             # )
#             # print(f"epoch {epoch}")
#             print(f"{args.layer_idx}_{args.layer_name}_{W.shape}_bs{bs} | step {step} / {args.ft_comp_steps}")
#             # print(f"loss {loss_per_epoch:.4f}")
#             # print(f"adaptive_loss {adaptive_loss_per_epoch:.4f}")
#             # print(f"bpp_loss {bpp_loss_epoch:.4f}")
#             # print(f"mse_loss {mse_loss_per_epoch:.4f}")

#             with torch.no_grad():
#                 err = torch.trace((W - W_hat) @ HR @ ((W - W_hat).T)).item()
#                 # trWHW = torch.trace(W @ HR @ W.T)
#                 proxy_err =  err / trWHW
#                 mse = mse_fn(W, W_hat).item()

#             ft_result['epoch'].append(epoch)
#             ft_result['loss_per_epoch'].append(loss_per_epoch)
#             ft_result['adaptive_loss_per_epoch'].append(adaptive_loss_per_epoch)
#             ft_result['bpp_loss_per_epoch'].append(bpp_loss_epoch)
#             ft_result['mse_loss_per_epoch'].append(mse_loss_per_epoch)
#             ft_result['best_loss_epoch'].append(best_loss_epoch)

#             ft_result['proxy_err'].append(proxy_err)
#             ft_result['err'].append(err)
#             ft_result['mse'].append(mse)
            
#             wandb.log({
#                 "epoch": epoch,
#                 "loss": loss_per_epoch,
#                 "best_loss": best_loss,
#                 "adaptive_loss": adaptive_loss_per_epoch,
#                 "bpp_loss": bpp_loss_epoch,
#                 "mse_loss": mse_loss_per_epoch,
#                 "proxy_err": proxy_err,
#                 "err":err,
#                 "mse":mse,
#                 "trWHW":trWHW,
#                 "base_proxy_err": base_proxy_err,
#                 "base_err":base_err,
#                 "base_bpp_loss":base_bpp_loss,
#                 "base_mse":base_mse,
#             })

#     wandb.finish()
    
#     comp_model.load_state_dict(best_state_dict)
#     print('best_code_latent: ', best_code.mean().item(), best_code.max().item(),best_code.min().item())

#     comp_model.eval()
#     comp_model.update()

#     out = comp_W(best_code, comp_model, args)

#     return out, ft_result