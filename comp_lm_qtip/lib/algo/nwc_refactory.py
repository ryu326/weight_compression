import torch
import math
# import utils
from lib import utils
import os
# from lib.algo import quip
# from lib.algo import code_optimize
# from lib.algo import optimize_qmap
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm
import copy
from torch.utils.data import TensorDataset, DataLoader
import sys
sys.path.append('/home/jgryu/workspace/weight_compression')
import wandb
from NWC.loss import get_loss_fn
import glog
import time

def compress_linear(W, H, comp_model, Qlevel, args, device='cpu'):

    # comp_model = comp_model.to(device)
    comp_model = copy.deepcopy(comp_model).to(device)
    comp_model.scale = comp_model.scale.to(device)
    comp_model.shift = comp_model.shift.to(device)
    
    W = W.to(device)
    H = H.to(device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()
    
    Wr, Hr, metadata = utils.standardize_W(W, H, args, device)
    if args.col_normalize or args.row_normalize or args.row_normalize2 or args.layer_normalize:
        # comp_model.scale = torch.tensor(1).to(device)
        # comp_model.shift = torch.tensor(0).to(device)
        try:
            comp_model.scale.copy_(torch.tensor(1))
            comp_model.shift.copy_(torch.tensor(0))
        except:
            pass
    if Qlevel is None:
        if args.ql or args.ql_invH:
            Qlevel = utils.get_ql_from_H(Hr, comp_model, args).to(device)
            if args.ql_search:
                assert torch.all(Qlevel == Qlevel[0]), "Qlevel의 모든 값이 동일하지 않습니다."
    
    metadata['qlevel'] = Qlevel  

    if args.perlayer_ft_epochs > 0:
        # Pass metadata (scale_cond, qlevel) to helper
        with torch.enable_grad():
            comp_model = fine_tune_comp_model(Wr, Hr, comp_model, args, device=device, **metadata)

    res = comp_W(Wr, Hr, comp_model, args, **metadata)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.time()
    elapsed_time = end_time - start_time
    glog.info(f"Total Compression Time: {elapsed_time*1000:.4f} ms")
    
    assert torch.isnan(res['hatWr']).any() == False
    
    # if args.fp_iter:
    #     U, D = utils.compute_U_from_H(Hr)
    #     W_hat_prev = res['hatWr']
    #     for _ in range(args.fp_iter_max):
    #         W_in = Wr + (Wr - W_hat_prev) @ U
    #         res_ = comp_W(W_in, Hr, comp_model, args, **metadata)
    #         W_hat = res_['hatWr']
    #         # if torch.norm(W_hat - W_hat_prev) < args.fp_tol:
    #         #     break
    #         glog.info(f'{args.layer_idx}_{args.layer_name} Step {_}, Convergence error: {torch.norm(W_hat - W_hat_prev)}')
    #         W_hat_prev = W_hat
    #     res = res_

    res['metadata'] = metadata

    total_metadata_bpp = utils.calculate_metadata_bpp(metadata, W.shape, args)
    # bpp_keys = ['bpp_loss_sum', 'bpp_loss_sum_init', 'bpp_loss_sum_sga', 'bpp_loss_sum_round', 'bpp_sum']
    bpp_keys = ['bpp_loss_sum', 'bpp_sum']
    for key in bpp_keys:
        if res.get(key) is not None:
            res[key] += total_metadata_bpp
    
    utils.clean()
    return res
    
def comp_W(W, H, model, args, **kwargs):    
    bs = min(W.shape[1], 4096*4096 // W.shape[0]) if args.comp_batch_size == -1 else args.comp_batch_size
    (m, n) = W.shape
    glog.info(f'{W.shape}')
    W_hat = torch.zeros_like(W)
    W_ldl = torch.zeros_like(W) if args.ldlq else None
    num_pixels = 0
    bpp_loss_sum = 0
    bpp_sum = 0
    codes = []
    y_list = [] if args.ft_y else None    
    
    # row_norm = kwargs.get('row_norm', None)  # (m, 1)
    # col_norm = kwargs.get('col_norm', None)  # (1, n)

    scale_cond = kwargs.get('scale_cond', None)  # (1, n)
    
    qlevel = kwargs.get('qlevel', None)
    qlevel = qlevel.reshape(W.shape[1], ) if qlevel is not None else None
    
    y_in_list = kwargs.get('y_in_list', None)
    
    if args.ldlq:
        bs = 128 if args.comp_batch_size == -1 else args.comp_batch_size
        ldl_bs = 128 if args.comp_batch_size < 128 else args.comp_batch_size
        L, D = block_LDL(H, ldl_bs)
        assert n % bs == 0
    
    for i,e in enumerate(range(n, 0, -bs)):
        s = max(0, e - bs)
        if args.ldlq:
            w = W[:, s:e] + (W[:, e:] - W_hat[:, e:]) @ L[e:, s:e]
            W_ldl[:, s:e] = w
        else:
            w = W[:, s:e]        
        
        ql = qlevel[s:e] if qlevel is not None else None
        sc = scale_cond[:, s:e] if scale_cond is not None else None
 
        x_hat, n_pixels, bpp_loss_, out, out_enc, nbits, in_data = model_foward_one_batch(w.clone(), model, args, ql = ql, sc = sc)
        if args.ft_y:
            y_list.append((out['y'], s, e))

        codes.append(out_enc)
        bpp_sum += nbits
        W_hat[:, s:e] = x_hat
        num_pixels += n_pixels
        bpp_loss_sum += bpp_loss_

    return {'hatWr': W_hat,
            'Wr_ldlq': W_ldl,
            'bpp_loss_sum': bpp_loss_sum.item(),
            'bpp_loss': bpp_loss_sum.item() / num_pixels,
            'num_pixels': num_pixels,
            'bpp_sum': bpp_sum,
            'bpp': bpp_sum / num_pixels,
            'codes': codes,
            'bpp_loss_for_train': bpp_loss_sum / num_pixels,
            'y_list': y_list,
            }   

def comp_W_from_y(Wshape, y_in_list, y_in_idx, model, args, **kwargs):    
    (m, n) = Wshape
    bs = min(n, 4096*4096 // m) if args.comp_batch_size == -1 else args.comp_batch_size
    W_hat = torch.zeros(m, n)
    num_pixels = 0
    bpp_loss_sum = 0
    bpp_sum = 0
    codes = []  

    scale_cond = kwargs.get('scale_cond', None)  # (1, n)
    
    qlevel = kwargs.get('qlevel', None)
    qlevel = qlevel.reshape(n, ) if qlevel is not None else None

    for (y_in, (s, e)) in zip(y_in_list, y_in_idx):
        ql = qlevel[s:e] if qlevel is not None else None

        sc = scale_cond[:, s:e] if scale_cond is not None else None
 
        x_hat, n_pixels, bpp_loss_, out, out_enc, nbits, in_data = model_foward_one_batch(None, model, args, ql = ql, sc = sc, y_in = y_in, shape = (m, e-s))

        codes.append(out_enc)
        bpp_sum += nbits
        W_hat[:, s:e] = x_hat
        num_pixels += n_pixels
        bpp_loss_sum += bpp_loss_
    
    return {'hatWr': W_hat,
            'bpp_loss_sum': bpp_loss_sum.item(),
            'bpp_loss': bpp_loss_sum.item() / num_pixels,
            'num_pixels': num_pixels,
            'bpp_sum': bpp_sum,
            'bpp': bpp_sum / num_pixels,
            'codes': codes,
            'bpp_loss_for_train': bpp_loss_sum / num_pixels,
            }   

def model_foward_one_batch(w, model, args, one_batch = True, **kwargs):
    y_in = kwargs.get('y_in', None)
    mode = kwargs.get('mode', 'init')
    ql = kwargs.get('ql', None)  # (n, )
    (m, n) = w.shape if w is not None else kwargs.get('shape', None)
    
    blks = model.input_size
    # assert (m if args.direction == 'col' else n) % blks == 0
    original_m, original_n = m, n
    pad_len = 0    
    if args.direction == 'col':
        # Col 방향일 때는 m(행)이 blks의 배수여야 함
        if m % blks != 0:
            pad_len = blks - (m % blks)
            # w shape: (m, n) -> pad bottom rows: (last_dim_left, last_dim_right, 2nd_last_left, 2nd_last_right)
            # (0, 0, 0, pad_len)
            if w is not None:
                w = F.pad(w, (0, 0, 0, pad_len))
            m = m + pad_len  # m 업데이트
    else:
        # Row 방향일 때는 n(열)이 blks의 배수여야 함
        if n % blks != 0:
            pad_len = blks - (n % blks)
            # w shape: (m, n) -> pad right cols
            if w is not None:
                w = F.pad(w, (0, pad_len))
            n = n + pad_len  # n 업데이트
    
    
    sc = kwargs.get('sc', None)  # (1, n)
    sc = sc.repeat(m, 1) if sc is not None else None #(m, n)
    
    if ql is not None:
        ql = ql.reshape(1, n).expand(m//blks, n)
    elif args.ql_search_value is not None:
        ql = torch.full((m//blks, n), args.ql_search_value, dtype=torch.int32, device=w.device)
    
    transpose = args.direction == 'col'
    w = w.T if (w is not None and transpose) else w
    ql = ql.T if (ql is not None and transpose) else ql
    # qm = qm.T if (qm is not None and transpose) else qm
    sc = sc.T if sc is not None and transpose else sc
    
    data = {}
    if w is not None:
        if one_batch:
            w = w.reshape(1, -1, blks)
        else:
            w = w.reshape(w.shape[0], -1, blks)
    data['weight_block'] = w
    # assert torch.isnan(w).any() == False
    
    if ql is not None:
        if one_batch:
            data['q_level'] = ql.reshape(1, m*n//blks)
        else:
            data['q_level'] = ql.reshape(ql.shape[0], m//blks)[:, 0:1]
    # if qm is not None:
    #     data['qmap'] = qm.reshape(1, w.shape[1])
    if hasattr(model, 'pe') and model.pe:
        wtype_mapping = {'q': 0, 'k': 1, 'v': 2, 'o': 3, 'gate': 4, 'up': 5, 'down': 6}
        depth = args.layer_idx
        ltype = wtype_mapping[args.layer_name]
        data['depth'] = torch.full((1, 1), depth, dtype=torch.long).to(w.device)
        data['ltype'] = torch.full((1, 1), ltype, dtype=torch.long).to(w.device)
    if sc is not None:
        if one_batch:
            sc = sc.reshape(1, -1, blks)
        else:
            sc =  sc.reshape(sc.shape[0], -1, blks)       
        if hasattr(model, 'scale_cond') and model.scale_cond:
            assert torch.all(sc == sc[..., :1])
            sc = sc[..., 0] #(1,  m*n//blks)
            assert sc.shape == (1, m*n//blks)
        data['scale_cond'] = sc
        
    num_pixels = m*n
    bpp_loss_sum = torch.tensor(0)
    nbits = 0
    out_enc = None
    out = None
    
    if args.use_codes:
        out_enc = model.compress(data)
        out_dec = model.decompress(out_enc)
        w_hat = out_dec['x_hat']
        # import ipdb; ipdb.set_trace()
        # import web_pdb; web_pdb.set_trace(host='0.0.0.0')
        for s in out_enc["strings"]:
            nbits += len(s[0]) * 8.0
    else:
        if hasattr(model, 'sga'):
            out = model(data, mode = mode, y_in = y_in)
        elif y_in is not None:
            out = model(data, y_in = y_in)
        else:
            out = model(data)
        w_hat = out['x_hat']
            
        if "likelihoods" in out:
            if isinstance(out["likelihoods"], dict):
                bpp_loss_sum = sum(
                    (torch.log(likelihoods).sum() / -math.log(2))
                    for likelihoods in out["likelihoods"].values()
                )
            else :
                bpp_loss_sum = (torch.log(out["likelihoods"]).sum() / -math.log(2))
        elif hasattr(model, 'bits'):
            nbits += model.bits * num_pixels
            
    # if args.direction == 'col':
    #     w_hat = w_hat.reshape(n, m).transpose(0, 1).contiguous()
    # else:
    #     w_hat = w_hat.reshape(m, n)
    
    # --- [수정 시작] Unpadding Logic ---
    if args.direction == 'col':
        # 현재 w_hat은 (n, m) 형태 (padding 포함)
        w_hat = w_hat.reshape(n, m).transpose(0, 1).contiguous()
        # 복원된 w_hat은 (m, n). 여기서 m은 패딩된 크기.
        if pad_len > 0:
            w_hat = w_hat[:original_m, :]
    else:
        w_hat = w_hat.reshape(m, n)
        if pad_len > 0:
            w_hat = w_hat[:, :original_n]
    # --- [수정 끝] ---

    # if args.use_codes:
    #     del out_dec['x_hat']
    # else:
    #     del out['x_hat']
    torch.cuda.empty_cache()
    
    return w_hat, num_pixels, bpp_loss_sum, out, out_enc, nbits, data

def block_LDL(H, b, check_nan=True):
    n = H.shape[0]
    assert (n % b == 0)
    m = n // b
    # try:
    #     L = torch.linalg.cholesky(H)
    # except:
    #     return None
    L = torch.linalg.cholesky(H)
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

def configure_optimizers(net, args, other_parms):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {n for n, p in net.named_parameters() if ".quantiles" not in n and p.requires_grad}
    aux_parameters = {n for n, p in net.named_parameters() if ".quantiles" in n and p.requires_grad}

    # print(aux_parameters)  # {'module.entropy_bottleneck_z.quantiles'}

    params_dict = dict(net.named_parameters())

    optimizer = optim.Adam(
        list((params_dict[n] for n in sorted(parameters))) + other_parms,
        lr=args.ft_comp_learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.ft_comp_aux_learning_rate,
    )
    # optimizer, aux_optimizer = None, None
    # code_optimizer = optim.Adam(
    #     other_parms,
    #     lr=args.code_optim_lr,
    # )
    return optimizer, aux_optimizer, None

def fine_tune_comp_model(Wr, Hr, comp_model, args, device, **metadata):
    """
    Fine-tunes the compression model on the standardized weight matrix Wr with Shuffling.
    """
    if args.perlayer_ft_epochs <= 0:
        return comp_model

    glog.info(f"Finetuning Comp Model for {args.perlayer_ft_epochs} epochs on layer {args.layer_name}...")
    
    comp_model.train()
    optimizer, aux_optimizer, _ = configure_optimizers(comp_model, args, [])
    
    train_std = Wr.std().item()
    criterion = get_loss_fn(args, std=train_std, device=device)

    bs = args.perlayer_ft_bs
    (m, n) = Wr.shape

    scale_cond = metadata.get('scale_cond', None)
    qlevel = metadata.get('qlevel', None)
    # qlevel shape 보정
    qlevel = qlevel.reshape(n, ) if qlevel is not None else None
    
    loop = range(args.perlayer_ft_epochs)
    if True:
        loop = tqdm(loop, desc="Finetuning")
            
    best_epoch_loss = float('inf')
    for epoch in loop:
        # [Shuffling Logic]
        # 매 Epoch마다 인덱스를 랜덤하게 섞습니다.
        # direction이 COL이면 n(열)을 섞고, ROW여도 데이터 로딩 방식에 따라 n(열) 방향으로 처리되므로 n을 섞습니다.
        # (앞선 코드에서 Wr을 Transpose해서 처리하는 로직이 있다면 그에 맞춰야 하지만, 
        #  보통 Loop가 n을 기준으로 돌기 때문에 n 차원을 섞습니다.)
        
        indices = torch.randperm(n, device=Wr.device)
        
        # 원본 데이터를 건드리지 않기 위해 셔플된 뷰 생성
        Wr_shuffled = Wr[:, indices]
        
        qlevel_shuffled = None
        if qlevel is not None:
            qlevel_shuffled = qlevel[indices]
            
        scale_cond_shuffled = None
        if scale_cond is not None:
            # scale_cond가 (1, n) 형태라고 가정
            scale_cond_shuffled = scale_cond[:, indices]

        epoch_loss_sum = 0.0
        epoch_recon_sum = 0.0
        epoch_bpp_sum = 0.0
        epoch_aux_sum = 0.0
        num_batches = 0

        for i, s in enumerate(range(0, n, bs)):
            e = min(s + bs, n)
            
            optimizer.zero_grad()
            aux_optimizer.zero_grad()
            
            w = Wr_shuffled[:, s:e]
            
            ql = qlevel_shuffled[s:e] if qlevel_shuffled is not None else None
            sc = scale_cond_shuffled[:, s:e] if scale_cond_shuffled is not None else None
            
            x_hat, n_pixels, bpp_loss_, out, out_enc, nbits, in_data = model_foward_one_batch(
                w.clone(), comp_model, args, one_batch=False, ql=ql, sc=sc
            )

            loss_out = criterion(data=in_data, output=out)
            loss = loss_out["loss"]
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(comp_model.parameters(), 1.0)
        
            optimizer.step()
        
            # Aux Loss (Entropy Model Update)
            aux_loss = comp_model.aux_loss()
            aux_loss.backward()
            aux_optimizer.step()

            # [Logging] 배치의 Loss 누적
            current_bs = e - s
            epoch_loss_sum += loss.item()
            epoch_recon_sum += loss_out.get('recon_loss', torch.tensor(0)).item()
            epoch_bpp_sum += loss_out.get('bpp_loss', torch.tensor(0)).item()
            epoch_aux_sum += aux_loss.item()
            num_batches += 1
            
        # [Logging] 에폭 종료 후 평균 계산 및 출력
        avg_loss = epoch_loss_sum / num_batches
        avg_recon = epoch_recon_sum / num_batches
        avg_bpp = epoch_bpp_sum / num_batches
        avg_aux = epoch_aux_sum / num_batches

        # Better / Worse 판단
        if avg_loss < best_epoch_loss:
            status = "Better"
            best_epoch_loss = avg_loss
        else:
            status = "Worse"

        # 매 에폭마다 glog 출력
        glog.info(
            f"Layer: {args.layer_name} | Epoch {epoch+1}/{args.perlayer_ft_epochs} "
            f"({100. * (epoch+1) / args.perlayer_ft_epochs:.1f}%): "
            f"[{status}] "
            f"\tLoss: {avg_loss:.6f}"
            f"\trecon_loss: {avg_recon:.6f}"
            f"\tbpp_loss: {avg_bpp:.6f}"
            f"\taux_loss: {avg_aux:.6f}"
        )

    comp_model.eval()
    comp_model.update()
    return comp_model

# def model_foward_one_batch2(w, model, args, **kwargs):
#     y_in = kwargs.get('y_in', None)
#     mode = kwargs.get('mode', 'init')
#     ql = kwargs.get('ql', None)  # (n, )
#     (m, n) = w.shape # (m, n), T안하고 들어오는 걸로 가정
#     blks = model.input_size
#     assert (m if args.direction == 'col' else n) % blks == 0
    
#     sc = kwargs.get('sc', None)  # (1, n)
#     sc = sc.repeat(m, 1) if sc is not None else None

#     if ql is not None:
#         ql = ql.reshape(1, n).expand(m//blks, n)
#     elif args.ql_search_value is not None:
#         ql = torch.full((m//blks, n), args.ql_search_value, dtype=torch.int32, device=w.device)
    
#     transpose = args.direction == 'col'
#     w = w.T if (w is not None and transpose) else w
#     ql = ql.T if (ql is not None and transpose) else ql
#     # qm = qm.T if (qm is not None and transpose) else qm
#     sc = sc.T if sc is not None and transpose else sc
    
#     data = {}
#     w = w.reshape(1, -1, blks)
#     data['weight_block'] = w
#     assert torch.isnan(w).any() == False
    
#     if ql is not None:
#         data['q_level'] = ql.reshape(1, w.shape[1])
#     # if qm is not None:
#     #     data['qmap'] = qm.reshape(1, w.shape[1])
#     if hasattr(model, 'pe') and model.pe:
#         wtype_mapping = {'q': 0, 'k': 1, 'v': 2, 'o': 3, 'gate': 4, 'up': 5, 'down': 6}
#         depth = args.layer_idx
#         ltype = wtype_mapping[args.layer_name]
#         data['depth'] = torch.full((1, 1), depth, dtype=torch.long).to(w.device)
#         data['ltype'] = torch.full((1, 1), ltype, dtype=torch.long).to(w.device)
#     if sc is not None:
#         data['scale_cond'] = sc.reshape(1, -1, blks)
        
#     num_pixels = m*n
#     bpp_loss_sum = torch.tensor(0)
#     nbits = 0
#     out_enc = None
#     out = None
    
#     if args.use_codes:
#         out_enc = model.compress(data)
#         out_dec = model.decompress(out_enc)
#         w_hat = out_dec['x_hat']
#         for s in out_enc["strings"]:
#             nbits += len(s[0]) * 8.0
#     else:
#         if hasattr(model, 'sga'):
#             out = model(data, mode = mode, y_in = y_in)
#         else:
#             out = model(data)
#         w_hat = out['x_hat']
            
#         if isinstance(out["likelihoods"], dict):
#             bpp_loss_sum = sum(
#                 (torch.log(likelihoods).sum() / -math.log(2))
#                 for likelihoods in out["likelihoods"].values()
#             )
#         else :
#             bpp_loss_sum = (torch.log(out["likelihoods"]).sum() / -math.log(2))
            
#     if args.direction == 'col':
#         w_hat = w_hat.reshape(n, m).transpose(0, 1).contiguous()
#     else:
#         w_hat = w_hat.reshape(m, n)

#     if args.use_codes:
#         del out_dec['x_hat']
#     else:
#         del out['x_hat']
#     torch.cuda.empty_cache()
    
#     return w_hat, num_pixels, bpp_loss_sum, out, out_enc, nbits

# def code_optimize(w, comp_model, init_out, args, **kwargs):
#     ql = kwargs.get('ql', None).reshape(w.shape[0], 1)
#     std = kwargs.get('std', None)
#     # mode = kwargs.get('mode', 'sga')
#     batch_idx = kwargs.get('batch_idx', -1)
#     ori_shape = w.shape
#     w = w.reshape(w.shape[0], -1, comp_model.input_size)
#     qs = kwargs.get('qs', None)
    
#     rnorm = kwargs.get('rnorm', None) # (1, m) 
#     cnorm = kwargs.get('cnorm', None) # (n, 1)
    
#     if args.optim_norm:
#         assert rnorm.dim() == 2 and cnorm.dim() == 2
#         assert rnorm.shape == (1, w.shape[1]) and cnorm.shape == (w.shape[0], 1)
#         assert cnorm == None or cnorm.dim() == 2 
#         assert cnorm == None or cnorm.shape == (w.shape[0], 1)
    
#     wandb.init(project=f"NWC_code_optim", name=f"{'_'.join(args.save_path.split('/')[-2:])}_{args.layer_idx}_{args.layer_name}_batch{batch_idx}", config=vars(args))

#     loss_fn =  get_loss_fn(args, std=std, device = w.device)

#     with torch.no_grad():
#         data = {'weight_block': w, 'q_level': ql}
#         init_loss = loss_fn(data, init_out)

#     y = init_out['y'].clone()
#     y = nn.Parameter(y, requires_grad=True)
    
#     tune_params = [y]
#     if rnorm is not None and args.optim_norm:
#         rnorm = nn.Parameter(rnorm, requires_grad=True)
#         tune_params.append(rnorm)
#     if cnorm is not None and args.optim_norm:
#         cnorm = nn.Parameter(cnorm, requires_grad=True)
#         tune_params.append(cnorm)
#     if args.optim_qs:
#         qs = nn.Parameter(qs, requires_grad=True)
#         tune_params.append(qs)
#     optimizer = optim.Adam(tune_params, lr=args.code_optim_lr)

#     # comp_model.train()
#     for param in comp_model.parameters():
#         param.requires_grad = False

#     best_loss = init_loss['loss'].item()
#     best_loss_bpp = init_loss['bpp_loss'].item()
#     best_loss_recon = init_loss['recon_loss'].item()
#     best_y = y.detach().clone()
#     best_rnorm = rnorm.detach().clone() if rnorm is not None else None
#     best_cnorm = cnorm.detach().clone() if cnorm is not None else None
#     best_w_hat = init_out['x_hat'].detach().clone()
#     best_qs = qs.detach().clone() if qs is not None else None
    
#     with torch.enable_grad():
#         for it in range(args.code_optim_it):
#             optimizer.zero_grad()
#             # qs = torch.clamp(qs, min=0.1)
#             qs.data.clamp_(min=0.1) if qs is not None else None
#             data = {'weight_block': None, 'q_level': ql}
#             out = comp_model(data, mode='sga', y_in = y, it = it, tot_it = args.code_optim_it, qs = qs)
#             data = {'weight_block': w, 'q_level': ql}
#             # loss = loss_fn(data, out, rnorm = rnorm, cnorm = cnorm)
#             loss = loss_fn(data, out)
#             loss['loss'].backward()
#             optimizer.step()
            
#             if loss['loss'].item() < best_loss:
#                 best_loss = loss['loss'].item()
#                 best_loss_bpp = loss['bpp_loss'].item()
#                 best_loss_recon = loss['recon_loss'].item()
#                 best_y = y.detach().clone()
#                 best_w_hat = out['x_hat'].detach().clone()
#                 best_rnorm = rnorm.detach().clone() if rnorm is not None else None
#                 best_cnorm = cnorm.detach().clone() if cnorm is not None else None
#                 best_qs = qs.detach().clone() if qs is not None else None
#             wandb.log({
#                 "loss": loss['loss'].item(),
#                 "bpp_loss": loss['bpp_loss'].item(),
#                 "recon_loss": loss['recon_loss'].item(),
#                 "it": it,
#                 "best_loss": best_loss,
#                 "best_loss_bpp": best_loss_bpp,
#                 "best_loss_recon": best_loss_recon,
#                 "init_loss": init_loss['loss'].item(),
#                 "init_bpp_loss": init_loss['bpp_loss'].item(),
#                 "init_recon_loss": init_loss['recon_loss'].item(),
#                 "best_qs_mean": best_qs.mean().item() if qs is not None else None,
#             })
            
#     glog.info(f'y == best_y :: {torch.equal(y, best_y)}')
#     glog.info(f'y == init_y :: {torch.equal(y, init_out["y"])}')
#     wandb.finish()

#     if isinstance(out["likelihoods"], dict):
#         bpp_loss = sum(
#             (torch.log(likelihoods).sum() / -math.log(2))
#             for likelihoods in out["likelihoods"].values()
#         ).item()
#     else :
#         bpp_loss = (torch.log(out["likelihoods"]).sum() / -math.log(2)).item()
#     num_pixels = ori_shape[0] * ori_shape[1]

#     return best_y, best_w_hat.reshape(ori_shape), bpp_loss, num_pixels, best_rnorm, best_cnorm, best_qs
    # return y, out['x_hat'].reshape(ori_shape)


# def describe_distribution(x):
#     assert isinstance(x, torch.Tensor), "Input must be a PyTorch tensor"
#     x = x.flatten().float()
#     n = x.numel()
    
#     # 중심 경향
#     mean = x.mean()
#     median = x.median()

#     # 산포도
#     std_dev = x.std(unbiased=False)
#     value_range = x.max() - x.min()
#     q1 = x.kthvalue(int(0.25 * n + 1)).values
#     q3 = x.kthvalue(int(0.75 * n + 1)).values
#     iqr = q3 - q1

#     # 모양
#     skewness = ((x - mean)**3).mean() / (std_dev**3)
#     kurtosis = ((x - mean)**4).mean() / (std_dev**4) - 3  # Fisher's definition

#     del x
#     return {
#         "mean": mean.item(),
#         "median": median.item(),
#         "std": std_dev.item(),
#         "range": value_range.item(),
#         "iqr": iqr.item(),
#         "skewness": skewness.item(),
#         "kurtosis": kurtosis.item()
#     }
