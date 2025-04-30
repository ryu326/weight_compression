import torch
import math
# import utils
from lib import utils
import os
from lib.algo import quip
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
from NWC.models.nwc_ql import NWC_ql_without_encoder
from NWC.models.nwc import NWC_without_encoder

def hessian_weighted_loss(W, W_hat, H):
    diff = W_hat - W
    H = H.float()
    trace_H = H.trace()
    if trace_H > 0:
        H = H / trace_H * H.shape[0] / W.numel()
    loss = torch.trace(diff @ H @ diff.T)     # scalar
    return loss

class RateDistortionLoss(nn.Module):
    def __init__(self, std, Hr, lmbda):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.std = std

        def clip_outliers_quantile_global(tensor, lower_q=0.03, upper_q=0.97):
            lower = torch.quantile(tensor, lower_q)
            upper = torch.quantile(tensor, upper_q)
            return torch.clip(tensor, min=lower.item(), max=upper.item())
        
        self.Hr = clip_outliers_quantile_global(Hr)

    def forward(self, ori_W, output, ori_shape):        
        out = {}
        num_pixels = output["x"].numel()
        # H = self.Hr[start_idx:end_idx][start_idx:end_idx]
        out["mse_loss"] = self.mse(ori_W,  output["x_hat"].reshape(ori_shape)) / self.std**2
        out["adaptive_loss"] = hessian_weighted_loss(ori_W.reshape(ori_shape), output["x_hat"].reshape(ori_shape), self.Hr) / self.std**2

        if isinstance(output["likelihoods"], dict):
            out["bpp_loss"] = sum(
                (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                for likelihoods in output["likelihoods"].values()
            )
        else :
            out["bpp_loss"] = (torch.log(output["likelihoods"]).sum() / (-math.log(2) * num_pixels))


        # out["loss"] = self.lmbda * out["adaptive_loss"] + out["bpp_loss"]
        # out["loss"] = self.lmbda * out["mse_loss"] + out["bpp_loss"]
        out["loss"] = self.lmbda * (out["mse_loss"]+ out["adaptive_loss"]) /2 + out["bpp_loss"]
        
        return out

def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {n for n, p in net.named_parameters() if ".quantiles" not in n and p.requires_grad}
    aux_parameters = {n for n, p in net.named_parameters() if ".quantiles" in n and p.requires_grad}

    # print(aux_parameters)  # {'module.entropy_bottleneck_z.quantiles'}

    params_dict = dict(net.named_parameters())

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.ft_comp_learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.ft_comp_aux_learning_rate,
    )
    return optimizer, aux_optimizer

def fine_tune_comp_model_v2(comp_model, W, HR, q_level, lstats, args, device):

    # start test
    out = pseudo_compress_tensor(W, comp_model, args.direction, args.comp_batch_size, q_level, None, lstats, device, args)
    W_hat = out['w_hat']
    
    err = torch.trace((W - W_hat) @ HR @ ((W - W_hat).T))
    trWHW = torch.trace(W @ HR @ W.T)
    proxy_err =  err / trWHW

    print(
        f'Before LayerFT {args.layer_idx}_{args.layer_name} | proxy err {proxy_err.item()} err {err.item()} tr(WHW.T) {trWHW.item()}'
    )    
    print(f"bpp_loss {out['bpp_loss_sum']/out['num_pixels']:.3f}")

    new_comp_model = NWC_without_encoder(comp_model.input_size,
                                            comp_model.dim_encoder,
                                            comp_model.n_resblock,
                                            comp_model.input_size,
                                            comp_model.scale,
                                            comp_model.shift
                                            )
    args.direction = 'row'
    comp_model = new_comp_model
    comp_model.train()
    comp_model.to(device)
    
    for param in comp_model.parameters():
        param.requires_grad = True
    for param in comp_model.g_s.parameters():
        param.requires_grad = True
    
    HR = HR.to(device)

    # mse_fn = torch.nn.MSELoss()
    loss_fn = RateDistortionLoss(std=comp_model.scale.mean(), Hr=HR, lmbda=args.ft_comp_lmbda)

    bs = 4096*1024 // W.shape[1]
    step = 0
    ft_result = defaultdict(list)
    ft_result['best_loss_epoch'] = []
    pe = nn.Parameter(torch.zeros(W.shape, device=device), requires_grad=True)
    comp_model.register_parameter("pe", pe)

    optimizer, aux_optimizer = configure_optimizers(comp_model, args)
    
    best_loss = float("inf")
    best_state_dict = copy.deepcopy(comp_model.state_dict())
    # best_pe  = comp_model.pe.detach().clone().cpu()
    assert 'pe' in best_state_dict.keys()
    best_loss_epoch = 0
    
    # if q_level is not None:
    #     # dataset = TensorDataset(W, q_level, comp_model.pe, comp_model.pe2)
    #     dataset = TensorDataset(W, q_level, comp_model.pe)
    # else:
        # dataset = TensorDataset(W, comp_model.pe, comp_model.pe2)
    dataset = TensorDataset(W, comp_model.pe)
    loader = DataLoader(dataset, batch_size=bs, shuffle=False, drop_last=False)
    
    total_samples = W.shape[0]
    with torch.enable_grad():
        with tqdm(range(args.ft_comp_ep), desc=f"{args.layer_idx}_{args.layer_name}_{W.shape}_bs{bs}") as pbar:
            for epoch in pbar:
        # for epoch in tqdm(range(args.ft_comp_ep), desc=f"{args.layer_idx}_{args.layer_name}_{W.shape}_bs{bs}") as pbar:
            
                num_pixels = 0
                bpp_loss_total = 0
                adaptive_loss_total = 0
                loss_total = 0
                mse_total = 0
                W_hat = torch.zeros_like(W)

                for batch in loader:
                    
                    optimizer.zero_grad()
                    aux_optimizer.zero_grad()            
                    
                    # if q_level is not None:
                    #     w_batch, ql_batch, pe_batch = batch
                    # else:
                    w_batch, pe_batch = batch
                    ql_batch = None
                    
                    x_hat, n_pixels, bpp_loss_, out = compress_weight_block_with_model(
                        pe_batch.to(device),
                        comp_model,
                        ql_batch.to(device) if ql_batch is not None else None,
                        lstats,
                        None,
                    )
                    ori_shape = w_batch.shape
                    
                    num_pixels += n_pixels
                    bpp_loss_total += bpp_loss_
                
                    loss = loss_fn(w_batch.to(device), out, ori_shape)
                    loss['loss'].backward()
                    
                    optimizer.step()
                    try:
                        aux_loss = comp_model.aux_loss()
                    except:
                        aux_loss = comp_model.module.aux_loss()
                        
                    aux_loss.backward()
                    aux_optimizer.step()

                    batch_size = w_batch.shape[0]                        
                    # MSE
                    mse_total += loss['mse_loss'].item() * batch_size  # ← 배치 크기 반영
                    # Adaptive Loss
                    adaptive_loss_total += loss['adaptive_loss'].item() * batch_size  # ← 배치 크기 반영
                    loss_total += loss['loss'].item() * batch_size            
                    
                    ft_result['loss'].append(loss['loss'].item())
                    ft_result['adaptive_loss'].append(loss['adaptive_loss'].item())
                    ft_result['bpp_loss'].append(loss['bpp_loss'].item())
                    ft_result['mse'].append(loss['mse_loss'].item())
                    ft_result['step'].append(step)
                    
                    step += 1
                    # W_hat[start_idx:end_idx] = out["x_hat"].detach().reshape(ori_shape)
                
                bpp_loss_epoch = bpp_loss_total / num_pixels
                adaptive_loss_per_epoch = adaptive_loss_total / total_samples
                mse_per_epoch = mse_total / total_samples
                loss_per_epoch = loss_total / total_samples
                # loss_per_epoch2 = args.ft_comp_lmbda * adaptive_loss_per_epoch + bpp_loss_epoch
                # assert math.isclose(loss_per_epoch, loss_per_epoch2, rel_tol=1e-6, abs_tol=1e-8)

                if loss_per_epoch < best_loss:
                    best_loss = loss_per_epoch
                    best_state_dict = copy.deepcopy(comp_model.state_dict())
                    # best_pe = comp_model.pe.detach().clone().cpu()
                    best_loss_epoch = epoch

                ft_result['epoch'].append(epoch)
                ft_result['loss_per_epoch'].append(loss_per_epoch)
                ft_result['adaptive_loss_per_epoch'].append(adaptive_loss_per_epoch)
                ft_result['bpp_loss_per_epoch'].append(bpp_loss_epoch)
                ft_result['mse_per_epoch'].append(mse_per_epoch)
                ft_result['best_loss_epoch'].append(best_loss_epoch)

                pbar.set_postfix(
                    epoch=ft_result['epoch'][-1],
                    loss=f"{ft_result['loss_per_epoch'][-1]:.4f}",
                    adaptive=f"{ft_result['adaptive_loss_per_epoch'][-1]:.4f}",
                    bpp=f"{ft_result['bpp_loss_per_epoch'][-1]:.4f}",
                    mse=f"{ft_result['mse_per_epoch'][-1]:.4f}",
                )
    comp_model.load_state_dict(best_state_dict)
    print('pe: ', comp_model.pe.mean().item(), comp_model.pe.max().item(),comp_model.pe.min().item())
    assert torch.any(comp_model.pe != 0)

    comp_model.eval()
    comp_model.update()

    pe = comp_model.pe.detach().clone()
    out = pseudo_compress_tensor(pe, comp_model, args.direction, args.comp_batch_size, q_level, None, lstats, device, args)

    return out, ft_result

def compress_linear(W, H, comp_model, ql, args, device='cpu'):
    
    comp_model = comp_model.to(device)
    comp_model.scale = comp_model.scale.to(device)
    comp_model.shift = comp_model.shift.to(device)
    W = W.to(device)
    H = H.to(device)
    
    SU, SV, scaleWH = None, None, None
    if args.incoh_mode != 'none':
        Lhr, H, W, SU, SV, scaleWH = quip.incoherence_preprocess(H, W, args)

    if args.ql == True:
        assert ql == None
        assert comp_model.Q == 4
        top = np.array([0.1, 1, 10])
        qlevel = [3, 2, 1]
        in_norm = torch.diag(H)
        topk = (top * len(in_norm)/100).astype(int)
        ql = torch.zeros_like(in_norm, dtype=torch.int32)
        _, topk_indices = torch.topk(in_norm, k=topk.sum())
        start = 0    
        for count, value in zip(topk , qlevel):
            indices = topk_indices[start:start + count]
            ql[indices] = value
            start += count
        # for test clip
        # ql = torch.full_like(in_norm, 3, dtype=torch.int32)
        
        
    if args.ql_invH == True:
        assert ql == None
        assert comp_model.Q == 4
        Lhr = torch.linalg.cholesky(H)
        H_inv = torch.cholesky_inverse(Lhr)
        top = np.array([0.1, 1, 10])
        qlevel = [3, 2, 1]
        diag = torch.diag(H_inv)
        topk = (top * len(diag)/100).astype(int)
        ql = torch.zeros_like(diag, dtype=torch.int32)
        _, topk_indices = torch.topk(diag, k=topk.sum(), largest=False)
        start = 0    
        for count, value in zip(topk , qlevel):
            indices = topk_indices[start:start + count]
            ql[indices] = value
            start += count
        
    lstats = None
    if args.layerwise_cdt == True:
        Wstats = describe_distribution(W)
        stat_keys = ["mean", "median", "std", "range", "iqr", "skewness", "kurtosis"]
        lstats = torch.tensor([Wstats[key] for key in stat_keys]).to(device)
        
    if args.ql_tuned:
        if args.layer_name == 'v':
            ql = torch.full_like(ql, 3)
        if args.layer_name == 'o':
            ql = torch.max(ql, torch.tensor(1))    
        if args.layer_idx == 0:
            ql = torch.max(ql, torch.tensor(1))

    if args.ql_search:
        ql_search_layer_idx = list(map(int, args.ql_search_layer_idx.split(',')))
        ql_search_layer_name = args.ql_search_layer_name.split(',')
        assert args.ql
        if args.layer_name in ql_search_layer_name and args.layer_idx in ql_search_layer_idx:
            ql = torch.full_like(ql, args.ql_search_value)    

    ql = ql.to(device) if ql is not None else None
    
    ft_result = None
    # comp_model_ft = copy.deepcopy(comp_model)
    # if args.ft_comp_model and args.layer_name in ['v', 'o', 'k', 'q']:
    # # if args.ft_comp_model and args.layer_name in ['v', 'o', 'k', 'q']:
    #     print(args.layer_name)
    #     assert args.direction == 'row'
    #     with torch.enable_grad():
    #         comp_model_ft, ft_result = fine_tune_comp_model(comp_model_ft, W, H, ql, lstats, args, device)
    
    # comp_model_ft.eval()
    # comp_model_ft.update()
    

    if args.ldlq:
        out = pseudo_compress_tensor_ldlq(W, comp_model, args.direction, args.comp_batch_size, ql, H, lstats, device, args)
    elif args.ft_comp_model2:
        out, ft_result = fine_tune_comp_model_v2(comp_model, W, H, ql, lstats, args, device)
    else:
        out = pseudo_compress_tensor(W, comp_model, args.direction, args.comp_batch_size, ql, None, lstats, device, args)   
    
    if args.incoh_mode != 'none':
        out['w_hat'] = quip.incoherence_process(out['w_hat'], SU, SV, scaleWH, args)
        
    utils.clean()
    return out['w_hat'].cpu(), out['bpp_loss_sum'], out['num_pixels'], SU, SV, scaleWH, ft_result

def compress_weight_block_with_model(weight_block, model, ql=None, lstats = None, pe = None, pe2 = None):
    ori_shape = weight_block.shape
    if ori_shape[-1] != model.input_size:
        weight_block = weight_block.reshape(ori_shape[0], -1, model.input_size)
        
    data = {}
    data['weight_block'] = weight_block
    if ql is not None:
        data['q_level'] = ql.reshape(ori_shape[0], 1)
    if lstats is not None:
        data['l_cdt'] = lstats.unsqueeze(0).repeat(ori_shape[0], 1)
    if pe is not None:
        data['pe'] = pe.reshape(ori_shape[0], -1, model.input_size)
    if pe2 is not None:
        data['pe2'] = pe2.reshape(ori_shape[0], -1, model.input_size)
        
    out = model(data)
    
    w_hat = out['x_hat'].reshape(ori_shape)

    num_pixels = weight_block.numel()
    if isinstance(out["likelihoods"], dict):
        bpp_loss = sum(
            (torch.log(likelihoods).sum() / -math.log(2))
            for likelihoods in out["likelihoods"].values()
        ).item()
    else :
        bpp_loss = (torch.log(out["likelihoods"]).sum() / -math.log(2)).item()

        # out_enc = model.compress(data)
    # try:
    #     out_dec = model.decompress(out_enc["strings"][0], out_enc["shape"], data["q_level"])
    # except:
    #     out_dec = model.decompress(out_enc["strings"][0], out_enc["shape"])
    
    # for s in out_enc["strings"]:
    #     bpp += len(s[0]) * 8.0


    return w_hat, num_pixels, bpp_loss, out
    
def pseudo_compress_tensor(w, model, direction, bs, q_level, hess_eigen, lstats, device, args):
    if direction == 'col':
        w = w.T
    ori_shape = w.shape
    
    if args.bundle:
        w = w.reshape(ori_shape[0], -1, model.input_size)  # (row, col) --> (row, -1, inputsize)
    else :
        w = w.reshape(-1, model.input_size)
        
    w_hat = torch.zeros(w.shape, dtype=w.dtype, device=w.device)

    num_pixels = 0
    bpp_loss = 0
    bpp = 0

    if q_level is not None:
        q_level = q_level.reshape(-1, )

    if hess_eigen is not None:
        R = hess_eigen['eigenvectors'].size(0)
        hess_eigen = hess_eigen['eigenvectors'].to(device)
        hess_eigen = hess_eigen.reshape(1, R, -1, model.input_size).repeat(bs, 1, 1, 1)

    for start_idx in range(0, w.shape[0], bs):
        end_idx = min(start_idx + bs, w.shape[0])
        batch = w[start_idx:end_idx]
        
        ql = None
        if q_level is not None:
            ql = q_level[start_idx:end_idx]

        pe_batch = None
        pe2_batch = None
        # if hasattr(model, 'pe'):
        #     pe_batch = model.pe[start_idx:end_idx].to(device)
        # if hasattr(model, 'pe2'):
        #     pe2_batch = model.pe2[start_idx:end_idx].to(device)

        x_hat, n_pixels, bpp_loss_, out = compress_weight_block_with_model(batch, model, ql, lstats, pe_batch, pe2_batch)
        
        w_hat[start_idx:end_idx] = x_hat
        num_pixels += n_pixels
        bpp_loss += bpp_loss_

    w_hat = w_hat.reshape(ori_shape)

    if direction == 'col':
        w_hat = w_hat.T
    
    return {'w_hat': w_hat,
            'bpp_loss_sum': bpp_loss,
            'num_pixels': num_pixels,
            'bpp': bpp}


#     # if args.gptq:
#     #     out = pseudo_compress_tensor_gptq(W, comp_model, args.direction, args.comp_batch_size, ql, H, device, args)     
# def pseudo_compress_tensor_gptq(W, model, direction, bs, q_level, H, device, args):
    
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
#             x_hat, n_pixels, bpp_loss_ = compress_weight_block_with_model(w_reshape, model, ql)
            
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

#     return {'w_hat': Q.cpu(),
#             'bpp_loss_sum': bpp_loss,
#             'num_pixels': num_pixels,
#             'bpp': bpp}

def pseudo_compress_tensor_ldlq(Wr, model, direction, bs, q_level, H, lstats, device, args):
    assert direction == 'col'
    
    L, D = block_LDL(H, 128)
    
    '''
    want eta = (Wr - hatWr) @ L
    want hatWr + eta = Wr + (Wr - hatWr) @ (L - I)
    want hatWr = Q( Wr + (Wr - hatWr) @ (L - I) )
    '''
    (m, n) = Wr.shape
    hatWr = torch.zeros_like(Wr, dtype=Wr.dtype, device=Wr.device)

    num_pixels = 0
    bpp_loss = 0
    bpp = 0

    for k in reversed(range(n)):
        WXWX = Wr[:, k:k+1] + (Wr[:, k+1:] - hatWr[:, k+1:]) @ L[k+1:, k:k+1]
    
        ql = None if q_level is None else q_level[k]
        pe_batch = model.pe[k].to(Wr.device) if hasattr(model, 'pe') else None
        pe2_batch = model.pe2[k].to(Wr.device) if hasattr(model, 'pe2') else None
                
        x_hat, n_pixels, bpp_loss_, out = compress_weight_block_with_model(WXWX.reshape(1, -1, model.input_size), model, ql, pe_batch, pe2_batch, lstats)
        
        q = x_hat.flatten()
        hatWr[:, k] = q
        num_pixels += n_pixels
        bpp_loss += bpp_loss_

    for ie in range(args.quip_tune_iters):
        raise Exception
        num_pixels = 0
        bpp_loss = 0
        bpp = 0
        
        for k in reversed(range(n)):
            WXWX = hatWr[:, k:k+1] + (Wr - hatWr) @ H[:, k:k+1] @ torch.linalg.inv(H[k:k+1, k:k+1])
            
            ql_ = None if q_level is None else q_level[i]
            x_hat, n_pixels, bpp_loss_ = compress_weight_block_with_model(WXWX.reshape(1, -1, model.input_size), model, ql)
            
            q = x_hat.flatten()
            hatWr[:, k] = q
            
            if ie == args.quip_tune_iters - 1:
                num_pixels += n_pixels
                bpp_loss += bpp_loss_

    return {'w_hat': hatWr.cpu(),
            'bpp_loss_sum': bpp_loss,
            'num_pixels': num_pixels,
            'bpp': bpp}


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
