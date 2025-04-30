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
from NWC.models.nwc_ql import NWC_ql_without_encoder
from NWC.models.nwc import NWC_without_encoder
import wandb

def compress_linear(W, H, comp_model, Qlevel, args, device='cpu'):
    
    W = W.to(device)
    H = H.to(device)
    
    comp_model = comp_model.to(device)
    if args.layerwise_scale:
        comp_model.scale = W.std().to(device)
        comp_model.shift = W.mean().to(device)
    elif args.channelwise_scale:
        comp_model.scale = torch.tensor(1).to(device)
        comp_model.shift = torch.tensor(0).to(device)
        
        if args.direction == 'row':
            channel_std = W.std(dim=-1, keepdim=True)
        elif args.direction == 'col':
            channel_std = W.std(dim=0, keepdim=True)
        W = W / channel_std
    else:
        comp_model.scale = comp_model.scale.to(device)
        comp_model.shift = comp_model.shift.to(device)

    
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
        # for test clip
        # ql = torch.full_like(in_norm, 3, dtype=torch.int32)
        
        
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
        out = pseudo_compress_tensor_ldlq(W, H, comp_model, args, qlevel = Qlevel)
    elif args.ft_comp_model2:
        out, ft_result = fine_tune_comp_model_v3(W, H, comp_model, args, qlevel = Qlevel)
    elif args.optim_code:
        out, ft_result, optimize_out = code_optimize.optim_code_delta_cnndec(W, H, comp_model, args, qlevel = Qlevel)
    else:
        out = pseudo_compress_tensor(W, comp_model, args, qlevel = Qlevel)   
    
    if args.incoh_mode != 'none':
        out['W_hat'] = quip.incoherence_process(out['W_hat'], SU, SV, scaleWH, args)
    
    if args.channelwise_scale:
        out['W_hat'] = out['W_hat'] * channel_std
    
    utils.clean()
    return out['W_hat'].cpu(), out['bpp_loss_sum'], out['num_pixels'], SU, SV, scaleWH, ft_result, optimize_out

def model_input(w, model, **kwargs):
    ori_shape = w.shape
    w = w.reshape(ori_shape[0], -1, model.input_size)
        
    data = {}
    data['weight_block'] = w
    
    ql = kwargs.get('ql', None)
    if ql is not None:
        data['q_level'] = ql.reshape(ori_shape[0], 1)
        
    lstats = kwargs.get('lstats', None)
    if lstats is not None:
        data['l_cdt'] = lstats.unsqueeze(0).repeat(ori_shape[0], 1)
        
    out = model(data)
    
    w_hat = out['x_hat'].reshape(ori_shape)

    num_pixels = w.numel()
    if isinstance(out["likelihoods"], dict):
        bpp_loss = sum(
            (torch.log(likelihoods).sum() / -math.log(2))
            for likelihoods in out["likelihoods"].values()
        ).item()
    else :
        bpp_loss = (torch.log(out["likelihoods"]).sum() / -math.log(2)).item()

    return w_hat, num_pixels, bpp_loss, out
    
def pseudo_compress_tensor(W, model, args, **kwargs):
    if args.direction == 'col':
        W = W.T
    W_hat = torch.zeros_like(W)

    num_pixels = 0
    bpp_loss_sum = 0
    bpp = 0
    
    qlevel = kwargs.get('qlevel', None)
    qlevel = qlevel.reshape(W.shape[0], ) if qlevel is not None else None

    bs = args.comp_batch_size
    for start_idx in range(0, W.shape[0], bs):
        end_idx = min(start_idx + bs, W.shape[0])
        batch = W[start_idx:end_idx]
        ql = qlevel[start_idx:end_idx] if qlevel is not None else None

        x_hat, n_pixels, bpp_loss_, out = model_input(batch, model, ql = ql)
        
        W_hat[start_idx:end_idx] = x_hat
        num_pixels += n_pixels
        bpp_loss_sum += bpp_loss_

    if args.direction == 'col':
        W_hat = W_hat.T
    
    return {'W_hat': W_hat,
            'bpp_loss_sum': bpp_loss_sum,
            'num_pixels': num_pixels,
            'bpp': bpp}

# def pseudo_compress_tensor(W, model, args, **kwargs):
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


def pseudo_compress_tensor_ldlq(Wr, H, model, args, **kwargs):
# def pseudo_compress_tensor_ldlq(Wr, model, direction, bs, q_level, H, lstats, device, args):
    assert args.direction == 'col'
    
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
    qlevel = kwargs.get('qlevel', None)
    for k in reversed(range(n)):
        WXWX = Wr[:, k:k+1] + (Wr[:, k+1:] - hatWr[:, k+1:]) @ L[k+1:, k:k+1]
    
        ql = None if qlevel is None else qlevel[k]
                
        x_hat, n_pixels, bpp_loss_, out = model_input(WXWX.reshape(1, -1), model, ql = ql)
        
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
            x_hat, n_pixels, bpp_loss_ = model_input(WXWX.reshape(1, -1, model.input_size), model, ql)
            
            q = x_hat.flatten()
            hatWr[:, k] = q
            
            if ie == args.quip_tune_iters - 1:
                num_pixels += n_pixels
                bpp_loss += bpp_loss_

    return {'W_hat': hatWr,
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
#             x_hat, n_pixels, bpp_loss_ = model_input(w_reshape, model, ql)
            
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


def hessian_proxy_loss(W, W_hat, H):
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

        # def clip_outliers_quantile_global(tensor, lower_q=0.03, upper_q=0.97):
        #     lower = torch.quantile(tensor, lower_q)
        #     upper = torch.quantile(tensor, upper_q)
        #     return torch.clip(tensor, min=lower.item(), max=upper.item())
        
        # self.Hr = clip_outliers_quantile_global(Hr)
        self.Hr = Hr

    def forward(self, ori_w, output):        
        out = {}
        num_pixels = output["x"].numel()
        w_hat = output["x_hat"].reshape(ori_w.shape)
        # H = self.Hr[start_idx:end_idx][start_idx:end_idx]
        
        out["mse_loss"] = self.mse(ori_w,  w_hat) / self.std**2
        out["adaptive_loss"] = hessian_proxy_loss(ori_w, w_hat, self.Hr) / self.std**2

        if isinstance(output["likelihoods"], dict):
            out["bpp_loss"] = sum(
                (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                for likelihoods in output["likelihoods"].values()
            )
        else :
            out["bpp_loss"] = (torch.log(output["likelihoods"]).sum() / (-math.log(2) * num_pixels))


        # out["loss"] = self.lmbda * out["adaptive_loss"] + out["bpp_loss"]
        out["loss"] = self.lmbda * out["mse_loss"] + out["bpp_loss"]
        # out["loss"] = self.lmbda * (out["mse_loss"]+ out["adaptive_loss"]) /2 + out["bpp_loss"]
        
        return out

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
    return optimizer, aux_optimizer

def fine_tune_comp_model_v3(W, HR, comp_model, args, **kwargs):
    
    ft_result = defaultdict(list)
    ft_result['best_loss_epoch'] = []

    qlevel = kwargs.get('qlevel', None)
    # start test
    mse_fn = nn.MSELoss()
    out = pseudo_compress_tensor(W, comp_model, args, qlevel = qlevel)    
    
    W_hat = out['W_hat']
    base_err = torch.trace((W - W_hat) @ HR @ ((W - W_hat).T)).item()
    trWHW = torch.trace(W @ HR @ W.T).item()
    base_proxy_err =  base_err / trWHW
    base_bpp_loss = out['bpp_loss_sum']/out['num_pixels']
    base_mse = mse_fn(W, W_hat).item()
    # print(f'Before LayerFT {args.layer_idx}_{args.layer_name} | proxy err {base_proxy_err.item()} err {base_err.item()} tr(WHW.T) {trWHW.item()}')    
    # print(f"bpp_loss :{base_bpp_loss:.3f}")
    del out
    
    ft_result['base_proxy_err'] = base_proxy_err
    ft_result['base_err'] = base_err
    ft_result['trWHW'] = trWHW
    ft_result['base_bpp_loss'] = base_bpp_loss
    ft_result['base_mse'] = base_mse
    # print(ft_result['base_bpp_loss'])
    # print(type(ft_result['base_bpp_loss']))
    
    new_comp_model = NWC_without_encoder(comp_model.input_size,
                                            comp_model.dim_encoder,
                                            comp_model.n_resblock,
                                            comp_model.input_size,
                                            comp_model.scale,
                                            comp_model.shift
                                            )
    new_comp_model.load_state_dict(comp_model.state_dict())
    new_comp_model.to(W.device)
    latent  = encode_latent(W, new_comp_model, args, qlevel = qlevel)
    code_latent = nn.Parameter(latent, requires_grad=True)
    # code_latent = nn.Parameter(torch.zeros(W.shape, device=device), requires_grad=True)
    
    # args.direction = 'row'
    assert args.direction == 'row'
    comp_model = new_comp_model
    comp_model.train()
    
    for param in comp_model.parameters():
        param.requires_grad = True
    for param in comp_model.g_s.parameters():
        param.requires_grad = args.ft_train_dec    

    loss_fn = RateDistortionLoss(std=comp_model.scale.mean(), Hr=HR, lmbda=args.ft_comp_lmbda)

    bs = 4096*1024 // W.shape[1]
    step = 0
    optimizer, aux_optimizer = configure_optimizers(comp_model, args, [code_latent])
    
    best_loss = float("inf")
    best_state_dict = copy.deepcopy(comp_model.state_dict())
    best_code  = code_latent.detach().clone()
    best_loss_epoch = 0
    
    dataset = TensorDataset(W, code_latent)
    loader = DataLoader(dataset, batch_size=bs, shuffle=False, drop_last=False)
    
    total_samples = W.shape[0]
    
    wandb.init(project="NWC_layerwise_ft3", name=f"{args.layer_idx}_{args.layer_name}_trdec{args.ft_train_dec}", config=vars(args))
    with torch.enable_grad():
        # with tqdm(range(args.ft_comp_ep), desc=f"{args.layer_idx}_{args.layer_name}_{W.shape}_bs{bs}") as pbar:\
            # for epoch in pbar: 
        for epoch in range(10000):
            if step >= args.ft_comp_steps:
                break     
           
            num_pixels = 0
            bpp_loss_total = 0
            adaptive_loss_total = 0
            loss_total = 0
            mse_total = 0
            W_hat = torch.zeros_like(W)

            start_idx = 0
            for w_batch, code_batch in loader:
                optimizer.zero_grad()
                aux_optimizer.zero_grad()                                      
                x_hat, n_pixels, bpp_loss_, out = model_input(
                    code_batch,
                    comp_model,
                )

                num_pixels += n_pixels
                bpp_loss_total += bpp_loss_
            
                loss = loss_fn(w_batch, out)
                loss['loss'].backward()
                
                optimizer.step()
                try:
                    aux_loss = comp_model.aux_loss()
                except:
                    aux_loss = comp_model.module.aux_loss()
                    
                aux_loss.backward()
                aux_optimizer.step()                
                
                ft_result['loss'].append(loss['loss'].item())
                ft_result['adaptive_loss'].append(loss['adaptive_loss'].item())
                ft_result['bpp_loss'].append(loss['bpp_loss'].item())
                ft_result['mse_loss'].append(loss['mse_loss'].item())
                ft_result['step'].append(step)
                
                batch_size = w_batch.shape[0]                        
                mse_total += loss['mse_loss'].item() * batch_size  # ← 배치 크기 반영
                adaptive_loss_total += loss['adaptive_loss'].item() * batch_size  # ← 배치 크기 반영
                loss_total += loss['loss'].item() * batch_size  
                
                step += 1
                end_idx = min(start_idx + bs, W.shape[0])
                W_hat[start_idx:end_idx] = x_hat.detach()
                start_idx += batch_size

            bpp_loss_epoch = bpp_loss_total / num_pixels
            adaptive_loss_per_epoch = adaptive_loss_total / total_samples
            mse_loss_per_epoch = mse_total / total_samples
            loss_per_epoch = loss_total / total_samples

            if loss_per_epoch < best_loss:
                best_loss = loss_per_epoch
                best_state_dict = copy.deepcopy(comp_model.state_dict())
                best_code = code_latent.detach().clone()
                best_loss_epoch = epoch

            # pbar.set_postfix(
            #     epoch=epoch,
            #     loss=f"{loss_per_epoch:.4f}",
            #     adaptive=f"{adaptive_loss_per_epoch:.4f}",
            #     bpp=f"{bpp_loss_epoch:.4f}",
            #     mse=f"{mse_loss_per_epoch:.4f}",
            # )
            # print(f"epoch {epoch}")
            print(f"{args.layer_idx}_{args.layer_name}_{W.shape}_bs{bs} | step {step} / {args.ft_comp_steps}")
            # print(f"loss {loss_per_epoch:.4f}")
            # print(f"adaptive_loss {adaptive_loss_per_epoch:.4f}")
            # print(f"bpp_loss {bpp_loss_epoch:.4f}")
            # print(f"mse_loss {mse_loss_per_epoch:.4f}")

            with torch.no_grad():
                err = torch.trace((W - W_hat) @ HR @ ((W - W_hat).T)).item()
                # trWHW = torch.trace(W @ HR @ W.T)
                proxy_err =  err / trWHW
                mse = mse_fn(W, W_hat).item()

            ft_result['epoch'].append(epoch)
            ft_result['loss_per_epoch'].append(loss_per_epoch)
            ft_result['adaptive_loss_per_epoch'].append(adaptive_loss_per_epoch)
            ft_result['bpp_loss_per_epoch'].append(bpp_loss_epoch)
            ft_result['mse_loss_per_epoch'].append(mse_loss_per_epoch)
            ft_result['best_loss_epoch'].append(best_loss_epoch)

            ft_result['proxy_err'].append(proxy_err)
            ft_result['err'].append(err)
            ft_result['mse'].append(mse)
            
            wandb.log({
                "epoch": epoch,
                "loss": loss_per_epoch,
                "best_loss": best_loss,
                "adaptive_loss": adaptive_loss_per_epoch,
                "bpp_loss": bpp_loss_epoch,
                "mse_loss": mse_loss_per_epoch,
                "proxy_err": proxy_err,
                "err":err,
                "mse":mse,
                "trWHW":trWHW,
                "base_proxy_err": base_proxy_err,
                "base_err":base_err,
                "base_bpp_loss":base_bpp_loss,
                "base_mse":base_mse,
            })

    wandb.finish()
    
    comp_model.load_state_dict(best_state_dict)
    print('best_code_latent: ', best_code.mean().item(), best_code.max().item(),best_code.min().item())

    comp_model.eval()
    comp_model.update()

    out = pseudo_compress_tensor(best_code, comp_model, args)

    return out, ft_result