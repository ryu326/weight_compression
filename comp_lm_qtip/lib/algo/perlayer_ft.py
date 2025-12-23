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
sys.path.append('/home/jgryu/workspace/weight_compression')
# from NWC.models.nwc_ql import NWC_ql_without_encoder
# from NWC.models.nwc import NWC_without_encoder
from NWC.models.cnn_dec import NWCC_dec_only
# import wandb
from NWC.loss import *
from NWC.models import get_model


class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)

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

# criterion = get_loss_fn(args, std=train_std, device = device)

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
    code_optimizer = optim.Adam(
        other_parms,
        lr=args.code_optim_lr,
    )
    return optimizer, aux_optimizer, code_optimizer

def code_optimize(W, HR, comp_model, args, **kwargs):
    from Weight_compression.comp_lm_qtip.lib.algo.archive.nwc import pseudo_compress_tensor, encode_latent
    
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
    del out
    
    ft_result['base_proxy_err'] = base_proxy_err
    ft_result['base_err'] = base_err
    ft_result['trWHW'] = trWHW
    ft_result['base_bpp_loss'] = base_bpp_loss
    ft_result['base_mse'] = base_mse
    
    # wandb.log({
    #     "bpp_loss": base_bpp_loss,
    #     "proxy_err": base_proxy_err,
    #     "mse": base_mse,
    #     "err": base_err,
    #     "trWHW": trWHW,
    # })
    print(f'--------------init {args.layer_idx}_{args.layer_name}------------------')
    print(f"bpp_loss :{base_bpp_loss:.3f}")
    print(f'proxy err {base_proxy_err:.4f}')
    print(f'mse {base_mse:.3f}')
    print(f'err {base_err:.3f}')
    print(f'tr(WHW.T) {trWHW:.3f}')    
    print('--------------------------------')

    # code_latent  = encode_latent(W, new_comp_model, args, qlevel = qlevel)
    code_latent  = encode_latent(W, comp_model, args, qlevel = qlevel)
    # code_latent = nn.Parameter(code_latent, requires_grad=True)
    # print(code_latent.grad)
    # code_latent = nn.Parameter(torch.zeros(W.shape, device=device), requires_grad=True)
    
    # args.direction = 'row'
    # assert args.direction == 'row'
    if args.direction == 'col':
        code_latent = code_latent.T
        W = W.T
    # comp_model = new_comp_model
    comp_model.train()
        
    for param in comp_model.parameters():
        param.requires_grad = args.ft_train
    for param in comp_model.g_s.parameters():
        param.requires_grad = args.ft_train_dec

    # loss_fn = RateDistortionLoss(std=comp_model.scale.mean(), Hr=HR, lmbda=args.ft_comp_lmbda)
    loss_fn =  get_loss_fn(args, std=W.std(), device = W.device)

    bs = min(W.shape[0], 4096*4096 // W.shape[1])
    batch_idx = 0
    W_hat = torch.zeros_like(W)
    num_pixels = 0
    bpp_loss_sum = 0
    bpp_sum = 0

    for start_idx in range(0, W.shape[0], bs):
        end_idx = min(start_idx + bs, W.shape[0])
        y = code_latent[start_idx:end_idx]
        W_batch = W[start_idx:end_idx]
        ql_batch = qlevel[start_idx:end_idx]
        print(y.shape, W_batch.shape, ql_batch.shape)

        y = nn.Parameter(y, requires_grad=True)
        optimizer, aux_optimizer, code_optimizer = configure_optimizers(comp_model, args, [y])
        wandb.init(project="NWC_code_optimize", name=f"{args.layer_idx}_{args.layer_name}_batch{batch_idx}/{W.shape[0]//bs-1}", config=vars(args))
        with torch.enable_grad():
            for it in range(args.code_optim_it):
                code_optimizer.zero_grad()
                x_hat, n_pixels, bpp_loss_, out, out_enc, bpp = model_forward_without_encoder(y, comp_model, args, it = it, tot_it = args.code_optim_it, ql = ql_batch)
                data = {'weight_block': W_batch, 'q_level': ql_batch}
                loss = loss_fn(data, out)
                loss['loss'].backward()
                code_optimizer.step()
                wandb.log({
                    "loss": loss['loss'].item(),
                    "bpp_loss": loss['bpp_loss'].item(),
                    "recon_loss": loss['recon_loss'].item(),
                    "it": it,
                })
        wandb.finish()
        batch_idx += 1
        code_latent[start_idx:end_idx] = y.detach().clone()
        W_hat[start_idx:end_idx] = x_hat.detach()
        num_pixels += n_pixels
        bpp_loss_sum += bpp_loss_
        bpp_sum += bpp
    
    bpp_loss = bpp_loss_sum / num_pixels

    if args.direction == 'col':
        W_hat = W_hat.T
        W = W.T
        
    with torch.no_grad():
        err = torch.trace((W - W_hat) @ HR @ ((W - W_hat).T)).item()
        # trWHW = torch.trace(W @ HR @ W.T)
        proxy_err =  err / trWHW
        mse = mse_fn(W, W_hat).item()
    
    ft_result['proxy_err'] = proxy_err
    ft_result['err'] = err
    ft_result['trWHW'] = trWHW
    ft_result['bpp_loss'] = bpp_loss
    ft_result['mse'] = mse
    
    # wandb.log({
    #     "bpp_loss": bpp_loss,
    #     "proxy_err": proxy_err,
    #     "mse": mse,
    #     "err": err,
    #     "trWHW": trWHW,
    # })
    print(f'--------------code optim {args.layer_idx}_{args.layer_name}------------------')
    print(f"bpp_loss :{bpp_loss:.3f}")
    print(f'proxy err {proxy_err:.4f}')
    print(f'mse {mse:.3f}')
    print(f'err {err:.3f}')
    print(f'tr(WHW.T) {trWHW:.3f}')    
    print('--------------------------------')

    return {'W_hat': W_hat,
            'bpp_loss_sum': bpp_loss_sum,
            'bpp_loss': bpp_loss,
            'num_pixels': num_pixels,
            'bpp_sum': bpp_sum,
            'bpp': bpp_sum / num_pixels, 
            'codes': None}, None
        
