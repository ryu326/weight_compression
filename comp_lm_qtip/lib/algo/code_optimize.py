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
from NWC.models.cnn_dec import NWCC_dec_only
import wandb


class RateDistortionLoss(nn.Module):
    def __init__(self, std, Hr, lmbda):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.std = std

        self.Hr = Hr
        
        log_I = torch.log(torch.diag(Hr) + 1e-6)
        min_val = log_I.amin()
        max_val = log_I.amax()
        self.I = (log_I - min_val) / (max_val - min_val + 1e-8).unsqueeze(0)
        
        
    def forward(self, ori_w, output):        
        out = {}
        num_pixels = ori_w.numel()
        w_hat = output["W_hat"].reshape(ori_w.shape)
        # H = self.Hr[start_idx:end_idx][start_idx:end_idx]
        
        out["mse_loss"] = self.mse(ori_w,  w_hat) / self.std**2
        out["importance_mse_loss"] = (((ori_w - w_hat)*self.I)**2).mean() / self.std**2

        if isinstance(output["likelihoods"], dict):
            out["bpp_loss"] = sum(
                (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                for likelihoods in output["likelihoods"].values()
            )
        else :
            out["bpp_loss"] = (torch.log(output["likelihoods"]).sum() / (-math.log(2) * num_pixels))

        # out["loss"] = self.lmbda * out["importance_mse_loss"] + out["bpp_loss"]
        out["loss"] = self.lmbda * out["mse_loss"] + out["bpp_loss"]
        
        return out

def optim_code_delta_cnndec(W, HR, comp_model, args, **kwargs):
    from lib.algo.nwc import pseudo_compress_tensor, configure_optimizers
    
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

    model = NWCC_dec_only()
    model.to(W.device)

    assert args.ft_train_dec == True
    
    comp_model = model
    comp_model.train()
    
    for param in comp_model.parameters():
        param.requires_grad = True
    for param in comp_model.g_s.parameters():
        param.requires_grad = args.ft_train_dec    

    loss_fn = RateDistortionLoss(std=W.std(), Hr=HR, lmbda=args.ft_comp_lmbda)

        # latent  = encode_latent(W, model, args, qlevel = qlevel)
    # code_latent = nn.Parameter(latent, requires_grad=True)
    
    code_latent = nn.Parameter(torch.zeros((1, model.in_channels, *W.shape), device=W.device), requires_grad=True)
    # delta = nn.Parameter(torch.ones((1, model.in_channels, *W.shape), device=W.device), requires_grad=True)
    delta = torch.ones((1, model.in_channels, *W.shape), device=W.device, requires_grad=False)

    optimizer, aux_optimizer = configure_optimizers(comp_model, args, [code_latent,delta])
    
    best_loss = float("inf")
    best_state_dict = copy.deepcopy(comp_model.state_dict())
    best_code  = code_latent.detach().clone()
    best_delta  = delta.detach().clone()
    best_loss_epoch = 0
    best_W = torch.zeros_like(W_hat)
    
    wandb.init(project="NWC_layerwise_ft_cnn", name=f"{args.layer_idx}_{args.layer_name}_trdec{args.ft_train_dec}", config=vars(args))
    with torch.enable_grad():
        for step in range(args.ft_comp_steps):
            optimizer.zero_grad()
            aux_optimizer.zero_grad()            
                                                  
            model_out = comp_model(code_latent, delta)
        
            loss = loss_fn(W, model_out)
            loss['loss'].backward()
            
            optimizer.step()
            try:
                aux_loss = comp_model.aux_loss()
            except:
                aux_loss = comp_model.module.aux_loss()
                
            aux_loss.backward()
            aux_optimizer.step()                
            
            ft_result['loss'].append(loss['loss'].item())
            ft_result['bpp_loss'].append(loss['bpp_loss'].item())
            ft_result['mse_loss'].append(loss['mse_loss'].item())
            ft_result['adaptive_loss'].append(loss['importance_mse_loss'].item())
            ft_result['step'].append(step)
            
            with torch.no_grad():
                W_hat = model_out["W_hat"].reshape(W.shape).detach()
                err = torch.trace((W - W_hat) @ HR @ ((W - W_hat).T)).item()
                proxy_err =  err / trWHW
                mse = mse_fn(W, W_hat).item()
            
            if loss['loss'].item() < best_loss:
                best_loss = loss['loss'].item()
                best_state_dict = copy.deepcopy(comp_model.state_dict())
                best_code = code_latent.detach().clone()
                best_delta  = delta.detach().clone()
                best_W = W_hat.clone()
                best_loss_epoch = step

            ft_result['epoch'].append(step)
            ft_result['loss_per_epoch'].append(loss['loss'].item())
            ft_result['adaptive_loss_per_epoch'].append(loss['importance_mse_loss'].item())
            ft_result['bpp_loss_per_epoch'].append(loss['bpp_loss'].item())
            ft_result['mse_loss_per_epoch'].append(loss['mse_loss'].item())
            ft_result['best_loss_epoch'].append(best_loss_epoch)
            ft_result['proxy_err'].append(proxy_err)
            ft_result['err'].append(err)
            ft_result['mse'].append(mse)
            
            wandb.log({
                "step": step,
                "loss": loss['loss'].item(),
                "best_loss": best_loss,
                "adaptive_loss": loss['importance_mse_loss'].item(),
                "bpp_loss": loss['bpp_loss'].item(),
                "mse_loss": loss['mse_loss'].item(),
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
    print('best_delta: ', best_delta.mean().item(), best_delta.max().item(), best_delta.min().item())

    comp_model.eval()
    # comp_model.update()

    # out = pseudo_compress_tensor(best_code, comp_model, args)

    bpp_loss_sum = loss['bpp_loss'] * W.numel()
    dec_parms = sum(p.numel() for p in model.parameters())
    effective_bpp32 = (dec_parms * 32 + bpp_loss_sum) / W.numel()
    effective_bpp16 = (dec_parms * 16 + bpp_loss_sum) / W.numel()
        
    out = {'W_hat': best_W,
            'bpp_loss_sum': bpp_loss_sum,
            'num_pixels': W.numel(),
            'bpp': 0,}

    optimize_out = {'dec_parms': dec_parms,
                    'effective_bpp32': effective_bpp32,
                    'effective_bpp16': effective_bpp16,
                    'model.config' : model.config}

    return out, ft_result, optimize_out