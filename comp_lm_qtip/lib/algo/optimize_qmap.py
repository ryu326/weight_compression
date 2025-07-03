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

class RateDistortionLoss(nn.Module):
    def __init__(self, std, Hr, lmbda):
        super().__init__()
        self.lmbda = lmbda
        self.std = std

        # def clip_outliers_quantile_global(tensor, lower_q=0.03, upper_q=0.97):
        #     lower = torch.quantile(tensor, lower_q)
        #     upper = torch.quantile(tensor, upper_q)
        #     return torch.clip(tensor, min=lower.item(), max=upper.item())
        
        # self.Hr = clip_outliers_quantile_global(Hr)
        self.Hr = Hr
        
    def hessian_proxy_loss(self, W, W_hat, H):
        diff = W_hat - W
        H = H.float()
        trace_H = H.trace()
        # if trace_H > 0:
        if trace_H <= 1e-6:
            trace_H = trace_H + 1e-6
        if True:
            H = H / trace_H * H.shape[0] / W.numel()
        # loss = torch.trace(diff @ H @ diff.T) / torch.trace(W @ H @ W.T)
        
        numerator = torch.sum((H @ diff.T) * diff.T)
        denominator = torch.sum((H @ W.T) * W.T)
        loss = numerator / denominator
        return loss

    def forward(self, w, w_hat, output, bpp_loss_sum, n_pixels):        
        out = {}
        num_pixels = w.numel()
        # w_hat = output["x_hat"].reshape(-1, ori_w.shape[0]).T.contiguous() ## qmap
        # w_hat = output["x_hat"] ## row norm
        
        # w_hat = output["x_hat"].mT.reshape(ori_w.shape).contiguous()
        # H = self.Hr[start_idx:end_idx][start_idx:end_idx]
        
        # out["mse_loss"] = F.mse_loss(w,  w_hat) / self.std**2
        out["mse_loss"] = F.mse_loss(w,  w_hat) / self.std**2
        # out["adaptive_loss"] = self.hessian_proxy_loss(w, w_hat, self.Hr) / self.std**2
        out["adaptive_loss"] = self.hessian_proxy_loss(w, w_hat, self.Hr) * w.shape[1]

        # if isinstance(output["likelihoods"], dict):
        #     out["bpp_loss"] = sum(
        #         (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
        #         for likelihoods in output["likelihoods"].values()
        #     )
        # else :
        #     out["bpp_loss"] = (torch.log(output["likelihoods"]).sum() / (-math.log(2) * num_pixels))
        
        out["bpp_loss"] = bpp_loss_sum / n_pixels
        
        out["loss"] = self.lmbda * out["adaptive_loss"] + out["bpp_loss"]
        # out["loss"] = self.lmbda * out["mse_loss"] + out["bpp_loss"]
        # out["loss"] = self.lmbda * (out["mse_loss"]+ out["adaptive_loss"]) /2 + out["bpp_loss"]
        return out
    
def normalize_qmap(qmap):
    return torch.sigmoid(qmap)

def get_norm(norm_param):
    return F.softplus(norm_param) + 1e-4 

def softplus_inverse(x):
    return x + torch.log(-torch.expm1(-x))
        
def optimize_qmap_rnorm(W, H, model, args, **kwargs):
    # comp_W와 비슷한 구조
    # Row 방향으로 자르기
    
    from lib.algo import nwc
    
    blks = model.input_size
    split_size = min(W.shape[0], 4096 * 1024 // W.shape[1])
    split_size = split_size // blks * blks
    (M, N) = W.shape
    
    init_qmap = kwargs.get('qmap', None) ## (M//blks, N)
    init_row_norm = kwargs.get('row_norm', None)  # (m, 1)
    init_col_norm = kwargs.get('col_norm', None)  # (1, n)
    
    qlevel = kwargs.get('qlevel', None)
    qlevel = qlevel.reshape(W.shape[1], ) if qlevel is not None else None
    
    if args.qmap_optim:
        qmap = torch.empty(M // blks, N) 
    if args.rnorm_optim:
        row_norm = torch.empty(M, 1)
    if args.cnorm_optim:
        col_norm = torch.empty(1, N)
        
    # std = W.std()
    std = model.scale.item()
    # std = 0.014042483642697334
    loss_fn = RateDistortionLoss(std, H, args.code_optim_lmbda)

    wandb.init(project=f"NWC_qmap_optim", name=f"{'_'.join(args.save_path.split('/')[-2:])}_{args.layer_idx}_{args.layer_name}", config=vars(args))

    for i,s in enumerate(range(0, M, split_size)):
        e = min(M, s + split_size)
        assert (e - s) % blks == 0
        
        
        w = W[s:e,:]
        ql = qlevel[:] if qlevel is not None else None
        init_rnorm = init_row_norm[s:e, :] if init_row_norm is not None else None
        init_cnorm = init_col_norm[:, :] if init_col_norm is not None else None
        init_qm = init_qmap[s//blks:e//blks, :] if init_qmap is not None else None       
        
        model.train()
        model.mode = 'ste'
        
        with torch.enable_grad():
            
            parm_list = []
            if args.qmap_optim:
                if init_qm is not None:
                    qm_logit = torch.special.logit(init_qm.clamp(min=1e-6, max=1-1e-6)) # (split_size//16, N)
                else:
                    qm_logit = torch.empty((e-s)//blks, N).normal_(mean=-2, std=1).to(w.device)  
                qm_logit = nn.Parameter(qm_logit)
                parm_list.append(qm_logit)
                
            if args.rnorm_optim:
                if init_rnorm is not None:
                    rnorm = init_rnorm
                else:
                    rnorm = torch.ones(e-s,1).to(w.device) 
                # rnorm_param = nn.Parameter(torch.log(torch.exp(rnorm) - 1.0))
                rnorm_param = nn.Parameter(softplus_inverse(rnorm.clamp(min=1e-2)))
                parm_list.append(rnorm_param)
                
            if args.cnorm_optim: ## 얘는 못함
                if init_cnorm is not None:
                    cnorm = init_cnorm
                else:
                    cnorm = torch.ones(1,N).to(w.device) 
                # cnorm_param = nn.Parameter(torch.log(torch.exp(cnorm) - 1.0))
                cnorm_param = nn.Parameter(softplus_inverse(cnorm.clamp(min=1e-2)))
                parm_list.append(cnorm_param)
                
            optimizer = torch.optim.LBFGS(parm_list, max_iter=5, line_search_fn="strong_wolfe")
                    
            train_param = {
                'w': w,
                'i': 0,
                'init_rnorm': init_rnorm,
                'init_cnorm': init_cnorm,
                'init_qm': init_qm,
                'loss_best': float('inf'),
                'qm_logit': qm_logit if args.qmap_optim else None,
                'qm_logit_best': qm_logit.clone().detach() if args.qmap_optim else None,
                'rnorm_param': rnorm_param if args.rnorm_optim else None,
                'rnorm_param_best': rnorm_param.clone().detach() if args.rnorm_optim else None,
                'cnorm_param': cnorm_param if args.cnorm_optim else None,
                'cnorm_param_best': cnorm_param.clone().detach() if args.cnorm_optim else None,
                'i_best': 0,
            }
            
            def closure_():
                w = train_param['w']
                
                if args.qmap_optim:
                    qm_norm = normalize_qmap(train_param['qm_logit'])
                else :
                    qm_norm = train_param['init_qm']
                if args.rnorm_optim:
                    r_norm = get_norm(train_param['rnorm_param'])
                else :
                    r_norm = train_param['init_rnorm']
                if args.cnorm_optim:
                    c_norm = get_norm(train_param['cnorm_param'])
                else :
                    c_norm = train_param['init_cnorm']
                    
                w_hat, n_pixels, bpp_loss_sum, out, out_enc, nbits = nwc.model_foward_one_batch(w.clone(), model, args, ql = ql, qm = qm_norm, rnorm = r_norm, cnorm = c_norm)

                loss_out = loss_fn(w, w_hat, out, bpp_loss_sum, n_pixels)
                optimizer.zero_grad()                
                loss_out['loss'].backward()
                train_param['i'] += 1
                                
                if train_param['loss_best'] >  loss_out['loss'] or train_param['i'] == 1:
                    train_param['loss_best'] = loss_out['loss'].item()
                    train_param['qm_logit_best'] = train_param['qm_logit'].clone().detach() if args.qmap_optim else None
                    train_param['rnorm_param_best'] = train_param['rnorm_param'].clone().detach() if args.rnorm_optim else None
                    train_param['cnorm_param_best'] = train_param['cnorm_param'].clone().detach() if args.cnorm_optim else None
                    train_param['i_best'] = train_param['i']

                log_dict = {
                    # "step": i * args.qmap_optim_iter*4 + train_param['i'],
                    "step": train_param['i'],
                    "i_best": train_param['i_best'],
                    "loss": loss_out['loss'].item(),
                    "bpp_loss": loss_out['bpp_loss'].item(),
                    "adaptive_loss": loss_out['adaptive_loss'].item(),
                    "mse_loss": loss_out['mse_loss'].item(),
                }
                if args.qmap_optim:
                    log_dict.update({
                        "qm_logit_mean": qm_logit.mean().item(),
                        "qm_logit_max": qm_logit.max().item(),
                        "qm_logit_min": qm_logit.min().item(),
                        "qmap_mean": qm_norm.mean().item(),
                        "qmap_max": qm_norm.max().item(),
                        "qmap_min": qm_norm.min().item(),
                        "qmap_grad_norm": qm_logit.grad.norm().item() if qm_logit.grad is not None else -1,
                    })
                if args.rnorm_optim:
                    log_dict.update({
                        "rnorm_mean": r_norm.mean().item(),
                        "rnorm_max": r_norm.max().item(),
                        "rnorm_min": r_norm.min().item(),
                        "rnorm_grad_norm": rnorm_param.grad.norm().item() if rnorm_param.grad is not None else -1,
                    })
                if args.cnorm_optim:
                    log_dict.update({
                        "cnorm_mean": c_norm.mean().item(),
                        "cnorm_max": c_norm.max().item(),
                        "cnorm_min": c_norm.min().item(),
                        "cnorm_grad_norm": cnorm_param.grad.norm().item() if cnorm_param.grad is not None else -1,
                    })

                wandb.log(log_dict)

                return loss_out['loss']
            
            for _ in range(args.qmap_optim_iter):
                optimizer.step(closure_)

        if args.qmap_optim:
            qmap[s//blks : e//blks, :] = normalize_qmap(train_param['qm_logit_best'])
        if args.rnorm_optim:
            row_norm[s:e, :] = get_norm(train_param['rnorm_param_best'])
        if args.cnorm_optim:
            col_norm[:, :] = get_norm(train_param['cnorm_param_best'])
    
    wandb.finish()
    
    
    qmap_final = qmap if args.qmap_optim else init_qmap
    row_norm_final = row_norm if args.rnorm_optim else init_row_norm
    col_norm_final = col_norm if args.cnorm_optim else init_col_norm

    return qmap_final, row_norm_final, col_norm_final



# def optimize_rnorm(W, H, model, args, **kwargs):
#     mse_fn = nn.MSELoss()
#     rnorm = kwargs.get('row_norm', None)
#     qlevel = kwargs.get('qlevel', None)
#     with torch.enable_grad():
#         blks = model.input_size
#         split_size = min(W.shape[0], 4096 * 1024 // W.shape[1])
#         split_size = split_size // blks * blks
#         (m, n) = W.shape
        
#         assert m % blks == 0
        
#         std = 1
#         # std = 0.014042483642697334
#         loss_fn = RateDistortionLoss(std, H, args.code_optim_lmbda)
        
#         # for i, w in enumerate(W_splits):
#         for i,s in enumerate(range(0, m, split_size)):
#             wandb.init(project=f"NWC_rnorm_optim", name=f"{'_'.join(args.save_path.split('/')[-2:])}_{args.layer_idx}_{args.layer_name}_batch{i}", config=vars(args))
            
#             e = min(m, s + split_size)
#             w = W[s:e,:]
#             ori_w = w
#             rn = 1 / rnorm[s:e,:]
#             rn = nn.Parameter(rn.to(w.device))
#             rn.requires_grad_()
#             optimizer = torch.optim.LBFGS([rn], max_iter=5)
#             model.mode = 'ste'
            
#             # w = w / rn
#             # w  = w.T.reshape(1, -1, blks)
#             ql = qlevel.reshape(1, n).expand((e-s)//blks, n)
            
#             train_param = {
#                 'w': w,
#                 'i': 0,
#                 'loss_best': float('inf'),
#                 'bpp_best': float('inf'),
#                 'recon_best': float('inf'),
#                 'mse_best': float('inf'),
#                 'rn': rn,
#                 'rn_best': rn,
#                 'i_best': 0,
#             }
            
#             def closure_():
#                 w = train_param['w']
#                 rn = train_param['rn']
                            
#                 w = w * rn
#                 w  = w.T.reshape(1, -1, blks)
                    
#                 data = {}
#                 data['weight_block'] = w
#                 data['q_level'] = ql.T.reshape(1, w.shape[1])
#                 out_net = model(data)
#                 out_net["x_hat"] = out_net["x_hat"].reshape(-1, ori_w.shape[0]).T.contiguous()
#                 out_net["x_hat"] = out_net["x_hat"] / rn

#                 loss_out = loss_fn(ori_w, out_net)
#                 optimizer.zero_grad()                
#                 loss_out['loss'].backward()
#                 train_param['i'] += 1
                                
#                 if train_param['loss_best'] >  loss_out['loss'] or train_param['i'] == 1:
#                     train_param['loss_best'] = loss_out['loss']
#                     train_param['rn_best'] = rn.clone().detach()
#                     train_param['i_best'] = train_param['i']
#                     # train_param['bpp_best'] = loss_out['bpp_loss'].clone().cpu().detach()
#                     # train_param['recon_best'] = loss_out['adaptive_loss'].clone().cpu().detach()
#                     # train_param['mse_best'] = loss_out['mse_loss'].clone().cpu().detach()

#                 wandb.log({
#                     "step": train_param['i'],
#                     "i_best": train_param['i_best'],
#                     "loss": loss_out['loss'].item(),
#                     "bpp_loss": loss_out['bpp_loss'].item(),
#                     "adaptive_loss": loss_out['adaptive_loss'].item(),
#                     "mse_loss": loss_out['mse_loss'].item(),
#                     "rn_mean": rn.mean().item(),
#                     "rn_max": rn.max().item(),
#                     "rn_min": rn.min().item(),
#                     "rn_grad_norm": rn.grad.norm().item() if rn.grad is not None else 0.0
#                 })

#                 return loss_out['loss']
            
#             for _ in range(args.qmap_optim_iter):
#                 optimizer.step(closure_)
            
#             wandb.finish()
            
#             assert (e - s) % blks == 0
#             rnorm[s:e,:] = 1/train_param['rn_best']
#         return rnorm.cpu()
        

# def optimize_qmap(W, H, model, args, **kwargs):
#     Qmap = kwargs.get('Qmap', None)
#     mse_fn = nn.MSELoss()
#     with torch.enable_grad():
    
#         blks = model.input_size
#         # split_size = min(W.shape[0], 4096*4096 // W.shape[1])
#         # split_size = min(W.shape[0], 4096*2048 // W.shape[1])
#         split_size = min(W.shape[0], 4096 * 1024 // W.shape[1])
#         split_size = split_size // blks * blks
#         (m, n) = W.shape
        
#         assert m % blks == 0
        
#         qmap = torch.empty(m // blks, n) # (m // 16 , n) col 방향 block 마다 값 한 개씩
            
#         # std = W.std()
#         # std = model.scale
#         std = 0.014042483642697334
#         loss_fn = RateDistortionLoss(std, H, args.code_optim_lmbda)
        
#         # for i, w in enumerate(W_splits):
#         for i,s in enumerate(range(0, m, split_size)):
#             e = min(m, s + split_size)
#             w = W[s:e,:]
            
#             wandb.init(project=f"NWC_qmap_optim", name=f"{'_'.join(args.save_path.split('/')[-2:])}_{args.layer_idx}_{args.layer_name}_batch{i}", config=vars(args))
            
#             ori_w = w
#             w  = w.T.reshape(1, -1, blks)
#             # qm = torch.empty(1, w.shape[1]).normal_(mean=0, std=1)
#             # qm = qm.clone().detach().to(w.device)
#             if Qmap is None:
#                 qm = nn.Parameter(torch.empty(1, w.shape[1]).normal_(mean=-2, std=1).to(w.device))
#             else:
#                 qm  = torch.special.logit(Qmap[s//16:e//16,:].T.reshape(1, -1), eps = 1e-8)
#             qm.requires_grad_()
#             model.mode = 'ste'
#             optimizer = torch.optim.LBFGS([qm], max_iter=5)
            
#             train_param = {
#                 'w': w,
#                 'i': 0,
#                 'loss_best': float('inf'),
#                 'bpp_best': float('inf'),
#                 'recon_best': float('inf'),
#                 'mse_best': float('inf'),
#                 'qmap_mean_best': 0,
#                 'qmap': qm,
#                 'qmap_best': qm,
#                 'i_best': 0,
#             }
            
#             def closure_():
#                 w = train_param['w']
#                 qm = train_param['qmap']
                
#                 qm_norm = normalize_qmap(qm)
#                 data = {}
#                 data['weight_block'] = w
#                 data['qmap'] = qm_norm
#                 out_net = model(data)

#                 loss_out = loss_fn(ori_w, out_net)
#                 optimizer.zero_grad()                
#                 loss_out['loss'].backward()
#                 train_param['i'] += 1
                                
#                 if train_param['loss_best'] >  loss_out['loss'] or train_param['i'] == 1:
#                     train_param['loss_best'] = loss_out['loss']
#                     train_param['qmap_best'] = qm.clone().detach()
#                     train_param['qmap_mean_best'] = qm_norm.detach().mean()
#                     train_param['i_best'] = train_param['i']
#                     # train_param['bpp_best'] = loss_out['bpp_loss'].clone().cpu().detach()
#                     # train_param['recon_best'] = loss_out['adaptive_loss'].clone().cpu().detach()
#                     # train_param['mse_best'] = loss_out['mse_loss'].clone().cpu().detach()

#                 wandb.log({
#                     "step": train_param['i'],
#                     "i_best": train_param['i_best'],
#                     "loss": loss_out['loss'].item(),
#                     "bpp_loss": loss_out['bpp_loss'].item(),
#                     "adaptive_loss": loss_out['adaptive_loss'].item(),
#                     "mse_loss": loss_out['mse_loss'].item(),
#                     "qmap_mean": qm.mean().item(),
#                     "qmap_max": qm.max().item(),
#                     "qmap_min": qm.min().item(),
#                     "qmap_norm_mean": qm_norm.mean().item(),
#                     "qmap_norm_max": qm_norm.max().item(),
#                     "qmap_norm_min": qm_norm.min().item(),
#                     "org_diff": mse_fn(
#                         qm_norm.detach().reshape(n, (e - s)//blks).T,
#                         Qmap[s//blks : e//blks, :].detach()
#                     ).item(),
#                     "qmap_grad_norm": qm.grad.norm().item() if qm.grad is not None else 0.0
#                 })

#                 return loss_out['loss']
            
#             for _ in range(args.qmap_optim_iter):
#                 optimizer.step(closure_)
            
#             wandb.finish()
            
#             qmap_norm_best = normalize_qmap(train_param['qmap_best'])
#             assert (e - s) % blks == 0
#             qmap_norm_best = qmap_norm_best.reshape(n, (e - s)//blks)
#             qmap_norm_best = qmap_norm_best.T
#             qmap[s//blks : e//blks, :] = qmap_norm_best
        
#         return qmap.cpu()