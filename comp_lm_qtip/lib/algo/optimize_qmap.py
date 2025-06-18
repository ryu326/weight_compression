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
        # if True:
        #     H = H / trace_H * H.shape[0] / W.numel()
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
        
        out["mse_loss"] = F.mse_loss(w,  w_hat) / self.std**2
        out["adaptive_loss"] = self.hessian_proxy_loss(w, w_hat, self.Hr) / self.std**2

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


def optimize_rnorm(W, H, model, args, **kwargs):
    mse_fn = nn.MSELoss()
    rnorm = kwargs.get('row_norm', None)
    qlevel = kwargs.get('qlevel', None)
    with torch.enable_grad():
        blks = model.input_size
        split_size = min(W.shape[0], 4096 * 1024 // W.shape[1])
        split_size = split_size // blks * blks
        (m, n) = W.shape
        
        assert m % blks == 0
        
        std = 1
        # std = 0.014042483642697334
        loss_fn = RateDistortionLoss(std, H, args.code_optim_lmbda)
        
        # for i, w in enumerate(W_splits):
        for i,s in enumerate(range(0, m, split_size)):
            wandb.init(project=f"NWC_rnorm_optim", name=f"{'_'.join(args.save_path.split('/')[-2:])}_{args.layer_idx}_{args.layer_name}_batch{i}", config=vars(args))
            
            e = min(m, s + split_size)
            w = W[s:e,:]
            ori_w = w
            rn = 1 / rnorm[s:e,:]
            rn = nn.Parameter(rn.to(w.device))
            rn.requires_grad_()
            optimizer = torch.optim.LBFGS([rn], max_iter=5)
            model.mode = 'ste'
            
            # w = w / rn
            # w  = w.T.reshape(1, -1, blks)
            ql = qlevel.reshape(1, n).expand((e-s)//blks, n)
            
            train_param = {
                'w': w,
                'i': 0,
                'loss_best': float('inf'),
                'bpp_best': float('inf'),
                'recon_best': float('inf'),
                'mse_best': float('inf'),
                'rn': rn,
                'rn_best': rn,
                'i_best': 0,
            }
            
            def closure_():
                w = train_param['w']
                rn = train_param['rn']
                            
                w = w * rn
                w  = w.T.reshape(1, -1, blks)
                    
                data = {}
                data['weight_block'] = w
                data['q_level'] = ql.T.reshape(1, w.shape[1])
                out_net = model(data)
                out_net["x_hat"] = out_net["x_hat"].reshape(-1, ori_w.shape[0]).T.contiguous()
                out_net["x_hat"] = out_net["x_hat"] / rn

                loss_out = loss_fn(ori_w, out_net)
                optimizer.zero_grad()                
                loss_out['loss'].backward()
                train_param['i'] += 1
                                
                if train_param['loss_best'] >  loss_out['loss'] or train_param['i'] == 1:
                    train_param['loss_best'] = loss_out['loss']
                    train_param['rn_best'] = rn.clone().detach()
                    train_param['i_best'] = train_param['i']
                    # train_param['bpp_best'] = loss_out['bpp_loss'].clone().cpu().detach()
                    # train_param['recon_best'] = loss_out['adaptive_loss'].clone().cpu().detach()
                    # train_param['mse_best'] = loss_out['mse_loss'].clone().cpu().detach()

                wandb.log({
                    "step": train_param['i'],
                    "i_best": train_param['i_best'],
                    "loss": loss_out['loss'].item(),
                    "bpp_loss": loss_out['bpp_loss'].item(),
                    "adaptive_loss": loss_out['adaptive_loss'].item(),
                    "mse_loss": loss_out['mse_loss'].item(),
                    "rn_mean": rn.mean().item(),
                    "rn_max": rn.max().item(),
                    "rn_min": rn.min().item(),
                    "rn_grad_norm": rn.grad.norm().item() if rn.grad is not None else 0.0
                })

                return loss_out['loss']
            
            for _ in range(args.qmap_optim_iter):
                optimizer.step(closure_)
            
            wandb.finish()
            
            assert (e - s) % blks == 0
            rnorm[s:e,:] = 1/train_param['rn_best']
        return rnorm.cpu()
        

def optimize_qmap(W, H, model, args, **kwargs):
    Qmap = kwargs.get('Qmap', None)
    mse_fn = nn.MSELoss()
    with torch.enable_grad():
    
        blks = model.input_size
        # split_size = min(W.shape[0], 4096*4096 // W.shape[1])
        # split_size = min(W.shape[0], 4096*2048 // W.shape[1])
        split_size = min(W.shape[0], 4096 * 1024 // W.shape[1])
        split_size = split_size // blks * blks
        (m, n) = W.shape
        
        assert m % blks == 0
        
        qmap = torch.empty(m // blks, n) # (m // 16 , n) col 방향 block 마다 값 한 개씩
            
        # std = W.std()
        # std = model.scale
        std = 0.014042483642697334
        loss_fn = RateDistortionLoss(std, H, args.code_optim_lmbda)
        
        # for i, w in enumerate(W_splits):
        for i,s in enumerate(range(0, m, split_size)):
            e = min(m, s + split_size)
            w = W[s:e,:]
            
            wandb.init(project=f"NWC_qmap_optim", name=f"{'_'.join(args.save_path.split('/')[-2:])}_{args.layer_idx}_{args.layer_name}_batch{i}", config=vars(args))
            
            ori_w = w
            w  = w.T.reshape(1, -1, blks)
            # qm = torch.empty(1, w.shape[1]).normal_(mean=0, std=1)
            # qm = qm.clone().detach().to(w.device)
            if Qmap is None:
                qm = nn.Parameter(torch.empty(1, w.shape[1]).normal_(mean=-2, std=1).to(w.device))
            else:
                qm  = torch.special.logit(Qmap[s//16:e//16,:].T.reshape(1, -1), eps = 1e-8)
            qm.requires_grad_()
            model.mode = 'ste'
            optimizer = torch.optim.LBFGS([qm], max_iter=5)
            
            train_param = {
                'w': w,
                'i': 0,
                'loss_best': float('inf'),
                'bpp_best': float('inf'),
                'recon_best': float('inf'),
                'mse_best': float('inf'),
                'qmap_mean_best': 0,
                'qmap': qm,
                'qmap_best': qm,
                'i_best': 0,
            }
            
            def closure_():
                w = train_param['w']
                qm = train_param['qmap']
                
                qm_norm = normalize_qmap(qm)
                data = {}
                data['weight_block'] = w
                data['qmap'] = qm_norm
                out_net = model(data)

                loss_out = loss_fn(ori_w, out_net)
                optimizer.zero_grad()                
                loss_out['loss'].backward()
                train_param['i'] += 1
                                
                if train_param['loss_best'] >  loss_out['loss'] or train_param['i'] == 1:
                    train_param['loss_best'] = loss_out['loss']
                    train_param['qmap_best'] = qm.clone().detach()
                    train_param['qmap_mean_best'] = qm_norm.detach().mean()
                    train_param['i_best'] = train_param['i']
                    # train_param['bpp_best'] = loss_out['bpp_loss'].clone().cpu().detach()
                    # train_param['recon_best'] = loss_out['adaptive_loss'].clone().cpu().detach()
                    # train_param['mse_best'] = loss_out['mse_loss'].clone().cpu().detach()

                wandb.log({
                    "step": train_param['i'],
                    "i_best": train_param['i_best'],
                    "loss": loss_out['loss'].item(),
                    "bpp_loss": loss_out['bpp_loss'].item(),
                    "adaptive_loss": loss_out['adaptive_loss'].item(),
                    "mse_loss": loss_out['mse_loss'].item(),
                    "qmap_mean": qm.mean().item(),
                    "qmap_max": qm.max().item(),
                    "qmap_min": qm.min().item(),
                    "qmap_norm_mean": qm_norm.mean().item(),
                    "qmap_norm_max": qm_norm.max().item(),
                    "qmap_norm_min": qm_norm.min().item(),
                    "org_diff": mse_fn(
                        qm_norm.detach().reshape(n, (e - s)//blks).T,
                        Qmap[s//blks : e//blks, :].detach()
                    ).item(),
                    "qmap_grad_norm": qm.grad.norm().item() if qm.grad is not None else 0.0
                })

                return loss_out['loss']
            
            for _ in range(args.qmap_optim_iter):
                optimizer.step(closure_)
            
            wandb.finish()
            
            qmap_norm_best = normalize_qmap(train_param['qmap_best'])
            assert (e - s) % blks == 0
            qmap_norm_best = qmap_norm_best.reshape(n, (e - s)//blks)
            qmap_norm_best = qmap_norm_best.T
            qmap[s//blks : e//blks, :] = qmap_norm_best
        
        return qmap.cpu()
        
        
def optimize_qmap_rnorm(W, H, model, args, **kwargs):
    # comp_W와 비슷한 구조
    # Row 방향으로 자르기
    
    from lib.algo import nwc
    
    blks = model.input_size
    split_size = min(W.shape[0], 4096 * 1024 // W.shape[1])
    split_size = split_size // blks * blks
    (M, N) = W.shape
    
    row_norm = kwargs.get('row_norm', None)  # (m, 1)
    col_norm = kwargs.get('col_norm', None)  # (1, n)
    
    qlevel = kwargs.get('qlevel', None)
    qlevel = qlevel.reshape(W.shape[1], ) if qlevel is not None else None
    
    qmap = kwargs.get('qmap', None) ## (M//blks, N)
    if qmap is None: 
        qmap = torch.empty(M // blks, N)
        
    # std = W.std()
    std = model.scale.item()
    # std = 0.014042483642697334
    loss_fn = RateDistortionLoss(std, H, args.code_optim_lmbda)

    for i,s in enumerate(range(0, M, split_size)):
        e = min(M, s + split_size)
        assert (e - s) % blks == 0
        
        wandb.init(project=f"NWC_qmap_optim", name=f"{'_'.join(args.save_path.split('/')[-2:])}_{args.layer_idx}_{args.layer_name}_batch{i}", config=vars(args))
        
        w = W[s:e,:]
        
        ql = qlevel[:] if qlevel is not None else None
        r_norm = row_norm[s:e, :] if row_norm is not None else None
        c_norm = col_norm[:, :] if col_norm is not None else None
        
        if qmap is not None:
            qm_logit = torch.special.logit(qmap[s//blks:e//blks, :]) # (split_size//16, N)
        else:
            qm_logit = torch.empty((s-e)//blks, N).normal_(mean=-2, std=1).to(w.device)

        model.mode = 'ste'
        
        with torch.enable_grad():
            
            parm_list = []
            if args.optim_qmap:
                qm_logit = nn.Parameter(qm_logit)
                # qm_logit.requires_grad_()
                parm_list.append(qm_logit)
            if args.optim_norm and r_norm is not None:
                r_norm = nn.Parameter(r_norm)
                # r_norm.requires_grad_()
                parm_list.append(r_norm)
            if args.optim_norm and c_norm is not None:
                c_norm = nn.Parameter(c_norm)
                # c_norm.requires_grad_()
                parm_list.append(c_norm)
                
            optimizer = torch.optim.LBFGS(parm_list, max_iter=5)
                    
            train_param = {
                'w': w,
                'i': 0,
                'loss_best': float('inf'),
                'qm_logit': qm_logit,
                'qm_logit_best': qm_logit.clone().detach() if args.optim_qmap else None,
                'r_norm': r_norm,
                'r_norm_best': r_norm.clone().detach() if r_norm is not None else None,
                'c_norm': c_norm,
                'c_norm_best': c_norm.clone().detach() if c_norm is not None else None,
                'i_best': 0,
            }
            
            def closure_():
                w = train_param['w']
                r_norm = train_param['r_norm']
                c_norm = train_param['c_norm']
                qm_logit = train_param['qm_logit']
                qm_norm = normalize_qmap(qm_logit)
                            
                w_hat, n_pixels, bpp_loss_sum, out, out_enc, nbits = nwc.model_foward_one_batch(w.clone(), model, args, ql = ql, qm = qm_norm, rnorm = r_norm, cnorm = c_norm)

                loss_out = loss_fn(w, w_hat, out, bpp_loss_sum, n_pixels)
                optimizer.zero_grad()                
                loss_out['loss'].backward()
                train_param['i'] += 1
                                
                if train_param['loss_best'] >  loss_out['loss'] or train_param['i'] == 1:
                    train_param['loss_best'] = loss_out['loss'].item()
                    train_param['qm_logit_best'] = qm_logit.clone().detach() if args.optim_qmap else None
                    train_param['r_norm_best'] = r_norm.clone().detach() if r_norm is not None else None
                    train_param['c_norm_best'] = c_norm.clone().detach() if c_norm is not None else None
                    train_param['i_best'] = train_param['i']

                wandb.log({
                    "step": train_param['i'],
                    "i_best": train_param['i_best'],
                    "loss": loss_out['loss'].item(),
                    "bpp_loss": loss_out['bpp_loss'].item(),
                    "adaptive_loss": loss_out['adaptive_loss'].item(),
                    "mse_loss": loss_out['mse_loss'].item(),
                    "qm_logit_mean": qm_logit.mean().item(),
                    "qm_logit_max": qm_logit.max().item(),
                    "qm_logit_min": qm_logit.min().item(),
                    "qmap_mean": qm_norm.mean().item(),
                    "qmap_max": qm_norm.max().item(),
                    "qmap_min": qm_norm.min().item(),
                    "qmap_grad_norm": qm_logit.grad.norm().item() if qm_logit.grad is not None else -1,

                    # r_norm 관련
                    "rnorm_mean": r_norm.mean().item() if r_norm is not None else -1,
                    "rnorm_max": r_norm.max().item() if r_norm is not None else -1,
                    "rnorm_min": r_norm.min().item() if r_norm is not None else -1,
                    "rnorm_grad_norm": r_norm.grad.norm().item() if r_norm is not None and r_norm.grad is not None else -1,

                    # c_norm 관련
                    "cnorm_mean": c_norm.mean().item() if c_norm is not None else -1,
                    "cnorm_max": c_norm.max().item() if c_norm is not None else -1,
                    "cnorm_min": c_norm.min().item() if c_norm is not None else -1,
                    "cnorm_grad_norm": c_norm.grad.norm().item() if c_norm is not None and c_norm.grad is not None else -1,
                })

                return loss_out['loss']
            
            for _ in range(args.qmap_optim_iter):
                optimizer.step(closure_)
        
        wandb.finish()
        
        qmap_best = normalize_qmap(train_param['qm_logit_best'])
        r_norm_best = train_param['r_norm_best']
        c_norm_best = train_param['c_norm_best']
        
        qmap[s//blks : e//blks, :] = qmap_best
        if row_norm is not None:
            row_norm[s:e, :] = r_norm_best
        if col_norm is not None:
            col_norm[:, :] = c_norm_best
    return qmap.cpu(), row_norm, col_norm