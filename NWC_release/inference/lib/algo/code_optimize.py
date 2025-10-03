# import torch
# import math
# # import utils
# from lib import utils
# import os
# from lib.algo import quip
# import numpy as np
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from collections import defaultdict
# from tqdm import tqdm
# import copy
# from torch.utils.data import TensorDataset, DataLoader
# import sys
# sys.path.append('/home/jgryu/workspace/weight_compression')
# # from NWC.models.nwc_ql import NWC_ql_without_encoder
# # from NWC.models.nwc import NWC_without_encoder
# from NWC.models.cnn_dec import NWCC_dec_only
# # import wandb
# from NWC.loss import *
# from NWC.models import get_model


# class Config:
#     def __init__(self, **entries):
#         self.__dict__.update(entries)

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

# # criterion = get_loss_fn(args, std=train_std, device = device)

# def configure_optimizers(net, args, other_parms):
#     """Separate parameters for the main optimizer and the auxiliary optimizer.
#     Return two optimizers"""

#     parameters = {n for n, p in net.named_parameters() if ".quantiles" not in n and p.requires_grad}
#     aux_parameters = {n for n, p in net.named_parameters() if ".quantiles" in n and p.requires_grad}

#     # print(aux_parameters)  # {'module.entropy_bottleneck_z.quantiles'}

#     params_dict = dict(net.named_parameters())

#     # optimizer = optim.Adam(
#     #     list((params_dict[n] for n in sorted(parameters))) + other_parms,
#     #     lr=args.ft_comp_learning_rate,
#     # )
#     # aux_optimizer = optim.Adam(
#     #     (params_dict[n] for n in sorted(aux_parameters)),
#     #     lr=args.ft_comp_aux_learning_rate,
#     # )
#     optimizer, aux_optimizer = None, None
#     code_optimizer = optim.Adam(
#         other_parms,
#         lr=args.code_optim_lr,
#     )
#     return optimizer, aux_optimizer, code_optimizer

# def code_optimize(W, HR, comp_model, args, **kwargs):
#     from Weight_compression.comp_lm_qtip.lib.algo.archive.nwc import pseudo_compress_tensor, encode_latent
    
#     ft_result = defaultdict(list)
#     ft_result['best_loss_epoch'] = []

#     qlevel = kwargs.get('qlevel', None)
#     # start test
#     mse_fn = nn.MSELoss()
#     out = pseudo_compress_tensor(W, comp_model, args, qlevel = qlevel)    
    
#     W_hat = out['W_hat']
#     base_err = torch.trace((W - W_hat) @ HR @ ((W - W_hat).T)).item()
#     trWHW = torch.trace(W @ HR @ W.T).item()
#     base_proxy_err =  base_err / trWHW
#     base_bpp_loss = out['bpp_loss_sum']/out['num_pixels']
#     base_mse = mse_fn(W, W_hat).item()
#     del out
    
#     ft_result['base_proxy_err'] = base_proxy_err
#     ft_result['base_err'] = base_err
#     ft_result['trWHW'] = trWHW
#     ft_result['base_bpp_loss'] = base_bpp_loss
#     ft_result['base_mse'] = base_mse
    
#     # wandb.log({
#     #     "bpp_loss": base_bpp_loss,
#     #     "proxy_err": base_proxy_err,
#     #     "mse": base_mse,
#     #     "err": base_err,
#     #     "trWHW": trWHW,
#     # })
#     print(f'--------------init {args.layer_idx}_{args.layer_name}------------------')
#     print(f"bpp_loss :{base_bpp_loss:.3f}")
#     print(f'proxy err {base_proxy_err:.4f}')
#     print(f'mse {base_mse:.3f}')
#     print(f'err {base_err:.3f}')
#     print(f'tr(WHW.T) {trWHW:.3f}')    
#     print('--------------------------------')

#     # code_latent  = encode_latent(W, new_comp_model, args, qlevel = qlevel)
#     code_latent  = encode_latent(W, comp_model, args, qlevel = qlevel)
#     # code_latent = nn.Parameter(code_latent, requires_grad=True)
#     # print(code_latent.grad)
#     # code_latent = nn.Parameter(torch.zeros(W.shape, device=device), requires_grad=True)
    
#     # args.direction = 'row'
#     # assert args.direction == 'row'
#     if args.direction == 'col':
#         code_latent = code_latent.T
#         W = W.T
#     # comp_model = new_comp_model
#     comp_model.train()
        
#     for param in comp_model.parameters():
#         param.requires_grad = args.ft_train
#     for param in comp_model.g_s.parameters():
#         param.requires_grad = args.ft_train_dec

#     # loss_fn = RateDistortionLoss(std=comp_model.scale.mean(), Hr=HR, lmbda=args.ft_comp_lmbda)
#     loss_fn =  get_loss_fn(args, std=W.std(), device = W.device)

#     bs = min(W.shape[0], 4096*4096 // W.shape[1])
#     batch_idx = 0
#     W_hat = torch.zeros_like(W)
#     num_pixels = 0
#     bpp_loss_sum = 0
#     bpp_sum = 0

#     for start_idx in range(0, W.shape[0], bs):
#         end_idx = min(start_idx + bs, W.shape[0])
#         y = code_latent[start_idx:end_idx]
#         W_batch = W[start_idx:end_idx]
#         ql_batch = qlevel[start_idx:end_idx]
#         print(y.shape, W_batch.shape, ql_batch.shape)

#         y = nn.Parameter(y, requires_grad=True)
#         optimizer, aux_optimizer, code_optimizer = configure_optimizers(comp_model, args, [y])
#         wandb.init(project="NWC_code_optimize", name=f"{args.layer_idx}_{args.layer_name}_batch{batch_idx}/{W.shape[0]//bs-1}", config=vars(args))
#         with torch.enable_grad():
#             for it in range(args.code_optim_it):
#                 code_optimizer.zero_grad()
#                 x_hat, n_pixels, bpp_loss_, out, out_enc, bpp = model_forward_without_encoder(y, comp_model, args, it = it, tot_it = args.code_optim_it, ql = ql_batch)
#                 data = {'weight_block': W_batch, 'q_level': ql_batch}
#                 loss = loss_fn(data, out)
#                 loss['loss'].backward()
#                 code_optimizer.step()
#                 wandb.log({
#                     "loss": loss['loss'].item(),
#                     "bpp_loss": loss['bpp_loss'].item(),
#                     "recon_loss": loss['recon_loss'].item(),
#                     "it": it,
#                 })
#         wandb.finish()
#         batch_idx += 1
#         code_latent[start_idx:end_idx] = y.detach().clone()
#         W_hat[start_idx:end_idx] = x_hat.detach()
#         num_pixels += n_pixels
#         bpp_loss_sum += bpp_loss_
#         bpp_sum += bpp
    
#     bpp_loss = bpp_loss_sum / num_pixels

#     if args.direction == 'col':
#         W_hat = W_hat.T
#         W = W.T
        
#     with torch.no_grad():
#         err = torch.trace((W - W_hat) @ HR @ ((W - W_hat).T)).item()
#         # trWHW = torch.trace(W @ HR @ W.T)
#         proxy_err =  err / trWHW
#         mse = mse_fn(W, W_hat).item()
    
#     ft_result['proxy_err'] = proxy_err
#     ft_result['err'] = err
#     ft_result['trWHW'] = trWHW
#     ft_result['bpp_loss'] = bpp_loss
#     ft_result['mse'] = mse
    
#     # wandb.log({
#     #     "bpp_loss": bpp_loss,
#     #     "proxy_err": proxy_err,
#     #     "mse": mse,
#     #     "err": err,
#     #     "trWHW": trWHW,
#     # })
#     print(f'--------------code optim {args.layer_idx}_{args.layer_name}------------------')
#     print(f"bpp_loss :{bpp_loss:.3f}")
#     print(f'proxy err {proxy_err:.4f}')
#     print(f'mse {mse:.3f}')
#     print(f'err {err:.3f}')
#     print(f'tr(WHW.T) {trWHW:.3f}')    
#     print('--------------------------------')

#     return {'W_hat': W_hat,
#             'bpp_loss_sum': bpp_loss_sum,
#             'bpp_loss': bpp_loss,
#             'num_pixels': num_pixels,
#             'bpp_sum': bpp_sum,
#             'bpp': bpp_sum / num_pixels, 
#             'codes': None}, None
        
# def model_forward_without_encoder(w, model, args, **kwargs):
#     ori_shape = w.shape
#     w = w.reshape(ori_shape[0], -1, model.input_size)
        
#     data = {}
#     # data['weight_block'] = w
    
#     it = kwargs.get('it', None)
#     tot_it = kwargs.get('tot_it', None)
#     ql = kwargs.get('ql', None)
#     if ql is not None:
#         data['q_level'] = ql.reshape(ori_shape[0], 1)
        
#     lstats = kwargs.get('lstats', None)
#     if lstats is not None:
#         data['l_cdt'] = lstats.unsqueeze(0).repeat(ori_shape[0], 1)
        
#     num_pixels = w.numel()
#     bpp_loss = 0
#     nbits = 0
#     out_enc = None
#     out = None
    
#     if args.use_codes:
#         raise NotImplementedError
#         # out_enc = model.compress_without_encoder(data)
#         # out_dec = model.decompress(out_enc)
#         # w_hat = out_dec['x_hat'].reshape(ori_shape)
        
#         # for s in out_enc["strings"]:
#         #     nbits += len(s[0]) * 8.0

#     else:
#         out = model(data, mode = 'sga', y_in = w, it = it, tot_it = tot_it)
#         w_hat = out['x_hat'].reshape(ori_shape)
        
#         if isinstance(out["likelihoods"], dict):
#             bpp_loss = sum(
#                 (torch.log(likelihoods).sum() / -math.log(2))
#                 for likelihoods in out["likelihoods"].values()
#             ).item()
#         else :
#             bpp_loss = (torch.log(out["likelihoods"]).sum() / -math.log(2)).item()
    
#     return w_hat, num_pixels, bpp_loss, out, out_enc, nbits


# # class RateDistortionLoss(nn.Module):
# #     def __init__(self, std, Hr, lmbda):
# #         super().__init__()
# #         self.mse = nn.MSELoss()
# #         self.lmbda = lmbda
# #         self.std = std

# #         self.Hr = Hr
        
# #         log_I = torch.log(torch.diag(Hr) + 1e-6)
# #         min_val = log_I.amin()
# #         max_val = log_I.amax()
# #         self.I = (log_I - min_val) / (max_val - min_val + 1e-8).unsqueeze(0)
        
        
# #     def forward(self, ori_w, output):        
# #         out = {}
# #         num_pixels = ori_w.numel()
# #         w_hat = output["W_hat"].reshape(ori_w.shape)
# #         # H = self.Hr[start_idx:end_idx][start_idx:end_idx]
        
# #         out["mse_loss"] = self.mse(ori_w,  w_hat) / self.std**2
# #         out["importance_mse_loss"] = (((ori_w - w_hat)*self.I)**2).mean() / self.std**2

# #         if isinstance(output["likelihoods"], dict):
# #             out["bpp_loss"] = sum(
# #                 (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
# #                 for likelihoods in output["likelihoods"].values()
# #             )
# #         else :
# #             out["bpp_loss"] = (torch.log(output["likelihoods"]).sum() / (-math.log(2) * num_pixels))

# #         # out["loss"] = self.lmbda * out["importance_mse_loss"] + out["bpp_loss"]
# #         out["loss"] = self.lmbda * out["mse_loss"] + out["bpp_loss"]
        
# #         return out

# # def optim_code_delta_cnndec(W, HR, comp_model, args, **kwargs):
# #     from lib.algo.nwc import pseudo_compress_tensor
    
# #     ft_result = defaultdict(list)
# #     ft_result['best_loss_epoch'] = []

# #     qlevel = kwargs.get('qlevel', None)
# #     # start test
# #     mse_fn = nn.MSELoss()
# #     out = pseudo_compress_tensor(W, comp_model, args, qlevel = qlevel)    
    
# #     W_hat = out['W_hat']
# #     base_err = torch.trace((W - W_hat) @ HR @ ((W - W_hat).T)).item()
# #     trWHW = torch.trace(W @ HR @ W.T).item()
# #     base_proxy_err =  base_err / trWHW
# #     base_bpp_loss = out['bpp_loss_sum']/out['num_pixels']
# #     base_mse = mse_fn(W, W_hat).item()
# #     # print(f'Before LayerFT {args.layer_idx}_{args.layer_name} | proxy err {base_proxy_err.item()} err {base_err.item()} tr(WHW.T) {trWHW.item()}')    
# #     # print(f"bpp_loss :{base_bpp_loss:.3f}")
# #     del out
    
# #     ft_result['base_proxy_err'] = base_proxy_err
# #     ft_result['base_err'] = base_err
# #     ft_result['trWHW'] = trWHW
# #     ft_result['base_bpp_loss'] = base_bpp_loss
# #     ft_result['base_mse'] = base_mse

# #     model = NWCC_dec_only()
# #     model.to(W.device)

# #     assert args.ft_train_dec == True
    
# #     comp_model = model
# #     comp_model.train()
    
# #     for param in comp_model.parameters():
# #         param.requires_grad = True
# #     for param in comp_model.g_s.parameters():
# #         param.requires_grad = args.ft_train_dec    

# #     loss_fn = RateDistortionLoss(std=W.std(), Hr=HR, lmbda=args.ft_comp_lmbda)

# #     # latent  = encode_latent(W, model, args, qlevel = qlevel)
# #     # code_latent = nn.Parameter(latent, requires_grad=True)
    
# #     code_latent = nn.Parameter(torch.zeros((1, model.in_channels, *W.shape), device=W.device), requires_grad=True)
# #     # delta = nn.Parameter(torch.ones((1, model.in_channels, *W.shape), device=W.device), requires_grad=True)
# #     delta = torch.ones((1, model.in_channels, *W.shape), device=W.device, requires_grad=False)

# #     optimizer, aux_optimizer = configure_optimizers(comp_model, args, [code_latent,delta])
    
# #     best_loss = float("inf")
# #     best_state_dict = copy.deepcopy(comp_model.state_dict())
# #     best_code  = code_latent.detach().clone()
# #     best_delta  = delta.detach().clone()
# #     best_loss_epoch = 0
# #     best_W = torch.zeros_like(W_hat)
    
# #     wandb.init(project="NWC_layerwise_ft_cnn", name=f"{args.layer_idx}_{args.layer_name}_trdec{args.ft_train_dec}", config=vars(args))
# #     with torch.enable_grad():
# #         for step in range(args.ft_comp_steps):
# #             optimizer.zero_grad()
# #             aux_optimizer.zero_grad()            
                                                  
# #             model_out = comp_model(code_latent, delta)
        
# #             loss = loss_fn(W, model_out)
# #             loss['loss'].backward()
            
# #             optimizer.step()
# #             try:
# #                 aux_loss = comp_model.aux_loss()
# #             except:
# #                 aux_loss = comp_model.module.aux_loss()
                
# #             aux_loss.backward()
# #             aux_optimizer.step()                
            
# #             ft_result['loss'].append(loss['loss'].item())
# #             ft_result['bpp_loss'].append(loss['bpp_loss'].item())
# #             ft_result['recon_loss'].append(loss['recon_loss'].item())
# #             ft_result['adaptive_loss'].append(loss['importance_mse_loss'].item())
# #             ft_result['step'].append(step)
            
# #             with torch.no_grad():
# #                 W_hat = model_out["W_hat"].reshape(W.shape).detach()
# #                 err = torch.trace((W - W_hat) @ HR @ ((W - W_hat).T)).item()
# #                 proxy_err =  err / trWHW
# #                 mse = mse_fn(W, W_hat).item()
            
# #             if loss['loss'].item() < best_loss:
# #                 best_loss = loss['loss'].item()
# #                 best_state_dict = copy.deepcopy(comp_model.state_dict())
# #                 best_code = code_latent.detach().clone()
# #                 best_delta  = delta.detach().clone()
# #                 best_W = W_hat.clone()
# #                 best_loss_epoch = step

# #             ft_result['epoch'].append(step)
# #             ft_result['loss_per_epoch'].append(loss['loss'].item())
# #             ft_result['adaptive_loss_per_epoch'].append(loss['importance_mse_loss'].item())
# #             ft_result['bpp_loss_per_epoch'].append(loss['bpp_loss'].item())
# #             ft_result['mse_loss_per_epoch'].append(loss['recon_loss'].item())
# #             ft_result['best_loss_epoch'].append(best_loss_epoch)
# #             ft_result['proxy_err'].append(proxy_err)
# #             ft_result['err'].append(err)
# #             ft_result['mse'].append(mse)
            
# #             wandb.log({
# #                 "step": step,
# #                 "loss": loss['loss'].item(),
# #                 "best_loss": best_loss,
# #                 "adaptive_loss": loss['importance_mse_loss'].item(),
# #                 "bpp_loss": loss['bpp_loss'].item(),
# #                 "mse_loss": loss['recon_loss'].item(),
# #                 "proxy_err": proxy_err,
# #                 "err":err,
# #                 "mse":mse,
# #                 "trWHW":trWHW,
# #                 "base_proxy_err": base_proxy_err,
# #                 "base_err":base_err,
# #                 "base_bpp_loss":base_bpp_loss,
# #                 "base_mse":base_mse,
# #             })

# #     wandb.finish()
    
# #     comp_model.load_state_dict(best_state_dict)
# #     print('best_code_latent: ', best_code.mean().item(), best_code.max().item(),best_code.min().item())
# #     print('best_delta: ', best_delta.mean().item(), best_delta.max().item(), best_delta.min().item())

# #     comp_model.eval()
# #     # comp_model.update()

# #     # out = pseudo_compress_tensor(best_code, comp_model, args)

# #     bpp_loss_sum = loss['bpp_loss'] * W.numel()
# #     dec_parms = sum(p.numel() for p in model.parameters())
# #     effective_bpp32 = (dec_parms * 32 + bpp_loss_sum) / W.numel()
# #     effective_bpp16 = (dec_parms * 16 + bpp_loss_sum) / W.numel()
        
# #     out = {'W_hat': best_W,
# #             'bpp_loss_sum': bpp_loss_sum,
# #             'num_pixels': W.numel(),
# #             'bpp': 0,}

# #     optimize_out = {'dec_parms': dec_parms,
# #                     'effective_bpp32': effective_bpp32,
# #                     'effective_bpp16': effective_bpp16,
# #                     'model.config' : model.config}

# #     return out, ft_result, optimize_out