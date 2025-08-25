"""
Utilities for fine tuning
"""
import copy
import math
from contextlib import contextmanager
from operator import attrgetter

import glog, json
import torch
from torch import multiprocessing as mp
from torch import nn
from transformers import AutoModelForCausalLM

from lib import utils
# from lib.linear import QuantizedLinear

# from . import ldlq
from .archive import nwc

@contextmanager
def use_tf32():
    fp32_matmul_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision('high')
    yield
    torch.set_float32_matmul_precision(fp32_matmul_precision)

def compress_finetune_decoder_layer_clip(mixed_layer, quant_order, idx, comp_model, ql_i, args,
                                    device, pre_orig_emb, orig_emb):
    torch.manual_seed(int(idx.split('_')[-1]))
    torch.set_num_threads(args.num_cpu_threads)
    torch.set_grad_enabled(False)

    dtype_ = torch.float64 if args.use_fp64 else torch.float32
    orig_dtype = None
    for p in mixed_layer.parameters():
        orig_dtype = p.dtype
        break
    mixed_layer = mixed_layer.float()

    # train_dl, valid_dl = utils.split_data(pre_orig_emb, orig_emb, args)

    # has_kernel = utils.has_kernel(args.decode_mode, args.L, args.K, args.V,
    #                               args.tlut_bits, args.td_x, args.td_y)

    for quant_i, (linear_attr, name, in_hess_name, out_hess_name,
                  rcp) in enumerate(quant_order):
        utils.clean()
        
        ql = ql_i[linear_attr] if ql_i is not None else None
        orig_linear = attrgetter(linear_attr)(mixed_layer)
        W = orig_linear.weight.to(dtype_)
        in_hess_path = f'{args.in_hess_path}/{idx}_{in_hess_name}.pt'
        
        H_data = torch.load(in_hess_path, map_location=torch.device('cpu'))
        HR = utils.flat_to_sym(H_data['flatH'], H_data['n'])
        n_h = H_data['n']
        if 'mu' in H_data:
            mu = H_data['mu']
            HR += mu[None, :] * mu[:, None]
            del mu
        del H_data

        # HR = utils.regularize_H(HR, args.sigma_reg)
        HR = utils.regularize_H2(HR, n_h, args.sigma_reg)
        
        comp_model.to(dtype_)
        args.layer_idx = idx
        args.layer_name = name
        # W_hat, bpp_loss_sum, num_pixels, SU, SV, scaleWH, ft_result = nwc.compress_linear(W.clone(), HR, comp_model, ql, args, device)
        W_hat, bpp_loss_sum, num_pixels, SU, SV, scaleWH, ft_result, optimize_out = nwc.compress_linear(W.clone(), HR, comp_model, ql, args, device)
        W_hat = W_hat.to(dtype_)
        
        # err = torch.trace(
        #     (W - W_hat) @ HR @ ((W - W_hat).T) / torch.trace(W @ HR @ W.T)
        # )
        
        err = torch.trace((W - W_hat) @ HR @ ((W - W_hat).T))
        trWHW = torch.trace(W @ HR @ W.T)
        proxy_err =  err / trWHW

        print(
            f'{idx}_{name} proxy err {proxy_err.item()} err {err.item()} tr(WHW.T) {trWHW.item()}'
        )
        
        print(f'bpp_loss {bpp_loss_sum/num_pixels}')
        
        save_path = f'{args.save_path}/{idx}_{name}.pt'

        torch.save(
            {
                'W_hat': W_hat,
                'SU': SU,
                'SV': SV,
                'scaleWH':scaleWH,
                'proxy_err': proxy_err.item(),
                'err': err.item(),
                'tr(WHW.T)': trWHW.item(),
                'mse': torch.mean((W - W_hat) ** 2).item(),
                'bpp_loss_sum': bpp_loss_sum,
                'num_pixels': num_pixels,
            }, save_path)

        if args.ft_comp_model and args.layer_name in ['v', 'o', 'k', 'q']:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  
            fig, axs = plt.subplots(2, 4, figsize=(12, 8))
            axs[0, 0].plot(ft_result['step'], ft_result['loss'], label='loss')
            axs[0, 0].set_title('loss')
            axs[0, 1].plot(ft_result['step'], ft_result['adaptive_loss'], label='adaptive_loss')
            axs[0, 1].set_title('adaptive_loss')
            axs[0, 2].plot(ft_result['step'], ft_result['bpp_loss'], label='bpp_loss')
            axs[0, 2].set_title('bpp_loss')
            axs[0, 3].plot(ft_result['step'], ft_result['mse'], label='mse')
            axs[0, 3].set_title('mse')
            axs[1, 1].plot(ft_result['epoch'], ft_result['adaptive_loss_per_epoch'], label='adaptive_loss_per_epoch')
            axs[1, 1].set_title('adaptive_loss_per_epoch')
            axs[1, 2].plot(ft_result['epoch'], ft_result['bpp_loss_per_epoch'], label='bpp_loss_per_epoch')
            axs[1, 2].set_title('bpp_loss_per_epoch')       
            axs[1, 0].plot(ft_result['epoch'], ft_result['loss_per_epoch'], label='loss_per_epoch')
            axs[1, 0].set_title('loss_per_epoch')      
            axs[1, 3].plot(ft_result['epoch'], ft_result['mse_per_epoch'], label='mse_per_epoch')
            axs[1, 3].set_title('mse_per_epoch')     
                 
            plt.savefig(f'{args.save_path}/{idx}_{name}_ft_result.png')
            with open(f'{args.save_path}/{idx}_{name}_ft_result.json', 'w') as f:
                json.dump(ft_result, f)

        comp_linear = copy.deepcopy(orig_linear)
        comp_linear.weight.copy_(W_hat)
        comp_linear.weight.requires_grad = False
        
        assert not torch.equal(orig_linear.weight.data, comp_linear.weight.data)
        del orig_linear

        split_attr = linear_attr.split('.')
        setattr(
            attrgetter('.'.join(split_attr[:-1]))(mixed_layer), split_attr[-1],
            comp_linear)

        if args.ft_epochs > 0:
            with torch.enable_grad():
                finetune_decoder_layer(mixed_layer, f'{idx}_{name}', device,
                                    train_dl, valid_dl, orig_dtype, args) 
        
        assert torch.equal(W_hat, attrgetter(linear_attr)(mixed_layer).weight)
        
        del HR, W, W_hat
        utils.clean()

    mixed_layer = mixed_layer.to(orig_dtype).cpu()
    utils.clean()
    torch.set_grad_enabled(False)

def infer(args, end_dev, n_layers, in_q, out_q):
    with torch.no_grad():
        fake_dev_map = {
            'model.embed_tokens': 0,
            'model.rotary_emb': 0,
            'model.norm': end_dev - 1,
            'lm_head': end_dev - 1
        }
        per_dev = math.ceil(n_layers / end_dev)
        for i in range(n_layers):
            fake_dev_map[f'model.layers.{i}'] = (i + 1) // per_dev

        model = AutoModelForCausalLM.from_pretrained(args.base_model,
                                                     torch_dtype='auto',
                                                     device_map=fake_dev_map,
                                                     low_cpu_mem_usage=True)
        while True:
            data = in_q.get()
            if data is None:
                return
            out_q.put(
                model(data.to(0))['logits'][:, :-1].contiguous().softmax(
                    dim=-1).cpu())


def finetune_susv_e2e(quant_model, start_dev, devset, orig_dtype, args):

    in_q = mp.Queue()
    out_q = mp.Queue()
    p = mp.Process(target=infer,
                   args=(args, start_dev, len(quant_model.model.layers), in_q,
                         out_q))
    p.start()

    train_dl, valid_dl = utils.split_data(devset, devset, args)

    optim = torch.optim.Adam(quant_model.parameters(), lr=args.ft_lr)

    best_loss = utils.calculate_ce_loss_model(quant_model, valid_dl, start_dev,
                                              in_q, out_q)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    best_sd = copy.deepcopy(quant_model.state_dict())
    glog.info(f'initial loss {best_loss}')
    worse_ct = 0
    for epoch in range(args.ft_epochs):
        for bidx, (source, _) in enumerate(train_dl):
            in_q.put(source)
            with torch.autocast(device_type='cuda',
                                dtype=orig_dtype,
                                enabled=True):
                output = quant_model(
                    source.to(start_dev))['logits'][:, :-1].contiguous()
                target = out_q.get().to(output.device)
                target = target.view(-1, target.shape[-1])
                loss = nn.CrossEntropyLoss()(output.view(-1, output.shape[-1]),
                                             target)
            scaler.scale(loss).backward()
            if bidx % args.ft_update_freq == args.ft_update_freq - 1 or bidx == len(
                    train_dl) - 1:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()

        if epoch % args.ft_valid_freq == (args.ft_valid_freq - 1):
            test_loss = utils.calculate_ce_loss_model(quant_model, valid_dl,
                                                      start_dev, in_q, out_q)
            if test_loss < best_loss:
                glog.info(
                    f'epoch {epoch} new loss {test_loss} old loss {best_loss} BETTER'
                )
                best_loss = test_loss
                best_sd = copy.deepcopy(quant_model.state_dict())
                worse_ct = 0
            else:
                glog.info(
                    f'epoch {epoch} new loss {test_loss} old loss {best_loss} WORSE'
                )
                worse_ct += 1
                if worse_ct >= args.ft_early_stop:
                    break

    in_q.put(None)
    p.join()
    with torch.no_grad():
        quant_model.load_state_dict(best_sd)
