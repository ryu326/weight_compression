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
from . import nwc

@contextmanager
def use_tf32():
    fp32_matmul_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision('high')
    yield
    torch.set_float32_matmul_precision(fp32_matmul_precision)


def finetune_decoder_layer(layer, name, device, train_dl, valid_dl, orig_dtype,
                           args):
    with use_tf32():
        layer = layer.to(device)

        source = next(iter(train_dl))[0]
        position_ids = torch.arange(source.shape[1], device=device).unsqueeze(0)
        # manifest tensor parallel attributes in layer
        output = layer(source.to(device),
                       position_ids=position_ids)[0]
        
        best_sd = {k: v.cpu() for k, v in layer.state_dict().items()}
        utils.clean()

        optim = torch.optim.Adam(layer.parameters(), lr=args.ft_lr)
        best_loss = utils.calculate_mse_loss(layer, valid_dl, device)
        glog.info(f'layer {name} initial loss {best_loss}')
        scaler = torch.cuda.amp.GradScaler(enabled=(orig_dtype==torch.float16))
        worse_ct = 0

        for epoch in range(args.ft_epochs):
            for bidx, (source, targets) in enumerate(train_dl):
                targets = targets.to(device, non_blocking=True)
                with torch.autocast(device_type='cuda',
                                    dtype=orig_dtype,
                                    enabled=True):
                    output = layer(source.to(device),
                                   position_ids=position_ids)[0]
                    loss = nn.MSELoss()(output, targets)
                scaler.scale(loss).backward()
                if bidx % args.ft_update_freq == args.ft_update_freq - 1 or bidx == len(
                        train_dl) - 1:
                    scaler.step(optim)
                    scaler.update()
                    optim.zero_grad()

            if epoch % args.ft_valid_freq == (args.ft_valid_freq - 1):
                test_loss = utils.calculate_mse_loss(layer, valid_dl, device)
                if test_loss < best_loss:
                    glog.info(
                        f'layer {name} @ epoch {epoch} new loss {test_loss} old loss {best_loss} BETTER'
                    )
                    best_loss = test_loss
                    best_sd = {k: v.cpu() for k, v in layer.state_dict().items()}
                    utils.clean()
                    worse_ct = 0
                else:
                    glog.info(
                        f'layer {name} @ epoch {epoch} new loss {test_loss} old loss {best_loss} WORSE'
                    )
                    worse_ct += 1
                    if worse_ct >= args.ft_early_stop:
                        break

    del optim, train_dl, valid_dl

    layer = layer.cpu()
    layer.load_state_dict(best_sd)
    utils.clean()


def compress_finetune_decoder_layer(mixed_layer, quant_order, idx, comp_model, ql_i, args,
                                    device, pre_orig_emb, orig_emb):
    torch.manual_seed(idx)
    torch.set_num_threads(args.num_cpu_threads)
    torch.set_grad_enabled(False)

    dtype_ = torch.float64 if args.use_fp64 else torch.float32
    orig_dtype = None
    for p in mixed_layer.parameters():
        orig_dtype = p.dtype
        break
    mixed_layer = mixed_layer.float()

    train_dl, valid_dl = utils.split_data(pre_orig_emb, orig_emb, args)

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
        # W_hat, bpp_loss_sum, num_pixels, SU, SV, scaleWH, ft_result, optimize_out = nwc.compress_linear(W.clone(), HR, comp_model, ql, args, device)
        out, SU, SV, scaleWH, ft_result, optimize_out = nwc.compress_linear(W.clone(), HR, comp_model, ql, args, device)
        
        glog.info(f'------------------------------------')
        trWHW = torch.trace(W @ HR @ W.T)
        if args.code_optim:
            W_hat_init = out['W_hat_init'].to(dtype_)
            bpp_loss_init = out['bpp_loss_init']
            bpp_init = out['bpp_init']        
            err_init = torch.trace((W - W_hat_init) @ HR @ ((W - W_hat_init).T))
            proxy_err_init =  err_init / trWHW
            glog.info(
                f'{idx}_{name} init proxy err {proxy_err_init.item():.5f} err {err_init.item():.3f} tr(WHW.T) {trWHW.item():.1f} bpp_loss {bpp_loss_init:.4f} bpp {bpp_init:.4f}'
            )
        if args.code_optim_test:
            W_hat_round = out['W_hat_round'].to(dtype_).cpu()
            err_round = torch.trace((W - W_hat_round) @ HR @ ((W - W_hat_round).T))
            proxy_err_round =  err_round / trWHW
            glog.info(
                f'{idx}_{name} rund proxy err {proxy_err_round.item():.5f} err {err_round.item():.3f} tr(WHW.T) {trWHW.item():.1f} bpp_loss {out["bpp_loss_round"]:.4f}'
            )
            W_hat_sga = out['W_hat_sga'].to(dtype_).cpu()
            err_sga = torch.trace((W - W_hat_sga) @ HR @ ((W - W_hat_sga).T))
            proxy_err_sga =  err_sga / trWHW
            glog.info(
                f'{idx}_{name} sga_ proxy err {proxy_err_sga.item():.5f} err {err_sga.item():.3f} tr(WHW.T) {trWHW.item():.1f} bpp_loss {out["bpp_loss_sga"]:.4f}'
            )

        W_hat = out['W_hat'].to(dtype_)
        bpp_loss = out['bpp_loss']
        bpp = out['bpp']        
        err = torch.trace((W - W_hat) @ HR @ ((W - W_hat).T))
        proxy_err =  err / trWHW
        glog.info(
            f'{idx}_{name} optm proxy err {proxy_err.item():.5f} err {err.item():.3f} tr(WHW.T) {trWHW.item():.1f} bpp_loss {bpp_loss:.4f} bpp {bpp:.4f}'
        )
        glog.info(f'------------------------------------')

        save_path = f'{args.save_path}/{idx}_{name}.pt'

        torch.save(
            {
                'W_hat': W_hat,
                'codes': out['codes'],
                'SU': SU,
                'SV': SV,
                'scaleWH':scaleWH,
                'proxy_err': proxy_err.item(),
                'proxy_err_init': proxy_err_init.item() if args.code_optim else None,
                'proxy_err_round': proxy_err_round.item() if args.code_optim_test else None,
                'proxy_err_sga': proxy_err_sga.item() if args.code_optim_test else None,
                'err': err.item(),
                'err_init': err_init.item() if args.code_optim else None,
                'err_round': err_round.item() if args.code_optim_test else None,
                'err_sga': err_sga.item() if args.code_optim_test else None,
                'tr(WHW.T)': trWHW.item(),
                'mse': torch.mean((W - W_hat) ** 2).item(),
                'mse_init': torch.mean((W - W_hat_init) ** 2).item() if args.code_optim else None,
                'bpp_loss': bpp_loss,
                'W_hat_init': W_hat_init if args.code_optim else None,
                'W_hat_round': W_hat_round if args.code_optim_test else None,
                'W_hat_sga': W_hat_sga if args.code_optim_test else None,
                'bpp_loss_init': bpp_loss_init if args.code_optim else None,
                'bpp_loss_round': out['bpp_loss_round'] if args.code_optim_test else None,
                'bpp_loss_sga': out['bpp_loss_sga'] if args.code_optim_test else None,
                'bpp_loss_sum': out['bpp_loss_sum'],
                'bpp_loss_sum_init': out['bpp_loss_sum_init'] if args.code_optim else None,
                'bpp_loss_sum_round': out['bpp_loss_sum_round'] if args.code_optim_test else None,
                'bpp_loss_sum_sga': out['bpp_loss_sum_sga'] if args.code_optim_test else None,
                'bpp': bpp,
                'bpp_init': bpp_init if args.code_optim else None,
                'bpp_sum': out['bpp_sum'],
                'num_pixels': out['num_pixels'],
                'optimize_out': optimize_out,
                'direction': args.direction,
            }, save_path)

        # if args.ft_comp_model2 and args.layer_name in ['v', 'o', 'k', 'q']:
        if args.ft_comp_model2 or args.code_optim:
            utils.plot_ft_comp_result(ft_result, args, idx, name)

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
