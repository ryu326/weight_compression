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
from lib.linear import CompLinear, CompLinear2, CompLinear3


# from . import ldlq
# from . import nwc
from . import nwc_refactory as nwc
from . import handcraft
# from . import nic
import copy

@contextmanager
def use_tf32():
    fp32_matmul_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision('high')
    yield
    torch.set_float32_matmul_precision(fp32_matmul_precision)


def finetune_decoder_layer(layer, name, device, train_dl, valid_dl, orig_dtype,
                           args, attention_mask, rotary_emb):
    with use_tf32():
        layer = layer.to(device)
        attention_mask = attention_mask.to(device=device, dtype=next(layer.parameters()).dtype)

        source = next(iter(train_dl))[0]
        position_ids = torch.arange(source.shape[1], device=device).unsqueeze(0)
        # manifest tensor parallel attributes in layer
        forward_kwargs = {
            "hidden_states": source.to(device),
            "position_ids": position_ids,
            "attention_mask": attention_mask
        }
        # [수정] Qwen용 Position Embeddings 계산 및 추가
        if rotary_emb is not None:
            position_embeddings = rotary_emb(source.to(device), position_ids)
            forward_kwargs["position_embeddings"] = position_embeddings
        output = layer(**forward_kwargs)[0]
        
        best_sd = {k: v.cpu() for k, v in layer.state_dict().items()}
        utils.clean()

        optim = torch.optim.Adam(layer.parameters(), lr=args.ft_lr)
        best_loss = utils.calculate_mse_loss_moe(layer, valid_dl, device, attention_mask, rotary_emb)
        glog.info(f'layer {name} initial loss {best_loss}')
        scaler = torch.cuda.amp.GradScaler(enabled=(orig_dtype==torch.float16))
        worse_ct = 0

        # --------- 사전 설정 ----------
        k, tau = 5, 0.5                 # SoftAdapt 창 길이·민감도
        loss_hist = {'mse': [], 'bpp': []}
        # initial_w = utils.get_initial_weights(layer, train_dl, device)
        
        for epoch in range(args.ft_epochs):
            for bidx, (source, targets) in enumerate(train_dl):
                targets = targets.to(device, non_blocking=True)
                forward_kwargs['hidden_states'] = source.to(device)
                with torch.autocast(device_type='cuda',
                                    dtype=orig_dtype,
                                    enabled=True):
                    output = layer(**forward_kwargs)[0]
                    # output = layer(source.to(device),
                    #                position_ids=position_ids,
                    #                attention_mask=attention_mask)[0]
                    loss = nn.MSELoss()(output, targets)
                    
                    # if args.ft_bpp_loss:
                    #     mse_loss = loss
                    #     bpp_loss = sum(m.bpp_loss for m in layer.modules()
                    #        if isinstance(m, CompLinear2))

                    #     loss_hist['mse'].append(mse_loss.detach())
                    #     loss_hist['bpp'].append(bpp_loss.detach())
                    #     if len(loss_hist['mse']) > k:
                    #         loss_hist['mse'].pop(0)
                    #         loss_hist['bpp'].pop(0)
                    #     if len(loss_hist['mse']) == k:
                    #         mse_ma  = torch.stack(loss_hist['mse']).mean()
                    #         bpp_ma  = torch.stack(loss_hist['bpp']).mean()
                    #         r_mse   = (loss - mse_ma) / mse_ma
                    #         r_bpp   = (bpp_loss - bpp_ma) / bpp_ma
                    #         w = torch.softmax(torch.stack([r_mse, r_bpp]) / tau, dim=0)
                    #     else:
                    #         w = initial_w
                    #     loss = w[0] * mse_loss + w[1] * bpp_loss   
                        # glog.info(
                        #     f'layer {name} @ epoch {epoch} bidx {bidx} loss {loss.item():.2f} mse {mse_loss.item():.4g} bpp {bpp_loss.item():.2f} w {w}'
                        # )                          
                                            
                scaler.scale(loss).backward()
                if bidx % args.ft_update_freq == args.ft_update_freq - 1 or bidx == len(
                        train_dl) - 1:
                    scaler.step(optim)
                    scaler.update()
                    optim.zero_grad()

            if epoch % args.ft_valid_freq == (args.ft_valid_freq - 1):
                if args.ft_bpp_loss:
                    avg_mse, avg_bpp = utils.calculate_test_loss(layer, valid_dl, device)
                    glog.info(
                        f'layer {name} @ epoch {epoch} mse {avg_mse:.4g} bpp {avg_bpp:.2f} w '
                    )   
                else:
                    test_loss = utils.calculate_mse_loss_moe(layer, valid_dl, device, attention_mask, rotary_emb)
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
                                    device, pre_orig_emb, orig_emb, attention_mask, rotary_emb):
    """
    [리팩토링됨] 레이어를 그룹별로 묶어 압축하고, 각 그룹이 끝날 때만 파인튜닝을 수행합니다.
    """
    try:
        torch.manual_seed(idx)
    except:
        torch.manual_seed(int(idx.split('_')[-1]))
    torch.set_num_threads(args.num_cpu_threads)
    torch.set_grad_enabled(False)

    dtype_ = torch.float64 if args.use_fp64 else torch.float32
    orig_dtype = None
    for p in mixed_layer.parameters():
        orig_dtype = p.dtype
        break
    mixed_layer = mixed_layer.float()

    train_dl, valid_dl = None, None
    if pre_orig_emb is not None and orig_emb is not None and args.ft_epochs > 0:
        train_dl, valid_dl = utils.split_data(pre_orig_emb, orig_emb, args)

    # --- [헬퍼 1] 압축 및 레이어 교체 (파인튜닝 X) ---
    def _compress_and_replace_layer(linear_attr, name, in_hess_name, out_hess_name, rcp):
        """단일 리니어 레이어를 압축하고 모듈을 교체합니다. (파인튜닝 없음)"""
        
        utils.clean()
        
        ql = ql_i[linear_attr] if ql_i is not None else None
        orig_linear = attrgetter(linear_attr)(mixed_layer)
        W = orig_linear.weight.to(dtype_)
        in_hess_path = f'{args.in_hess_path}/{idx}_{in_hess_name}.pt'
        args.in_hess_name = in_hess_name
        
        H_data = torch.load(in_hess_path, map_location=torch.device('cpu'))
        HR = utils.flat_to_sym(H_data['flatH'], H_data['n'])
        n_h = H_data['n']
        if 'mu' in H_data:
            mu = H_data['mu']
            HR += mu[None, :] * mu[:, None]
            del mu
        del H_data
        HR = utils.regularize_H2(HR, n_h, args.sigma_reg)
        
        args.layer_idx = idx
        args.layer_name = name

        if args.handcraft_mode is not None:
            glog.info(f'Using handcraft compression method {args.handcraft_mode}')
            out, SU, SV, scaleWH, ft_result, optimize_out = handcraft.compress_linear(W.clone(), HR, args, 'cpu')
        elif args.nic_model is not None:
            out, SU, SV, scaleWH, ft_result, optimize_out = nic.compress_linear(W.clone(), comp_model, args, device = device)
        else:
            W, HR = W.to(device), HR.to(device)
            out = nwc.compress_linear(W.clone(), HR, comp_model, ql, args, device)

        glog.info(f'------------------------------------')
        trWHW = torch.trace(W @ HR @ W.T)
        hatWr = out['hatWr'].to(dtype_)
        W_hat = utils.de_standardize_Wr(hatWr, out['metadata'], args)
        assert torch.isnan(W_hat).any() == False
        
        bpp_loss = out['bpp_loss']
        bpp = out['bpp']        
        err = torch.trace((W - W_hat) @ HR @ ((W - W_hat).T))
        proxy_err =  err / trWHW
        mse =  torch.mean((W - W_hat) ** 2).item()        
        glog.info(
            f'{idx}_{name} optm proxy err {proxy_err.item():.5f} err {err.item():.3f} tr(WHW.T) {trWHW.item():.1f} bpp_loss {bpp_loss:.4f} bpp {bpp:.4f} mse {mse:.4g}'
        )
        glog.info(f'------------------------------------')

        metadata = out['metadata']
        if args.ft_comp_model2 or args.code_optim:
            utils.plot_ft_comp_result(ft_result, args, idx, name)

        if args.ft_rnorm == True:
            comp_linear = CompLinear(orig_linear.in_features,
                                    orig_linear.out_features,
                                    orig_linear.bias,
                                    orig_linear.weight.device,
                                    orig_linear.weight.dtype,
                                    )
            comp_linear.Wr.copy_(hatWr)
            comp_linear.Wr.requires_grad = False
            comp_linear.row_norm = nn.Parameter(metadata['row_std'], requires_grad=True)
        elif args.ft_metadata:
            comp_linear = CompLinear2(orig_linear.in_features,
                                    orig_linear.out_features,
                                    orig_linear.bias,
                                    orig_linear.weight.device,
                                    orig_linear.weight.dtype,
                                    )
            comp_linear.args = utils.filter_compression_args(args)
            params_to_register = {
                key: nn.Parameter(value) 
                for key, value in metadata.items() 
                if isinstance(value, torch.Tensor) and value.is_floating_point()
            }
            params_to_register['scale_cond'].requires_grad = False
            comp_linear.metadata = nn.ParameterDict(params_to_register)
            if args.ft_Wr == True:
                assert args.ldlq == True
                comp_linear.Wr.copy_(out['Wr_ldlq'])
                comp_linear.Wr.requires_grad = True
                comp_linear.model = comp_model
                for param in comp_linear.model.parameters():
                    param.requires_grad = False
                comp_linear.model.eval()
                comp_linear.model.mode = 'ste'
                comp_linear.args.ldlq = False
                comp_linear.args.comp_batch_size = 1024
            else:
                comp_linear.hatWr = hatWr
                comp_linear.hatWr.requires_grad = False
        elif args.ft_y:
            comp_linear = CompLinear3(orig_linear.in_features,
                                    orig_linear.out_features,
                                    orig_linear.bias,
                                    orig_linear.weight.device,
                                    orig_linear.weight.dtype,
                                    )
            y_params = [
                nn.Parameter(y, requires_grad=False)
                for (y, s, e) in out['y_list'] 
                if isinstance(y, torch.Tensor)
            ]
            comp_linear.y_in_list = nn.ParameterList(y_params)
            y_idx = [
                (s, e)
                for (y, s, e) in out['y_list'] 
                if isinstance(y, torch.Tensor)
            ]
            comp_linear.y_in_idx = y_idx
            comp_linear.Wshape = W.shape
            comp_linear.model = comp_model
            for param in comp_linear.model.parameters():
                param.requires_grad = False
            comp_linear.model.eval()
            comp_linear.model.mode = 'ste'
            comp_linear.args = utils.filter_compression_args(args)
            comp_linear.args.ldlq = False
            comp_linear.args.comp_batch_size = 1024
            params_to_register = {
                key: nn.Parameter(value) 
                for key, value in metadata.items() 
                if isinstance(value, torch.Tensor) and value.is_floating_point()
            }
            comp_linear.metadata = nn.ParameterDict(params_to_register)
            comp_linear.qlevel = metadata['qlevel']  
        else:
            comp_linear = copy.deepcopy(orig_linear)
            comp_linear.weight.copy_(W_hat)
            comp_linear.weight.requires_grad = False
        del orig_linear

        split_attr = linear_attr.split('.')
        setattr(
            attrgetter('.'.join(split_attr[:-1]))(mixed_layer), split_attr[-1],
            comp_linear)

        save_path = f'{args.save_path}/{idx}_{name}.pt'
        
        for k, v in metadata.items():
            if isinstance(v, torch.Tensor):
                metadata[k] = v.cpu()
                
        if all(value is None for value in metadata.values()):
            W_hat = W_hat.cpu()
            hatWr = None
        else:
            W_hat = None
            hatWr = hatWr.cpu()
        torch.save(
            {
                'W_hat': W_hat if not args.use_codes else None,
                'hatWr': hatWr if not args.use_codes else None,
                'codes': out['codes'],
                'bpp_loss': bpp_loss,
                'bpp': bpp,
                'proxy_err': proxy_err.item(),
                'err': err.item(),
                'tr(WHW.T)': trWHW.item(),
                'mse': mse,
                'bpp_sum': out['bpp_sum'],
                'bpp_loss_sum': out['bpp_loss_sum'],
                'direction': args.direction,
                'num_pixels': out['num_pixels'],
                'metadata': metadata
            }, save_path)
        
        glog.info(f"Compressed and replaced: {idx}_{name}")

        del HR, W, W_hat, hatWr
        utils.clean()

    # --- [헬퍼 2] 파인튜닝 수행 ---
    def _finetune_if_needed(stage_name):
        """지정된 스테이지 이후에 파인튜닝을 수행합니다."""
        if args.ft_epochs > 0 and train_dl is not None:
            glog.info(f"--- Layer {idx}: Finetuning after {stage_name} compression ---")
            with torch.enable_grad():
                finetune_decoder_layer(mixed_layer, f'{idx}_{stage_name}', device,
                                   train_dl, valid_dl, orig_dtype, args, attention_mask, rotary_emb)
            glog.info(f"--- Layer {idx}: Finished finetuning after {stage_name} ---")
        else:
            glog.info(f"--- Layer {idx}: Skipping finetuning for {stage_name} ---")

    # --- [리팩토링] 메인 로직: 스테이지별 압축 및 파인튜닝 ---

    # 1. quant_order를 그룹으로 분리
    attn_layers = [q for q in quant_order if 'self_attn' in q[0]]
    gate_layers = [q for q in quant_order if q[1] == 'gate'] # 'gate' save_name 기준
    w1_layers = [q for q in quant_order if 'w1' in q[1]] # 'expert..._w1' save_name 기준
    w3_layers = [q for q in quant_order if 'w3' in q[1]]
    w2_layers = [q for q in quant_order if 'w2' in q[1]]

    # --- STAGE 1: Attention + Gate Layers (수정됨) ---
    combined_attn_gate = attn_layers + gate_layers
    glog.info(f"--- Layer {idx}: Processing {len(combined_attn_gate)} Attention + Gate Layers ---")
    for (linear_attr, name, in_hess_name, out_hess_name, rcp) in combined_attn_gate:
        _compress_and_replace_layer(linear_attr, name, in_hess_name, out_hess_name, rcp)
    _finetune_if_needed("Attention_and_Gate") # <--- 파인튜닝을 한 번만 호출

    # --- STAGE 2: W1 Experts ---
    glog.info(f"--- Layer {idx}: Processing {len(w1_layers)} W1 Expert Layers ---")
    for (linear_attr, name, in_hess_name, out_hess_name, rcp) in w1_layers:
        _compress_and_replace_layer(linear_attr, name, in_hess_name, out_hess_name, rcp)
    _finetune_if_needed("W1_Experts")

    # --- STAGE 3: W3 Experts ---
    glog.info(f"--- Layer {idx}: Processing {len(w3_layers)} W3 Expert Layers ---")
    for (linear_attr, name, in_hess_name, out_hess_name, rcp) in w3_layers:
        _compress_and_replace_layer(linear_attr, name, in_hess_name, out_hess_name, rcp)
    _finetune_if_needed("W3_Experts")

    # --- STAGE 4: W2 Experts ---
    glog.info(f"--- Layer {idx}: Processing {len(w2_layers)} W2 Expert Layers ---")
    for (linear_attr, name, in_hess_name, out_hess_name, rcp) in w2_layers:
        _compress_and_replace_layer(linear_attr, name, in_hess_name, out_hess_name, rcp)
    _finetune_if_needed("W2_Experts")

    glog.info(f"--- Layer {idx}: All compression stages complete. ---")

    # --- [기존 로직] 파인튜닝된 파라미터 저장 ---
    if args.ft_epochs > 0:
        if args.ft_rnorm:
            for quant_i, (linear_attr, name, in_hess_name, out_hess_name,
                        rcp) in enumerate(quant_order):
                quant_linear = attrgetter(linear_attr)(mixed_layer)
                save_path = f'{args.save_path}/{idx}_{name}.pt'
                data = torch.load(save_path)
                # data['row_norm'] = quant_linear.row_norm.data.to(orig_dtype).cpu()
                data['row_norm'] = quant_linear.row_norm.data.to(dtype_).cpu()
                torch.save(data, save_path)
        elif args.ft_metadata:
            for quant_i, (linear_attr, name, in_hess_name, out_hess_name,
                        rcp) in enumerate(quant_order):
                quant_linear = attrgetter(linear_attr)(mixed_layer)
                save_path = f'{args.save_path}/{idx}_{name}.pt'
                data = torch.load(save_path)
                if args.ft_Wr == True:
                    data['hatWr'] = quant_linear.out['hatWr'].data.to(dtype_).cpu()
                    data['bpp_loss'] = quant_linear.out['bpp_loss']
                    data['bpp_loss_sum'] = quant_linear.out['bpp_loss_sum']
                    data['bpp'] = quant_linear.out['bpp']
                    data['bpp_sum'] = quant_linear.out['bpp_sum']
                else:
                    assert torch.equal(data['hatWr'], quant_linear.hatWr.data.to(dtype_).cpu())
                tensors_to_update = {key: param.data for key, param in quant_linear.metadata.items()}
                data['metadata'].update(tensors_to_update)
                torch.save(data, save_path)
        # (ft_y 관련 로직은 생략되었으므로 유지)

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


def finetune_e2e(quant_model, start_dev, devset, orig_dtype, args):

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
    # best_loss = 0
    
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
