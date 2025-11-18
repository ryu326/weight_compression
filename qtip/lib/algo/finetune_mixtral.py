"""
Utilities for fine tuning
"""
import copy
import math
from contextlib import contextmanager
from operator import attrgetter

import glog
import torch
from torch import multiprocessing as mp
from torch import nn
from transformers import AutoModelForCausalLM

from lib import codebook, utils
from lib.linear import QuantizedLinear

from . import ldlq


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
        # attention_mask = attention_mask.to(device)
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
                scaler.scale(loss).backward()
                if bidx % args.ft_update_freq == args.ft_update_freq - 1 or bidx == len(
                        train_dl) - 1:
                    scaler.step(optim)
                    scaler.update()
                    optim.zero_grad()

            if epoch % args.ft_valid_freq == (args.ft_valid_freq - 1):
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

# --- [리팩토링] ---
def quantize_finetune_decoder_layer(mixed_layer, quant_order, idx, cb, args,
                                    device, pre_orig_emb, orig_emb, attention_mask, rotary_emb):
    torch.manual_seed(idx)
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

    # --- [헬퍼 1] 양자화 및 레이어 교체 (파인튜닝 X) ---
    def _quantize_and_replace_layer(linear_attr, name, in_hess_name, out_hess_name, rcp):
        """단일 리니어 레이어를 양자화하고 모듈을 교체합니다. (파인튜닝 없음)"""
        
        # 1. 원본 td_x 값을 저장 (td_y가 아닌 td_x를 수정하고 있었음)
        original_td_x = args.td_x
        
        if name == 'gate':
            args.td_x = max(1, args.td_x // 2)
            glog.info(f"Layer {idx}_{name}: Halving td_x to {args.td_x}")

        # 3. 수정된 td_x/td_y를 기반으로 has_kernel을 결정
        has_kernel = utils.has_kernel(args.decode_mode, args.L, args.K, args.V,
                                      args.tlut_bits, args.td_x, args.td_y)
        
        nonlocal cb
        
        utils.clean()
        cb_device = cb.to(device).to(orig_dtype)
        orig_linear = attrgetter(linear_attr)(mixed_layer)
        W = orig_linear.weight.to(dtype_)
        use_bias = (orig_linear.bias is not None)
        bias = orig_linear.bias
        del orig_linear
        (m, n) = W.shape
        SU = (torch.randn(n, device=device).sign() + 1e-5).sign().to(dtype_)
        SV = (torch.randn(m, device=device).sign() + 1e-5).sign().to(dtype_)

        in_hess_path = f'{args.in_hess_path}/{idx}_{in_hess_name}.pt'
        H_data = torch.load(in_hess_path, map_location=torch.device('cpu'), weights_only=False)
        HR = utils.flat_to_sym(H_data['flatH'], H_data['n'])
        if 'mu' in H_data:
            mu = H_data['mu']
            HR += mu[None, :] * mu[:, None]
            del mu
        del H_data

        HR = utils.regularize_H(HR, args.sigma_reg)

        if args.split_for_tp:
            # ... (split_for_tp 로직은 동일) ...
            if rcp == 'col':
                Wr = utils.matmul_hadUt(
                    utils.matmul_hadUt((W.T.to(device) * SV).reshape(
                        n * args.tp_rank, m // args.tp_rank)).reshape(
                            W.T.shape).T * SU)
                HRr = utils.matmul_hadUt(
                    utils.matmul_hadUt(HR.to(device) * SU).T * SU)
                Wscale = Wr.reshape(
                    args.tp_rank, m * n // args.tp_rank).square().mean(
                        dim=-1).sqrt() / (cb_device.lut.to(
                            torch.float64).square().mean().sqrt().float() *
                                        args.scale_override)
                Wr = Wr.reshape(args.tp_rank,
                                m * n // args.tp_rank) / Wscale.unsqueeze(-1)
                Wr = Wr.reshape(m, n)
            elif rcp == 'row':
                Wr = utils.matmul_hadUt(
                    (utils.matmul_hadUt(W.T.to(device) * SV).T * SU).reshape(
                        m * args.tp_rank, n // args.tp_rank)).reshape(W.shape)
                HRr = utils.matmul_hadUt(
                    (utils.matmul_hadUt((HR.to(device) * SU).reshape(
                        n * args.tp_rank, n // args.tp_rank)).reshape(n, n).T *
                    SU).reshape(n * args.tp_rank,
                                n // args.tp_rank)).reshape(n, n)
                Wscale = Wr.reshape(
                    m, args.tp_rank,
                    n // args.tp_rank).transpose(0, 1).reshape(
                        args.tp_rank, m * n // args.tp_rank).square().mean(
                            dim=-1).sqrt() / (cb_device.lut.to(
                                torch.float64).square().mean().sqrt().float() *
                                            args.scale_override)
                Wr = Wr.reshape(m, args.tp_rank, n // args.tp_rank).transpose(
                    0, 1).reshape(args.tp_rank,
                                m * n // args.tp_rank) / Wscale.unsqueeze(-1)
                Wr = Wr.reshape(args.tp_rank, m,
                                n // args.tp_rank).transpose(0,
                                                            1).reshape(m, n)
        else:
            Wr = utils.matmul_hadUt(
                utils.matmul_hadUt(W.T.to(device) * SV).T * SU)
            HRr = utils.matmul_hadUt(
                utils.matmul_hadUt(HR.to(device) * SU).T * SU)
            Wscale = Wr.square().mean().sqrt() / (
                cb_device.lut.to(torch.float64).square().mean().sqrt().float() *
                args.scale_override)
            Wr /= Wscale

        LRr, _ = utils.block_LDL(HRr, args.td_y)
        diag = torch.arange(n, device=LRr.device)
        LRr[diag, diag] = 0

        hatWr, Qidxs = ldlq.LDLQ(Wr, LRr, cb_device, args, for_kernel=has_kernel)

        Qidxs = Qidxs.cpu()
        packed = cb_device.pack_trellis(
            Qidxs.reshape(m // args.td_x, args.td_x, n // args.td_y,
                            args.td_y // args.V).transpose(1, 2).reshape(
                                -1, args.td_x * args.td_y // args.V))

        if has_kernel:
            packed = packed.view(torch.uint8).view(-1, 2).flip(
                (-1, )).reshape(m // 16 // 2, 2, n // 16 // 2, 2, 16 * 16 // 8,
                                args.K).permute(0, 2, 4, 3, 1, 5).flip(
                                    (-1, )).contiguous().flatten().view(
                                        torch.int16).reshape(packed.shape)
        else:
            packed = packed.view(torch.int16)

        if rcp == 'col':
            Wr = (Wr.reshape(args.tp_rank, m * n // args.tp_rank) *
                Wscale.unsqueeze(-1)).reshape(m, n)
            hatWr = (hatWr.reshape(args.tp_rank, m * n // args.tp_rank) *
                Wscale.unsqueeze(-1)).reshape(m, n)
        elif rcp == 'row':
            Wr = Wr.reshape(m, args.tp_rank, n // args.tp_rank).transpose(
                0, 1).reshape(args.tp_rank, -1) * Wscale.unsqueeze(-1)
            Wr = Wr.reshape(args.tp_rank, m,
                            n // args.tp_rank).transpose(0, 1).reshape(m, n)
            hatWr = hatWr.reshape(m, args.tp_rank,
                                    n // args.tp_rank).transpose(0, 1).reshape(
                                        args.tp_rank, -1) * Wscale.unsqueeze(-1)
            hatWr = hatWr.reshape(args.tp_rank, m,
                                    n // args.tp_rank).transpose(0, 1).reshape(
                                        m, n)
        else:
            Wr *= Wscale
            hatWr *= Wscale

        err = torch.trace(
            (Wr - hatWr) @ HRr @ (Wr - hatWr).T) / torch.trace(Wr @ HRr @ Wr.T)
        print(
            f'{idx}_{name} proxy err {err.item()} tr(WHW.T) {torch.trace(Wr @ HRr @ Wr.T)}'
        )
        
        save_path = f'{args.save_path}/{idx}_{name}.pt'
        rcp_int = 0
        if args.split_for_tp:
            rcp_int = 1 if rcp == 'row' else 2

        torch.save(
            {
                'trellis': packed.cpu(),
                'SU': SU.to(orig_dtype).cpu(),
                'SV': SV.to(orig_dtype).cpu(),
                'Wscale': Wscale,
                'proxy_err': err.item(),
                'tr(WHW.T)': torch.trace(Wr @ HRr @ Wr.T).item(),
                'mse': torch.mean((Wr - hatWr) ** 2).item(),
                'tlut':
                cb_device.tlut.data.to(orig_dtype).cpu()
                if hasattr(cb_device, 'tlut') else None,
                'rcp': rcp_int,
                'tp_rank': args.tp_rank,
                'bias': bias
            }, save_path)

        del HRr, Wr, hatWr, LRr, Qidxs
        utils.clean()
        
        q_linear = QuantizedLinear(
            n, m, args.td_x, args.td_y, args.L, args.K, args.V,
            args.tlut_bits, args.decode_mode,
            mode='train-recons' if args.ft_train_lut else 'train-fixW',
            dtype=orig_dtype, grad_ckpt=args.ft_grad_ckpt, bias=use_bias
        )
        q_linear.trellis.copy_(packed)
        q_linear.SU.copy_(SU)
        q_linear.SV.copy_(SV)
        q_linear.rcp.copy_(rcp_int)
        q_linear.tp_rank.copy_(args.tp_rank)
        if use_bias:
            q_linear.bias.copy_(bias)
        q_linear = q_linear.to(device).float()
        del packed, SU, SV
        utils.clean()
        
        if rcp == 'row':
            q_linear.SU = nn.Parameter(
                (q_linear.SU.reshape(args.tp_rank, -1) *
                    Wscale.unsqueeze(-1)).reshape(q_linear.SU.shape),
                requires_grad=True)
            q_linear.SV = nn.Parameter(q_linear.SV, requires_grad=True)
        elif rcp == 'col':
            q_linear.SU = nn.Parameter(q_linear.SU, requires_grad=True)
            q_linear.SV = nn.Parameter(
                (q_linear.SV.reshape(args.tp_rank, -1) *
                    Wscale.unsqueeze(-1)).reshape(q_linear.SV.shape),
                requires_grad=True)
        else:
            q_linear.SU = nn.Parameter(q_linear.SU, requires_grad=True)
            q_linear.SV = nn.Parameter(q_linear.SV * Wscale,
                                        requires_grad=True)

        if q_linear.tlut is not None:
            q_linear.tlut.copy_(cb_device.tlut.data)
            q_linear.tlut.requires_grad = args.ft_train_lut

        split_attr = linear_attr.split('.')
        setattr(
            attrgetter('.'.join(split_attr[:-1]))(mixed_layer), split_attr[-1],
            q_linear)
        
        glog.info(f"Quantized and replaced: {idx}_{name}")

        # 4. td_x 값을 원래대로 복원
        args.td_x = original_td_x
        utils.clean()
        cb = cb.cpu() # cb_device는 로컬 변수이므로 cb를 cpu로 보냄

    # --- [헬퍼 2] 파인튜닝 수행 ---
    def _finetune_if_needed(stage_name):
        """지정된 스테이지 이후에 파인튜닝을 수행합니다."""
        if args.ft_epochs > 0 and train_dl is not None:
            glog.info(f"--- Layer {idx}: Finetuning after {stage_name} quantization ---")
            with torch.enable_grad():
                finetune_decoder_layer(mixed_layer, f'{idx}_{stage_name}', device,
                                   train_dl, valid_dl, orig_dtype, args, attention_mask, rotary_emb)
            glog.info(f"--- Layer {idx}: Finished finetuning after {stage_name} ---")
        else:
            glog.info(f"--- Layer {idx}: Skipping finetuning for {stage_name} ---")

    # --- [리팩토링] 메인 로직: 스테이지별 양자화 및 파인튜닝 ---

    # 1. quant_order를 그룹으로 분리
    attn_layers = [q for q in quant_order if 'self_attn' in q[0]]
    gate_layers = [q for q in quant_order if q[1] == 'gate'] # 'gate' save_name 기준
    w1_layers = [q for q in quant_order if 'w1' in q[1]] # 'expert..._w1' save_name 기준
    w3_layers = [q for q in quant_order if 'w3' in q[1]]
    w2_layers = [q for q in quant_order if 'w2' in q[1]]

    # # --- STAGE 1: Attention Layers ---
    # glog.info(f"--- Layer {idx}: Processing {len(attn_layers)} Attention Layers ---")
    # for (linear_attr, name, in_hess_name, out_hess_name, rcp) in attn_layers:
    #     _quantize_and_replace_layer(linear_attr, name, in_hess_name, out_hess_name, rcp)
    # _finetune_if_needed("Attention")

    # # --- STAGE 2: Gate Layer ---
    # glog.info(f"--- Layer {idx}: Processing {len(gate_layers)} Gate Layer ---")
    # for (linear_attr, name, in_hess_name, out_hess_name, rcp) in gate_layers:
    #     _quantize_and_replace_layer(linear_attr, name, in_hess_name, out_hess_name, rcp)
    # _finetune_if_needed("Gate")
    
    combined_attn_gate = attn_layers + gate_layers
    glog.info(f"--- Layer {idx}: Processing {len(combined_attn_gate)} Attention + Gate Layers ---")
    for (linear_attr, name, in_hess_name, out_hess_name, rcp) in combined_attn_gate:
        _quantize_and_replace_layer(linear_attr, name, in_hess_name, out_hess_name, rcp)
    _finetune_if_needed("Attention_and_Gate")

    # --- STAGE 3: W1 Experts ---
    glog.info(f"--- Layer {idx}: Processing {len(w1_layers)} W1 Expert Layers ---")
    for (linear_attr, name, in_hess_name, out_hess_name, rcp) in w1_layers:
        _quantize_and_replace_layer(linear_attr, name, in_hess_name, out_hess_name, rcp)
    _finetune_if_needed("W1_Experts")

    # --- STAGE 4: W3 Experts ---
    glog.info(f"--- Layer {idx}: Processing {len(w3_layers)} W3 Expert Layers ---")
    for (linear_attr, name, in_hess_name, out_hess_name, rcp) in w3_layers:
        _quantize_and_replace_layer(linear_attr, name, in_hess_name, out_hess_name, rcp)
    _finetune_if_needed("W3_Experts")

    # --- STAGE 5: W2 Experts ---
    glog.info(f"--- Layer {idx}: Processing {len(w2_layers)} W2 Expert Layers ---")
    for (linear_attr, name, in_hess_name, out_hess_name, rcp) in w2_layers:
        _quantize_and_replace_layer(linear_attr, name, in_hess_name, out_hess_name, rcp)
    _finetune_if_needed("W2_Experts")

    glog.info(f"--- Layer {idx}: All quantization stages complete. ---")

    # --- [기존 로직] 파인튜닝된 SU/SV/tlut 파라미터 저장 ---
    # 이 루프는 모든 파인튜닝이 끝난 후, 최종 튜닝된 파라미터를 .pt 파일에 덮어쓰기 위해
    # 원래부터 존재했습니다. 이 로직은 그대로 유지되어야 합니다.
    glog.info(f"--- Layer {idx}: Saving final finetuned parameters... ---")
    for quant_i, (linear_attr, name, in_hess_name, out_hess_name,
                  rcp) in enumerate(quant_order):
        quant_linear = attrgetter(linear_attr)(mixed_layer)
        save_path = f'{args.save_path}/{idx}_{name}.pt'
        data = torch.load(save_path, map_location=torch.device('cpu'), weights_only=False)
        
        # Wscale은 device에 있어야 함
        wscale_device = quant_linear.SU.device
        wscale = data['Wscale'].to(wscale_device)

        if rcp == 'row':
            data['SU'] = (
                ((quant_linear.SU.data).reshape(args.tp_rank, -1) /
                 wscale.unsqueeze(-1)
                 ).reshape(quant_linear.SU.data.shape)).to(orig_dtype).cpu()
            data['SV'] = quant_linear.SV.data.to(orig_dtype).cpu()
        elif rcp == 'col':
            data['SU'] = quant_linear.SU.data.to(orig_dtype).cpu()
            data['SV'] = (
                ((quant_linear.SV.data).reshape(args.tp_rank, -1) /
                 wscale.unsqueeze(-1)
                 ).reshape(quant_linear.SV.data.shape)).to(orig_dtype).cpu()
        else:
            data['SU'] = quant_linear.SU.data.to(orig_dtype).cpu()
            data['SV'] = (quant_linear.SV.data / wscale).to(orig_dtype).cpu()

        if quant_linear.tlut is not None:
            data['tlut'] = quant_linear.tlut.data.to(orig_dtype).cpu()
        
        if quant_linear.bias is not None: # [추가] 튜닝된 bias 저장
            data['bias'] = quant_linear.bias.data.to(orig_dtype).cpu()
            
        torch.save(data, save_path)

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
    # start_dev = 2
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
    # best_loss = 0 # for test
    
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