"""
Utilities for fine tuning
"""
import copy
import math
import os
from contextlib import contextmanager
from operator import attrgetter
from types import SimpleNamespace

import glog
import torch
from torch import multiprocessing as mp
from torch import nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from transformers import AutoModelForCausalLM

try:
    import wandb
except ImportError:
    wandb = None
try:
    from transformers.masking_utils import (create_causal_mask,
                                            create_sliding_window_causal_mask)
except Exception:
    create_causal_mask = None
    create_sliding_window_causal_mask = None

from lib import utils
from lib.utils import ecft
# from lib.linear import QuantizedLinear
from lib.linear import CompLinear, CompLinear2, CompLinear3
from lib.linear.ec_linear import EntropyConstrainedLinear


# from . import ldlq
# from . import nwc
from . import nwc_refactory as nwc
from . import handcraft
from . import ecsq
from . import ec_linear_ft
# from . import nic

@contextmanager
def use_tf32():
    fp32_matmul_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision('high')
    yield
    torch.set_float32_matmul_precision(fp32_matmul_precision)


def _is_gemma3_layer(layer):
    return getattr(getattr(layer, "config", None), "model_type",
                   "") in ("gemma3", "gemma3_text")


def _qwen3_layer_config(layer):
    """Qwen3DecoderLayer doesn't store config on itself; pull it from
    layer.self_attn.config (set by Qwen3Attention's __init__)."""
    cfg = getattr(getattr(layer, "self_attn", None), "config", None)
    if cfg is None:
        cfg = getattr(layer, "config", None)
    return cfg


def _is_qwen3_layer(layer):
    cfg = _qwen3_layer_config(layer)
    return getattr(cfg, "model_type", "") in ("qwen3", "qwen3_moe")


def _build_qwen3_rotary_emb(layer, device):
    """Construct a Qwen3RotaryEmbedding from the layer's config so that we
    can pass position_embeddings=(cos, sin) to the layer forward — required
    by the externalized rotary API used in transformers >=4.51 for Qwen3."""
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    cfg = _qwen3_layer_config(layer)
    if cfg is None:
        raise RuntimeError(
            "Could not locate Qwen3 config on the decoder layer for rotary build.")
    return Qwen3RotaryEmbedding(config=cfg, device=device)


def _bidirectional_window_overlay(sliding_window):
    def inner_mask(batch_idx, head_idx, q_idx, kv_idx):
        return abs(q_idx - kv_idx) < sliding_window

    return inner_mask


def _build_gemma3_attention_masks(config, input_embeds, position_ids,
                                  cache_position):
    if create_causal_mask is None or create_sliding_window_causal_mask is None:
        raise RuntimeError(
            "Gemma3 masking utils are unavailable. "
            "Please upgrade transformers to >=4.57."
        )
    mask_kwargs = {
        "config": config,
        "input_embeds": input_embeds,
        "attention_mask": None,
        "cache_position": cache_position,
        "past_key_values": None,
        "position_ids": position_ids,
    }
    sliding_mask_kwargs = mask_kwargs.copy()
    if getattr(config, "use_bidirectional_attention", False):
        mask_kwargs["or_mask_function"] = lambda *args: torch.tensor(
            True, dtype=torch.bool)
        sliding_mask_kwargs["or_mask_function"] = _bidirectional_window_overlay(
            config.sliding_window)

    try:
        return {
            "full_attention": create_causal_mask(**mask_kwargs),
            "sliding_attention": create_sliding_window_causal_mask(
                **sliding_mask_kwargs),
        }
    except RuntimeError as exc:
        if ('Please clone()' in str(exc)
                and getattr(config, "_attn_implementation", "") == "sdpa"):
            config._attn_implementation = "eager"
            return {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(
                    **sliding_mask_kwargs),
            }
        raise


def finetune_decoder_layer(layer, name, device, train_dl, valid_dl, orig_dtype,
                           args):
    with use_tf32():
        layer = layer.to(device)

        source = next(iter(train_dl))[0]
        position_ids = torch.arange(source.shape[1], device=device).unsqueeze(0)
        is_gemma3 = _is_gemma3_layer(layer)
        is_qwen3 = _is_qwen3_layer(layer)
        gemma3_ctx = None
        qwen3_rotary = None
        layer_dtype = next(layer.parameters()).dtype
        if is_gemma3:
            try:
                from transformers.models.gemma3.modeling_gemma3 import \
                    Gemma3RotaryEmbedding
            except Exception as exc:
                raise RuntimeError(
                    "Gemma3 support requires transformers with "
                    "Gemma3RotaryEmbedding available.") from exc
            config = layer.config
            rotary_emb = Gemma3RotaryEmbedding(config=config, device=device)
            local_config = copy.deepcopy(config)
            local_config.rope_theta = local_config.rope_local_base_freq
            local_config.rope_scaling = {"rope_type": "default"}
            rotary_emb_local = Gemma3RotaryEmbedding(config=local_config,
                                                     device=device)
            cache_position = torch.arange(source.shape[1], device=device)
            dummy_input = source[:1].to(device=device, dtype=layer_dtype)
            position_embeddings_global = rotary_emb(dummy_input, position_ids)
            position_embeddings_local = rotary_emb_local(
                dummy_input, position_ids)
            causal_mask_mapping = _build_gemma3_attention_masks(
                config, dummy_input, position_ids, cache_position)
            gemma3_ctx = {
                "cache_position": cache_position,
                "position_ids": position_ids,
                "position_embeddings_global": position_embeddings_global,
                "position_embeddings_local": position_embeddings_local,
                "causal_mask_mapping": causal_mask_mapping,
            }

        if is_qwen3:
            qwen3_rotary = _build_qwen3_rotary_emb(layer, device)

        use_grad_ckpt = getattr(args, "ft_grad_ckpt", False)

        def _forward_layer_inner(input_states):
            input_states = input_states.to(device=device, dtype=layer_dtype)
            if not is_gemma3:
                kwargs = {"position_ids": position_ids}
                if is_qwen3:
                    kwargs["position_embeddings"] = qwen3_rotary(
                        input_states, position_ids)
                return layer(input_states, **kwargs)[0]
            layer_attention_mask = gemma3_ctx["causal_mask_mapping"][
                layer.attention_type]
            return layer(
                input_states,
                position_ids=gemma3_ctx["position_ids"],
                attention_mask=layer_attention_mask,
                use_cache=False,
                cache_position=gemma3_ctx["cache_position"],
                position_embeddings_global=gemma3_ctx[
                    "position_embeddings_global"],
                position_embeddings_local=gemma3_ctx[
                    "position_embeddings_local"],
                output_attentions=False)[0]

        def _forward_layer(input_states):
            if use_grad_ckpt and torch.is_grad_enabled():
                return grad_checkpoint(
                    _forward_layer_inner,
                    input_states,
                    use_reentrant=False,
                )
            return _forward_layer_inner(input_states)

        # manifest tensor parallel attributes in layer
        output = _forward_layer(source)
        
        best_sd = {k: v.cpu() for k, v in layer.state_dict().items()}
        utils.clean()

        ec_modules = [
            m for m in layer.modules() if isinstance(m, EntropyConstrainedLinear)
        ]
        use_ecft_decoder = bool(getattr(args, "ecft_decoder", False))
        ec_entropy_chunk_rows = int(getattr(args, "ec_entropy_chunk_rows", 0) or 0)
        for module in ec_modules:
            module.entropy_chunk_rows = ec_entropy_chunk_rows
        if use_ecft_decoder and not ec_modules:
            glog.warning(
                f'layer {name} ecft_decoder=True but no EntropyConstrainedLinear found; fallback to MSE fine-tuning'
            )
            use_ecft_decoder = False
        elif use_ecft_decoder:
            glog.info(
                f'layer {name} ecft_decoder entropy_chunk_rows={ec_entropy_chunk_rows}'
            )


        if use_ecft_decoder:
            # Block-level decoder fine-tune uses ft_lr (shared with non-ecft path).
            # ecft_learning_rate is reserved for single-sublayer ecft in ec_linear_ft.py.
            opt_args = SimpleNamespace(
                learning_rate=float(getattr(args, "ft_lr", 1e-4)),
                aux_learning_rate=float(getattr(args, "ecft_aux_learning_rate", 1e-3)),
            )
            optim, aux_optim = ec_linear_ft.configure_ec_optimizers(layer, opt_args)
        else:
            optim = torch.optim.Adam(layer.parameters(), lr=args.ft_lr)
            aux_optim = None

        def _calculate_mse_loss_local(dataloader):
            layer.eval()
            total_loss = 0
            ct = 0
            with torch.no_grad():
                for source, target in dataloader:
                    target = target.to(device, non_blocking=True)
                    total_loss += nn.MSELoss()(_forward_layer(source), target)
                    ct += 1
            layer.train()
            return (total_loss / ct).cpu().item()

        def _calculate_test_loss_local(dataloader):
            layer.eval()
            total_mse = 0.0
            total_bpp = 0.0
            ct = 0
            with torch.no_grad():
                for source, target in dataloader:
                    target = target.to(device, non_blocking=True)
                    output = _forward_layer(source)
                    mse_loss = nn.MSELoss()(output, target)
                    bpp_loss = sum(
                        m.bpp_loss for m in layer.modules()
                        if hasattr(m, 'bpp_loss'))
                    total_mse += mse_loss
                    total_bpp += bpp_loss
                    ct += 1
            layer.train()
            return (total_mse / ct).cpu().item(), (total_bpp / ct).cpu().item()

        def _get_initial_weights_local(dataloader):
            layer.eval()
            source, target = next(iter(dataloader))
            target = target.to(device)
            output = _forward_layer(source)
            l_mse = nn.MSELoss()(output, target)
            l_bpp = sum(
                m.bpp_loss for m in layer.modules() if hasattr(m, 'bpp_loss'))
            epsilon = 1e-8
            w_mse = l_bpp / (l_mse + l_bpp + epsilon)
            w_bpp = l_mse / (l_mse + l_bpp + epsilon)
            layer.train()
            return torch.tensor([w_mse, w_bpp], device=device)

        ecft_lmbda = float(getattr(args, "ecft_lmbda", 0.0) or 0.0)
        use_adaptive_lambda = use_ecft_decoder and bool(
            getattr(args, "ecft_adaptive_lambda", False)
        )
        if use_adaptive_lambda and getattr(args, "R_target", None) is None:
            raise ValueError(
                "ecft_adaptive_lambda=True requires --R_target to set target rate."
            )
        target_rate_value = (
            float(getattr(args, "R_target"))
            if getattr(args, "R_target", None) is not None
            else 0.0
        )
        lambda_rate = float(ecft_lmbda)
        lambda_rate_lr = float(getattr(args, "ecft_lambda_lr", 0.1))
        lambda_rate_min = float(getattr(args, "ecft_lambda_min", 0.0))
        lambda_rate_max = float(getattr(args, "ecft_lambda_max", 1e8))
        rate_ema_beta = float(getattr(args, "ecft_rate_ema_beta", 0.9))
        rate_tolerance = float(getattr(args, "ecft_rate_tolerance", 0.0))
        rate_ema = None
        ecft_dec_log_path = None
        wandb_run = None
        if use_ecft_decoder:
            ecft_dec_log_path = ecft.dec_log_path(args, name)
            ecft.dec_log_init(
                ecft_dec_log_path,
                epochs=int(args.ft_epochs),
                lmbda=ecft_lmbda,
                adaptive=use_adaptive_lambda,
                target_rate=target_rate_value if use_adaptive_lambda else None,
            )
            if bool(getattr(args, "use_wandb", False)) and wandb is not None:
                project = getattr(args, "wandb_project", "ecft_decoder")
                save_path = getattr(args, "save_path", None) or "run"
                run_name = f"{os.path.basename(os.path.dirname(save_path))}/{os.path.basename(save_path)}/{str(name)}"
                wandb_run = wandb.init(
                    project=project,
                    name=run_name,
                    group=getattr(args, "wandb_group", None),
                    config=vars(args) if hasattr(args, "__dict__") else None,
                    reinit=True,
                )

        init_mse = (
            _calculate_mse_loss_local(valid_dl)
            if is_gemma3
            else utils.calculate_mse_loss(layer, valid_dl, device)
        )
        if use_ecft_decoder:
            with torch.no_grad():
                init_rate = float(ecft.rate_loss_from_modules(ec_modules, training=False, device=device).item())
            if use_adaptive_lambda:
                best_loss = init_mse + lambda_rate * (init_rate - target_rate_value)
            else:
                best_loss = init_mse + ecft_lmbda * init_rate
            glog.info(
                f'layer {name} initial ecft loss {best_loss:.6f} '
                f'mse {init_mse:.6f} rate {init_rate:.6f} lambda {lambda_rate:.6f}'
            )
        else:
            best_loss = init_mse
            glog.info(f'layer {name} initial loss {best_loss}')

        scaler = torch.cuda.amp.GradScaler(enabled=(orig_dtype==torch.float16))
        worse_ct = 0

        # --------- 사전 설정 ----------
        k, tau = 5, 0.5                 # SoftAdapt 창 길이·민감도
        loss_hist = {'mse': [], 'bpp': []}
        initial_w = None
        if args.ft_bpp_loss:
            initial_w = _get_initial_weights_local(
                train_dl) if is_gemma3 else utils.get_initial_weights(
                    layer, train_dl, device)
        
        global_step = 0
        for epoch in range(args.ft_epochs):
            last_ec_rate = None
            last_ec_aux = 0.0
            ecft_epoch_loss_sum = 0.0
            ecft_epoch_mse_sum = 0.0
            ecft_epoch_rate_sum = 0.0
            ecft_epoch_aux_sum = 0.0
            ecft_epoch_batches = 0
            ecft_epoch_aux_steps = 0
            for bidx, (source, targets) in enumerate(train_dl):
                if use_ecft_decoder:
                    ecft.clear_forward_cache(ec_modules)
                targets = targets.to(device, non_blocking=True)
                used_ecft_objective = False
                with torch.autocast(device_type='cuda',
                                    dtype=orig_dtype,
                                    enabled=True):
                    output = _forward_layer(source)
                    loss = nn.MSELoss()(output, targets)
                    
                    if args.ft_bpp_loss:
                        mse_loss = loss
                        bpp_loss = sum(m.bpp_loss for m in layer.modules()
                           if isinstance(m, CompLinear2))

                        loss_hist['mse'].append(mse_loss.detach())
                        loss_hist['bpp'].append(bpp_loss.detach())
                        if len(loss_hist['mse']) > k:
                            loss_hist['mse'].pop(0)
                            loss_hist['bpp'].pop(0)
                        if len(loss_hist['mse']) == k:
                            mse_ma  = torch.stack(loss_hist['mse']).mean()
                            bpp_ma  = torch.stack(loss_hist['bpp']).mean()
                            r_mse   = (loss - mse_ma) / mse_ma
                            r_bpp   = (bpp_loss - bpp_ma) / bpp_ma
                            w = torch.softmax(torch.stack([r_mse, r_bpp]) / tau, dim=0)
                        else:
                            w = initial_w
                        loss = w[0] * mse_loss + w[1] * bpp_loss
                        # glog.info(
                        #     f'layer {name} @ epoch {epoch} bidx {bidx} loss {loss.item():.2f} mse {mse_loss.item():.4g} bpp {bpp_loss.item():.2f} w {w}'
                        # )
                    elif use_ecft_decoder:
                        mse_loss = loss
                        # Use likelihoods produced by the same forward pass
                        # so MSE and rate are computed from one quantization sample.
                        rate_loss = ecft.rate_loss_from_cache(ec_modules, device=device)
                        if use_adaptive_lambda:
                            loss = mse_loss + lambda_rate * (
                                rate_loss - target_rate_value
                            )
                        else:
                            loss = mse_loss + ecft_lmbda * rate_loss
                        last_ec_rate = float(rate_loss.detach().item())
                        used_ecft_objective = True

                if used_ecft_objective:
                    step_loss = float(loss.detach().item())
                    step_mse = float(mse_loss.detach().item())
                    step_rate = float(rate_loss.detach().item())
                    ecft_epoch_loss_sum += step_loss
                    ecft_epoch_mse_sum += step_mse
                    ecft_epoch_rate_sum += step_rate
                    ecft_epoch_batches += 1
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "train/loss": step_loss,
                                "train/mse": step_mse,
                                "train/rate": step_rate,
                                "train/lambda_rate": float(lambda_rate),
                                "epoch": epoch,
                                "bidx": bidx,
                            },
                            step=global_step,
                        )
                    global_step += 1
                                            
                scaler.scale(loss).backward()
                if bidx % args.ft_update_freq == args.ft_update_freq - 1 or bidx == len(
                        train_dl) - 1:
                    scaler.step(optim)
                    scaler.update()
                    optim.zero_grad()
                    if use_ecft_decoder and aux_optim is not None:
                        aux_optim.zero_grad(set_to_none=True)
                        aux_loss = ecft.aux_loss_from_modules(ec_modules, device=device)
                        aux_loss.backward()
                        aux_optim.step()
                        last_ec_aux = float(aux_loss.detach().item())
                        ecft_epoch_aux_sum += last_ec_aux
                        ecft_epoch_aux_steps += 1
                        if wandb_run is not None:
                            wandb_run.log({"train/aux": last_ec_aux}, step=global_step)
                    if use_ecft_decoder and use_adaptive_lambda and last_ec_rate is not None:
                        if rate_ema is None:
                            rate_ema = last_ec_rate
                        else:
                            rate_ema = (
                                rate_ema_beta * rate_ema
                                + (1.0 - rate_ema_beta) * last_ec_rate
                            )
                        rate_error = rate_ema - target_rate_value
                        if abs(rate_error) < rate_tolerance:
                            rate_error = 0.0
                        lambda_rate = float(
                            max(
                                lambda_rate_min,
                                min(
                                    lambda_rate_max,
                                    lambda_rate + lambda_rate_lr * rate_error,
                                    ),
                            )
                        )
                    if use_ecft_decoder:
                        ecft.clear_forward_cache(ec_modules)

            if use_ecft_decoder and ecft_epoch_batches > 0:
                epoch_row = {
                    "stage": "train",
                    "epoch": float(epoch + 1),
                    "loss": ecft_epoch_loss_sum / ecft_epoch_batches,
                    "mse_loss": ecft_epoch_mse_sum / ecft_epoch_batches,
                    "rate_loss": ecft_epoch_rate_sum / ecft_epoch_batches,
                    "aux_loss": (
                        ecft_epoch_aux_sum / ecft_epoch_aux_steps
                        if ecft_epoch_aux_steps > 0 else 0.0
                    ),
                    "lambda_rate": float(lambda_rate),
                }
                ecft.dec_log_append(ecft_dec_log_path, epoch_row)
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "epoch_train/loss": epoch_row["loss"],
                            "epoch_train/mse": epoch_row["mse_loss"],
                            "epoch_train/rate": epoch_row["rate_loss"],
                            "epoch_train/aux": epoch_row["aux_loss"],
                            "epoch_train/lambda_rate": epoch_row["lambda_rate"],
                            "epoch": epoch + 1,
                        },
                        step=global_step,
                    )

            if epoch % args.ft_valid_freq == (args.ft_valid_freq - 1):
                if args.ft_bpp_loss:
                    if is_gemma3:
                        avg_mse, avg_bpp = _calculate_test_loss_local(
                            valid_dl)
                    else:
                        avg_mse, avg_bpp = utils.calculate_test_loss(
                            layer, valid_dl, device)
                    glog.info(
                        f'layer {name} @ epoch {epoch} mse {avg_mse:.4g} bpp {avg_bpp:.2f} w '
                    )   
                elif use_ecft_decoder:
                    ecft.clear_forward_cache(ec_modules)
                    valid_mse = _calculate_mse_loss_local(
                        valid_dl) if is_gemma3 else utils.calculate_mse_loss(
                            layer, valid_dl, device)
                    with torch.no_grad():
                        valid_rate = float(ecft.rate_loss_from_modules(ec_modules, training=False, device=device).item())
                    if use_adaptive_lambda:
                        test_loss = valid_mse + lambda_rate * (
                            valid_rate - target_rate_value
                        )
                    else:
                        test_loss = valid_mse + ecft_lmbda * valid_rate
                    valid_row = {
                        "stage": "valid",
                        "epoch": float(epoch + 1),
                        "loss": float(test_loss),
                        "mse_loss": float(valid_mse),
                        "rate_loss": float(valid_rate),
                        "aux_loss": float(last_ec_aux),
                        "lambda_rate": float(lambda_rate),
                    }
                    ecft.dec_log_append(ecft_dec_log_path, valid_row)
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "valid/loss": valid_row["loss"],
                                "valid/mse": valid_row["mse_loss"],
                                "valid/rate": valid_row["rate_loss"],
                                "valid/aux": valid_row["aux_loss"],
                                "valid/lambda_rate": valid_row["lambda_rate"],
                                "epoch": epoch + 1,
                            },
                            step=global_step,
                        )

                    if test_loss < best_loss:
                        glog.info(
                            f'layer {name} @ epoch {epoch} new loss {test_loss:.6f} old loss {best_loss:.6f} BETTER '
                            f'(mse={valid_mse:.6f} rate={valid_rate:.6f} aux={last_ec_aux:.6f} lambda={lambda_rate:.6f})'
                        )
                        best_loss = test_loss
                        best_sd = {k: v.cpu() for k, v in layer.state_dict().items()}
                        utils.clean()
                        worse_ct = 0
                    else:
                        glog.info(
                            f'layer {name} @ epoch {epoch} new loss {test_loss:.6f} old loss {best_loss:.6f} WORSE '
                            f'(mse={valid_mse:.6f} rate={valid_rate:.6f} aux={last_ec_aux:.6f} lambda={lambda_rate:.6f})'
                        )
                        worse_ct += 1
                        if worse_ct >= args.ft_early_stop:
                            break
                    ecft.clear_forward_cache(ec_modules)
                else:
                    test_loss = _calculate_mse_loss_local(
                        valid_dl) if is_gemma3 else utils.calculate_mse_loss(
                            layer, valid_dl, device)
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

    if aux_optim is not None:
        del aux_optim
    del optim, train_dl, valid_dl

    if wandb_run is not None:
        wandb_run.finish()

    layer = layer.cpu()
    layer.load_state_dict(best_sd)
    utils.clean()


def compress_finetune_decoder_layer(mixed_layer, quant_order, idx, comp_model, ql_i, args,
                                    device, pre_orig_emb, orig_emb):
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

    if pre_orig_emb != None and orig_emb != None:
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
        # in_hess_path = f'{args.in_hess_path}/lang_{idx}_{in_hess_name}.pt'
        args.in_hess_name = in_hess_name
        # try:
        H_data = torch.load(in_hess_path, map_location=torch.device('cpu'), weights_only=False)
        HR = utils.flat_to_sym(H_data['flatH'], H_data['n'])
        n_h = H_data['n']
        if 'mu' in H_data:
            mu = H_data['mu']
            HR += mu[None, :] * mu[:, None]
            del mu
        del H_data
        # HR = utils.regularize_H(HR, args.sigma_reg)
        HR = utils.regularize_H2(HR, n_h, args.sigma_reg)
        # except:
        #     HR = torch.eye(W.shape[1])
        # comp_model.to(dtype_) ## TODO: check if this is needed
        args.layer_idx = idx
        args.layer_name = name

        # ===== helper: 1회 압축 실행 + metric 계산 =====
        def _run_compress_and_eval(W0, HR0, norm_tag: str):
            extras = {}
            if args.handcraft_mode is not None:
                glog.info(f'Using handcraft compression method {args.handcraft_mode}')
                W_cpu = W0.detach().to('cpu')
                HR_cpu = HR0.detach().to('cpu')
                out = handcraft.compress_linear(W_cpu.clone(), HR_cpu, args, 'cpu')
                W_dev, HR_dev = W_cpu, HR_cpu

            elif args.nic_model is not None:
                W_dev = W0.detach().to(device)
                HR_dev = HR0.detach().to(device)
                out, ft_result = nic.compress_linear(W_dev.clone(), comp_model, args, device=device)
            elif getattr(args, "ec_linear", False):
                W_dev = W0.detach().to(device)
                HR_dev = HR0.detach().to(device)
                out = ec_linear_ft.ec_linear_train(W_dev.clone(), HR_dev, args, device=device)
            elif getattr(args, 'ecsq', False):
                W_dev = W0.detach().to(device)
                HR_dev = HR0.detach().to(device)
                out = ecsq.uniform_ecsq_gpu(W_dev.clone(), HR_dev, args, device=device)
            else:
                W_dev = W0.detach().to(device)
                HR_dev = HR0.detach().to(device)
                out = nwc.compress_linear(W_dev.clone(), HR_dev, comp_model, ql, args, device)
            # ===== metric 계산 =====
            trWHW = torch.trace(W_dev @ HR_dev @ W_dev.T)
            hatWr = out['hatWr'].to(dtype_)
            W_hat = utils.de_standardize_Wr(hatWr, out['metadata'], args, comp_model)
            assert torch.isnan(W_hat).any() == False
            if W_hat.device != W_dev.device:
                W_hat = W_hat.to(W_dev.device)

            bpp_loss = out['bpp_loss']
            bpp = out['bpp']
            err = torch.trace((W_dev - W_hat) @ HR_dev @ ((W_dev - W_hat).T))
            proxy_err = err / trWHW
            mse = torch.mean((W_dev - W_hat) ** 2).item()

            glog.info(
                f'{idx}_{name} [{norm_tag}] optm proxy err {proxy_err.item():.5f} err {err.item():.3f} '
                f'tr(WHW.T) {trWHW.item():.1f} bpp_loss {bpp_loss:.4f} bpp {bpp:.4f} mse {mse:.4g} '
                f'(row_norm={args.row_normalize}, col_norm={args.col_normalize})'
            )

            return dict(
                out=out,
                W=W_dev,
                W_hat=W_hat,
                hatWr=hatWr,
                HR=HR_dev,
                proxy_err=float(proxy_err.item()),
                err=float(err.item()),
                trWHW=float(trWHW.item()),
                bpp_loss=float(bpp_loss),
                bpp=float(bpp),
                mse=float(mse),
                row_norm=bool(args.row_normalize),
                col_norm=bool(args.col_normalize),
            )

        # ===== normalization_search 처리 =====
        orig_row_norm = getattr(args, "row_normalize", False)
        orig_col_norm = getattr(args, "col_normalize", False)

        if getattr(args, 'normalization_search', False):
            glog.info('normalization_search=True: try row-only vs col-only and pick smaller proxy_err')

            candidates = [
                ("row_only", True,  False),
                ("col_only", False, True),
            ]

            best = None
            for tag, rnorm, cnorm in candidates:
                args.row_normalize = rnorm
                args.col_normalize = cnorm

                try:
                    res = _run_compress_and_eval(W, HR, tag)
                except RuntimeError as exc:
                    if '[NaN]' not in str(exc):
                        raise
                    glog.warning(
                        'normalization_search candidate failed: '
                        f'tag={tag} row_norm={rnorm} col_norm={cnorm} err={exc}'
                    )
                    continue

                if (best is None) or (res["proxy_err"] < best["proxy_err"]):
                    best = res

            if best is None:
                # 둘 다 실패하면 원래 설정으로 1회 실행
                args.row_normalize = orig_row_norm
                args.col_normalize = orig_col_norm
                glog.warning(
                    'normalization_search: all candidates failed; '
                    'falling back to original normalization settings'
                )
                best = _run_compress_and_eval(W, HR, "fallback_orig")

            # 선택된 설정을 args에 반영(이후 저장/로깅/후처리가 선택 결과 기준으로 진행됨)
            args.row_normalize = best["row_norm"]
            args.col_normalize = best["col_norm"]

            out = best["out"]
            W_hat = best["W_hat"]
            hatWr = best["hatWr"]

            glog.info(
                f'normalization_search selected: row_norm={args.row_normalize}, col_norm={args.col_normalize}, '
                f'proxy_err={best["proxy_err"]:.5f}'
            )

        else:
            best = _run_compress_and_eval(W, HR, "single")

            out = best["out"]
            W_hat = best["W_hat"]
            hatWr = best["hatWr"]

        glog.info('------------------------------------')

        metadata = out['metadata']
 
        if getattr(args, 'ecft_decoder', False) and out['metadata_fused']:
            comp_linear = out['layer']
        else:
            comp_linear = copy.deepcopy(orig_linear)
            comp_linear.weight.copy_(W_hat)
            comp_linear.weight.requires_grad = False
            # assert not torch.equal(orig_linear.weight.data, comp_linear.weight.data)
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

        metadata_total_raw = float(utils.calculate_metadata_bpp(metadata, W.shape, args))
        num_pixels = float(ecft.to_scalar(out["num_pixels"]))
        metadata_total = metadata_total_raw / num_pixels
        bpp_sum = float(ecft.to_scalar(out["bpp_sum"]))
        bpp_loss_sum = float(ecft.to_scalar(out["bpp_loss_sum"]))
        bpp_with_metadata = (bpp_sum + metadata_total_raw) / num_pixels
        bpp_loss_with_metadata = (bpp_loss_sum + metadata_total_raw) / num_pixels

        decoder_total, decoder_total_raw = ecft.decoder_totals_from_out(out)
        torch.save(
            {
                'W_hat': W_hat if not args.use_codes else None,
                'hatWr': hatWr if not args.use_codes else None,
                'codes': out['codes'],
                'bpp_loss': best['bpp_loss'],
                'bpp': best['bpp'],
                'proxy_err': best['proxy_err'],
                'err': best['err'],
                'tr(WHW.T)': best['trWHW'],
                'mse': best['mse'],
                'mse_normed': out['mse_normed'].item() if isinstance(out['mse_normed'], torch.Tensor) else out['mse_normed'],
                'bpp_sum': out['bpp_sum'],
                'bpp_loss_sum': out['bpp_loss_sum'],
                'bpp_with_metadata': bpp_with_metadata,
                'bpp_loss_with_metadata': bpp_loss_with_metadata,
                'metadata_total': metadata_total,
                'metadata_total_raw': metadata_total_raw,
                'direction': args.direction,
                'num_pixels': out['num_pixels'],
                'decoder_total': decoder_total,
                'decoder_total_raw': decoder_total_raw,
                'metadata': metadata
            }, save_path)

        if args.ft_epochs > 0:
            # should_run_ft = (not args.ecft_decoder) or (quant_i == len(quant_order) - 1)
            should_run_ft = True
            if should_run_ft:
                with torch.enable_grad():
                    finetune_decoder_layer(mixed_layer, f'{idx}_{name}', device, train_dl, valid_dl, orig_dtype, args,)
            # else:
            #     assert torch.equal(W_hat, attrgetter(linear_attr)(mixed_layer).weight)

        del HR, W, W_hat, hatWr
        utils.clean()

    mixed_layer = mixed_layer.to(orig_dtype).cpu()
    if int(getattr(args, "ft_epochs", 0) or 0) > 0 and args.ecft_decoder:
        decoder_ft_path = f'{args.save_path}/{idx}_decoder_ft.pt'
        torch.save(
            {
                "state_dict": mixed_layer.state_dict(),
                "layer_idx": idx,
                "ft_epochs": int(getattr(args, "ft_epochs", 0) or 0),
                "ecft_decoder": bool(getattr(args, "ecft_decoder", False)),
            },
            decoder_ft_path,
        )
        glog.info(f"saved finetuned decoder state: {decoder_ft_path}")
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
