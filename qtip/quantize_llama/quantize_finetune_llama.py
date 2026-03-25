import argparse
import os
import time

import glog

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

import torch
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_attn_mask_utils import \
    _prepare_4d_causal_attention_mask
try:
    from transformers.masking_utils import (create_causal_mask,
                                            create_sliding_window_causal_mask)
except Exception:
    create_causal_mask = None
    create_sliding_window_causal_mask = None

from lib import utils
from lib.algo import finetune
from lib.codebook import bitshift
from operator import attrgetter

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_cpu_threads', default=8, type=int)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--devset_size', default=384, type=int)
parser.add_argument('--ctx_size', default=4096, type=int)
parser.add_argument('--save_path', type=str)
parser.add_argument('--in_hess_path', type=str)
parser.add_argument('--base_model', type=str)
parser.add_argument('--attn_implementation', default=None, type=str)
parser.add_argument('--sigma_reg', default=1e-2, type=float)
parser.add_argument('--sigma_reg2', default=1e-2, type=float)
parser.add_argument('--scale_override', default=-1, type=float)
parser.add_argument('--codebook', type=str)
parser.add_argument('--use_fp64', action='store_true')
parser.add_argument('--no_use_buffered', action='store_true')
parser.add_argument('--sample_proc', default=1, type=int)
parser.add_argument('--lowmem_ldlq', action='store_true')
parser.add_argument('--ft_lr', default=3e-6, type=float)
parser.add_argument('--ft_bs', default=4, type=int)
parser.add_argument('--ft_update_freq', default=1, type=int)
parser.add_argument('--ft_epochs', default=5, type=int)
parser.add_argument('--ft_valid_freq', default=1, type=int)
parser.add_argument('--ft_valid_size', default=128, type=float)
parser.add_argument('--ft_early_stop', default=5, type=int)
parser.add_argument('--ft_grad_ckpt', action='store_true')
parser.add_argument('--td_x', default=16, type=int)
parser.add_argument('--td_y', default=16, type=int)
parser.add_argument('--L', default=16, type=int)
parser.add_argument('--K', default=2, type=int)
parser.add_argument('--V', default=2, type=int)
parser.add_argument('--tlut_bits', default=0, type=int)
parser.add_argument('--decode_mode', default='lut', type=str)
parser.add_argument('--ft_train_lut', action='store_true')
parser.add_argument('--split_for_tp', action='store_true')
parser.add_argument('--tp_rank', default=8, type=int)
parser.add_argument('--skip_list', default=None, type=str)


def check_exist(idx, args):
    suffix = ['q', 'k', 'v', 'o', 'up', 'down', 'layernorm']
    for _ in suffix:
        test = f'{args.save_path}/{idx}_{_}.pt'
        if not os.path.exists(test):
            return False
    return True


def _is_gemma3_model(config):
    return getattr(config, "model_type", "") in ("gemma3", "gemma3_text")


def _bidirectional_window_overlay(sliding_window):
    def inner_mask(batch_idx, head_idx, q_idx, kv_idx):
        return abs(q_idx - kv_idx) < sliding_window

    return inner_mask


def _build_gemma3_attention_masks(text_model, input_embeds, position_ids,
                                  cache_position):
    if create_causal_mask is None or create_sliding_window_causal_mask is None:
        raise RuntimeError(
            "Gemma3 masking utils are unavailable. "
            "Please upgrade transformers to >=4.57."
        )
    mask_kwargs = {
        "config": text_model.config,
        "input_embeds": input_embeds,
        "attention_mask": None,
        "cache_position": cache_position,
        "past_key_values": None,
        "position_ids": position_ids,
    }
    sliding_mask_kwargs = mask_kwargs.copy()
    if getattr(text_model.config, "use_bidirectional_attention", False):
        mask_kwargs["or_mask_function"] = lambda *args: torch.tensor(
            True, dtype=torch.bool)
        sliding_mask_kwargs["or_mask_function"] = _bidirectional_window_overlay(
            text_model.config.sliding_window)

    return {
        "full_attention": create_causal_mask(**mask_kwargs),
        "sliding_attention": create_sliding_window_causal_mask(
            **sliding_mask_kwargs),
    }


def quantize_llama_decoder(layer, idx, cb, args, device, pre_orig_emb,
                           orig_emb, model_config, skip_list):
    if check_exist(idx, args):
        return

    if skip_list is None:
        skip_list = []
        
    # layer name, save_name, input hessian file, output hessian file
    quant_order = []
    for thing in [
                ('self_attn.v_proj', 'v', 'qkv', 'v', 'col'),
                  ('self_attn.q_proj', 'q', 'qkv', 'q', 'col'),
                  ('self_attn.k_proj', 'k', 'qkv', 'k', 'col'),
                  ('self_attn.o_proj', 'o', 'o', 'o', 'row'),
                  ('mlp.up_proj', 'up', 'up', 'up', 'col'),
                  ('mlp.gate_proj', 'gate', 'up', 'gate', 'col'),
                  ('mlp.down_proj', 'down', 'down', 'down', 'row')
                ]:
        if f'{idx}_{thing[1]}' not in skip_list:
            quant_order.append(thing)
        else:
            attrgetter(thing[0])(layer).weight.requires_grad = False
            print(f'skipping {idx}_{thing[1]}')
        
    finetune.quantize_finetune_decoder_layer(layer, quant_order, idx, cb, args,
                                             device, pre_orig_emb, orig_emb)
    norm_dict = {
        'input_layernorm': layer.input_layernorm.weight,
        'post_attention_layernorm': layer.post_attention_layernorm.weight,
    }
    if hasattr(layer, 'pre_feedforward_layernorm'):
        norm_dict['pre_feedforward_layernorm'] = \
            layer.pre_feedforward_layernorm.weight
    if hasattr(layer, 'post_feedforward_layernorm'):
        norm_dict['post_feedforward_layernorm'] = \
            layer.post_feedforward_layernorm.weight
    torch.save(norm_dict, f'{args.save_path}/{idx}_layernorm.pt')


def main(args):
    if args.skip_list is not None:
        args.skip_list = args.skip_list.split(',')
        
    dtype_ = torch.float64 if args.use_fp64 else torch.float32

    cb = bitshift.bitshift_codebook(L=args.L,
                                    K=args.K,
                                    V=args.V,
                                    tlut_bits=args.tlut_bits,
                                    decode_mode=args.decode_mode)
    model = AutoModelForCausalLM.from_pretrained(args.base_model,
                                                 torch_dtype='auto',
                                                 low_cpu_mem_usage=True)
    text_model = model.language_model if hasattr(model,
                                                 'language_model') else model.model
    is_gemma3 = _is_gemma3_model(text_model.config)
    if args.attn_implementation is not None:
        text_model.config._attn_implementation = args.attn_implementation

    # save configs
    all_config = {'quant_args': args, 'model_config': text_model.config}
    quip_params = {
        'codebook': args.codebook,
        'codebook_version': cb.version,
        'L': args.L,
        'K': args.K,
        'V': args.V,
        'tlut_bits': args.tlut_bits,
        'decode_mode': args.decode_mode,
        'td_x': args.td_x,
        'td_y': args.td_y,
        'split_for_tp': args.split_for_tp,
        'skip_list': args.skip_list,
    }
    all_config['model_config'].update({'quip_params': quip_params})
    torch.save(all_config, os.path.join(args.save_path, 'config.pt'))

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    glog.info('loaded model')

    devset = utils.sample_rp1t(tokenizer, args.devset_size, args.ctx_size,
                               args.sample_proc)
    glog.info('loaded dataset and devset')

    nproc = torch.cuda.device_count()
    orig_emb_cache = [text_model.embed_tokens(devset)]

    for _ in range(nproc):
        orig_emb_cache.append(
            torch.zeros(orig_emb_cache[0].shape,
                        dtype=orig_emb_cache[0].dtype,
                        device=orig_emb_cache[0].device))

    position_ids = torch.arange(args.ctx_size, dtype=torch.int64)[None, :] + \
        torch.zeros(args.batch_size, args.ctx_size, dtype=torch.int64)
    attention_mask = None
    causal_mask_mapping = None
    cache_position = None
    if is_gemma3:
        cache_position = torch.arange(args.ctx_size, dtype=torch.int64)
        try:
            causal_mask_mapping = _build_gemma3_attention_masks(
                text_model, orig_emb_cache[0][:args.batch_size], position_ids,
                cache_position)
        except RuntimeError as exc:
            if ('Please clone()' in str(exc)
                    and text_model.config._attn_implementation == 'sdpa'):
                glog.warning(
                    'SDPA mask path failed; falling back to eager attention.')
                text_model.config._attn_implementation = 'eager'
                causal_mask_mapping = _build_gemma3_attention_masks(
                    text_model, orig_emb_cache[0][:args.batch_size],
                    position_ids, cache_position)
            else:
                raise
    else:
        attention_mask = _prepare_4d_causal_attention_mask(
            None, (args.batch_size, args.ctx_size),
            orig_emb_cache[0][:args.batch_size], 0)

    cur_device = 0
    proc_list = [None for _ in range(nproc)]
    for i in range(len(text_model.layers)):
        glog.info(f'layer {i} gpu {cur_device}')
        if proc_list[cur_device] is not None:
            proc_list[cur_device][0].join()
            text_model.layers[proc_list[cur_device][1]] = None
            utils.clean()
            if cur_device == 0:
                orig_emb_cache[0].copy_(orig_emb_cache[-1])
        ### ryu

        if args.ft_epochs > 0:
            if cur_device + 1 < nproc and proc_list[cur_device + 1] is not None:
                proc_list[cur_device + 1][0].join()
        else:
            # ft_epochs == 0 일 때는 기다리지 않음 (병렬성 극대화)
            pass
        # if cur_device + 1 < nproc and proc_list[cur_device + 1] is not None:
        #     proc_list[cur_device + 1][0].join()
        
        utils.clean()
        st = time.time()
        position_ids = position_ids.to(cur_device)
        if is_gemma3:
            cache_position = cache_position.to(cur_device)
            layer_attention_mask = causal_mask_mapping[
                text_model.layers[i].attention_type]
            if layer_attention_mask is not None:
                layer_attention_mask = layer_attention_mask.to(cur_device)
        else:
            attention_mask = attention_mask.to(cur_device)
        text_model.layers[i].to(cur_device)
        ##
        if args.ft_epochs > 0:
            for j in range(args.devset_size // args.batch_size):
                utils.clean()
                if is_gemma3:
                    tmp_input = orig_emb_cache[cur_device][
                        args.batch_size * j:args.batch_size * (j + 1)].to(
                            cur_device)
                    position_embeddings_global = text_model.rotary_emb(
                        tmp_input, position_ids)
                    position_embeddings_local = text_model.rotary_emb_local(
                        tmp_input, position_ids)
                    orig_emb_cache[cur_device + 1][
                        args.batch_size * j:args.batch_size *
                        (j + 1)] = text_model.layers[i](
                            tmp_input,
                            position_ids=position_ids,
                            attention_mask=layer_attention_mask,
                            use_cache=False,
                            cache_position=cache_position,
                            position_embeddings_global=
                            position_embeddings_global,
                            position_embeddings_local=position_embeddings_local,
                            output_attentions=False)[0].cpu()
                else:
                    orig_emb_cache[cur_device + 1][args.batch_size * j : args.batch_size * (j + 1)] = \
                        text_model.layers[i](
                            orig_emb_cache[cur_device][args.batch_size * j : args.batch_size * (j + 1)].to(cur_device),
                            position_ids=position_ids,
                            attention_mask=attention_mask,
                            use_cache=False,
                            output_attentions=False)[0].cpu()
        else:
            orig_emb_cache[cur_device + 1] = torch.zeros_like(orig_emb_cache[cur_device])
            # orig_emb_cache[cur_device + 1] = orig_emb_cache[cur_device]
        text_model.layers[i].cpu()
        position_ids = position_ids.cpu()
        if is_gemma3:
            cache_position = cache_position.cpu()
            if layer_attention_mask is not None:
                layer_attention_mask = layer_attention_mask.cpu()
        else:
            attention_mask = attention_mask.cpu()
        utils.clean()
        glog.info('computed original embedding for layer {} in {}s'.format(i, time.time() - st))

        proc_list[cur_device] = (mp.Process(target=quantize_llama_decoder,
                                            args=(
                                                text_model.layers[i],
                                                i,
                                                cb,
                                                args,
                                                cur_device,
                                                orig_emb_cache[cur_device],
                                                orig_emb_cache[cur_device + 1],
                                                all_config['model_config'],
                                                args.skip_list
                                            )), i)
        proc_list[cur_device][0].start()

        cur_device = (cur_device + 1) % nproc

    for p in proc_list:
        p[0].join()


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    mp.set_start_method('spawn')
    mp.set_sharing_strategy('file_system')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.save_path, exist_ok=True)
    main(args)
