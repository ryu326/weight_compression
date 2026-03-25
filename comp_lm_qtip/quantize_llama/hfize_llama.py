import argparse
import os
import time

import glog
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from lib import codebook, utils
# from lib.utils.unsafe_import import model_from_hf_path
# from model.llama import LlamaForCausalLM
from transformers import LlamaForCausalLM as OrigLlama
from torch import nn
import json
try:
    from model.llama import LlamaForCausalLM as LocalLlama
except Exception:
    LocalLlama = None

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--quantized_path', type=str)
parser.add_argument('--hf_output_path', type=str)
parser.add_argument('--skip_list', type=str, default='')
parser.add_argument('--use_codes', action='store_true')
parser.add_argument('--W_key', type=str, default='')
parser.add_argument('--sep_rnorm', action='store_true')
parser.add_argument('--base_model', type=str)


def _is_gemma3_config(config):
    return getattr(config, 'model_type', '') in ('gemma3', 'gemma3_text')


def _get_text_model(model):
    return model.language_model if hasattr(model, 'language_model') else model.model


def main(args):
    assert os.path.exists(args.quantized_path)
    saved_config = torch.load(os.path.join(args.quantized_path, 'config.pt'), weights_only=False)
    model_config = saved_config['model_config']
    comp_config = saved_config['quant_args']
    glog.info(model_config)

    _name_or_path = args.base_model
    tokenizer = AutoTokenizer.from_pretrained(_name_or_path)
    is_gemma3 = _is_gemma3_config(model_config)
    if is_gemma3:
        model = AutoModelForCausalLM.from_pretrained(
            _name_or_path,
            torch_dtype='auto',
            low_cpu_mem_usage=True,
            config=model_config)
        orig_model = AutoModelForCausalLM.from_pretrained(
            _name_or_path,
            torch_dtype='auto',
            low_cpu_mem_usage=True,
            config=model_config)
    else:
        model_cls = OrigLlama
        if args.sep_rnorm and LocalLlama is not None:
            model_cls = LocalLlama
        model = model_cls.from_pretrained(_name_or_path,
                                          torch_dtype='auto',
                                          low_cpu_mem_usage=True,
                                          config=model_config)
        orig_model = OrigLlama.from_pretrained(_name_or_path,
                                               torch_dtype='auto',
                                               low_cpu_mem_usage=True,
                                               config=model_config)

    text_model = _get_text_model(model)
    orig_text_model = _get_text_model(orig_model)

    skip_list = args.skip_list.split(',') if args.skip_list else []
    glog.info(f'skipping {skip_list}')
    
    comp_result = {
        'bpp_loss': 0,
        'bpp': 0,
        'ppl': 0,
        'num_pixels': 0
    }    
    try:
        with open(os.path.join(args.quantized_path, 'config.json'), 'r') as f:
            saved_config = json.load(f)
        comp_result['config'] = saved_config
    except Exception as e:
        print(f"Failed to load config: {e}")

    cpu = torch.device('cpu')
    if os.path.exists(f'{args.quantized_path}/lmhead.pt'):
        lmhead_data = torch.load(f'{args.quantized_path}/lmhead.pt',
                                 map_location=cpu, weights_only=False)
        model.lm_head.weight.copy_(lmhead_data['lm_head'].to(model.lm_head.weight.dtype))
        text_model.norm.weight.copy_(lmhead_data['norm'].to(text_model.norm.weight.dtype))
    else:
        glog.info("lmhead.pt not found. Asserting model and orig_model heads/norms are identical...")
        assert torch.equal(model.lm_head.weight, orig_model.lm_head.weight), "LM heads do not match!"
        assert torch.equal(text_model.norm.weight, orig_text_model.norm.weight), "Final norms do not match!"
        assert torch.equal(text_model.embed_tokens.weight, orig_text_model.embed_tokens.weight), "Embeddings do not match!"
        glog.info("Assert OK: Heads and norms are identical.")
        
        
    weight_types = [
        ('q', 'self_attn', 'q_proj'),
        ('k', 'self_attn', 'k_proj'),
        ('v', 'self_attn', 'v_proj'),
        ('o', 'self_attn', 'o_proj'),
        ('up', 'mlp', 'up_proj'),
        ('gate', 'mlp', 'gate_proj'),
        ('down', 'mlp', 'down_proj'),
    ]

    for ii in range(len(text_model.layers)):
        layer = text_model.layers[ii]
        orig_layer = orig_text_model.layers[ii]

        if os.path.exists(f'{args.quantized_path}/{ii}_layernorm.pt'):
            ln_data = torch.load(f'{args.quantized_path}/{ii}_layernorm.pt',
                                 map_location=cpu, weights_only=False)
            layer.input_layernorm.weight.copy_(ln_data['input_layernorm'].to(
                layer.input_layernorm.weight.dtype))
            layer.post_attention_layernorm.weight.copy_(
                ln_data['post_attention_layernorm'].to(
                    layer.post_attention_layernorm.weight.dtype))
            if hasattr(layer, 'pre_feedforward_layernorm') and 'pre_feedforward_layernorm' in ln_data:
                layer.pre_feedforward_layernorm.weight.copy_(
                    ln_data['pre_feedforward_layernorm'].to(
                        layer.pre_feedforward_layernorm.weight.dtype))
            if hasattr(layer, 'post_feedforward_layernorm') and 'post_feedforward_layernorm' in ln_data:
                layer.post_feedforward_layernorm.weight.copy_(
                    ln_data['post_feedforward_layernorm'].to(
                        layer.post_feedforward_layernorm.weight.dtype))
        
        # --- 개선점: 정의된 가중치 타입 리스트를 순회하며 반복 작업 처리 ---
        for name, submodule_name, proj_name in weight_types:
            skip_key = f'{ii}_{name}'
            
            if skip_key not in skip_list:
                file_path = f'{args.quantized_path}/{skip_key}.pt'
                saved_layer = torch.load(file_path, map_location=cpu, weights_only=False)
                
                submodule = getattr(layer, submodule_name)
                proj_layer = getattr(submodule, proj_name)
                
                comp_result[f'{skip_key}.pt'] = {k: v for k, v in saved_layer.items() if not isinstance(v, torch.Tensor) and k != 'codes' and k != 'metadata'}
                comp_result['bpp_loss'] += saved_layer['bpp_loss_sum' + args.W_key]
                comp_result['num_pixels'] += saved_layer['num_pixels']
                comp_result['bpp'] += saved_layer['bpp_sum']
                
                if args.sep_rnorm:
                    try:
                        proj_layer.Wr.copy_(saved_layer['hatWr'])
                        # proj_layer.row_norm = nn.Parameter(saved_layer['metadata']['row_std'], requires_grad=True)         
                        proj_layer.row_norm.copy_(saved_layer['metadata']['row_std'])
                    except:
                        # proj_layer.Wr.copy_(saved_layer['W_hat']/saved_layer['row_norm'])
                        proj_layer.Wr.copy_(saved_layer['W_hat'])
                        proj_layer.row_norm = nn.Parameter(saved_layer['row_norm'])         
                        # proj_layer.row_norm.copy_(saved_layer['row_norm'])
                else:
                    W_hat = saved_layer['W_hat' + args.W_key]
                    if W_hat == None:
                        W_hat = utils.de_standardize_Wr(saved_layer['hatWr'], saved_layer['metadata'], comp_config)
                    proj_layer.weight.copy_(W_hat.to(proj_layer.weight.dtype))                

            else:
                glog.info(f'### skipping {skip_key} ###')
                # getattr과 setattr을 사용해 원본 모델의 레이어로 교체
                orig_submodule = getattr(orig_layer, submodule_name)
                submodule_to_update = getattr(layer, submodule_name)
                setattr(submodule_to_update, proj_name, getattr(orig_submodule, proj_name))

        glog.info(f'loaded layer {ii}')
        
    if comp_result['num_pixels'] > 0:
        comp_result['bpp_loss'] = comp_result['bpp_loss'] / comp_result['num_pixels']
        comp_result['bpp'] = comp_result['bpp'] / comp_result['num_pixels']
    
    if isinstance(comp_result['bpp_loss'], torch.Tensor):
        comp_result['bpp_loss'] = comp_result['bpp_loss'].item()
    if isinstance(comp_result['bpp'], torch.Tensor):
        comp_result['bpp'] = comp_result['bpp'].item()

    glog.info(f'saving model...')
    model.save_pretrained(args.hf_output_path, safe_serialization=True)
    tokenizer.save_pretrained(args.hf_output_path)
    del model
    
    file_path = f'{args.hf_output_path}_result.json'
    if os.path.exists(file_path):
        os.rename(file_path, f'{args.hf_output_path}_result_.json')
    with open(file_path, 'w') as f:
        json.dump(comp_result, f, indent=2)
        
    # (The rest of the script for generation remains commented out as in the original)
    # ...

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    args = parser.parse_args()
    main(args)
