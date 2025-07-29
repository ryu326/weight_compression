import argparse
import os
import time

import glog
import torch
from transformers import AutoTokenizer

from lib import codebook, utils
from lib.utils.unsafe_import import model_from_hf_path
# from model.llama import LlamaForCausalLM
from transformers import LlamaForCausalLM as OrigLlama
import json

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--quantized_path', type=str)
parser.add_argument('--hf_output_path', type=str)
parser.add_argument('--skip_list', type=str, default='')
parser.add_argument('--use_codes', action='store_true')
parser.add_argument('--W_key', type=str, default='')

def main(args):
    assert os.path.exists(args.quantized_path)
    saved_config = torch.load(os.path.join(args.quantized_path, 'config.pt'), weights_only=False)
    model_config = saved_config['model_config']
    glog.info(model_config)

    tokenizer = AutoTokenizer.from_pretrained(model_config._name_or_path)
    
    model = OrigLlama.from_pretrained(model_config._name_or_path,
                                      torch_dtype='auto',
                                      low_cpu_mem_usage=True,
                                      config=model_config)

    orig_model = OrigLlama.from_pretrained(model_config._name_or_path,
                                           torch_dtype='auto',
                                           low_cpu_mem_usage=True,
                                           config=model_config)

    skip_list = args.skip_list.split(',') if args.skip_list else []
    
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
        model.model.norm.weight.copy_(lmhead_data['norm'].to(model.model.norm.weight.dtype))

    # --- 개선점: 반복되는 가중치 타입 정보를 리스트로 정의 ---
    # 각 튜플은 (가중치 약어, 상위 모듈 이름, projection 레이어 이름)을 가집니다.
    weight_types = [
        ('q', 'self_attn', 'q_proj'),
        ('k', 'self_attn', 'k_proj'),
        ('v', 'self_attn', 'v_proj'),
        ('o', 'self_attn', 'o_proj'),
        ('up', 'mlp', 'up_proj'),
        ('gate', 'mlp', 'gate_proj'),
        ('down', 'mlp', 'down_proj'),
    ]

    for ii in range(len(model.model.layers)):
        layer = model.model.layers[ii]
        orig_layer = orig_model.model.layers[ii]

        if os.path.exists(f'{args.quantized_path}/{ii}_layernorm.pt'):
            ln_data = torch.load(f'{args.quantized_path}/{ii}_layernorm.pt',
                                 map_location=cpu, weights_only=False)
            layer.input_layernorm.weight.copy_(ln_data['input_layernorm'].to(
                layer.input_layernorm.weight.dtype))
            layer.post_attention_layernorm.weight.copy_(
                ln_data['post_attention_layernorm'].to(
                    layer.post_attention_layernorm.weight.dtype))
        
        # --- 개선점: 정의된 가중치 타입 리스트를 순회하며 반복 작업 처리 ---
        for name, submodule_name, proj_name in weight_types:
            skip_key = f'{ii}_{name}'
            
            if skip_key not in skip_list:
                file_path = f'{args.quantized_path}/{skip_key}.pt'
                saved_layer = torch.load(file_path, map_location=cpu, weights_only=False)
                
                W_hat = saved_layer['W_hat' + args.W_key]
                # if model_config.get('comp_params', {}).get('ft_rnorm'):
                if hasattr(model_config, 'comp_params') and model_config.comp_params.get('ft_rnorm'):
                    rnorm = saved_layer['row_norm']
                    W_hat = W_hat * rnorm  
                
                # getattr을 사용해 동적으로 모듈과 가중치에 접근
                submodule = getattr(layer, submodule_name)
                proj_layer = getattr(submodule, proj_name)
                proj_layer.weight.copy_(W_hat.to(proj_layer.weight.dtype))

                comp_result[f'{skip_key}.pt'] = {k: v for k, v in saved_layer.items() if not isinstance(v, torch.Tensor) and k != 'codes'}
                comp_result['bpp_loss'] += saved_layer['bpp_loss_sum' + args.W_key]
                comp_result['num_pixels'] += saved_layer['num_pixels']
                comp_result['bpp'] += saved_layer['bpp_sum']
            else:
                print(f'### skipping {skip_key} ###')
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