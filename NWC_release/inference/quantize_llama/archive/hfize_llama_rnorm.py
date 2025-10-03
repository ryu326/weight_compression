import argparse
import os
import time

import glog
import torch
from transformers import AutoTokenizer

from lib import codebook, utils
from lib.utils.unsafe_import import model_from_hf_path
from model.llama import LlamaForCausalLM
from transformers import LlamaForCausalLM as OrigLlama
import json

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--quantized_path', type=str)
parser.add_argument('--hf_output_path', type=str)
parser.add_argument('--skip_list', type=str, default='') # 기본값을 빈 문자열로 설정
parser.add_argument('--use_codes', action='store_true')
parser.add_argument('--W_key', type=str, default='')

def main(args):
    assert os.path.exists(args.quantized_path)
    saved_config = torch.load(os.path.join(args.quantized_path, 'config.pt'), weights_only=False)
    model_config = saved_config['model_config']
    glog.info(model_config)

    tokenizer = AutoTokenizer.from_pretrained(model_config._name_or_path)

    model = LlamaForCausalLM.from_pretrained(model_config._name_or_path,
                                                torch_dtype='auto',
                                                low_cpu_mem_usage=True,
                                                config=model_config)

    orig_model = OrigLlama.from_pretrained(model_config._name_or_path,
                                           torch_dtype='auto',
                                           low_cpu_mem_usage=True,
                                           config=model_config)

    print("--- 모델 생성 직후 메모리 주소 확인 ---")
    addr1 = id(model.model.layers[27].self_attn.o_proj.row_norm)
    addr2 = id(model.model.layers[29].self_attn.q_proj.row_norm)
    print(f"Layer 27 row_norm 주소: {addr1}")
    print(f"Layer 29 row_norm 주소: {addr2}")
    print(f"주소 동일 여부: {addr1 == addr2}")

    skip_list = args.skip_list.split(',') if args.skip_list else []

    comp_result = {
        'bpp_loss': 0,
        'bpp': 0,
        'ppl': 0,
        'num_pixels': 0,
    }

    try:
        with open(os.path.join(args.quantized_path, 'config.json'), 'r') as f:
            comp_result['config'] = json.load(f)
    except Exception as e:
        print(f"Failed to load config: {e}")

    cpu = torch.device('cpu')
    if os.path.exists(f'{args.quantized_path}/lmhead.pt'):
        lmhead_data = torch.load(f'{args.quantized_path}/lmhead.pt',
                                 map_location=cpu, weights_only=False)
        model.lm_head.weight.copy_(lmhead_data['lm_head'].to(model.lm_head.weight.dtype))
        model.model.norm.weight.copy_(lmhead_data['norm'].to(model.model.norm.weight.dtype))

    for ii in range(len(model.model.layers)):
        layer = model.model.layers[ii]
        orig_layer = orig_model.model.layers[ii]

        if os.path.exists(f'{args.quantized_path}/{ii}_layernorm.pt'):
            ln_data = torch.load(f'{args.quantized_path}/{ii}_layernorm.pt',
                                 map_location=cpu, weights_only=False)
            layer.input_layernorm.weight.copy_(ln_data['input_layernorm'].to(layer.input_layernorm.weight.dtype))
            layer.post_attention_layernorm.weight.copy_(ln_data['post_attention_layernorm'].to(layer.post_attention_layernorm.weight.dtype))

        # ------------------- ## 코드 수정 부분 시작 ## -------------------
        # 각 프로젝션에 대한 정보를 딕셔너리로 정의
        proj_map = {
            'q': (layer.self_attn, 'q_proj', orig_layer.self_attn),
            'k': (layer.self_attn, 'k_proj', orig_layer.self_attn),
            'v': (layer.self_attn, 'v_proj', orig_layer.self_attn),
            'o': (layer.self_attn, 'o_proj', orig_layer.self_attn),
            'up': (layer.mlp, 'up_proj', orig_layer.mlp),
            'gate': (layer.mlp, 'gate_proj', orig_layer.mlp),
            'down': (layer.mlp, 'down_proj', orig_layer.mlp),
        }

        for suffix, (target_parent, proj_name, orig_parent) in proj_map.items():
            if f'{ii}_{suffix}' not in skip_list:
                file_path = f'{args.quantized_path}/{ii}_{suffix}.pt'
                saved_layer = torch.load(file_path, map_location=cpu, weights_only=False)
                
                target_module = getattr(target_parent, proj_name)

                rnorm = saved_layer['row_norm']
                W_hat = saved_layer['W_hat' + args.W_key] / rnorm
                
                target_module.weight.copy_(W_hat.to(target_module.weight.dtype))
                # target_module.row_norm.data = rnorm.clone().to(target_module.row_norm.dtype)
                target_module.row_norm.copy_(rnorm.clone().to(target_module.row_norm.dtype))
                # 결과 집계
                comp_result[f'{file_path}'] = {k:v for k, v in saved_layer.items() if not isinstance(v, torch.Tensor) and k != 'codes'}
                comp_result['bpp_loss'] += saved_layer['bpp_loss_sum' + args.W_key]
                comp_result['num_pixels'] += saved_layer['num_pixels']
                comp_result['bpp'] += saved_layer['bpp_sum']
            else:
                print(f'### skip {ii}_{suffix} ####')
                orig_module = getattr(orig_parent, proj_name)
                setattr(target_parent, proj_name, orig_module)
        # ------------------- ## 코드 수정 부분 끝 ## -------------------
        
        glog.info(f'loaded layer {ii}')

    print("\n--- 가중치 로딩 후 메모리 주소 확인 ---")
    addr1 = id(model.model.layers[27].self_attn.o_proj.row_norm)
    addr2 = id(model.model.layers[29].self_attn.q_proj.row_norm)
    print(f"Layer 27 row_norm 주소: {addr1}")
    print(f"Layer 29 row_norm 주소: {addr2}")
    print(f"주소 동일 여부: {addr1 == addr2}")

    if comp_result['num_pixels'] > 0:
        comp_result['bpp_loss'] /= comp_result['num_pixels']
        comp_result['bpp'] /= comp_result['num_pixels']

    if isinstance(comp_result['bpp_loss'], torch.Tensor):
        comp_result['bpp_loss'] = comp_result['bpp_loss'].item()
    if isinstance(comp_result['bpp'], torch.Tensor):
        comp_result['bpp'] = comp_result['bpp'].item()

    glog.info(f'saving model...')
    model.save_pretrained(args.hf_output_path, safe_serialization=False)
    tokenizer.save_pretrained(args.hf_output_path)
    del model

    file_path = f'{args.hf_output_path}_result.json'
    if os.path.exists(file_path):
        os.rename(file_path, f'{args.hf_output_path}_result_.json')
    with open(file_path, 'w') as f:
        json.dump(comp_result, f, indent=2)

if __name__ == '__main__':
    torch.manual_seed(0)
    args = parser.parse_args()
    main(args)