# functions in this file cause circular imports so they cannot be loaded into __init__

import json
import os

import accelerate
import torch
import transformers

from model.llama import LlamaForCausalLM
from transformers import Qwen2ForCausalLM

from model.mixtral import MixtralForCausalLM
from transformers import MixtralConfig, LlamaConfig

def model_from_hf_path(path, max_mem_ratio=0.7, device_map=None):

    # AutoConfig fails to read name_or_path correctly
    bad_config = transformers.AutoConfig.from_pretrained(path)
    is_quantized = hasattr(bad_config, 'quip_params')
    model_type = bad_config.model_type
    
    # --- [수정 2] 분기할 레이어 클래스 이름 변수 ---
    no_split_modules = []

    if is_quantized:
        if 'qwen' in path.lower():
            model_str = transformers.Qwen2Config.from_pretrained(
                path)._name_or_path
            model_cls = Qwen2ForCausalLM
            no_split_modules = ['Qwen2DecoderLayer'] # [추가]
        else:
            if model_type == 'llama':
                model_str = LlamaConfig.from_pretrained(
                    path)._name_or_path
                model_cls = LlamaForCausalLM
                no_split_modules = ['LlamaDecoderLayer'] # [추가]
            # --- [수정 3] Mixtral 분기 추가 ---
            elif model_type == 'mixtral':
                model_str = MixtralConfig.from_pretrained(
                    path)._name_or_path
                model_cls = MixtralForCausalLM # 사용자 정의 Mixtral
                no_split_modules = ['MixtralDecoderLayer'] # [추가]
            # --- [수정 3 완료] ---
            else:
                raise Exception(f"Unsupported quantized model type: {model_type}") # [수정]
    else:
        model_cls = transformers.AutoModelForCausalLM
        model_str = path
        # [수정 4] 비양자화 모델도 분기 처리
        if model_type == 'llama':
            no_split_modules = ['LlamaDecoderLayer']
        elif model_type == 'mixtral':
            no_split_modules = ['MixtralDecoderLayer']
        elif 'qwen' in path.lower():
            no_split_modules = ['Qwen2DecoderLayer']

    if device_map is None:
        mmap = {
            i: f"{torch.cuda.mem_get_info(i)[1]*max_mem_ratio/(1 << 30)}GiB"
            for i in range(torch.cuda.device_count())
        }
        # 임시 모델 로드 (device_map 추론용)
        model = model_cls.from_pretrained(path,
                                          torch_dtype='auto',
                                          low_cpu_mem_usage=True,
                                          attn_implementation='sdpa')
        
        # --- [수정 5] no_split_module_classes 수정 ---
        device_map = accelerate.infer_auto_device_map(
            model,
            no_split_module_classes=no_split_modules, # 'LlamaDecoderLayer' -> no_split_modules
            max_memory=mmap)
        # --- [수정 5 완료] ---
        
        del model # 임시 모델 삭제
        torch.cuda.empty_cache() # 메모리 정리

    model = model_cls.from_pretrained(path,
                                      torch_dtype='auto',
                                      low_cpu_mem_usage=True,
                                      attn_implementation='sdpa',
                                      device_map=device_map)

    return model, model_str

# def model_from_hf_path(path, max_mem_ratio=0.7, device_map=None):

#     # AutoConfig fails to read name_or_path correctly
#     bad_config = transformers.AutoConfig.from_pretrained(path)
#     is_quantized = hasattr(bad_config, 'quip_params')
#     model_type = bad_config.model_type
#     if is_quantized:
#         if 'qwen' in path.lower():
#             model_str = transformers.Qwen2Config.from_pretrained(
#                 path)._name_or_path
#             model_cls = Qwen2ForCausalLM
#         else:
#             if model_type == 'llama':
#                 model_str = transformers.LlamaConfig.from_pretrained(
#                     path)._name_or_path
#                 model_cls = LlamaForCausalLM
#             else:
#                 raise Exception
#     else:
#         model_cls = transformers.AutoModelForCausalLM
#         model_str = path

#     if device_map is None:
#         mmap = {
#             i: f"{torch.cuda.mem_get_info(i)[1]*max_mem_ratio/(1 << 30)}GiB"
#             for i in range(torch.cuda.device_count())
#         }
#         model = model_cls.from_pretrained(path,
#                                           torch_dtype='auto',
#                                           low_cpu_mem_usage=True,
#                                           attn_implementation='sdpa')
#         device_map = accelerate.infer_auto_device_map(
#             model,
#             no_split_module_classes=['LlamaDecoderLayer'],
#             max_memory=mmap)
#     model = model_cls.from_pretrained(path,
#                                       torch_dtype='auto',
#                                       low_cpu_mem_usage=True,
#                                       attn_implementation='sdpa',
#                                       device_map=device_map)

#     return model, model_str

def model_from_hf_path_clip(path, max_mem_ratio=0.7, device_map=None):

    bad_config = transformers.AutoConfig.from_pretrained(path)
    is_quantized = hasattr(bad_config, 'quip_params')
    model_type = bad_config.model_type
    if is_quantized:
        model_str = transformers.CLIPConfig.from_pretrained(
            path)._name_or_path
        model_cls = CLIPModel
    else:
        model_cls = transformers.AutoModel
        model_str = path

    model = model_cls.from_pretrained(path,
                                      torch_dtype='auto',
                                      low_cpu_mem_usage=True,
                                      attn_implementation='sdpa',
                                      device_map='cuda')

    return model, model_str