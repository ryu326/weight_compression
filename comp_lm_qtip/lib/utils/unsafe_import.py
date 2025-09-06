# functions in this file cause circular imports so they cannot be loaded into __init__

import json
import os

import accelerate
import torch
import transformers

from model.llama import LlamaForCausalLM

import accelerate
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoConfig

def model_from_hf_path(path, max_mem_ratio=0.7, device_map=None):
    config = AutoConfig.from_pretrained(path)
    is_quantized = hasattr(config, 'quip_params')

    if is_quantized and config.model_type == 'llama':
        model_cls = LlamaForCausalLM
    else:
        # 다른 모델 타입에 대한 처리
        model_cls = transformers.AutoModelForCausalLM

    # 1. 메타 텐서로 빈 모델을 생성 (메모리 사용 X)
    with init_empty_weights():
        model = model_cls(config)

    model.tie_weights() # 일부 모델에 필요한 단계

    # 2. device_map이 없으면 추론
    if device_map is None:
        mmap = {
            i: f"{torch.cuda.mem_get_info(i)[1] * max_mem_ratio / (1 << 30)}GiB"
            for i in range(torch.cuda.device_count())
        }
        device_map = accelerate.infer_auto_device_map(
            model,
            no_split_module_classes=['LlamaDecoderLayer'],
            max_memory=mmap
        )

    # 3. 체크포인트에서 가중치를 불러와 디바이스 맵에 맞게 모델에 직접 로드
    # path는 .safetensors 같은 체크포인트 파일이 있는 디렉토리여야 합니다.
    model = load_checkpoint_and_dispatch(
        model,
        path,
        device_map=device_map,
    )
    
    # 로딩 후 수동으로 설정해야 할 수 있습니다.
    model.config.attn_implementation = 'sdpa'

    return model, path

# def model_from_hf_path(path, max_mem_ratio=0.7, device_map=None):

#     # AutoConfig fails to read name_or_path correctly
#     bad_config = transformers.AutoConfig.from_pretrained(path)
#     is_quantized = hasattr(bad_config, 'quip_params')
#     model_type = bad_config.model_type
#     if is_quantized:
#         if model_type == 'llama':
#             # model_str = transformers.LlamaConfig.from_pretrained(
#             #     path)._name_or_path
#             model_cls = LlamaForCausalLM
#             model_str = path
#         else:
#             raise Exception
#     else:
#         model_cls = transformers.AutoModelForCausalLM
#         model_str = path
        
#     import ipdb; ipdb.set_trace()
    
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


# def model_from_hf_path(path, max_mem_ratio=0.7, device_map=None):

#     # AutoConfig fails to read name_or_path correctly
#     bad_config = transformers.AutoConfig.from_pretrained(path)
#     # is_quantized = hasattr(bad_config, 'quip_params')
#     is_quantized = False
#     model_type = bad_config.model_type
#     if is_quantized:
#         if model_type == 'llama':
#             model_str = transformers.LlamaConfig.from_pretrained(
#                 path)._name_or_path
#             model_cls = LlamaForCausalLM
#         else:
#             raise Exception
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
