# functions in this file cause circular imports so they cannot be loaded into __init__

import json
import os

import accelerate
import torch
import transformers
import glog

try:
    from model.llama import LlamaForCausalLM
    from transformers import LlamaConfig
except:
    LlamaForCausalLM = None
    LlamaConfig = None
    
try:
    from transformers import Qwen2ForCausalLM
except ImportError:
    Qwen2ForCausalLM = None

try:
    from model.mixtral import MixtralForCausalLM
    from transformers import MixtralConfig
except ImportError:
    MixtralConfig = None
    MixtralForCausalLM = None

try:
    from transformers import Qwen3MoeConfig
    from model.qwen3moe import Qwen3MoeForCausalLM
except ImportError:
    Qwen3MoeConfig = None
    Qwen3MoeForCausalLM = None

def model_from_hf_path(path, max_mem_ratio=0.7, device_map=None):

    # AutoConfig fails to read name_or_path correctly
    # trust_remote_code=True는 Qwen 등 최신 모델을 위해 필요할 수 있음
    bad_config = transformers.AutoConfig.from_pretrained(path, trust_remote_code=True)
    is_quantized = hasattr(bad_config, 'quip_params')
    model_type = bad_config.model_type.lower()
    
    no_split_modules = []
    if is_quantized:
        # 1. Qwen 계열 확인
        if 'qwen' in model_type or 'qwen' in path.lower():
            # MoE 모델인지 확인 (config type 혹은 이름으로 유추)
            if 'moe' in model_type or 'moe' in path.lower():
                try:
                    # [수정] Qwen3MoeConfig 우선 사용
                    if Qwen3MoeConfig:
                        model_str = Qwen3MoeConfig.from_pretrained(path, trust_remote_code=True)._name_or_path
                    else:
                        model_str = path
                except:
                    model_str = path
                
                # [수정] 사용자 정의 Qwen3MoE 사용 (임포트 보장됨)
                model_cls = Qwen3MoeForCausalLM 

                # [수정] Qwen3 MoE 디코더 레이어 이름
                no_split_modules = ['Qwen3MoeDecoderLayer'] 
            else:
                # Dense Qwen
                if Qwen2ForCausalLM:
                    model_str = transformers.Qwen2Config.from_pretrained(path)._name_or_path
                    model_cls = Qwen2ForCausalLM
                else:
                    # Qwen2 클래스가 없으면 AutoModel 사용
                    model_str = path
                    model_cls = transformers.AutoModelForCausalLM
                no_split_modules = ['Qwen2DecoderLayer']
        
        # 2. Mixtral 확인
        elif 'mixtral' in model_type:
            if MixtralConfig:
                model_str = MixtralConfig.from_pretrained(path)._name_or_path
            else:
                model_str = path

            # [수정] 사용자 정의 Mixtral 사용 (임포트 보장됨)
            model_cls = MixtralForCausalLM 
            
            no_split_modules = ['MixtralDecoderLayer']
            
        # 3. Llama 확인
        elif 'llama' in model_type:
            model_str = LlamaConfig.from_pretrained(path)._name_or_path
            # [수정] 사용자 정의 Llama 사용 (임포트 보장됨)
            model_cls = LlamaForCausalLM
            no_split_modules = ['LlamaDecoderLayer']
            
        else:
            raise Exception(f"Unsupported quantized model type: {model_type}")
            
    else:
        # --- 비양자화(원본) 모델 로딩 ---
        model_cls = transformers.AutoModelForCausalLM
        model_str = path
        
        # Device Map을 위한 No Split Module 설정
        if 'mixtral' in model_type:
            no_split_modules = ['MixtralDecoderLayer']
        elif 'qwen' in model_type:
            if 'moe' in model_type:
                 # [수정] Qwen3MoeDecoderLayer
                 no_split_modules = ['Qwen3MoeDecoderLayer']
            else:
                 no_split_modules = ['Qwen2DecoderLayer']
        elif 'llama' in model_type:
            no_split_modules = ['LlamaDecoderLayer']

    if device_map is None:
        glog.info("Start computing device map")
        mmap = {
            i: f"{torch.cuda.mem_get_info(i)[1]*max_mem_ratio/(1 << 30)}GiB"
            for i in range(torch.cuda.device_count())
        }
        
        print("Computing device_map with meta skeleton...")
        
        # 1. Config 불러오기
        try:
            # AutoConfig 혹은 특정 Config 클래스 사용
            if 'qwen' in model_type and 'moe' in model_type and Qwen3MoeConfig:
                 config = Qwen3MoeConfig.from_pretrained(path, trust_remote_code=True)
            elif 'mixtral' in model_type and MixtralConfig:
                 config = MixtralConfig.from_pretrained(path)
            elif 'llama' in model_type and LlamaConfig:
                 config = LlamaConfig.from_pretrained(path)
            else:
                 config = transformers.AutoConfig.from_pretrained(path, trust_remote_code=True)
        except:
            config = transformers.AutoConfig.from_pretrained(path, trust_remote_code=True)

        # 2. 가중치 로드 없이 껍데기(Skeleton)만 생성 (Meta Device)
        # 중요: 실제 weights를 읽지 않으므로 순식간에 끝납니다.
        with accelerate.init_empty_weights():
            # model_cls가 AutoModel이 아니라 커스텀 클래스인 경우 config로 초기화
            # 대부분의 HF 모델은 model_cls(config)를 지원합니다.
            try:
                meta_model = model_cls(config)
            except Exception as e:
                # 커스텀 모델이 config init을 지원하지 않는 예외적 경우 fallback
                print(f"Warning: Skeleton load failed ({e}). Fallback to AutoModel structure.")
                meta_model = transformers.AutoModelForCausalLM.from_config(config)

        # 3. Skeleton 모델로 device_map 계산
        device_map = accelerate.infer_auto_device_map(
            meta_model,
            no_split_module_classes=no_split_modules, 
            max_memory=mmap,
        )
        
        glog.info("End computing device map")
        del meta_model


    # if device_map is None:
    #     glog.info("Start computing device map")
    #     mmap = {
    #         i: f"{torch.cuda.mem_get_info(i)[1]*max_mem_ratio/(1 << 30)}GiB"
    #         for i in range(torch.cuda.device_count())
    #     }
        
    #     # 임시 모델 로드 (구조 파악용)
    #     # trust_remote_code=True 추가
    #     model = model_cls.from_pretrained(path,
    #                                       torch_dtype='bfloat16',
    #                                       low_cpu_mem_usage=True,
    #                                       attn_implementation='sdpa')
        
    #     # no_split_module_classes 적용
    #     device_map = accelerate.infer_auto_device_map(
    #         model,
    #         no_split_module_classes=no_split_modules, 
    #         max_memory=mmap)
        
    #     del model
    #     glog.info("End computing device map")
        torch.cuda.empty_cache()
    # 최종 모델 로드
    glog.info("Start model loading")
    model = model_cls.from_pretrained(path,
                                    torch_dtype='bfloat16',
                                    #   low_cpu_mem_usage=True,
                                    #   attn_implementation='sdpa',
                                    use_safetensors=True,
                                    device_map=device_map,
                                    # trust_remote_code=True,
                                    )
                                    #   offload_folder="offload_temp") very slow
    glog.info("End model loading")
    return model, model_str

# def model_from_hf_path(path, max_mem_ratio=0.7, device_map=None):

#     # AutoConfig fails to read name_or_path correctly
#     bad_config = transformers.AutoConfig.from_pretrained(path)
#     is_quantized = hasattr(bad_config, 'quip_params')
#     model_type = bad_config.model_type
    
#     # --- [수정 2] 분기할 레이어 클래스 이름 변수 ---
#     no_split_modules = []

#     if is_quantized:
#         if 'qwen' in path.lower():
#             model_str = transformers.Qwen2Config.from_pretrained(
#                 path)._name_or_path
#             model_cls = Qwen2ForCausalLM
#             no_split_modules = ['Qwen2DecoderLayer'] # [추가]
#         else:
#             if model_type == 'llama':
#                 model_str = LlamaConfig.from_pretrained(
#                     path)._name_or_path
#                 model_cls = LlamaForCausalLM
#                 no_split_modules = ['LlamaDecoderLayer'] # [추가]
#             # --- [수정 3] Mixtral 분기 추가 ---
#             elif model_type == 'mixtral':
#                 model_str = MixtralConfig.from_pretrained(
#                     path)._name_or_path
#                 model_cls = MixtralForCausalLM # 사용자 정의 Mixtral
#                 no_split_modules = ['MixtralDecoderLayer'] # [추가]
#             # --- [수정 3 완료] ---
#             else:
#                 raise Exception(f"Unsupported quantized model type: {model_type}") # [수정]
#     else:
#         model_cls = transformers.AutoModelForCausalLM
#         model_str = path
#         # [수정 4] 비양자화 모델도 분기 처리
#         if model_type == 'llama':
#             no_split_modules = ['LlamaDecoderLayer']
#         elif model_type == 'mixtral':
#             no_split_modules = ['MixtralDecoderLayer']
#         elif 'qwen' in path.lower():
#             no_split_modules = ['Qwen2DecoderLayer']

#     if device_map is None:
#         mmap = {
#             i: f"{torch.cuda.mem_get_info(i)[1]*max_mem_ratio/(1 << 30)}GiB"
#             for i in range(torch.cuda.device_count())
#         }
#         # 임시 모델 로드 (device_map 추론용)
#         model = model_cls.from_pretrained(path,
#                                           torch_dtype='auto',
#                                           low_cpu_mem_usage=True,
#                                           attn_implementation='sdpa')
        
#         # --- [수정 5] no_split_module_classes 수정 ---
#         device_map = accelerate.infer_auto_device_map(
#             model,
#             no_split_module_classes=no_split_modules, # 'LlamaDecoderLayer' -> no_split_modules
#             max_memory=mmap)
#         # --- [수정 5 완료] ---
        
#         del model # 임시 모델 삭제
#         torch.cuda.empty_cache() # 메모리 정리

#     model = model_cls.from_pretrained(path,
#                                       torch_dtype='auto',
#                                       low_cpu_mem_usage=True,
#                                       attn_implementation='sdpa',
#                                       device_map=device_map)

#     return model, model_str

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