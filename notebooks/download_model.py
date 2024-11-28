## huggingface model download

import torch
from transformers import CLIPVisionModelWithProjection, ViTForImageClassification, AutoModelForCausalLM
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import os

def latest_version_path(cache_dir, model_name, branch = 'main'):
    model_name_dir =  "models--" + model_name.replace('/', '--')
    path = os.path.join(cache_dir, model_name_dir)

    if not os.path.isdir(os.path.join(path, 'snapshots')):
        return None
    
    branch_file =  os.path.join(path, 'refs', branch)

    with open(branch_file, 'r', encoding='utf-8') as file:
        revision = file.read()

    return os.path.join(path, 'snapshots', revision)

# 모델과 토크나이저 로드 (캐시 디렉토리 지정)
# model = AutoModel.from_pretrained(model_name, cache_dir=cache_directory)
# tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_directory)
cache_directory = "/home/jgryu/Weight_compression/model_zoo/huggingface" 

for model_name in ['phi', "llama"]:
    if model_name == 'VIT':
        model_cls = ViTForImageClassification
        ver_list = ['google/vit-base-patch16-224', 
                    'google/vit-base-patch16-224-in21k', 
                    'google/vit-base-patch16-384',
                    'google/vit-base-patch32-224-in21k',
                    'google/vit-base-patch32-384',
                    'google/vit-large-patch16-384',
                    'google/vit-large-patch32-224-in21k',
                    'google/vit-large-patch16-224-in21k',
                    'google/vit-large-patch16-224',
                    'google/vit-huge-patch14-224-in21k',]
    elif model_name == 'CLIP':
        ver_list = ['openai/clip-vit-base-patch14', 'openai/clip-vit-large-patch14']
        model_cls = CLIPVisionModelWithProjection
    elif model_name == 'llama':
        model_cls = AutoModelForCausalLM
        ver_list = [
            "meta-llama/Meta-Llama-3-70B-Instruct",
            "meta-llama/Llama-Guard-3-8B",
            "meta-llama/Meta-Llama-3.1-405B-Instruct-FP8",
            "meta-llama/Meta-Llama-3.1-405B-Instruct",
            "meta-llama/Meta-Llama-3.1-70B-Instruct",
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "meta-llama/Meta-Llama-3.1-405B-FP8",
            "meta-llama/Meta-Llama-3.1-405B",
            "meta-llama/Llama-Guard-3-8B-INT8",
            "meta-llama/Prompt-Guard-86M",
            "meta-llama/Meta-Llama-3.1-70B",
            "meta-llama/Meta-Llama-3.1-8B",
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "meta-llama/Meta-Llama-Guard-2-8B",
            "meta-llama/Meta-Llama-3-8B",
            "meta-llama/Meta-Llama-3-70B",
            "meta-llama/LlamaGuard-7b",
            "meta-llama/Llama-2-70b-chat-hf",
            "meta-llama/Llama-2-13b-chat-hf",
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-2-70b-hf",
            "meta-llama/Llama-2-13b-hf",
            "meta-llama/Llama-2-7b-hf",
            "meta-llama/Llama-2-70b-chat",
            "meta-llama/Llama-2-13b-chat",
            "meta-llama/Llama-2-7b-chat",
            "meta-llama/Llama-2-70b",
            "meta-llama/Llama-2-13b",
            "meta-llama/Llama-2-7b",
            "meta-llama/CodeLlama-70b-Instruct-hf",
            "meta-llama/CodeLlama-70b-Python-hf",
            "meta-llama/CodeLlama-70b-hf",
            "meta-llama/CodeLlama-34b-Instruct-hf",
            "meta-llama/CodeLlama-34b-Python-hf",
            "meta-llama/CodeLlama-34b-hf",
            "meta-llama/CodeLlama-13b-Instruct-hf",
            "meta-llama/CodeLlama-13b-Python-hf",
            "meta-llama/CodeLlama-13b-hf",
            "meta-llama/CodeLlama-7b-Instruct-hf",
            "meta-llama/CodeLlama-7b-Python-hf",
            "meta-llama/CodeLlama-7b-hf"
            "meta-llama/Llama-3.2-1B",
            "meta-llama/Llama-3.2-3B",
            "meta-llama/Llama-3.2-1B-Instruct",
            "meta-llama/Llama-3.2-3B-Instruct",
            "meta-llama/Llama-Guard-3-1B",
            "meta-llama/Llama-Guard-3-1B-INT4",
            "meta-llama/Llama-3.2-11B-Vision",
            "meta-llama/Llama-3.2-11B-Vision-Instruct",
            "meta-llama/Llama-3.2-90B-Vision",
            "meta-llama/Llama-3.2-90B-Vision-Instruct",
            "meta-llama/Llama-Guard-3-11B-Vision"
        ]
    elif model_name == "qwen":
        model_cls = AutoModelForCausalLM
        ver_list = [
            "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "Qwen/Qwen2.5-Math-RM-72B",
            "Qwen/Qwen2-Math-RM-72B",
            "Qwen/Qwen2.5-Math-1.5B-Instruct",
            "Qwen/Qwen2.5-Math-72B-Instruct",
            "Qwen/Qwen2.5-Math-7B-Instruct",
            "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
            "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF",
            "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
            "Qwen/Qwen2.5-3B-Instruct-GGUF",
            "Qwen/Qwen2.5-14B-Instruct-GGUF",
            "Qwen/Qwen2.5-7B-Instruct-GGUF",
            "Qwen/Qwen2.5-32B-Instruct-GGUF",
            "Qwen/Qwen2.5-72B-Instruct-GGUF",
            "Qwen/Qwen2.5-Coder-7B-Instruct",
            "Qwen/Qwen2.5-Coder-7B",
            "Qwen/Qwen2.5-Coder-1.5B",
            "Qwen/Qwen2.5-7B",
            "Qwen/Qwen2.5-72B",
            "Qwen/Qwen2.5-3B",
            "Qwen/Qwen2.5-32B",
            "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8",
            "Qwen/Qwen2.5-14B",
            "Qwen/Qwen2.5-1.5B",
            "Qwen/Qwen2.5-0.5B",
            "Qwen/Qwen2.5-Math-7B",
            "Qwen/Qwen2.5-Math-1.5B",
            "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8",
            "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int8",
            "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int8",
            "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8",
            "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8",
            "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8",
            "Qwen/Qwen2.5-Math-72B",
            "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
            "Qwen/Qwen2.5-7B-Instruct-AWQ",
            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4",
            "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4",
            "Qwen/Qwen2.5-3B-Instruct-AWQ",
            "Qwen/Qwen2.5-3B-Instruct",
            "Qwen/Qwen2.5-32B-Instruct-AWQ",
            "Qwen/Qwen2.5-32B-Instruct",
            "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4",
            "Qwen/Qwen2.5-14B-Instruct-AWQ",
            "Qwen/Qwen2.5-14B-Instruct",
            "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4",
            "Qwen/Qwen2.5-1.5B-Instruct-AWQ",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4",
            "Qwen/Qwen2.5-0.5B-Instruct-AWQ",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "Qwen/Qwen2-VL-72B-Instruct",
            "Qwen/Qwen2-VL-72B-Instruct-AWQ",
            "Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int8",
            "Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int4",
            "Qwen/Qwen2.5-72B-Instruct",
            "Qwen/Qwen2.5-72B-Instruct-AWQ",
            "Qwen/Qwen2-7B-Instruct-MLX",
            "Qwen/Qwen2-VL-7B-Instruct-AWQ",
            "Qwen/Qwen2-Math-72B-Instruct",
            "Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4",
            "Qwen/Qwen2-VL-2B-Instruct-AWQ",
            "Qwen/Qwen2-VL-7B-Instruct",
            "Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4",
            "Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int8",
            "Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int8",
            "Qwen/Qwen2-VL-2B-Instruct",
            "Qwen/Qwen2-0.5B-Instruct-GPTQ-Int4",
            "Qwen/Qwen2-0.5B-Instruct-GPTQ-Int8",
            "Qwen/Qwen2-0.5B-Instruct-AWQ",
            "Qwen/Qwen2-1.5B-Instruct-GPTQ-Int4",
            "Qwen/Qwen2-1.5B-Instruct-GPTQ-Int8",
            "Qwen/Qwen2-1.5B-Instruct-AWQ",
            "Qwen/Qwen2-57B-A14B-Instruct-GPTQ-Int4",
            "Qwen/Qwen2-7B-Instruct-GPTQ-Int4",
            "Qwen/Qwen2-7B-Instruct-GPTQ-Int8",
            "Qwen/Qwen2-7B-Instruct-AWQ",
            "Qwen/Qwen2-0.5B-Instruct-GGUF",
            "Qwen/Qwen2-7B-Instruct",
            "Qwen/Qwen2-7B-Instruct-GGUF",
            "Qwen/Qwen2-72B-Instruct-GPTQ-Int4",
            "Qwen/Qwen2-72B-Instruct-GPTQ-Int8",
            "Qwen/Qwen2-57B-A14B-Instruct",
            "Qwen/Qwen2-0.5B-Instruct",
            "Qwen/Qwen2-72B-Instruct-AWQ",
            "Qwen/Qwen2-72B-Instruct",
            "Qwen/Qwen2-Math-7B-Instruct",
            "Qwen/Qwen2-Math-1.5B-Instruct",
            "Qwen/Qwen2-Audio-7B",
            "Qwen/Qwen2-Audio-7B-Instruct",
            "Qwen/Qwen2-Math-1.5B",
            "Qwen/Qwen2-Math-72B",
            "Qwen/Qwen2-Math-7B",
            "Qwen/Qwen2-1.5B-Instruct-GGUF",
            "Qwen/Qwen2-72B-Instruct-GGUF",
            "Qwen/Qwen2-57B-A14B-Instruct-GGUF",
            "Qwen/Qwen2-57B-A14B",
            "Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4",
            "Qwen/Qwen2-7B",
            "Qwen/Qwen2-72B",
            "Qwen/Qwen2-1.5B-Instruct",
            "Qwen/Qwen2-1.5B",
            "Qwen/Qwen2-0.5B",
            "Qwen/Qwen2-1.5B-Instruct-MLX",
            "Qwen/Qwen2-0.5B-Instruct-MLX",
            "Qwen/CodeQwen1.5-7B",
            "Qwen/Qwen1.5-32B-Chat-AWQ",
            "Qwen/Qwen1.5-MoE-A2.7B-Chat",
            "Qwen/Qwen1.5-7B-Chat-GPTQ-Int8",
            "Qwen/Qwen1.5-7B-Chat-GPTQ-Int4",
            "Qwen/Qwen1.5-7B-Chat-AWQ",
            "Qwen/Qwen1.5-72B-Chat-GPTQ-Int8",
            "Qwen/Qwen1.5-72B-Chat-GPTQ-Int4",
            ]
    elif model_name == 'gemma':
        model_cls = AutoModelForCausalLM
        ver_list = ["google/gemma-2-2b",
            "google/gemma-2-2b-it",
            "google/gemma-2-9b",
            "google/gemma-2-9b-it",
            "google/gemma-2-27b",
            "google/gemma-2-27b-it",
            "google/gemma-2-2b-pytorch",
            "google/gemma-2-2b-it-pytorch",
            "google/gemma-2-9b-pytorch",
            "google/gemma-2-9b-it-pytorch",
            "google/gemma-2-27b-pytorch",
            "google/gemma-2-27b-it-pytorch",
            "google/gemma-2-9b-keras",
            "google/gemma-2-instruct-9b-keras",
            "google/gemma-2-2b-it-GGUF",
            "google/gemma-1.1-2b-it",
            "google/gemma-1.1-7b-it",
            "google/gemma-1.1-7b-it-GGUF",
            "google/gemma-1.1-2b-it-GGUF",
            "google/gemma-1.1-2b-it-pytorch",
            "google/gemma-1.1-7b-it-pytorch",
            "google/gemma-7b",
            "google/gemma-7b-it",
            "google/gemma-2b",
            "google/gemma-2b-it",
            "google/gemma-7b-it-GGUF",
            "google/gemma-7b-GGUF",
            "google/gemma-2b-GGUF",
            "google/gemma-2b-it-GGUF",
            "google/gemma-7b-it-pytorch",
            "google/gemma-7b-pytorch",
            "google/gemma-2b-pytorch",
            "google/gemma-7b-quant-pytorch",
            "google/gemma-7b-it-quant-pytorch",
            "google/gemma-2b-it-pytorch",
            "google/gemma-7b-flax",
            "google/gemma-7b-it-flax",
            "google/gemma-2b-flax",
            "google/gemma-2b-it-flax",
            "google/gemma-2b-it-keras",
            "google/gemma-7b-keras",
            "google/gemma-2b-keras",
            "google/gemma-7b-it-keras",
            "google/gemma-2b-sfp-cpp",
            "google/gemma-2b-it-cpp",
            "google/gemma-7b-sfp-cpp",
            "google/gemma-7b-cpp",
            "google/gemma-7b-it-cpp",
            "google/gemma-7b-it-sfp-cpp",
            "google/gemma-2b-it-sfp-cpp",
            "google/gemma-2b-cpp",
            "google/gemma-1.1-2b-it-keras",
            "google/gemma-1.1-7b-it-keras",
            "google/gemma-1.1-2b-it-tflite",
            "google/gemma-2b-it-tflite",
            "google/codegemma-2b",
            "google/codegemma-7b",
            "google/codegemma-7b-it",
            "google/codegemma-7b-it-GGUF",
            "google/codegemma-2b-GGUF",
            "google/codegemma-7b-GGUF",
            "google/codegemma-7b-pytorch",
            "google/codegemma-2b-pytorch",
            "google/codegemma-7b-it-pytorch",
            "google/codegemma-2b-keras",
            "google/codegemma-7b-keras",
            "google/codegemma-7b-it-keras",
            "google/codegemma-1.1-7b-it",
            "google/codegemma-1.1-2b",
            "google/codegemma-1.1-2b-pytorch",
            "google/codegemma-1.1-7b-it-pytorch",
            "google/codegemma-1.1-2b-GGUF",
            "google/codegemma-1.1-7b-it-GGUF",]
    elif model_name == 'paligemma':
        model_cls = AutoModelForCausalLM
        ver_list = [
            "google/paligemma-3b-pt-224",
            "google/paligemma-3b-pt-448",
            "google/paligemma-3b-pt-896",
            "google/paligemma-3b-pt-224-jax",
            "google/paligemma-3b-pt-448-jax",
            "google/paligemma-3b-pt-896-jax",
            "google/paligemma-3b-mix-224",
            "google/paligemma-3b-mix-448",
            "google/paligemma-3b-mix-224-jax",
            "google/paligemma-3b-mix-448-jax",
            "google/paligemma-3b-pt-448-keras",
            "google/paligemma-3b-mix-448-keras",
            "google/paligemma-3b-pt-224-keras",
            "google/paligemma-3b-pt-896-keras",
            "google/paligemma-3b-mix-224-keras"
        ]
    elif model_name == 'phi':
        model_cls = AutoModelForCausalLM
        ver_list = [
                "microsoft/Phi-3.5-mini-instruct",
                "microsoft/Phi-3.5-MoE-instruct",
                "microsoft/Phi-3.5-vision-instruct",
                "microsoft/Phi-3.5-mini-instruct-onnx",
                "microsoft/Phi-3-mini-4k-instruct",
                "microsoft/Phi-3-mini-128k-instruct",
                "microsoft/Phi-3-small-8k-instruct",
                "microsoft/Phi-3-small-128k-instruct",
                "microsoft/Phi-3-medium-4k-instruct",
                "microsoft/Phi-3-medium-128k-instruct",
                "microsoft/Phi-3-vision-128k-instruct",
                "microsoft/Phi-3-mini-4k-instruct-onnx",
                "microsoft/Phi-3-mini-4k-instruct-onnx-web",
                "microsoft/Phi-3-mini-128k-instruct-onnx",
                "microsoft/Phi-3-small-8k-instruct-onnx-cuda",
                "microsoft/Phi-3-small-128k-instruct-onnx-cuda",
                "microsoft/Phi-3-medium-4k-instruct-onnx-cpu",
                "microsoft/Phi-3-medium-4k-instruct-onnx-cuda",
                "microsoft/Phi-3-medium-4k-instruct-onnx-directml",
                "microsoft/Phi-3-medium-128k-instruct-onnx-cpu",
                "microsoft/Phi-3-medium-128k-instruct-onnx-cuda",
                "microsoft/Phi-3-medium-128k-instruct-onnx-directml",
                "microsoft/Phi-3-vision-128k-instruct-onnx-cpu",
                "microsoft/Phi-3-vision-128k-instruct-onnx-cuda",
                "microsoft/Phi-3-vision-128k-instruct-onnx-directml",
                "microsoft/Phi-3-mini-4k-instruct-gguf"
            ]

    for ver in tqdm(ver_list):
        if "70b" in ver.lower() or "405b" in ver.lower() or "72b" in ver.lower() or "onnx" in ver.lower():
            print("Too big model")
            continue
        if isinstance(latest_version_path(cache_directory, ver), str):
            print("Already downloaded")
            continue

        try:
            pass
            print(f'##### {ver} ######')
            net = model_cls.from_pretrained(ver, cache_dir = cache_directory, token="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", trust_remote_code=True)
        except Exception as e:
            print("Fail download model from huggingface")
            print(f"An error occurred: {e}")

        # print(ver)
        # net = model_cls.from_pretrained(ver, cache_dir = cache_directory, token="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", trust_remote_code=True)

