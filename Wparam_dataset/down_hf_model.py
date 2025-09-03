import torch
from transformers import CLIPVisionModelWithProjection, ViTForImageClassification, AutoModelForCausalLM
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from huggingface_hub import snapshot_download


model_list = [
    # 'meta-llama/Llama-2-7b-hf',
    # 'meta-llama/Llama-2-13b-hf',
    # 'meta-llama/Meta-Llama-3-8B',
    # 'lmsys/vicuna-7b-v1.5',
    # 'lmsys/vicuna-13b-v1.5',
    # 'facebook/opt-6.7b',
    # 'relaxml/Llama-2-7b-QTIP-4Bit',
    # 'relaxml/Llama-2-7b-QTIP-3Bit',
    # 'relaxml/Llama-2-7b-QTIP-2Bit',
    # 'relaxml/Llama-2-13b-QTIP-4Bit',
    # 'relaxml/Llama-2-13b-QTIP-3Bit',
    # 'relaxml/Llama-2-13b-QTIP-2Bit',
    # 'meta-llama/Llama-4-Scout-17B-16E',
    # 'meta-llama/Llama-3.2-11B-Vision',
    # 'meta-llama/Llama-3.2-1B',
    # 'meta-llama/Llama-3.2-3B',
    # "Qwen/Qwen2.5-7B",
    # 'meta-llama/Llama-3.2-11B-Vision',
    # 'meta-llama/Llama-2-70b-hf'
    'mistralai/Mistral-7B-Instruct-v0.2'
]

cache_directory = "./hf_model/cache"

for model_name in tqdm(model_list):
    print(model_name)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir = cache_directory, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir = cache_directory, trust_remote_code=True, force_download=True)
        save_directory = f"./hf_model/{model_name.replace('/', '--')}"  # 저장할 경로
        # save_directory = f"./hf_model/{model_name.replace('/', '--')}_awq"  # 저장할 경로
        
        # tokenizer.save_pretrained(save_directory)
        # snapshot_download(
        #     repo_id="meta-llama/Llama-2-70b-hf",
        #     local_dir='save_directory',
        #     resume_download=True,
        #     local_dir_use_symlinks=False 
        # )
        try:
            model.save_pretrained(save_directory)
            tokenizer.save_pretrained(save_directory)
            print('########## Saved!! ##########')
        except Exception as e:
            print("Generation Config Error:", e)
            model.generation_config.do_sample = True
            model.save_pretrained(save_directory)
            tokenizer.save_pretrained(save_directory)
            print('########## Saved!! ##########')
    except Exception as e:
        print("Download Error:", e)