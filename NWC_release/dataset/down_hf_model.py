import torch
from transformers import CLIPVisionModelWithProjection, ViTForImageClassification, AutoModelForCausalLM
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from tqdm import tqdm
from huggingface_hub import snapshot_download
from transformers import LlavaForConditionalGeneration
from transformers import AutoImageProcessor, AutoModelForImageClassification

import shutil
import os    
import time 

model_list = [
    'meta-llama/Meta-Llama-3-8B',
    'meta-llama/Llama-2-7b-hf',
    'meta-llama/Llama-2-13b-hf',
    # 'meta-llama/Llama-2-70b-hf'
]

cache_directory = "./hf_model/cache"
for model_name in tqdm(model_list):
    print(model_name)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir = cache_directory, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir = cache_directory, trust_remote_code=True, force_download=True)
        save_directory = f"./hf_model/{model_name.replace('/', '--')}"
        try:
            model.save_pretrained(save_directory)
            tokenizer.save_pretrained(save_directory)
            print('########## Saved!! ##########')
        except Exception as e:
            print("Generation Config Error:", e)
        
    except Exception as e:
        print("Download Error:", e)
        
    if os.path.exists(cache_directory):
        try:
            shutil.rmtree(cache_directory)
            print("Cache directory cleared successfully.")
        except OSError as oe:
            print(f"Error clearing cache directory: {oe}")
    else:
        print("Cache directory does not exist, no need to clear.")