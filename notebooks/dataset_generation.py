import numpy as np
import os
dtype = np.float32

import torch
import torchvision
# import tqdm
import os

from transformers import CLIPVisionModelWithProjection, ViTForImageClassification, AutoModelForCausalLM
from transformers import AutoModel, AutoTokenizer
import numpy

from huggingface_hub import try_to_load_from_cache, _CACHED_NO_EXIST
from huggingface_hub import scan_cache_dir

def get_ckpt_path(path, branch = 'main'):
    if not os.path.isdir(os.path.join(path, 'snapshots')):
        return None
    branch_file =  os.path.join(path, 'refs', branch)
    with open(branch_file, 'r', encoding='utf-8') as file:
        revision = file.read()
    return os.path.join(path, 'snapshots', revision)

def check_contains_any_substrings(string, substrings):
    return any(substring in string for substring in substrings)

model_zoo_path = '/home/jgryu/Weight_compression/model_zoo/huggingface'
# model_filter = ['0.5b','1.5b','2b','3b', '7b', '8b', '9b', '13b', 'mini', 'small']
model_filter = ['0.5b','1.5b','2b','3b', '7b', '8b', '9b','mini']
model_filter = ['-13b', '-small']


model_list = os.listdir(model_zoo_path)
ckpt_path_list = []
for ck in model_list:
    if check_contains_any_substrings(ck.lower(), model_filter):
        ckpt_path = get_ckpt_path(os.path.join(model_zoo_path, ck))
        if ckpt_path is not None:
            ckpt_path_list.append(ckpt_path)
            
            
save_dir = '/home/jgryu/Weight_compression/model_param_tensor'

for ckpt_path in ckpt_path_list:
    model_name = ckpt_path.split('/')[-3]
    print(model_name)
    model_name = model_name.split('--')
    
    save_path = os.path.join(save_dir, model_name[1], model_name[2])

    if os.path.isdir(save_path) and bool(os.listdir(save_path)):
        print('### Skip ###')
        continue

    try:
        model = AutoModelForCausalLM.from_pretrained(ckpt_path, local_files_only=True, trust_remote_code=True)
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        state_dict = model.state_dict()
        for k, v in state_dict.items():
            np.save(os.path.join(save_path, k.replace(".", "-")), v)
            
    except Exception as e:
        print(f"An error occurred: {e}")
        print(f"Fail load model from {ckpt_path}")

