import numpy as np
import os
dtype = np.float32

import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm
import os

from transformers import CLIPVisionModelWithProjection, AutoModelForCausalLM, LlamaForCausalLM
from transformers import AutoModel, AutoTokenizer, OPTForCausalLM, BloomForCausalLM
import numpy

from huggingface_hub import try_to_load_from_cache, _CACHED_NO_EXIST
from huggingface_hub import scan_cache_dir

import glob
import random
import json
import os

from datasets import load_dataset

import functools
import gc
from collections import defaultdict
from typing import List

import torch
import torch.nn as nn
import tqdm
# from tinychat.models import LlavaLlamaForCausalLM
from transformers.models.bloom.modeling_bloom import BloomForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.opt.modeling_opt import OPTForCausalLM

import numpy as np
from scipy.linalg import eigh

def topk_eigenvectors(A, device):
    A = A.to(device)
    eigenvalues, eigenvectors = torch.linalg.eigh(A)
    eigenvalues = eigenvalues.cpu()
    eigenvectors = eigenvectors.cpu()
    # print("최대 고유값:", eigenvalues[-1])
    return eigenvalues, eigenvectors

device = "cuda" if torch.cuda.is_available() else "cpu"

model_list = [
    'meta-llama/Meta-Llama-3-8B',
    'meta-llama/Llama-2-7b-hf',
    'meta-llama/Llama-2-13b-hf',
    # 'lmsys/vicuna-7b-v1.5',
    # 'lmsys/vicuna-13b-v1.5',
    # 'facebook/opt-6.7b',
]

n_samples=128
seqlen=512
calib_data="pileval"
batch_size = 12

for model_name in model_list:
    
    model_name = model_name.replace('/', '--')
    print('model_name: ', model_name)

    model_path = f"./hf_model/{model_name}"

    save_path = f'./hessian/{model_name}'
    os.makedirs(save_path, exist_ok=True)
    hess  = torch.load(save_path + f'/{calib_data}_n_samples{n_samples}_seqlen{seqlen}.pt')
    
    hess_eigen = []
    for i in tqdm.tqdm(range(len(hess))):
        
        eigen_one_layer = defaultdict(list)
        
        for n, h in hess[i].items():
            # print(i, n, h.shape)
            
            eigen = {}
            
            eigenvalues, eigenvectors = topk_eigenvectors(h, device)
            
            eigen['eigenvalues'] = eigenvalues
            eigen['eigenvectors'] = eigenvectors
            
            print(eigen['eigenvalues'].shape)
            print(eigen['eigenvectors'].shape)
            
            eigen_one_layer[n] = eigen
        
        hess_eigen.append(eigen_one_layer)
    
    torch.save(hess_eigen, save_path + f'/{calib_data}_n_samples{n_samples}_seqlen{seqlen}_eigen.pt')
    