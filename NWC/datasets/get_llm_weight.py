import glob
import json
import os
import random

import numpy as np
import PIL.Image as Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import torch.nn as nn
import pprint

from transformers import CLIPVisionModelWithProjection, AutoModelForCausalLM
from transformers import AutoModel, AutoTokenizer, OPTForCausalLM, BloomForCausalLM,AutoModelForImageClassification

def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}

def get_blocks(model):
    if model.__class__.__name__ in ("LlamaForCausalLM", "Qwen2ForCausalLM"):
        layers = model.model.layers
    elif model.__class__.__name__ == "LlavaLlamaForCausalLM":
        layers = model.model.layers
    elif isinstance(model, OPTForCausalLM):
        layers = model.model.decoder.layers
    elif isinstance(model, BloomForCausalLM):
        layers = model.transformer.h
    elif "mpt" in str(model.__class__).lower():
        layers = model.transformer.blocks
    elif "falcon" in str(model.__class__).lower():
        layers = model.transformer.h
    elif "bigcode" in str(model.__class__).lower():
        layers = model.transformer.h
    elif "neox" in str(model.__class__).lower():
        layers = model.gpt_neox.layers
    elif model.__class__.__name__ == "LlavaLlamaModel":
        layers = model.llm.model.layers
    elif model.__class__.__name__ in ("CLIPModel"):
        vision_layers = model.vision_model.encoder.layers
        text_layers = model.text_model.encoder.layers
        layers = {'vision': vision_layers,
                  'text': text_layers}
    elif model.__class__.__name__ in ("SiglipModel"):
        vision_layers = model.vision_model.encoder.layers
        text_layers = model.text_model.encoder.layers
        layers = {'vision': vision_layers,
                  'text': text_layers}
    elif model.__class__.__name__ == "Dinov2ForImageClassification":
        layers = model.dinov2.encoder.layer
    else:
        raise NotImplementedError(type(model))
    # if not isinstance(layers, dict):
    #     layers = {'': layers}
    return layers

def get_normed_weight_from_hf(hf_path, direction, size, normalize = None):    
    device = torch.device("cuda")
    assert direction in ['col', 'row']
    assert normalize in ['col', 'row', None]
    model = AutoModelForCausalLM.from_pretrained(hf_path, local_files_only=True)
    layers = get_blocks(model)
    
    raw_data = {}
    raw_data['weight'] = []
    # raw_data['idx'] = []
    # raw_data['layer_type'] = []
    # raw_data['scale'] = []
        
    for i in tqdm(range(len(layers))):
        named_linears = get_named_linears(layers[i])
        
        for n, m in named_linears.items():
            W = m.weight.data.detach().to(device)

            if normalize == 'col':
                col_std = W.std(dim=0, keepdim=True)
                Wr = W / col_std
            elif normalize == 'row':
                row_std = W.std(dim=1, keepdim=True)
                Wr = W / row_std
            elif normalize == None:
                Wr = W 


            if direction == 'col':
                w = Wr.T.to('cpu')
            else:
                w = Wr.to('cpu')
                           
            if w.shape[-1] % size == 0:
                w = w.reshape(-1, size)
            else:
                raise
            
            raw_data['weight'].append(w)
            # raw_data['scale'].append(s)
            # idx = torch.tensor([i], dtype = torch.int8)
            # raw_data['idx'].extend([idx] * w.shape[0])
            # layer_type = torch.tensor([wtype_mapping[n]], dtype = torch.int8)
            # raw_data['layer_type'].extend([layer_type] * w.shape[0])
    
    for k in raw_data.keys():
        raw_data[k] = torch.cat(raw_data[k], dim = 0)
        print(f'{k} total shape: ', raw_data[k].shape)
    
    ###################### split, compute stats ######################
    
    indices = torch.randperm(len(raw_data['weight']))
    split_index = int(len(raw_data['weight']) - 1000)
    train_indices = indices[:split_index]
    val_indices = indices[split_index:]

    dataset = {}
    dataset['train'] = {}
    dataset['val'] = {}
    for k in raw_data.keys():
        dataset['train'][k] = raw_data[k][train_indices]
        dataset['val'][k] = raw_data[k][val_indices]
        
    print('train Weight: ', dataset['train']['weight'].shape, 'val: ', dataset['val']['weight'].shape)
    # print('train Scale: ', dataset['train']['scale'].shape, 'val: ', dataset['val']['scale'].shape)

    dataset_stats = {}
    for split in ['train', 'val']:
        data = dataset[split]
        
        # mean_dim0 = data['weight'].mean(dim=0)
        # std_dim0 = data['weight'].std(dim=0)        
        mean_all = data['weight'].mean()
        std_all = data['weight'].std()
        
        dataset_stats[split] = {
            'mean': mean_all.item(),
            'std': std_all.item(),
            'mean_channel': None,
            'std_channel': None
        }
            
    return dataset, dataset_stats



def get_normed_patch_weight_from_hf(hf_path, direction, L = 1024, I = 16, normalize = None):
    import torch
    from einops import rearrange

    device = torch.device("cuda")
    assert direction in ['col', 'row']
    assert normalize in ['col', 'row', None]
    model = AutoModelForCausalLM.from_pretrained(hf_path, local_files_only=True)
    layers = get_blocks(model)
    
    raw_data = {}
    raw_data['weight'] = []
    # raw_data['idx'] = []
    # raw_data['layer_type'] = []
    # raw_data['scale'] = []
        
    for i in tqdm(range(len(layers))):
        named_linears = get_named_linears(layers[i])
        
        for n, m in named_linears.items():
            W = m.weight.data.detach().to(device)

            if normalize == 'col':
                col_std = W.std(dim=0, keepdim=True)
                Wr = W / col_std
            elif normalize == 'row':
                row_std = W.std(dim=1, keepdim=True)
                Wr = W / row_std
            elif normalize == None:
                Wr = W 
            
            if direction == 'col':
                w = Wr.T.to('cpu')
            else:
                w = Wr.to('cpu')
                           
            patches = rearrange(
                w, 
                '(h p1) (w p2) -> (h w) p1 p2', 
                p1=L, 
                p2=I
            )
            
            raw_data['weight'].append(patches)
    
    for k in raw_data.keys():
        raw_data[k] = torch.cat(raw_data[k], dim = 0)
        print(f'{k} total shape: ', raw_data[k].shape)
    
    ###################### split, compute stats ######################
    
    indices = torch.randperm(len(raw_data['weight']))
    split_index = int(len(raw_data['weight']) - 500)
    train_indices = indices[:split_index]
    val_indices = indices[split_index:]

    dataset = {}
    dataset['train'] = {}
    dataset['val'] = {}
    for k in raw_data.keys():
        dataset['train'][k] = raw_data[k][train_indices]
        dataset['val'][k] = raw_data[k][val_indices]
        
    print('train Weight: ', dataset['train']['weight'].shape, 'val: ', dataset['val']['weight'].shape)
    # print('train Scale: ', dataset['train']['scale'].shape, 'val: ', dataset['val']['scale'].shape)

    dataset_stats = {}
    for split in ['train', 'val']:
        data = dataset[split]
        
        # mean_dim0 = data['weight'].mean(dim=0)
        # std_dim0 = data['weight'].std(dim=0)        
        mean_all = data['weight'].mean()
        std_all = data['weight'].std()
        
        dataset_stats[split] = {
            'mean': mean_all.item(),
            'std': std_all.item(),
            'mean_channel': None,
            'std_channel': None
        }
    ## v2
    print('---- Dataset_stats before std normalization ----')
    pprint.pprint(dataset_stats)
    
    if normalize == None:
        dataset['train']['weight'] = dataset['train']['weight'] / dataset_stats['train']['std']
        dataset['val']['weight'] = dataset['val']['weight'] / dataset_stats['val']['std']        
    
    dataset_stats['val']['std'] = 1
    dataset_stats['train']['std'] = 1
    print('---- Dataset_stats after std normalization -----')
    pprint.pprint(dataset_stats)
    
    return dataset, dataset_stats