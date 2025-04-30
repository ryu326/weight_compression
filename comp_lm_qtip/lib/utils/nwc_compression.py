import json
import argparse
import json
import os
import sys
import torch
import torch.nn as nn
import re
import math
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    OPTForCausalLM,
    BloomForCausalLM,    
)
from torch.utils.data import DataLoader
import logging


def setup_logging(log_file):
    # Remove any pre-existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Configure logging settings
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file),  # Log to file
            logging.StreamHandler(sys.stdout)  # Log to console
        ]
    )

def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}

def get_blocks(model):
    if model.__class__.__name__ in ("LlamaForCausalLM", "Qwen2ForCausalLM"):
        layers = model.model.layers
    elif model.__class__.__name__ == "LlavaLlamaForCausalLM":
        # layers = [model.model.layers, model.model.vision_tower.vision_tower.vision_model.encoder.layers]
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
    else:
        raise NotImplementedError(type(model))
    return layers

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
    return layers


def get_model_weight_stats(model, args, size):
    
    # if args.diag_scale == True:
    if False:
    # if True:
        with open('/workspace/Weight_compression/Wparam_dataset/block_pt/meta-llama--Llama-2-7b-hf/scaled_sig0.001_row_4096_dataset_stats.json', 'r') as f:
            data_stats = json.load(f)
        mean = torch.tensor(data_stats['train']['mean'])
        std = torch.tensor(data_stats['train']['std'])
    else:
        dataset_stats = {}
        weights = []
        layers = get_blocks(model)
        
        if isinstance(layers, dict):
            layers_ = []
            for k, v in layers.items():
                layers_ += v
            assert len(layers_) == 12 + 24
            layers = layers_
        
        for i in tqdm(range(len(layers)), desc="calculating model weight mean & std"):
            named_linears = get_named_linears(layers[i])
            for n, m in named_linears.items():
                w = m.weight.data.detach()
                if args.direction == 'col':
                    w = w.T    
                w = w.reshape(-1, size)
                weights.append(w)
        
        weights = torch.cat(weights, dim = 0)
        
        # mean = weights.mean(0)
        # std = weights.std(0)
        mean = weights.mean()
        std = weights.std()
            
    return mean, std

def plot_ft_comp_result(ft_result, args, idx, name):                
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  
    fig, axs = plt.subplots(3, 4, figsize=(16, 12))
    axs[0, 0].plot(ft_result['step'], ft_result['loss'], label='loss')
    axs[0, 0].set_title('loss')
    axs[0, 0].set_xlabel('step')

    axs[0, 1].plot(ft_result['step'], ft_result['adaptive_loss'], label='adaptive_loss')
    axs[0, 1].set_title('adaptive_loss')    
    axs[0, 1].set_xlabel('step')

    axs[0, 2].plot(ft_result['step'], ft_result['bpp_loss'], label='bpp_loss')
    axs[0, 2].set_title('bpp_loss')
    axs[0, 2].set_xlabel('step')

    axs[0, 3].plot(ft_result['step'], ft_result['mse_loss'], label='mse_loss')
    axs[0, 3].set_title('mse_loss')
    axs[0, 3].set_xlabel('step')
    
    axs[1, 0].plot(ft_result['epoch'], ft_result['loss_per_epoch'], label='loss_epoch')
    axs[1, 0].set_title('loss_epoch')
    axs[1, 0].set_xlabel('epoch')
    
    axs[1, 1].plot(ft_result['epoch'], ft_result['adaptive_loss_per_epoch'], label='adaptive_loss_epoch')
    axs[1, 1].set_title('adaptive_loss_epoch')
    axs[1, 1].set_xlabel('epoch')
    
    axs[1, 2].plot(ft_result['epoch'], ft_result['bpp_loss_per_epoch'], label='bpp_loss_per_epoch')
    axs[1, 2].set_title('bpp_loss_per_epoch')
    axs[1, 2].axhline(y=ft_result['base_bpp_loss'], color='r', linestyle='--', label='base_bpp_loss')
    axs[1, 2].legend()
    axs[1, 2].set_xlabel('epoch')
    
    axs[1, 3].plot(ft_result['epoch'], ft_result['mse_loss_per_epoch'], label='mse_loss_per_epoch')
    axs[1, 3].set_title('mse_loss_per_epoch')
    axs[1, 3].set_xlabel('epoch')
    
    axs[2, 0].plot(ft_result['epoch'], ft_result['proxy_err'], label='proxy_err')
    axs[2, 0].set_title('proxy_err')
    axs[2, 0].axhline(y=ft_result['base_proxy_err'], color='r', linestyle='--', label='base_proxy_err')
    axs[2, 0].legend()
    axs[2, 0].set_xlabel('epoch')
    
    axs[2, 1].plot(ft_result['epoch'], ft_result['mse'], label='mse')
    axs[2, 1].set_title('mse')
    axs[2, 1].axhline(y=ft_result['base_mse'], color='r', linestyle='--', label='base_mse')
    axs[2, 1].legend()
    axs[2, 1].set_xlabel('epoch')

    axs[2, 2].plot(ft_result['epoch'], ft_result['err'], label='err')
    axs[2, 2].set_title('err')
    axs[2, 2].axhline(y=ft_result['base_err'], color='r', linestyle='--', label='base_err')
    axs[2, 2].legend()
    axs[2, 2].set_xlabel('epoch')
    
    os.makedirs(args.save_path + '/plots', exist_ok=True)
    os.makedirs(args.save_path + '/jsons', exist_ok=True)
    plt.savefig(f'{args.save_path}/plots/{idx}_{name}_ft_result.png')
    with open(f'{args.save_path}/jsons/{idx}_{name}_ft_result.json', 'w') as f:
        json.dump(ft_result, f)