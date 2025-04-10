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