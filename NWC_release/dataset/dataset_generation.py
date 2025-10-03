import numpy as np
import os
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm
import os
from transformers import AutoModelForCausalLM
import json
import argparse # argparse 라이브러리 import

device = torch.device("cpu")

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
    return layers

def main(args):
    model_name_hf = args.model_name
    size = args.size
    direction = args.direction
    
    sanitized_model_name = model_name_hf.replace('/', '--')
    print(f"Processing model: {model_name_hf}")
    print(f"Sanitized name for paths: {sanitized_model_name}")
    print(f"Block size: {size}")
    print(f"Direction: {direction}")
    
    model_path = f"./hf_model/{sanitized_model_name}"

    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
    layers = get_blocks(model)
    
    datas = []
    
    for i in tqdm(range(len(layers)), desc=f"Extracting weights from {model_name_hf}"):
        named_linears = get_named_linears(layers[i])
        for n, m in named_linears.items():
            w = m.weight.data.detach()
            
            if direction == 'col':
                w = w.T

            w = w.reshape(-1, size)
            
            datas.append(w)
    
    datas = torch.cat(datas, dim = 0)
    print('Total dataset shape: ', datas.shape)
    
    indices = torch.randperm(len(datas))
    split_index = int(len(datas) - 1000)
    train_indices = indices[:split_index]
    val_indices = indices[split_index:]

    dataset = {}
    dataset['train'] = datas[train_indices]
    dataset['val'] = datas[val_indices]
    print('Train shape: ', dataset['train'].shape, 'Val shape: ', dataset['val'].shape)

    dataset_stats = {}
    for split in ['train', 'val']:
        data = dataset[split]
        
        mean_all = data.mean()
        std_all = data.std()
        
        dataset_stats[split] = {
            'mean': mean_all.item(),
            'std': std_all.item(),
        }
    print("Dataset stats:", dataset_stats)

    output_dir = f'./block_pt/{sanitized_model_name}'
    os.makedirs(output_dir, exist_ok = True)
    
    pt_path = f'{output_dir}/{direction}_{size}.pt'
    torch.save(dataset, pt_path)
    print(f"Dataset saved to {pt_path}")
    
    json_path = f'{output_dir}/{direction}_{size}_dataset_stats.json'
    with open(json_path, 'w') as f:
        json.dump(dataset_stats, f, indent=4)
    print(f"Stats saved to {json_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extracts and processes model weights into block datasets.")
    
    parser.add_argument(
        '--model_name', 
        type=str, 
        required=True, 
        help="Name of the Hugging Face model (e.g., 'meta-llama/Meta-Llama-3-8B')."
    )
    parser.add_argument(
        '--size', 
        type=int, 
        required=True, 
        help="The block size for reshaping the weights."
    )
    parser.add_argument(
        '--direction', 
        type=str, 
        default='col', 
        choices=['col', 'row'], 
        help="The direction for reshaping ('col' or 'row'). Default is 'col'."
    )

    args = parser.parse_args()
    
    main(args)