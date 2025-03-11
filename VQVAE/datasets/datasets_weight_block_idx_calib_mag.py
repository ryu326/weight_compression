import glob
import json
import os
import random

import numpy as np
import PIL.Image as Image
import torch
from torch.utils.data import Dataset
# from torchvision import transforms

from transformers import CLIPVisionModelWithProjection, ViTForImageClassification, AutoModelForCausalLM
from transformers import AutoModel, AutoTokenizer

def latest_version_path(cache_dir, model_name, branch = 'main'):
    model_name_dir =  "models--" + model_name.replace('/', '--')
    path = os.path.join(cache_dir, model_name_dir)

    if not os.path.isdir(os.path.join(path, 'snapshots')):
        return None
    
    branch_file =  os.path.join(path, 'refs', branch)

    with open(branch_file, 'r', encoding='utf-8') as file:
        revision = file.read()

    return os.path.join(path, 'snapshots', revision)

class LayerInputs:
    def __init__(self, num_layers):
        self.layers = [
            {
                "self_attn": {
                    "q_proj": None,
                    "k_proj": None,
                    "v_proj": None,
                    "o_proj": None,
                },
                "mlp": {
                    "gate_proj": None,
                    "up_proj": None,
                    "down_proj": None,
                },
            }
            for _ in range(num_layers)
        ]

def tensor_2_block_idx(tensor_data):
    ltype_mapping = {0: 'self_attn', 1: 'mlp'}
    wtype_mapping = {
        0: 'q_proj', 1: 'k_proj', 2: 'v_proj', 3: 'o_proj',
        4: 'gate_proj', 5: 'up_proj', 6: 'down_proj'
    }

    # Extract individual elements from the tensor
    layer_idx = tensor_data[0].item()
    ltype = ltype_mapping[tensor_data[1].item()]
    wtype = wtype_mapping[tensor_data[2].item()]
    idx = tensor_data[3].item()
    slice_start = tensor_data[4].item()
    slice_stop = tensor_data[5].item()

    # Construct the dictionary
    reconstructed_data = {
        'layer_idx': layer_idx,
        'ltype': ltype,
        'wtype': wtype,
        'idx': idx,
        'slice': (slice_start, slice_stop)
    }

    return reconstructed_data


class ColBlock_Mag(Dataset):
    def __init__(self, net, tensor_block_idx, layer_inputs, dataset_stats):

        self.model = net
        self.layer_inputs = layer_inputs

        self.tensor_block_idx = tensor_block_idx
        print("Dataset Shape: ", tensor_block_idx.size())
        
        self.mean = torch.Tensor(dataset_stats["mean_channel"])
        self.std = torch.Tensor(dataset_stats["std_channel"])

    def __len__(self):
        return len(self.tensor_block_idx)

    def __getitem__(self, idx):
        
        t_block_idx = self.tensor_block_idx[idx]
        block_idx = tensor_2_block_idx(t_block_idx)
        
        layer_idx = block_idx['layer_idx']
        ltype = block_idx['ltype']
        wtype = block_idx['wtype']
        col_idx = block_idx['idx']
        row_slice = slice(block_idx['slice'][0], block_idx['slice'][1])

        weight = getattr(self.model.model.layers[layer_idx], ltype)
        weight = getattr(weight, wtype).weight
        weight = weight[row_slice, col_idx]
        
        input_block = self.layer_inputs.layers[layer_idx][ltype][wtype][col_idx].unsqueeze(-1).expand(16)
        
        # assert weight.shape == input_block.shape
        
        return {'tensor_block_idx': t_block_idx,
                'weight_block': weight,
                'input_block': input_block
                }

class RowBlock_Mag(Dataset):
    def __init__(self, net, tensor_block_idx, layer_inputs, dataset_stats):

        self.model = net
        self.layer_inputs = layer_inputs

        self.tensor_block_idx = tensor_block_idx
        print("Dataset Shape: ", tensor_block_idx.size())
        
        self.mean = torch.Tensor(dataset_stats["mean_channel"])
        self.std = torch.Tensor(dataset_stats["std_channel"])

    def __len__(self):
        return len(self.tensor_block_idx)

    def __getitem__(self, idx):
        
        t_block_idx = self.tensor_block_idx[idx]
        block_idx = tensor_2_block_idx(t_block_idx)
        
        layer_idx = block_idx['layer_idx']
        ltype = block_idx['ltype']
        wtype = block_idx['wtype']
        row_idx = block_idx['idx']
        col_slice = slice(block_idx['slice'][0], block_idx['slice'][1])

        weight = getattr(self.model.model.layers[layer_idx], ltype)
        weight = getattr(weight, wtype).weight
        weight = weight[row_idx, col_slice]
        
        input_block = self.layer_inputs.layers[layer_idx][ltype][wtype][col_slice]
        # assert weight.shape == input_block.shape
        
        return {'tensor_block_idx': t_block_idx,
                'weight_block': weight,
                'input_block': input_block
                }

def get_dataset_block_16_calib(direction):
    model_name = 'meta-llama/Meta-Llama-3-8B'
    cache_directory = "../Wparam_dataset_v0/model_zoo/huggingface" 
    ckpt_path = latest_version_path(cache_directory, model_name)
    net = AutoModelForCausalLM.from_pretrained(ckpt_path, local_files_only=True)
    net = net.to(torch.device('cpu'))

    for param in net.parameters():
        param.requires_grad = False

    layer_inputs = torch.load('../Wparam_dataset/calib_data/layer_inputs_channelwise_mag.pt')
    
    if direction == 'col':
        tensor_block_idx_train = torch.load('../Wparam_dataset/block_16_idx/tensor_block_col_idx_train.pt')
        tensor_block_idx_val = torch.load('../Wparam_dataset/block_16_idx/tensor_block_col_idx_val.pt')       
        
        with open('../Wparam_dataset/dataset_block/meta-llama/Meta-Llama-3-8B/mlp_attn_16_col_dataset_stats.json', "r", encoding="utf-8") as file:
            dataset_stats = json.load(file)  # JSON 파일을 Python 객체로 변환

        train_dataset = ColBlock_Mag(net, tensor_block_idx_train, layer_inputs, dataset_stats["train"])
        valid_dataset = ColBlock_Mag(net, tensor_block_idx_val, layer_inputs, dataset_stats["val"])
        
    elif direction == 'row':
        tensor_block_idx_train = torch.load('../Wparam_dataset/block_16_idx/tensor_block_row_idx_train.pt')
        tensor_block_idx_val = torch.load('../Wparam_dataset/block_16_idx/tensor_block_row_idx_val.pt')       
        
        with open('../Wparam_dataset/dataset_block/meta-llama/Meta-Llama-3-8B/mlp_attn_16_row_dataset_stats.json', "r", encoding="utf-8") as file:
            dataset_stats = json.load(file)  # JSON 파일을 Python 객체로 변환

        train_dataset = RowBlock_Mag(net, tensor_block_idx_train, layer_inputs, dataset_stats["train"])
        valid_dataset = RowBlock_Mag(net, tensor_block_idx_val, layer_inputs, dataset_stats["val"])
    else:
        raise
    return train_dataset, valid_dataset, dataset_stats["train"]["std"], dataset_stats["val"]["std"]