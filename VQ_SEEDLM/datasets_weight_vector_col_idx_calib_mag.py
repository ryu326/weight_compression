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
    col_idx = tensor_data[3].item()
    row_slice_start = tensor_data[4].item()
    row_slice_stop = tensor_data[5].item()

    # Construct the dictionary
    reconstructed_data = {
        'layer_idx': layer_idx,
        'ltype': ltype,
        'wtype': wtype,
        'col_idx': col_idx,
        'row_slice': (row_slice_start, row_slice_stop)
    }

    return reconstructed_data


class Weight_Vector_Dataset_Calib_Mag(Dataset):
    def __init__(self, net, tensor_block_idx, layer_inputs, dataset_stats, split):
        # split = 'train' or 'val'            
            
        self.model = net
        self.layer_inputs = layer_inputs
        
        ## v1
        # if split == 'val':
        #     np.random.seed(100)
        #     selected_indices = np.random.permutation(len(tensor_block_idx))[:1000]
        #     self.tensor_block_idx = [tensor_block_idx[i] for i in selected_indices]
        # else:
        #     self.tensor_block_idx = tensor_block_idx
        
        ## v2
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
        col_idx = block_idx['col_idx']
        row_slice = slice(block_idx['row_slice'][0], block_idx['row_slice'][1])

        layer = getattr(self.model.model.layers[layer_idx], ltype)
        weight = getattr(layer, wtype).weight
        weight_block = weight[row_slice, col_idx]
        
        input_block = self.layer_inputs.layers[layer_idx][ltype][wtype][col_idx].unsqueeze(-1)
        # import ipdb; ipdb.set_trace()
        # assert input_block.dim() == 1
        
        return {'block_idx': block_idx, 
                'tensor_block_idx': t_block_idx,
                'weight_block': weight_block,
                'input_block': input_block
                }


def get_dataset_col_16_calib():

    model_name = 'meta-llama/Meta-Llama-3-8B'
    cache_directory = "../Wparam_dataset_v0/model_zoo/huggingface" 
    ckpt_path = latest_version_path(cache_directory, model_name)
    net = AutoModelForCausalLM.from_pretrained(ckpt_path, local_files_only=True)
    net = net.to(torch.device('cpu'))

    for param in net.parameters():
        param.requires_grad = False

    tensor_block_idx_train = torch.load('../Wparam_dataset/per_row_16_calib/tensor_block_col_idx_train.pt')
    tensor_block_idx_val = torch.load('../Wparam_dataset/per_row_16_calib/tensor_block_col_idx_val.pt')
    
    layer_inputs = torch.load('../Wparam_dataset/per_row_16_calib/layer_inputs_channelwise_mag.pt')
    
    with open('../Wparam_dataset/dataset_per_row/meta-llama/Meta-Llama-3-8B/mlp_attn_16_col_dataset_stats.json', "r", encoding="utf-8") as file:
        dataset_stats = json.load(file)  # JSON 파일을 Python 객체로 변환

    train_dataset = Weight_Vector_Dataset_Calib_Mag(net, tensor_block_idx_train, layer_inputs, dataset_stats["train"], 'train')
    valid_dataset = Weight_Vector_Dataset_Calib_Mag(net, tensor_block_idx_val, layer_inputs, dataset_stats["val"], 'val')

    return train_dataset, valid_dataset, dataset_stats["train"]["std"], dataset_stats["val"]["std"]
