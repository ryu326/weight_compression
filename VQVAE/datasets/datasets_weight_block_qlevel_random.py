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


class ColBlock_qlevel(Dataset):
    def __init__(self, dataset, dataset_stats):
        # split = 'train' or 'val'

        self.dataset = dataset

        self.mean = torch.Tensor(dataset_stats["mean_channel"])
        self.std = torch.Tensor(dataset_stats["std_channel"])
        
        self.random_values = [torch.tensor(i, dtype=torch.long) for i in range(4)]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        # q_level = torch.randint(0, 4, (1,))
        q_level = random.choice(self.random_values)
        return {'weight_block': img,
                'q_level': q_level}
        
def get_dataset_block_random_qlevel(block_direction, input_size):
    dataset_folder_path = f'../Wparam_dataset_v1/dataset_block/meta-llama/Meta-Llama-3-8B/mlp_attn_{input_size}_{block_direction}_dataset.pt'
    data = torch.load(dataset_folder_path)
    with open(dataset_folder_path.replace("dataset.pt", "dataset_stats.json"), "r", encoding="utf-8") as file:
        dataset_stats = json.load(file)  # JSON 파일을 Python 객체로 변환

    train_dataset = ColBlock_qlevel(data["train"], dataset_stats["train"])
    valid_dataset = ColBlock_qlevel(data["val"], dataset_stats["val"])

    return train_dataset, valid_dataset, dataset_stats["train"]["std"], dataset_stats["val"]["std"]