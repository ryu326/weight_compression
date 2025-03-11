import glob
import json
import os
import random

import numpy as np
import PIL.Image as Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class Weight_Vector_Dataset(Dataset):
    def __init__(self, dataset, dataset_stats, input_size):
        # split = 'train' or 'val'

        self.dataset = dataset

        self.mean = torch.Tensor(dataset_stats["mean_channel"]).view(-1, input_size)
        self.std = torch.Tensor(dataset_stats["std_channel"]).view(-1, input_size)

        self.input_size = input_size
        
        self.random_values = [torch.tensor(i, dtype=torch.long) for i in range(4)]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx].view(-1, self.input_size)
        q_level = random.choice(self.random_values)
        return {'weight_block': img,
                'input_block': torch.zeros(1),
                'q_level': q_level}


def get_datasets_block_seq_random_qlevel(block_direction, input_size):
    dataset_folder_path = f'../Wparam_dataset_v1/dataset_block/meta-llama/Meta-Llama-3-8B/mlp_attn_2048_{block_direction}_dataset.pt'
    data = torch.load(dataset_folder_path)
    
    # dataset_stats = torch.load(dataset_folder_path.replace('dataset.pt', 'dataset_stats.pt'))
    
    with open(dataset_folder_path.replace("dataset.pt", "dataset_stats.json"), "r", encoding="utf-8") as file:
        dataset_stats = json.load(file)  # JSON 파일을 Python 객체로 변환

    train_dataset = Weight_Vector_Dataset(data["train"], dataset_stats["train"], input_size)
    valid_dataset = Weight_Vector_Dataset(data["val"], dataset_stats["val"], input_size)

    return train_dataset, valid_dataset, dataset_stats["train"]["std"], dataset_stats["val"]["std"]
