import glob
import json
import os
import random

import numpy as np
import PIL.Image as Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class GaussianBlock(Dataset):
    def __init__(self, length, dataset_stats, input_size):
        self.len = length
        self.input_size = input_size
        self.mean = torch.Tensor(dataset_stats["mean_channel"])
        self.std = torch.Tensor(dataset_stats["std_channel"])

        # 모든 데이터를 미리 생성
        self.cached_samples = torch.randn(self.len, self.input_size) * self.std + self.mean

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return {
            'weight_block': self.cached_samples[idx],
            'input_block': torch.zeros(1)
        }



def get_datasets_gaussian_block(block_direction, input_size):
    dataset_folder_path = f'../Wparam_dataset/dataset_block/meta-llama/Meta-Llama-3-8B/mlp_attn_{input_size}_{block_direction}_dataset.pt'
    data = torch.load(dataset_folder_path)
    # dataset_stats = torch.load(dataset_folder_path.replace('dataset.pt', 'dataset_stats.pt'))
    
    with open(dataset_folder_path.replace("dataset.pt", "dataset_stats.json"), "r", encoding="utf-8") as file:
        dataset_stats = json.load(file)  # JSON 파일을 Python 객체로 변환

    # train_dataset = GaussianBlock(data["train"], dataset_stats["train"], input_size)
    # valid_dataset = GaussianBlock(data["val"], dataset_stats["val"], input_size)

    length = len(data['train'])

    train_dataset = GaussianBlock(length, dataset_stats["train"], input_size)
    valid_dataset = GaussianBlock(1000, dataset_stats["val"], input_size)

    return train_dataset, valid_dataset, dataset_stats["train"]["std"], dataset_stats["val"]["std"]
