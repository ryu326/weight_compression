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
    def __init__(self, dataset, dataset_stats, input_size, R):
        # split = 'train' or 'val'

        self.dataset = dataset
        self.R = R
        
        self.mean = torch.Tensor(dataset_stats["mean_channel"]).view(-1, input_size)
        self.std = torch.Tensor(dataset_stats["std_channel"]).view(-1, input_size)

        self.input_size = input_size
        
        random_hess = torch.randn(8000, dataset.size(1))
        norms = torch.norm(random_hess, p=2, dim=-1, keepdim=True)  # 각 열의 2-노름 계산
        
        random_hess = random_hess / norms
        
        self.random_hess = random_hess.view(8000, -1, input_size)
        
    def __len__(self):
        return len(self.dataset)


    # def shuffle_hesseigen(self):
    #     perm = torch.randperm(self.random_hess.size(0))  # 10개의 샘플을 섞음
    #     self.random_hess = self.random_hess[perm, :, :, :]


    def __getitem__(self, idx):
        img = self.dataset[idx].view(-1, self.input_size)
        
        indices = torch.randperm(len(self.random_hess))[:self.R]
        hesseigen = self.random_hess[indices]
        
        return {'weight_block': img,
                'input_block': torch.zeros(1),
                'hesseigen': hesseigen}


def get_datasets_block_seq_random_hesseigen(dataset_pt_path, input_size, R):
    
    data = torch.load(dataset_pt_path)
    
    with open(dataset_pt_path.replace(".pt", "_dataset_stats.json"), "r", encoding="utf-8") as file:
        dataset_stats = json.load(file)  # JSON 파일을 Python 객체로 변환

    train_dataset = Weight_Vector_Dataset(data["train"], dataset_stats["train"], input_size, R)
    valid_dataset = Weight_Vector_Dataset(data["val"], dataset_stats["val"], input_size, R)

    return train_dataset, valid_dataset, dataset_stats["train"]["std"], dataset_stats["val"]["std"]
