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
    def __init__(self, dataset, dataset_stats, input_size, Q, args):

        self.Q = int(Q)
        self.std = torch.zeros(1)
        self.mean = torch.zeros(1)

        self.weight = dataset['weight']

        stat_keys = ["mean", "median", "std", "range", "iqr", "skewness", "kurtosis"]
        stats = [dataset[key] for key in stat_keys]
        self.lstats = torch.stack(stats, dim=-1)
        self.input_size = input_size
    
        self.random_values = [torch.tensor(i, dtype=torch.long) for i in range(self.Q)]

    def __len__(self):
        return len(self.weight)

    def __getitem__(self, idx):
        weight = self.weight[idx].view(-1, self.input_size)
        q_level = random.choice(self.random_values)
        return {'weight_block': weight,
                'l_cdt': self.lstats[idx],
                'q_level': q_level.unsqueeze(0)}


def get_datasets_block_seq_random_qlevel_lstats(dataset_pt_path, input_size, Q, args):
    
    data = torch.load(dataset_pt_path)
    
    with open(dataset_pt_path.replace(".pt", "_dataset_stats.json"), "r", encoding="utf-8") as file:
        dataset_stats = json.load(file)  # JSON 파일을 Python 객체로 변환

    train_dataset = Weight_Vector_Dataset(data["train"], dataset_stats["train"], input_size, Q, args)
    valid_dataset = Weight_Vector_Dataset(data["val"], dataset_stats["val"], input_size, Q, args)

    return train_dataset, valid_dataset, dataset_stats["train"]["std"], dataset_stats["val"]["std"]
