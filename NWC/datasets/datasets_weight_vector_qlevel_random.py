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
    def __init__(self, dataset, dataset_stats, Q, args, split='train'):
        self.dataset = dataset
        self.Q = int(Q)
        self.split = split

        if args.dataset_stat_type == 'scaler':
            self.mean = torch.tensor(dataset_stats["mean"])
            self.std = torch.tensor(dataset_stats["std"])
        elif args.dataset_stat_type == 'channel':
            self.mean = torch.Tensor(dataset_stats["mean_channel"])
            self.std = torch.Tensor(dataset_stats["std_channel"])

        
        self.random_values = [torch.tensor(i, dtype=torch.long) for i in range(self.Q)]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]

        # if self.split == 'val':
        #     prob = random.random()
        #     if prob < 0.001:         # 0.1%
        #         q_level = torch.tensor(3, dtype=torch.long)
        #     elif prob < 0.011:        # 0.1% + 1% = 1.1%
        #         q_level = torch.tensor(2, dtype=torch.long)
        #     elif prob < 0.111:        # 0.1% + 1% + 10% = 11.1%
        #         q_level = torch.tensor(1, dtype=torch.long)
        #     else:
        #         q_level = torch.tensor(0, dtype=torch.long)
        # else:
        #     q_level = random.choice(self.random_values)
        q_level = random.choice(self.random_values)

        return {
            'weight_block': img,
            'q_level': q_level,
        }


# class Weight_Vector_Dataset(Dataset):
#     def __init__(self, dataset, dataset_stats, input_size, Q, args):
#         # split = 'train' or 'val'

#         self.dataset = dataset
#         self.Q = int(Q)

#         if args.dataset_stat_type == 'scaler':
#             self.mean = torch.tensor(dataset_stats["mean"])
#             self.std = torch.tensor(dataset_stats["std"])
#         elif args.dataset_stat_type == 'channel':
#             self.mean = torch.Tensor(dataset_stats["mean_channel"]).view(-1, input_size)
#             self.std = torch.Tensor(dataset_stats["std_channel"]).view(-1, input_size)

#         self.input_size = input_size
        
#         self.random_values = [torch.tensor(i, dtype=torch.long) for i in range(self.Q)]

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         img = self.dataset[idx].view(-1, self.input_size)
#         q_level = random.choice(self.random_values)
#         return {'weight_block': img,
#                 'q_level': q_level.unsqueeze(0)}


def get_datasets_vector_random_qlevel(dataset_pt_path, Q, args):
    
    data = torch.load(dataset_pt_path)
    
    with open(dataset_pt_path.replace(".pt", "_dataset_stats.json"), "r", encoding="utf-8") as file:
        dataset_stats = json.load(file)  # JSON 파일을 Python 객체로 변환

    train_dataset = Weight_Vector_Dataset(data["train"], dataset_stats["train"], Q, args)
    valid_dataset = Weight_Vector_Dataset(data["val"], dataset_stats["val"], Q, args, split='val')

    return train_dataset, valid_dataset, dataset_stats["train"]["std"], dataset_stats["val"]["std"]
