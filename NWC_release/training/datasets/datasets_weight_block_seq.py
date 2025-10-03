import glob
import json
import os
import random

import numpy as np
import PIL.Image as Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

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
    def __init__(self, dataset, dataset_stats, input_size, args):
        # split = 'train' or 'val'

        self.dataset = dataset

        if args.dataset_stat_type == 'scaler':
            self.mean = torch.tensor(dataset_stats["mean"])
            self.std = torch.tensor(dataset_stats["std"])
        elif args.dataset_stat_type == 'channel':
            self.mean = torch.Tensor(dataset_stats["mean_channel"]).view(-1, input_size)
            self.std = torch.Tensor(dataset_stats["std_channel"]).view(-1, input_size)

        self.input_size = input_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx].view(-1, self.input_size)
        return {'weight_block': img,
                'input_block': torch.zeros(1),
                }


def get_datasets_block_seq(dataset_pt_path, input_size, args):
    
    data = torch.load(dataset_pt_path)
    
    with open(dataset_pt_path.replace(".pt", "_dataset_stats.json"), "r", encoding="utf-8") as file:
        dataset_stats = json.load(file)  # JSON 파일을 Python 객체로 변환

    train_dataset = Weight_Vector_Dataset(data["train"], dataset_stats["train"], input_size, args)
    valid_dataset = Weight_Vector_Dataset(data["val"], dataset_stats["val"], input_size, args)

    return train_dataset, valid_dataset, dataset_stats["train"]["std"], dataset_stats["val"]["std"]