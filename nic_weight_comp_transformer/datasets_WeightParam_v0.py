import json
import os
import random

import numpy as np
import PIL.Image as Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class WParam_dataset(Dataset):
    def __init__(self, dataset_folder, split="train", param_type="mlp", data_dim=1024, length=64, seed=100):
        path = os.path.join(dataset_folder, "path_json", f"{param_type}_tensor_path_{split}.json")

        with open(path, "r") as f:
            path_list = json.load(f)
        print(len(path_list))
        count = 0

        self.data_dim = data_dim
        self.length = length

        tensor_path_list = []
        for tensor_path in path_list:
            try:
                path = dataset_folder + tensor_path
                t = np.load(path)
                # t.reshape(-1, self.length, self.data_dim)
                if t.size % (data_dim * length) != 0:
                    continue
                count += 1
                tensor_path_list.append(path)
            except:
                continue

        self.num = count
        self.tensor_path_list = tensor_path_list

        if split == "val":
            self.tensor_path_list = self.tensor_path_list[:100]
            self.num = 100

        if param_type == "mlp":
            self.mean = -5.42295056421355e-06
            self.std = 0.011819059083636133

        print(f"####### data num: {self.num} ########")
        print(f"####### data num: {len(self.tensor_path_list)} ########")

    def __len__(self):
        return len(self.tensor_path_list)

    def load_tensor(self, tensor_path):
        X = np.load(tensor_path)
        X = torch.from_numpy(X).requires_grad_(False)
        X = X.view(-1, self.length, self.data_dim)
        random_index = torch.randint(0, X.size(0), (1,)).item()

        return (X[random_index] - self.mean) / self.std

    def __getitem__(self, idx):
        img = self.load_tensor(self.tensor_path_list[idx])
        return img

    # def __init__(self, dataset_folder, split='train', dim = 1024, length = 128, seed = 100):
    #     ## //homejgryu/Weight_compression/Wparam_dataset/path_json/meta-llama-3-8b_mlp_train.json

    #     self.data = torch.from_numpy(np.load(dataset_folder))

    # def __len__(self):
    #     return self.data.shape[0]

    # def __getitem__(self, idx):
    #     img= self.data[idx]
    #     return img
