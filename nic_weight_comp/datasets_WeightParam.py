from torch.utils.data import Dataset
from torchvision import transforms
import os
import PIL.Image as Image
import json, random
import numpy as np
import torch
import glob


class WParam_dataset(Dataset):
    def __init__(self, dataset_folder, split="train", seed=100):

        directories = [
            os.path.join(dataset_folder, d)
            for d in os.listdir(dataset_folder)
            if os.path.isdir(os.path.join(dataset_folder, d))
        ]
        print(directories)
        self.wp_path_list = glob.glob(f"{directories[0]}/**/*.npy", recursive=True)
        print(len(self.wp_path_list))
        if split == "val":
            random.seed(seed)
            random_list = random.sample(self.wp_path_list, 1000)
            self.wp_path_list = random_list

        self.mean = np.load(dataset_folder + f"/mean_value.npy")
        self.std = np.load(dataset_folder + f"/std_value.npy")

    def __len__(self):
        return len(self.wp_path_list)

    def load_tensor(self, tensor_path):
        X = np.load(tensor_path)
        X = (X - self.mean) / self.std
        X = np.repeat(np.expand_dims(X, axis=0), 3, axis=0)
        return torch.from_numpy(X).requires_grad_(False)

    def __getitem__(self, idx):
        img = self.load_tensor(self.wp_path_list[idx])
        return img
