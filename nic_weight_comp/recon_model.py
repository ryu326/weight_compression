import os, random, sys, socket, lpips, shutil, operator

# 시간 측정해보기

import pandas as pd
import numpy as np

import torch
import torch.optim as optim
import torch.distributed as dist
import torchvision
import torch.nn.functional as F

from torch.utils.data import DataLoader

from datasets_Imagenet_best_worst import Imagenet_best_worst
from datasets_ImageNet import ImageNet_dataset
from datasets_WeightParam import WParam_dataset

# from datasets_openimages_v6 import Openimages_v6_dataset

from pytorch_msssim import ms_ssim as ms_ssim_func

from models.TCM import TCM
from models.FTIC import FrequencyAwareTransFormer
from models.ELIC import ELIC, model_config

from utils.optimizers import *
from utils.util import *

from tqdm import tqdm


def pad(x, p):
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )
    return x_padded, (padding_left, padding_right, padding_top, padding_bottom)


def crop(x, padding):
    return F.pad(
        x,
        (-padding[0], -padding[1], -padding[2], -padding[3]),
    )


def make_image_format(W, wp_mean, wp_std, normalize):
    if normalize:
        W = (W - wp_mean) / wp_std
    W = W.unsqueeze(1).repeat(1, 3, 1, 1)
    return W


def reverse_image_format(W, wp_mean, wp_std, normalize):
    # 이미지를 채널 축에서 3 -> 1로 줄이기
    # W = W[:, 0, :, :]  # 첫 번째 채널만 유지
    W = W.mean(1)  # 첫 번째 채널만 유지
    # Normalize를 반대로 적용
    if normalize:
        W = W * wp_std + wp_mean
    return W


def reconstruct_model(state_dict, model, save_path, logger, size, weight_condition, mean, std, batch=4, normalize=True):
    avg_bpp = 0.0
    mean_MSE = 0
    count = 0
    mse_func = nn.MSELoss()

    device = next(model.parameters()).device

    recon_state_dict = {}

    for k, W in state_dict.items():
        if not weight_condition in k:
            continue
        print(f"### Reconstructing {k} ####")

        W_reshaped = W.reshape(-1, size, size)  # ( -1, -1) --> (-1, size, size)
        W_reshaped = W_reshaped.to(device)
        W_reshaped = make_image_format(W_reshaped, mean, std, normalize)  # (-1, size, size) --> (-1, 3, size, size)

        # try :
        #     W_reshaped = W_reshaped.reshape(-1, batch, 3, size, size)  # (-1, 3, size, size) --> (-1, batch, 3, size, size)
        # except:
        #     W_reshaped = W_reshaped.reshape(-1, 1, 3, size, size)  # (-1, 3, size, size) --> (-1, 1, 3, size, size)

        W_reshaped = W_reshaped.reshape(-1, 1, 3, size, size)  # (-1, 3, size, size) --> (-1, 1, 3, size, size)
        W_recon = torch.zeros(W_reshaped.shape, dtype=W_reshaped.dtype, device=W_reshaped.device)

        for idx, W_slice in tqdm(enumerate(W_reshaped)):  # (bath, 3, size, size) in (-1, bath, 3, size, size)
            # print(W_slice.shape)
            count += 1
            x = W_slice.to(device)  # (bach3, size, size) --> (1, 3, size, size)

            try:
                x_paddeimg, padding = pad(x, p=128)
                out_enc = model.compress(x_paddeimg.to(device))
            except:
                x_paddeimg, padding = pad(x, p=256)
                out_enc = model.compress(x_paddeimg.to(device))

            out_dec = model.decompress(out_enc["strings"], out_enc["shape"])

            num_pixels = x.size(0) * x.size(2) * x.size(3)
            bpp = 0
            for s in out_enc["strings"]:
                if s != [0]:  #
                    bpp += len(s[0]) * 8.0 / num_pixels

            x_hat = crop(out_dec["x_hat"], padding).clone().detach()  # (1, 3, size, size)
            mse = mse_func(x, x_hat).item()
            avg_bpp += bpp
            mean_MSE += mse

            W_recon_slice = x_hat
            W_recon[idx] = W_recon_slice
            # logger.info(f"File name: {idx}, MSE: {mse}, BPP: {bpp}")

        W_recon = W_recon.reshape(-1, 3, size, size).to("cpu")  # (-1, batch, 3, size, size) --> (-1, 3, size, size)
        W_recon = reverse_image_format(W_recon, mean, std, normalize)  #  (-1, 3, size, size) --> (-1, size, size)
        recon_state_dict[k] = W_recon

    avg_bpp /= count
    mean_MSE /= count
    # logger.info(f'Average_MSE: {mean_MSE}, Average_Bit-rate: {avg_bpp} bpp')

    return recon_state_dict, avg_bpp, mean_MSE


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
from transformers import CLIPVisionModelWithProjection, ViTForImageClassification, AutoModelForCausalLM
from transformers import AutoModel, AutoTokenizer

ckpt_path = "/home/jgryu/Weight_compression/llm-awq/model_cache/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920"
net = AutoModelForCausalLM.from_pretrained(ckpt_path, local_files_only=True)

mean = np.load(
    f"/home/jgryu/Weight_compression/Wparam_dataset/TFRecord/meta-llama--Meta-Llama-3-8B/mlp/d16/mlp_d16_train_mean.npy"
)
std = np.load(
    f"/home/jgryu/Weight_compression/Wparam_dataset/TFRecord/meta-llama--Meta-Llama-3-8B/mlp/d16/mlp_d16_train_std.npy"
)
mean = torch.from_numpy(mean)
std = torch.from_numpy(std)

size = 256
weight_condition = "mlp"

path = "checkpoints_image_pretrained"
pt_list = os.listdir(path)
lmbdas = []
for pt in pt_list:
    lm = pt.replace(".pth", "")
    lmbdas.append(float(lm))
lmbdas = sorted(lmbdas)[-2:-1]
print(lmbdas)

for lm in lmbdas:
    print(f"##### lambda: {lm} #####")
    pt = f"{lm}.pth"
    ck_path = f"checkpoints_image_pretrained/{lm}.pth"

    try:
        checkpoint = torch.load(ck_path, map_location=device)
        assert isinstance(checkpoint, dict), "Checkpoint is not a dictionary"
        assert "state_dict" in checkpoint, "Missing 'state_dict' in checkpoint"
        print(f"Checkpoint for {lm} loaded successfully.")
    except Exception as e:
        print(f"Failed to load checkpoint for {lm}: {e}")

    model = TCM(N=64)
    try:
        model.load_state_dict(checkpoint["state_dict"])
        print(f"Model state_dict loaded successfully for {lm}.")
    except RuntimeError as e:
        print(f"Failed to load model state_dict for {lm}: {e}")

    model = model.eval().to(device)
    model.requires_grad_(False)
    model.update()

    recon_state_dict, avg_bpp, mean_MSE = reconstruct_model(
        net.state_dict(),
        model,
        save_path=None,
        logger=None,
        size=size,
        weight_condition=weight_condition,
        mean=mean,
        std=std,
    )

print(avg_bpp, mean_MSE)
torch.save(recon_state_dict, "reconstruncted_state_dict/meta-llama--Meta-Llama-3-8B_mlp_d256_256.pth")
