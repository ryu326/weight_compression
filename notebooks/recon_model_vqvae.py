import json
import os
import sys

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          CLIPVisionModelWithProjection,
                          ViTForImageClassification)

notebook_dir = os.path.dirname(os.path.abspath("__file__"))
project_root = os.path.abspath(os.path.join(notebook_dir, ".."))

if project_root not in sys.path:
    sys.path.append(project_root)

from VQ_SEEDLM import models


def reconstruct_model(state_dict, model, weight_condition, batch_size=32768 // 4):
    with torch.no_grad():
        mean_MSE = 0
        count = 0
        mse_func = nn.MSELoss()
        device = next(model.parameters()).device
        recon_state_dict = {}

        for k, W in tqdm(state_dict.items()):
            # if not weight_condition in k: continue
            if not "mlp" in k and not "attn" in k:
                continue
            # print(f'### Reconstructing {k} ####')

            W_reshaped = W.reshape(-1, model.input_size)  # ( -1, -1) --> (-1, size, size)
            W_recon = torch.zeros(W_reshaped.shape, dtype=W_reshaped.dtype, device=W_reshaped.device)

            for start_idx in range(0, W_reshaped.shape[0], batch_size):
                end_idx = min(start_idx + batch_size, W_reshaped.shape[0])  # 마지막 배치를 처리할 때 범위 조정
                batch = W_reshaped[start_idx:end_idx]  # batch_size 크기로 슬라이싱
                batch = batch.to(device)  # 배치를 GPU로 이동

                out = model(batch)
                x_hat = out["x_hat"]
                W_recon[start_idx:end_idx] = x_hat

                # print(mse_func(out["x"], out["x_hat"]).item())
                mean_MSE += mse_func(out["x"], out["x_hat"]).item()
                count += 1

            W_recon = W_recon.reshape(W.shape).cpu()
            recon_state_dict[k] = W_recon

        mean_MSE /= count

    return recon_state_dict, mean_MSE


def latest_version_path(cache_dir, model_name, branch="main"):
    model_name_dir = "models--" + model_name.replace("/", "--")
    path = os.path.join(cache_dir, model_name_dir)
    if not os.path.isdir(os.path.join(path, "snapshots")):
        return None
    branch_file = os.path.join(path, "refs", branch)
    with open(branch_file, "r", encoding="utf-8") as file:
        revision = file.read()
    return os.path.join(path, "snapshots", revision)


# model_path_list = [
#    '../VQ_SEEDLM/checkpoint/vqvae/Meta-Llama-3-8B/mlp_attn_16_row_dataset.pt/size16_ne16_denc512_P16_K8_de16_batch_size2048_total_iter2000000_lr0.0001_seed100/best_mse_model_MSE_0.00197_total_iter_350000.pth.tar',
#    '../VQ_SEEDLM/checkpoint/vqvae/Meta-Llama-3-8B/mlp_attn_16_row_dataset.pt/size16_ne16_denc512_P32_K8_de16_batch_size2048_total_iter2000000_lr0.0001_seed100/best_mse_model_MSE_0.0001_total_iter_300000.pth.tar',
#    '../VQ_SEEDLM/checkpoint/vqvae/Meta-Llama-3-8B/mlp_attn_16_row_dataset.pt/size16_neNone_de16_K8_P4_encdim512_batch_size2048_total_iter2000000_lr0.0001_seed100/best_mse_model_MSE_0.11166_total_iter_850000.pth.tar',
#    '../VQ_SEEDLM/checkpoint/vqvae/Meta-Llama-3-8B/mlp_attn_16_row_dataset.pt/size16_neNone_de16_K8_P6_encdim512_batch_size2048_total_iter2000000_lr0.0001_seed100/best_mse_model_MSE_0.03877_total_iter_800000.pth.tar',
#    ]

model_path_list = [
    "/home/jgryu/Weight_compression/VQ_SEEDLM/checkpoint/vqvae/Meta-Llama-3-8B/mlp_attn_16_row_dataset.pt/size16_enmse_neNone_de16_K8_P32_encdim512_batch_size2048_total_iter2000000_lr0.0001_seed100/best_mse_model_MSE_0.00335_total_iter_2000000.pth.tar",
    "/home/jgryu/Weight_compression/VQ_SEEDLM/checkpoint/vqvae/Meta-Llama-3-8B/mlp_attn_16_row_dataset.pt/size16_nmse_neNone_de16_K8_P10_encdim512_batch_size2048_total_iter2000000_lr0.0001_seed100/best_mse_model_MSE_0.00484_total_iter_1350000.pth.tar",
    "/home/jgryu/Weight_compression/VQ_SEEDLM/checkpoint/vqvae/Meta-Llama-3-8B/mlp_attn_16_row_dataset.pt/size16_enmse_neNone_de16_K8_P16_encdim512_batch_size2048_total_iter2000000_lr0.0001_seed100/best_mse_model_MSE_0.02292_total_iter_150000.pth.tar",
    "/home/jgryu/Weight_compression/VQ_SEEDLM/checkpoint/vqvae/Meta-Llama-3-8B/mlp_attn_16_row_dataset.pt/size16_ne16_denc512_P32_K8_de16_batch_size2048_total_iter2000000_lr0.0001_seed100/best_mse_model_MSE_0.0001_total_iter_300000.pth.tar",
    "/home/jgryu/Weight_compression/VQ_SEEDLM/checkpoint/vqvae/Meta-Llama-3-8B/mlp_attn_16_row_dataset.pt/size16_neNone_de16_K8_P4_encdim512_batch_size2048_total_iter2000000_lr0.0001_seed100/best_mse_model_MSE_0.11166_total_iter_850000.pth.tar",
    "/home/jgryu/Weight_compression/VQ_SEEDLM/checkpoint/vqvae/Meta-Llama-3-8B/mlp_attn_16_row_dataset.pt/size16_enmse_neNone_de16_K8_P4_encdim512_batch_size2048_total_iter2000000_lr0.0001_seed100/best_mse_model_MSE_0.60428_total_iter_1200000.pth.tar",
    "/home/jgryu/Weight_compression/VQ_SEEDLM/checkpoint/vqvae/Meta-Llama-3-8B/mlp_attn_16_row_dataset.pt/size16_enmse_neNone_de16_K8_P8_encdim512_batch_size2048_total_iter2000000_lr0.0001_seed100/best_mse_model_MSE_0.31604_total_iter_650000.pth.tar",
    "/home/jgryu/Weight_compression/VQ_SEEDLM/checkpoint/vqvae/Meta-Llama-3-8B/mlp_attn_16_row_dataset.pt/size16_nmse_neNone_de16_K8_P12_encdim512_batch_size2048_total_iter2000000_lr0.0001_seed100/best_mse_model_MSE_0.00263_total_iter_1850000.pth.tar",
    "/home/jgryu/Weight_compression/VQ_SEEDLM/checkpoint/vqvae/Meta-Llama-3-8B/mlp_attn_16_row_dataset.pt/size16_nmse_neNone_de16_K8_P8_encdim512_batch_size2048_total_iter2000000_lr0.0001_seed100/best_mse_model_MSE_0.01813_total_iter_1500000.pth.tar",
    "/home/jgryu/Weight_compression/VQ_SEEDLM/checkpoint/vqvae/Meta-Llama-3-8B/mlp_attn_16_row_dataset.pt/size16_neNone_de16_K8_P6_encdim512_batch_size2048_total_iter2000000_lr0.0001_seed100/best_mse_model_MSE_0.03877_total_iter_800000.pth.tar",
    "/home/jgryu/Weight_compression/VQ_SEEDLM/checkpoint/vqvae/Meta-Llama-3-8B/mlp_attn_16_row_dataset.pt/size16_enmse_neNone_de16_K8_P12_encdim512_batch_size2048_total_iter2000000_lr0.0001_seed100/best_mse_model_MSE_0.1122_total_iter_1350000.pth.tar",
    "/home/jgryu/Weight_compression/VQ_SEEDLM/checkpoint/vqvae/Meta-Llama-3-8B/mlp_attn_16_row_dataset.pt/size16_ne16_denc512_P16_K8_de16_batch_size2048_total_iter2000000_lr0.0001_seed100/best_mse_model_MSE_0.00197_total_iter_350000.pth.tar",
]


config_list = [
    {"input_size": 16, "dim_encoder": 512, "P": 32, "K": 8, "n_resblock": 4, "dim_embeddings": 16},
    {"input_size": 16, "dim_encoder": 512, "P": 10, "K": 8, "n_resblock": 4, "dim_embeddings": 16},
    {"input_size": 16, "dim_encoder": 512, "P": 16, "K": 8, "n_resblock": 4, "dim_embeddings": 16},
    {"input_size": 16, "dim_encoder": 512, "P": 32, "K": 8, "n_resblock": 4, "dim_embeddings": 16},
    {"input_size": 16, "dim_encoder": 512, "P": 4, "K": 8, "n_resblock": 4, "dim_embeddings": 16},
    {"input_size": 16, "dim_encoder": 512, "P": 4, "K": 8, "n_resblock": 4, "dim_embeddings": 16},
    {"input_size": 16, "dim_encoder": 512, "P": 8, "K": 8, "n_resblock": 4, "dim_embeddings": 16},
    {"input_size": 16, "dim_encoder": 512, "P": 12, "K": 8, "n_resblock": 4, "dim_embeddings": 16},
    {"input_size": 16, "dim_encoder": 512, "P": 8, "K": 8, "n_resblock": 4, "dim_embeddings": 16},
    {"input_size": 16, "dim_encoder": 512, "P": 6, "K": 8, "n_resblock": 4, "dim_embeddings": 16},
    {"input_size": 16, "dim_encoder": 512, "P": 12, "K": 8, "n_resblock": 4, "dim_embeddings": 16},
    {"input_size": 16, "dim_encoder": 512, "P": 16, "K": 8, "n_resblock": 4, "dim_embeddings": 16},
]

# config_list = [
#     # {"input_size": 16, "dim_encoder": 512, "P": 16, "ne": 256, "n_resblock": 4, "dim_embeddings": 16},
#     # {"input_size": 16, "dim_encoder": 512, "P": 32, "ne": 256, "n_resblock": 4, "dim_embeddings": 16},
#     # {"input_size": 16, "dim_encoder": 512, "P": 4, "ne": 256, "n_resblock": 4, "dim_embeddings": 16},
#     # {"input_size": 16, "dim_encoder": 512, "P": 6, "ne": 256, "n_resblock": 4, "dim_embeddings": 16},
# ]

with open(
    "/home/jgryu/Weight_compression/Wparam_dataset/dataset_per_row/meta-llama/Meta-Llama-3-8B/mlp_16_row_dataset_stats.json",
    "r",
    encoding="utf-8",
) as file:
    dataset_stats = json.load(file)  # JSON 파일을 Python 객체로 변환

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

for model_path, config in zip(model_path_list, config_list):
    model = models.VQVAE(
        input_size=config["input_size"],
        dim_encoder=config["dim_encoder"],
        P=config["P"],
        dim_embeddings=config["dim_embeddings"],
        n_embeddings=2 ** config["K"],
        n_resblock=config["n_resblock"],
        beta=0.25,
        scale=torch.Tensor(dataset_stats["train"]["std_channel"]).to(device),
        shift=torch.Tensor(dataset_stats["train"]["mean_channel"]).to(device),
    )
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)

    cache_directory = "../Wparam_dataset_v0/model_zoo/huggingface"
    ckpt_path = latest_version_path(cache_directory, "meta-llama/Meta-Llama-3-8B")
    net = AutoModelForCausalLM.from_pretrained(ckpt_path, local_files_only=True)

    ckpt_path = "/home/jgryu/Weight_compression/model_cache/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920"
    # net = AutoModelForCausalLM.from_pretrained(ckpt_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, local_files_only=True)
    state_dict = net.state_dict()

    recon_state_dict, mean_MSE = reconstruct_model(state_dict, model, weight_condition="mlp")

    print(mean_MSE / dataset_stats["train"]["std"] ** 2)

    for k, v in state_dict.items():
        if k not in recon_state_dict.keys():
            recon_state_dict[k] = v
            # print(k, v.shape)
        else:
            mse = ((recon_state_dict[k] - state_dict[k]) ** 2).mean()
            # print(k, f'{mse.item():-20f}')

    net.load_state_dict(recon_state_dict)
    save_directory = (
        f"/home/jgryu/Weight_compression/model_cache_reconstructed/vqvae/{os.path.join(*model_path.split('/')[-3:])}"
    )
    net = net.to(dtype=torch.bfloat16)

    net.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
