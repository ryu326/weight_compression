import argparse
import json
import os
import sys
import torch
import torch.nn as nn
import re
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    CLIPVisionModelWithProjection,
    ViTForImageClassification,
)
from torch.utils.data import DataLoader

# device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

notebook_dir = os.path.dirname(os.path.abspath("__file__"))
project_root = os.path.abspath(os.path.join(notebook_dir, ".."))

std = 0.012528747320175171

if project_root not in sys.path:
    sys.path.append(project_root)

import models
from datasets_weight_block_idx_calib_mag import get_dataset_block_16_calib, LayerInputs

def parse_args():
    parser = argparse.ArgumentParser(description="Reconstruct model using specified configuration.")
    parser.add_argument("--cuda", type=int, default=0, help="CUDA device number (e.g., 0, 1, 2, etc.)")
    parser.add_argument("--model_path", type=str, default=
                        '/home/jgryu/Weight_compression/VQVAE/checkpoint/vqvae_mag/col_16_calib/bpp3.0_size16_smse_ne256_de16_K8_P6_encdim512_batch_size2048_total_iter1500000_lr0.0001_seed100/best_mse_model_MSE_0.08167_total_iter_50000.pth.tar')
    parser.add_argument("--direction", type=str, default='row')
    return parser.parse_args()

def tensor_2_block_idx(tensor_data):
    ltype_mapping = {0: 'self_attn', 1: 'mlp'}
    wtype_mapping = {
        0: 'q_proj', 1: 'k_proj', 2: 'v_proj', 3: 'o_proj',
        4: 'gate_proj', 5: 'up_proj', 6: 'down_proj'
    }

    layer_idx = tensor_data[0].item()
    ltype = ltype_mapping[tensor_data[1].item()]
    wtype = wtype_mapping[tensor_data[2].item()]
    row_idx = tensor_data[3].item()
    col_slice_start = tensor_data[4].item()
    col_slice_stop = tensor_data[5].item()

    reconstructed_data = {
        'layer_idx': layer_idx,
        'ltype': ltype,
        'wtype': wtype,
        'row_idx': row_idx,
        'col_slice': (col_slice_start, col_slice_stop)
    }

    return reconstructed_data

def latest_version_path(cache_dir, model_name, branch="main"):
    model_name_dir = "models--" + model_name.replace("/", "--")
    path = os.path.join(cache_dir, model_name_dir)
    if not os.path.isdir(os.path.join(path, "snapshots")):
        return None
    branch_file = os.path.join(path, "refs", branch)
    with open(branch_file, "r", encoding="utf-8") as file:
        revision = file.read()
    return os.path.join(path, "snapshots", revision)


def reconstruct_model(state_dict, model, input_mag, direction, batch_size=32768):
    with torch.no_grad():
        mean_MSE = 0
        count = 0
        mse_func = nn.MSELoss()
        device = next(model.parameters()).device
        recon_state_dict = {}

        for k, W in tqdm(state_dict.items()):
            if not "mlp" in k and not "self_attn" in k:
                continue
            
            if direction == 'col':
                W = W.T
            
            W_reshaped = W.reshape(-1, model.input_size)  # ( -1, -1) --> (-1, size, size)
            W_recon = torch.zeros(W_reshaped.shape, dtype=W_reshaped.dtype, device=W_reshaped.device)

            for start_idx in range(0, W_reshaped.shape[0], batch_size):
                end_idx = min(start_idx + batch_size, W_reshaped.shape[0])  # 마지막 배치를 처리할 때 범위 조정
                batch = W_reshaped[start_idx:end_idx]  # batch_size 크기로 슬라이싱
                batch = batch.to(device)  # 배치를 GPU로 이동

                data = {}
                data['weight_block'] = batch
                out = model(data)
                x_hat = out["x_hat"]
                W_recon[start_idx:end_idx] = x_hat

                # print(mse_func(out["x"], out["x_hat"]).item())
                mean_MSE += mse_func(out["x"], out["x_hat"]).item()
                count += 1

            W_recon = W_recon.reshape(W.shape).cpu()
            
            if direction == 'col':
                W_recon = W_recon.T
            
            recon_state_dict[k] = W_recon

        mean_MSE /= count

    return recon_state_dict


if __name__ == "__main__":
    args = parse_args()

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    import models
    config = os.path.join(os.path.dirname(args.model_path), 'config.json')
    with open(config, 'r', encoding='utf-8') as file:
        config = json.load(file)

    comp_model = models.VQVAE(
        input_size=config['input_size'],
        dim_encoder=config['dim_encoder'],
        P=config['P'],
        dim_embeddings=config['dim_embeddings'],
        n_embeddings=2 ** config['K'],
        n_resblock=config['n_resblock'],
        beta=0.25,
        scale=torch.zeros(16),
        shift=torch.zeros(16)
    )

    ckpt = torch.load(args.model_path)
    comp_model.load_state_dict(ckpt["state_dict"])
    comp_model.to(device)
    
    input_mag = torch.load('/home/jgryu/Weight_compression/Wparam_dataset/calib_data/layer_inputs_channelwise_mag.pt', weights_only=False)    

    cache_directory = "../Wparam_dataset_v0/model_zoo/huggingface"
    ckpt_path = latest_version_path(cache_directory, "meta-llama/Meta-Llama-3-8B")
    net = AutoModelForCausalLM.from_pretrained(ckpt_path, local_files_only=True)
    ckpt_path = "/home/jgryu/Weight_compression/model_cache/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920"
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, local_files_only=True)
    state_dict = net.state_dict()

    recon_state_dict= reconstruct_model(state_dict, comp_model, input_mag, args.direction )

    n_total = 0
    mse_total = 0
    for k, v in state_dict.items():
        if k not in recon_state_dict.keys():
            recon_state_dict[k] = v
        else:
            mse = ((recon_state_dict[k] - state_dict[k]) ** 2).sum().item()
            n = v.numel()
            print(k, f'mse: {mse/n/std**2}')            
            
            mse_total += mse
            n_total += n

    mse_total = mse_total / n_total / std **2 
    print(f'Total MSE : {mse_total}')

    net.load_state_dict(recon_state_dict)
    save_directory = (
        f"/home/jgryu/Weight_compression/model_reconstructed/vqvae_qlike/{args.direction}_recon_{os.path.join(*args.model_path.split('/')[-3:])}_MSE_{round(mse_total, 5)}"
    )
    net = net.to(dtype=torch.bfloat16)
    
    print('Strart saving')
    net.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    print('End saving')