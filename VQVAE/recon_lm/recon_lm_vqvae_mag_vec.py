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


def reconstruct_model(state_dict, model, input_mag, direction, batch_size=32768//256):

    wtype_mapping = {'q_proj': 0, 'k_proj': 1, 'v_proj': 2, 'o_proj': 3, 'gate_proj': 4, 'up_proj': 5, 'down_proj': 6}
    with torch.no_grad():
        # mse_func = nn.MSELoss()
        device = next(model.parameters()).device
        recon_state_dict = {}

        # for k, W in tqdm(state_dict.items()):
        for k, W in tqdm(state_dict.items()):
            if not "mlp" in k and not "attn" in k: continue
            print(k)
            match = re.search(r"layers\.(\d+).", k)
            if match:
                layer_index = int(match.group(1))  # 찾은 숫자를 정수형으로 변환
                print("Layer index:", layer_index)
            else:
                print("No layer index found in the string.")
            
            # if layer_index == 1: break
            
            if 'self_attn' in k:
                ltype = 'self_attn'
            elif 'mlp' in k:
                ltype = 'mlp'
            
            if 'q_proj' in k:
                mapping = wtype_mapping['q_proj']
                wtype = 'q_proj'
            elif 'k_proj' in k:
                mapping = wtype_mapping['k_proj']
                wtype = 'k_proj'
            elif 'v_proj' in k:
                mapping = wtype_mapping['v_proj']
                wtype = 'v_proj'
            elif 'o_proj' in k:
                mapping = wtype_mapping['o_proj']
                wtype = 'o_proj'
            elif 'gate_proj' in k:
                mapping = wtype_mapping['gate_proj']
                wtype = 'gate_proj'
            elif 'up_proj' in k:
                mapping = wtype_mapping['up_proj']
                wtype = 'up_proj'
            elif 'down_proj' in k:
                mapping = wtype_mapping['down_proj']
                wtype = 'down_proj'

            rows, cols = W.shape
            input_block = input_mag.layers[layer_index][ltype][wtype]
            # print(input_block.shape)
            input_block = input_block / input_block.max(-1, keepdim=True).values
            assert input_block.dim() == 1
            print(input_block.shape)
            
            indices = torch.randperm(len(input_block))
            input_block = input_block[indices]
            
            input_block = input_block.expand(rows, cols)
            # print(input_block.shape)
            
            if direction == 'col':
                W = W.T
                input_block = input_block.T
                
            try: 
                W_reshaped = W.reshape(-1, 256, 16)
                input_block = input_block.reshape(-1, 256, 16)
            except:
                padding_size = (cols // 4096 + 1) * 4096 - cols
                W = torch.cat([W, torch.zeros(rows, padding_size, device=W.device)], dim=-1)
                W_reshaped = W.reshape(-1, 256, 16)
                
                input_block = torch.cat([input_block, torch.zeros(rows, padding_size, device=input_block.device)], dim=-1)
                input_block = input_block.reshape(-1, 256, 16)
                            
            W_recon = torch.zeros(W_reshaped.shape, dtype=W_reshaped.dtype, device=W_reshaped.device)

            for start_idx in range(0, W_reshaped.shape[0], batch_size):
                end_idx = min(start_idx + batch_size, W_reshaped.shape[0])  # 마지막 배치를 처리할 때 범위 조정
                w = W_reshaped[start_idx:end_idx]  # batch_size 크기로 슬라이싱
                w = w.to(device)  # 배치를 GPU로 이동

                a = input_block[start_idx:end_idx]
                a = a.to(device)
                
                data = {}
                data['weight_block'] = w
                data['input_block'] = a
                out = model(data)
                
                x_hat = out["x_hat"]
                W_recon[start_idx:end_idx] = x_hat
                
            W_recon = W_recon.reshape(rows, -1).cpu()
            W_recon = W_recon[:, :cols]
            
            assert W.shape == W_recon.shape
            # mse = mse_func(W, W_recon)        
            # print(k, mse / std**2 )
                        
            if direction == 'col':
                W_recon = W_recon.T
            
            recon_state_dict[k] = W_recon

    return recon_state_dict


if __name__ == "__main__":
    args = parse_args()

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    import models
    config = os.path.join(os.path.dirname(args.model_path), 'config.json')
    with open(config, 'r', encoding='utf-8') as file:
        config = json.load(file)

    comp_model = models.VQVAE_MAG(
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
        f"/home/jgryu/Weight_compression/model_reconstructed/vqvae_mag_vec_random/{os.path.join(*args.model_path.split('/')[-3:])}_MSE_{round(mse_total, 5)}"
    )
    net = net.to(dtype=torch.bfloat16)
    
    print('Strart saving')
    net.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    print('End saving')