import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from safetensors import safe_open
import os
from typing import Dict, List, Tuple
from multiprocessing import Pool, cpu_count

# plot_multiple_weight_distributions 함수는 수정할 필요가 없습니다.
# (이하 동일)
def plot_multiple_weight_distributions(weights_dict: Dict[str, torch.Tensor], title: str, save_path: str):
    num_weights = len(weights_dict)
    if num_weights == 0:
        print(f"No weights found for {title}.")
        return

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.suptitle(title, fontsize=16, y=1.0)
    epsilon = 1e-8
    
    for name, tensor_2d in weights_dict.items():
        if tensor_2d.dim() != 2:
            continue
        tensor_2d = tensor_2d.to(torch.float32)
        tensor_norm_global = tensor_2d / (torch.std(tensor_2d) + epsilon)
        tensor_norm_row = tensor_2d / (torch.std(tensor_2d, dim=1, keepdim=True) + epsilon)
        tensor_norm_col = tensor_2d / (torch.std(tensor_2d, dim=0, keepdim=True) + epsilon)
        col_stds_of_row_normed = torch.std(tensor_norm_row, dim=0, keepdim=True)
        tensor_norm_row_col = tensor_norm_row / (col_stds_of_row_normed + epsilon)

        sns.kdeplot(tensor_norm_global.flatten().cpu().numpy(), ax=axes[0], label=name, fill=True, alpha=0.1)
        sns.kdeplot(tensor_norm_row.flatten().cpu().numpy(), ax=axes[1], label=name, fill=True, alpha=0.1)
        sns.kdeplot(tensor_norm_col.flatten().cpu().numpy(), ax=axes[2], label=name, fill=True, alpha=0.1)
        sns.kdeplot(tensor_norm_row_col.flatten().cpu().numpy(), ax=axes[3], label=name, fill=True, alpha=0.1)

    axes[0].set_title('1. Normalized by Global Std Dev', fontsize=12)
    axes[1].set_title('2. Normalized by Row-wise Std Dev', fontsize=12)
    axes[2].set_title('3. Normalized by Column-wise Std Dev', fontsize=12)
    axes[3].set_title('4. Row then Column Normalization', fontsize=12)
    
    for ax in axes:
        ax.set_xlabel('Normalized Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Comparison plot saved to: {save_path}")

# =================================================================================
# 병렬 처리를 위한 Worker 함수 (수정된 부분)
# =================================================================================
def process_layer_worker(args):
    """하나의 레이어를 처리하는 독립적인 함수"""
    model_name_fs, model_path, safetensors_files, idx, device = args # 인자 수정
    print(f"--- Processing Layer {idx} on process {os.getpid()} ---")

    weights_to_compare = {}
    linear_keys = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    
    # ***** 수정된 부분 시작 *****
    # 모든 safetensors 파일을 순회하며 해당 레이어의 가중치를 찾습니다.
    for sf_file in safetensors_files:
        file_path = os.path.join(model_path, sf_file)
        with safe_open(file_path, framework="pt", device=device) as f:
            for key in linear_keys:
                # 이미 찾은 키는 다시 찾지 않음
                if key in weights_to_compare:
                    continue
                
                # 어텐션 블록에서 텐서 찾기
                tensor_name = f"model.layers.{idx}.self_attn.{key}.weight"
                if tensor_name in f.keys():
                    weights_to_compare[key] = f.get_tensor(tensor_name)
                
                # MLP 블록에서 텐서 찾기
                tensor_name = f"model.layers.{idx}.mlp.{key}.weight"
                if tensor_name in f.keys():
                    weights_to_compare[key] = f.get_tensor(tensor_name)
    # ***** 수정된 부분 끝 *****

    if weights_to_compare:
        plot_multiple_weight_distributions(
            weights_dict=weights_to_compare,
            title=f'{model_name_fs} Layer {idx} - Weight Distribution Comparison',
            save_path=f"./plot/comparison/{model_name_fs}/layer_{idx}_comparison.png"
        )
    else:
        print(f"Warning: No weights found for layer {idx}.")
        
    return f"Layer {idx} done."

# =================================================================================
# 메인 실행 로직 (수정된 부분)
# =================================================================================
if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    
    model_name = 'meta-llama/Meta-Llama-3-8B'
    model_name_fs = model_name.replace('/', '--')
    model_path = f"./hf_model/{model_name_fs}"
    
    try:
        # 파일 목록을 이름순으로 정렬하여 일관성을 유지합니다.
        safetensors_files = sorted([f for f in os.listdir(model_path) if f.endswith('.safetensors')])
    except FileNotFoundError:
        print(f"Model directory not found at {model_path}. Please ensure the model is downloaded.")
        exit()

    if not safetensors_files:
        raise FileNotFoundError("Safetensors file not found in the directory. Please ensure the model is downloaded correctly.")
    
    layer_indices = [0, 1, 10]
    device = 'cpu'

    # ***** 수정된 부분 시작 *****
    # worker에 파일 리스트 전체를 전달하도록 tasks를 수정합니다.
    tasks = [(model_name_fs, model_path, safetensors_files, idx, device) for idx in layer_indices]
    # ***** 수정된 부분 끝 *****
    
    num_processes = min(cpu_count(), 8)
    print(f"Starting parallel processing with {num_processes} processes for layers {layer_indices}...")
    
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_layer_worker, tasks)

    for r in results:
        print(r)
        
    print("All tasks finished.")