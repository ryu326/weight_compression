import json
import argparse
import json
import os
import sys
import torch
import torch.nn as nn
import re
import math
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    OPTForCausalLM,
    BloomForCausalLM,    
)
from torch.utils.data import DataLoader
import logging
import numpy as np
from lib.algo import quip
import glog

def setup_logging(log_file):
    # Remove any pre-existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Configure logging settings
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file),  # Log to file
            logging.StreamHandler(sys.stdout)  # Log to console
        ]
    )

def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}

# def get_blocks(model):
#     if model.__class__.__name__ in ("LlamaForCausalLM", "Qwen2ForCausalLM"):
#         layers = model.model.layers
#     elif model.__class__.__name__ == "LlavaLlamaForCausalLM":
#         # layers = [model.model.layers, model.model.vision_tower.vision_tower.vision_model.encoder.layers]
#         layers = model.model.layers
#     elif isinstance(model, OPTForCausalLM):
#         layers = model.model.decoder.layers
#     elif isinstance(model, BloomForCausalLM):
#         layers = model.transformer.h
#     elif "mpt" in str(model.__class__).lower():
#         layers = model.transformer.blocks
#     elif "falcon" in str(model.__class__).lower():
#         layers = model.transformer.h
#     elif "bigcode" in str(model.__class__).lower():
#         layers = model.transformer.h
#     elif "neox" in str(model.__class__).lower():
#         layers = model.gpt_neox.layers
#     elif model.__class__.__name__ == "LlavaLlamaModel":
#         layers = model.llm.model.layers
#     else:
#         raise NotImplementedError(type(model))
#     return layers

def get_blocks(model):
    if model.__class__.__name__ in ("LlamaForCausalLM", "Qwen2ForCausalLM"):
        layers = model.model.layers
    elif model.__class__.__name__ == "LlavaLlamaForCausalLM":
        layers = model.model.layers
    elif isinstance(model, OPTForCausalLM):
        layers = model.model.decoder.layers
    elif isinstance(model, BloomForCausalLM):
        layers = model.transformer.h
    elif "mpt" in str(model.__class__).lower():
        layers = model.transformer.blocks
    elif "falcon" in str(model.__class__).lower():
        layers = model.transformer.h
    elif "bigcode" in str(model.__class__).lower():
        layers = model.transformer.h
    elif "neox" in str(model.__class__).lower():
        layers = model.gpt_neox.layers
    elif model.__class__.__name__ == "LlavaLlamaModel":
        layers = model.llm.model.layers
    elif model.__class__.__name__ in ("CLIPModel"):
        vision_layers = model.vision_model.encoder.layers
        text_layers = model.text_model.encoder.layers
        layers = {'vision': vision_layers,
                  'text': text_layers}
    elif model.__class__.__name__ in ("SiglipModel"):
        vision_layers = model.vision_model.encoder.layers
        text_layers = model.text_model.encoder.layers
        layers = {'vision': vision_layers,
                  'text': text_layers}
    return layers


def get_model_weight_stats(model, args, size):
    
    # if args.diag_scale == True:
    if False:
    # if True:
        with open('/workspace/Weight_compression/Wparam_dataset/block_pt/meta-llama--Llama-2-7b-hf/scaled_sig0.001_row_4096_dataset_stats.json', 'r') as f:
            data_stats = json.load(f)
        mean = torch.tensor(data_stats['train']['mean'])
        std = torch.tensor(data_stats['train']['std'])
    else:
        dataset_stats = {}
        weights = []
        layers = get_blocks(model)
        
        if isinstance(layers, dict):
            layers_ = []
            for k, v in layers.items():
                layers_ += v
            assert len(layers_) == 12 + 24
            layers = layers_
        
        for i in tqdm(range(len(layers)), desc="calculating model weight mean & std"):
            named_linears = get_named_linears(layers[i])
            for n, m in named_linears.items():
                w = m.weight.data.detach()
                if args.direction == 'col':
                    w = w.T    
                w = w.reshape(-1, size)
                weights.append(w)
        
        weights = torch.cat(weights, dim = 0)
        
        # mean = weights.mean(0)
        # std = weights.std(0)
        mean = weights.mean()
        std = weights.std()
            
    return mean, std

def plot_ft_comp_result(ft_result, args, idx, name):                
    # import matplotlib.pyplot as plt
    # import matplotlib
    # matplotlib.use('Agg')  
    # fig, axs = plt.subplots(3, 4, figsize=(16, 12))
    # axs[0, 0].plot(ft_result['step'], ft_result['loss'], label='loss')
    # axs[0, 0].set_title('loss')
    # axs[0, 0].set_xlabel('step')

    # axs[0, 1].plot(ft_result['step'], ft_result['adaptive_loss'], label='adaptive_loss')
    # axs[0, 1].set_title('adaptive_loss')    
    # axs[0, 1].set_xlabel('step')

    # axs[0, 2].plot(ft_result['step'], ft_result['bpp_loss'], label='bpp_loss')
    # axs[0, 2].set_title('bpp_loss')
    # axs[0, 2].set_xlabel('step')

    # axs[0, 3].plot(ft_result['step'], ft_result['mse_loss'], label='mse_loss')
    # axs[0, 3].set_title('mse_loss')
    # axs[0, 3].set_xlabel('step')
    
    # axs[1, 0].plot(ft_result['epoch'], ft_result['loss_per_epoch'], label='loss_epoch')
    # axs[1, 0].set_title('loss_epoch')
    # axs[1, 0].set_xlabel('epoch')
    
    # axs[1, 1].plot(ft_result['epoch'], ft_result['adaptive_loss_per_epoch'], label='adaptive_loss_epoch')
    # axs[1, 1].set_title('adaptive_loss_epoch')
    # axs[1, 1].set_xlabel('epoch')
    
    # axs[1, 2].plot(ft_result['epoch'], ft_result['bpp_loss_per_epoch'], label='bpp_loss_per_epoch')
    # axs[1, 2].set_title('bpp_loss_per_epoch')
    # axs[1, 2].axhline(y=ft_result['base_bpp_loss'], color='r', linestyle='--', label='base_bpp_loss')
    # axs[1, 2].legend()
    # axs[1, 2].set_xlabel('epoch')
    
    # axs[1, 3].plot(ft_result['epoch'], ft_result['mse_loss_per_epoch'], label='mse_loss_per_epoch')
    # axs[1, 3].set_title('mse_loss_per_epoch')
    # axs[1, 3].set_xlabel('epoch')
    
    # axs[2, 0].plot(ft_result['epoch'], ft_result['proxy_err'], label='proxy_err')
    # axs[2, 0].set_title('proxy_err')
    # axs[2, 0].axhline(y=ft_result['base_proxy_err'], color='r', linestyle='--', label='base_proxy_err')
    # axs[2, 0].legend()
    # axs[2, 0].set_xlabel('epoch')
    
    # axs[2, 1].plot(ft_result['epoch'], ft_result['mse'], label='mse')
    # axs[2, 1].set_title('mse')
    # axs[2, 1].axhline(y=ft_result['base_mse'], color='r', linestyle='--', label='base_mse')
    # axs[2, 1].legend()
    # axs[2, 1].set_xlabel('epoch')

    # axs[2, 2].plot(ft_result['epoch'], ft_result['err'], label='err')
    # axs[2, 2].set_title('err')
    # axs[2, 2].axhline(y=ft_result['base_err'], color='r', linestyle='--', label='base_err')
    # axs[2, 2].legend()
    # axs[2, 2].set_xlabel('epoch')
    
    # os.makedirs(args.save_path + '/plots', exist_ok=True)
    os.makedirs(args.save_path + '/jsons', exist_ok=True)
    # plt.savefig(f'{args.save_path}/plots/{idx}_{name}_ft_result.png')
    with open(f'{args.save_path}/jsons/{idx}_{name}_ft_result.json', 'w') as f:
        json.dump(ft_result, f)
        
        
def get_ql_from_H(H, comp_model, args):
    if args.ql == True:
        if args.Q == 4:
            top = np.array([0.1, 1, 10])
            qlevels = [3, 2, 1]
            in_norm = torch.diag(H)
            topk = (top * len(in_norm)/100).astype(int)
            Qlevel = torch.zeros_like(in_norm, dtype=torch.int32)
            _, topk_indices = torch.topk(in_norm, k=topk.sum())
            start = 0
            for count, value in zip(topk , qlevels):
                indices = topk_indices[start:start + count]
                Qlevel[indices] = value
                start += count
        elif args.Q == 2:
            # top = np.array([0.1])
            top = np.array([args.ql_search_r])
            qlevels = [args.ql_search_value] if comp_model.Q == 4 else [1]
            in_norm = torch.diag(H)
            topk = (top * len(in_norm)/100).astype(int)
            Qlevel = torch.zeros_like(in_norm, dtype=torch.int32)
            _, topk_indices = torch.topk(in_norm, k=topk.sum())
            start = 0    
            for count, value in zip(topk , qlevels):
                indices = topk_indices[start:start + count]
                Qlevel[indices] = value
                start += count
        # unique_vals, counts = torch.unique(Qlevel, return_counts=True)
        # print(unique_vals)
        # print(counts)    
        
    if args.ql_invH == True:
        assert comp_model.Q == 4
        Lhr = torch.linalg.cholesky(H)
        H_inv = torch.cholesky_inverse(Lhr)
        top = np.array([0.1, 1, 10])
        qlevels = [3, 2, 1]
        diag = torch.diag(H_inv)
        topk = (top * len(diag)/100).astype(int)
        Qlevel = torch.zeros_like(diag, dtype=torch.int32)
        _, topk_indices = torch.topk(diag, k=topk.sum(), largest=False)
        start = 0    
        for count, value in zip(topk , qlevels):
            indices = topk_indices[start:start + count]
            Qlevel[indices] = value
            start += count
        assert Qlevel == None

    # if args.ql_tuned:
    #     if args.layer_name == 'v':
    #         Qlevel = torch.full_like(Qlevel, 3)
    #     if args.layer_name == 'o':
    #         Qlevel = torch.max(Qlevel, torch.tensor(1))    
    #     if args.layer_idx == 0:
    #         Qlevel = torch.max(Qlevel, torch.tensor(1))

    if args.ql_search:
        ql_search_layer_idx = list(map(int, args.ql_search_layer_idx.split(',')))
        ql_search_layer_name = args.ql_search_layer_name.split(',')
        # assert args.ql
        if args.layer_name in ql_search_layer_name and args.layer_idx in ql_search_layer_idx:
            Qlevel = torch.full_like(Qlevel, args.ql_search_value)    
        Qlevel = torch.full((H.shape[1],), args.ql_search_value, dtype=torch.int32) 
    #  
    return Qlevel

def compute_U_from_H(H: torch.Tensor):
    """
    주어진 대칭 양의 정부호 행렬 H (n×n)에 대해,
    H = (U + I) @ D @ (U + I).T 를 만족하는 U와 D를 계산합니다.
    여기서 U는 순상삼각행렬(strictly upper-triangular)이고 D는 대각행렬(diagonal)입니다.
    이전 PyTorch 버전과의 호환성을 위해 ldl 함수가 없을 경우 Cholesky 분해를 사용합니다.

    Args:
        H (torch.Tensor): n x n 크기의 대칭 양의 정부호 행렬.

    Returns:
        (torch.Tensor, torch.Tensor): 계산된 U와 D.
    """
    n = H.shape[-1]
    device = H.device
    dtype = H.dtype

    # UDU^T 분해를 계산하기 위한 정확한 방법:
    # 1. 반대각 행렬(anti-diagonal matrix) P를 만듭니다. P @ A @ P는 행렬 A의 행과 열 순서를 뒤집습니다.
    P = torch.fliplr(torch.eye(n, device=device, dtype=dtype))
    
    # 2. H의 행과 열 순서를 뒤집은 H_rev를 계산합니다.
    H_rev = P @ H @ P
    
    # H_rev = L_rev @ D_rev @ L_rev.T 를 만족하는 L_rev, D_rev를 찾습니다.
    # 여기서 L_rev는 단위 하삼각행렬(unit lower-triangular)입니다.
    
    try:
        # 2-1. H_rev에 대해 Cholesky 분해를 수행합니다: H_rev = L_chol @ L_chol.T
        L_chol_rev = torch.linalg.cholesky(H_rev)
        
        # 2-2. L_chol_rev의 대각 성분을 추출합니다.
        d_rev_diag = torch.diagonal(L_chol_rev)
        
        # 2-3. 대각행렬 D_rev를 계산합니다.
        D_rev = torch.diag_embed(d_rev_diag * d_rev_diag)
        
        # 2-4. 단위 하삼각행렬 L_rev를 계산합니다.
        #      L_chol_rev의 각 열을 해당 대각 성분으로 나누어줍니다.
        L_rev = L_chol_rev / d_rev_diag
    except torch.linalg.LinAlgError:
        # 행렬이 양의 정부호가 아닐 경우를 대비한 예외 처리
        return None, None

    # 3. 이제 원래 분해로 변환합니다.
    # H = (P @ L_rev @ P) @ (P @ D_rev @ P) @ (P @ L_rev.T @ P)
    # 여기서 U_unit = P @ L_rev @ P 는 단위 상삼각행렬(unit upper-triangular)이 됩니다.
    U_unit = P @ L_rev @ P
    D = P @ D_rev @ P
    
    # 4. 순상삼각행렬 U는 U_unit에서 단위행렬을 빼서 얻습니다.
    U = U_unit - torch.eye(n, device=device, dtype=dtype)
    return U, D


def standardize_W(W, H, args, device):
    Wr = W.to(device)
    Hr = H.to(device)

    SU, SV, scaleWH = None, None, None
    scaleH = None
    layer_std = None
    layer_mean = None
    row_std, col_std = None, None
    scale_cond = None

    # 전처리 및 표준화 단계
    if args.incoh_mode != 'none':
        Lhr, Hr, Wr, SU, SV, scaleWH = quip.incoherence_preprocess(Hr, Wr, args)
    
    if args.scaleH:
        diagH = torch.diag(Hr)
        diagH = torch.clamp(diagH, min=1e-8)
        # scaleH = diagH.sqrt().to(torch.float16)
        scaleH = diagH.sqrt().to(torch.float32)
        # print(scaleH.max(), scaleH.min(), scaleH.mean())
        if args.lb_scaleH is not None:
            scaleH = torch.clamp(scaleH, min=args.lb_scaleH)
        if args.smooth_scaleH_alpha is not None:
            glog.info(f'--{args.layer_idx}_{args.layer_name} smooth_scaleH_alpha: {args.smooth_scaleH_alpha}')
            scaleH = scaleH ** args.smooth_scaleH_alpha


        # print(scaleH.max(), scaleH.min(), scaleH.mean())
        Wr = Wr * scaleH[None, :]
        Hr = Hr / scaleH[None, :]
        Hr = Hr / scaleH[:, None]
    
    if args.scaleHinv:
        assert args.row_normalize == True
        Lhr = torch.linalg.cholesky(Hr)
        H_inv = torch.cholesky_inverse(Lhr)
        diagH_inv = torch.diag(H_inv)
        scaleH = 1 / diagH_inv
        scaleH = torch.clamp(scaleH, min=1e-8)
        scaleH = scaleH.sqrt()
        Wr = Wr * scaleH[None, :]
        Hr = Hr / scaleH[None, :]
        Hr = Hr / scaleH[:, None]

    U = None
    inv_sqrtLam = None
    if args.whiten:
        H_eig = torch.load(f'{args.in_hess_eig_path}/{args.layer_idx}_{args.in_hess_name}_eig.pt')
        U = H_eig['eigenvectors'].to(device)       # [n, k] (k=n이면 full)
        Lam = H_eig['eigenvalues'].to(device).flatten()  # [k]

        eps = 1e-12
        sqrtLam = Lam.clamp_min(eps).pow(0.5)                    # [k]
        inv_sqrtLam = Lam.clamp_min(eps).pow(-0.5)               # 복원 시 필요

        Wr = (Wr @ U) * sqrtLam                                              # [out, k]
        scale_cond = Wr.std(dim = 0, keepdim=True)

    if args.layer_normalize:
        layer_std = Wr.std()
        layer_mean = Wr.mean()
        Wr = (Wr - layer_mean) / layer_std
    
    if args.row_normalize:
        row_std = Wr.std(dim=1, keepdim=True).to(torch.float32)
        Wr /= row_std
    
    if args.row_normalize2:
        diagH = torch.diag(Hr)
        diagH = torch.clamp(diagH, min=1e-8)
        scaleH = diagH.sqrt()
        row_std = (Wr * scaleH[None, :]).std(dim=1, keepdim=True).to(torch.float16)
        # L_chol = torch.linalg.cholesky(Hr)
        # row_std = (Wr @ L_chol).std(dim=1, keepdim=True)
        Wr /= row_std
        
    if args.col_normalize:
        col_std = Wr.std(dim=0, keepdim=True).to(torch.float16)
        Wr /= col_std
        
    if args.scale_cond:
        # assert args.scaleH == True
        # assert args.col_normalize == False
        scale_cond = Wr.std(dim=0, keepdim=True)

        # if comp_model.config.uniform_scale_max is not None:
            # comp_model.config.uniform_scale_max = 1 ## for test
            # glog.info(f'== clamp col_std {comp_model.config.uniform_scale_max} ==')
            # glog.info(f'{col_std.mean()} {col_std.min()} {col_std.max()}')
            # col_std = torch.clamp(col_std, max = comp_model.config.uniform_scale_max)
        # if args.scale_cond_test is not None:
        #     col_std = torch.full_like(col_std, args.scale_cond_test)
        #     glog.info(f'{col_std.mean()} {col_std.min()} {col_std.max()}')

    if args.scale_cond0:
        assert args.scaleH == False
        assert args.col_normalize == False
        diagH = torch.diag(Hr)
        diagH = torch.clamp(diagH, min=1e-8)
        _scaleH = diagH.sqrt()

        col_std = Wr.std(dim=0, keepdim=False)
        # scale_cond = _scaleH / col_std  ## scale_cond0
        scale_cond = _scaleH ## scale_cond2
        # scale_cond = _scaleH * col_std ## scale_cond3
        scale_cond = scale_cond[None, :].to(torch.float32)

    if args.scale_cond_ub is not None and scale_cond is not None:
        scale_cond = torch.clamp(scale_cond, max=args.scale_cond_ub)
        glog.info(f'--{args.layer_idx}_{args.layer_name} scale_cond0: {scale_cond.mean()}, {scale_cond.max()}, {scale_cond.min()}')
            
        # glog.info(f'--{args.layer_idx}_{args.layer_name} col_std: {col_std.mean()}, {col_std.max()}, {col_std.min()}')
        # glog.info(f'--{args.layer_idx}_{args.layer_name} scaleH: {_scaleH.mean()}, {_scaleH.max()}, {_scaleH.min()}')
        # glog.info(f'--{args.layer_idx}_{args.layer_name} scale_cond0: {scale_cond.mean()}, {scale_cond.max()}, {scale_cond.min()}')
        
    # lstats = None
    # if args.layerwise_cdt == True:
    #     Wstats = describe_distribution(W)
    #     stat_keys = ["mean", "median", "std", "range", "iqr", "skewness", "kurtosis"]
    #     lstats = torch.tensor([Wstats[key] for key in stat_keys]).to(device)

    # 메타데이터 패키징
    metadata = {
        'SU': SU, 'SV': SV, 'scaleWH': scaleWH,
        'scaleH': scaleH,
        'layer_std': layer_std, 'layer_mean': layer_mean,
        'row_std': row_std, 'col_std': col_std,
        'scale_cond': scale_cond,
        'U': U,
        'inv_sqrtLam': inv_sqrtLam,
    }
    
    return Wr, Hr, metadata

def de_standardize_Wr(W_hat, metadata, args):

    SU = metadata.get('SU')
    SV = metadata.get('SV')
    scaleWH = metadata.get('scaleWH')
    col_std = metadata.get('col_std')
    row_std = metadata.get('row_std')
    layer_mean = metadata.get('layer_mean')
    layer_std = metadata.get('layer_std')
    scaleH = metadata.get('scaleH')
    
    # 표준화의 역순으로 역연산 수행
    if args.col_normalize and col_std is not None:
        W_hat = W_hat * col_std

    if args.row_normalize2 and row_std is not None:
        W_hat = W_hat * row_std

    if args.row_normalize and row_std is not None:
        # print(W_hat.device, row_std.device)
        W_hat = W_hat * row_std
        
    if args.layer_normalize and layer_std is not None and layer_mean is not None:
        W_hat = W_hat * layer_std + layer_mean
        
    if hasattr(args, 'whiten') and args.whiten:
        inv_sqrtLam = metadata.get('inv_sqrtLam')
        U = metadata.get('U')
        W_hat = W_hat * inv_sqrtLam
        W_hat = W_hat @ U.T        
        
    if args.scaleHinv and scaleH is not None:
        W_hat = W_hat / scaleH[None, :]

    if args.scaleH and scaleH is not None:
        W_hat = W_hat / scaleH[None, :]

    return W_hat
import math

def calculate_metadata_bpp(metadata, Wshape, args):
    """
    metadata에 저장된 텐서들과 Qlevel의 총 비트 수를 계산합니다.
    """
    total_bits = 0

    def _get_tensor_bits(tensor):
        """Helper function to calculate bits for a single tensor."""
        if tensor is None:
            return 0
        return tensor.numel() * tensor.element_size() * 8

    # 1. 기존 metadata 텐서들의 BPP 계산
    if args.col_normalize:
        total_bits += _get_tensor_bits(metadata.get('col_std'))
    
    if args.row_normalize:
        total_bits += _get_tensor_bits(metadata.get('row_std'))

    if metadata.get('scaleH') is not None:
        total_bits += _get_tensor_bits(metadata.get('scaleH'))
        
    # 2. Qlevel에 대한 BPP 계산 추가
    if metadata.get('qlevel') is not None:
        total_bits += Wshape[1] * math.ceil(math.log2(args.Q))
        
    return total_bits

import argparse

def filter_compression_args(source_args):
    """
    기존 args 객체에서 압축 관련 특정 인자들만 추출하여
    새로운 Namespace 객체를 만듭니다.
    """
    keys_to_extract = [
        'ql', 'ldlq', 'direction', 'comp_batch_size', 'ql_invH', 
        'layer_normalize', 'row_normalize', 'row_normalize2', 'col_normalize', 
        'Q', 'use_codes', 'scaleH', 'scaleHinv', 'scale_cond0', 'scale_cond', 
        'fp_iter', 'fp_iter_max', 'fp_tol',
        'incoh_mode',
        'ql_search_value',
        'ql_search_r',
        'layer_idx',
        'layer_name',
        'ft_y',
        'whiten'
    ]
    # 2. 새로운 빈 Namespace 객체 생성
    filtered_args = argparse.Namespace()
    
    # 3. 리스트에 있는 각 키에 대해, 원본 args에서 값을 가져와 새 args에 설정
    for key in keys_to_extract:
        if hasattr(source_args, key):
            value = getattr(source_args, key)
            setattr(filtered_args, key, value)
            
    return filtered_args


def ldl_decomposition(H: torch.Tensor, check_nan=True):
    # 1. 먼저 콜레스키(Cholesky) 분해를 수행합니다: H = L_chol @ L_chol.T
    #    여기서 L_chol은 하삼각행렬이지만, 대각 성분이 1이 아닙니다.
    try:
        L_chol = torch.linalg.cholesky(H)
    except torch.linalg.LinAlgError:
        # 행렬이 양의 정부호가 아니면 분해 실패
        return None

    # 2. L_chol의 대각 성분(d)을 추출합니다.
    d = torch.diagonal(L_chol)

    # 3. 대각행렬 D를 계산합니다. D는 d의 각 원소를 제곱한 값을 대각선에 가집니다.
    D = torch.diag_embed(d * d)

    # 4. 단위 하삼각행렬 L을 계산합니다.
    #    L_chol의 각 열을 해당 대각 성분으로 나누어주면 됩니다.
    #    L = L_chol @ diag(1/d) 와 동일한 연산입니다.
    L = L_chol / d

    # NaN 체크
    if check_nan and L.isnan().any():
        return None

    return L, D.to(L.device)

if __name__ == '__main__':
    # 테스트를 위한 대칭 양의 정부호 행렬 H 생성
    A = torch.randn(4, 4, dtype=torch.float64)
    H = A.T @ A + torch.eye(4, dtype=torch.float64) * 1e-3
    
    print("입력 행렬 H:")
    print(H)
    print("-" * 30)
    print(f"PyTorch 버전: {torch.__version__}")
    if hasattr(torch.linalg, "ldl"):
        print("`torch.linalg.ldl` 사용 가능. ldl 경로로 테스트합니다.")
    else:
        print("`torch.linalg.ldl` 사용 불가. Cholesky 대체 경로로 테스트합니다.")
    print("-" * 30)

    # 함수를 호출하여 U와 D 계산
    U, D = compute_U_from_H(H)

    if U is not None:
        print("\n계산된 순상삼각행렬 U:")
        print(U)

        print("\n계산된 대각행렬 D:")
        print(D)
        print("-" * 30)

        # 결과 검증
        n = H.shape[-1]
        I = torch.eye(n, device=H.device, dtype=H.dtype)
        H_reconstructed = (U + I) @ D @ (U + I).T
        
        print("복원된 행렬 H_reconstructed:")
        print(H_reconstructed)
        
        is_close = torch.allclose(H, H_reconstructed)
        print(f"\n원본 H와 복원된 H가 일치하는가? {is_close}")

        if not is_close:
            diff = torch.max(torch.abs(H - H_reconstructed))
            print(f"최대 오차: {diff.item()}")
    else:
        print("행렬 분해에 실패했습니다.")