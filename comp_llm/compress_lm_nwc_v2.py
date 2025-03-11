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
from matmul_had import *

notebook_dir = os.path.dirname(os.path.abspath("__file__"))
project_root = os.path.abspath(os.path.join(notebook_dir, ".."))

std = 0.012528747320175171
if project_root not in sys.path:
    sys.path.append(project_root)
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from NWC.models import get_model
import utils

def parse_args():
    parser = argparse.ArgumentParser(description="Reconstruct model using specified configuration.")
    parser.add_argument("--lm_model_path", type=str)
    parser.add_argument("--comp_model_path", type=str)
    parser.add_argument("--direction", type=str, default='col')
    parser.add_argument("--batch_size", type=int, default=32768)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--no_save", action='store_true', default = False)
    parser.add_argument("--bundle", action='store_true', default = True)
    parser.add_argument("--ql", type=str, default = None)
    parser.add_argument("--hesseigen", type=str, default = None)
    parser.add_argument("--gptq", action='store_true', default = False)
    parser.add_argument("--ldlq", action='store_true', default = False)
    parser.add_argument("--hess", type=str, default = None)
    parser.add_argument("--quip_hess", type=str, default = None)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--hf_path', default='hfized/quantized_hada_70b', type=str)
    parser.add_argument('--seqlen', default=2048, type=int)
    parser.add_argument('--no_use_cuda_graph', action='store_true')
    parser.add_argument('--no_use_flash_attn', action='store_true')
    parser.add_argument('--sigma_reg', default=1e-3, type=float)
    parser.add_argument('--hess_proj', action='store_true')
    parser.add_argument('--hess_proj_dim', default=128, type=int)
    parser.add_argument('--diag_scale', action='store_true')
    parser.add_argument('--quip_tune_iters', default=0, type=int)
    
    return parser.parse_args()

args = parse_args()

class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def compress_weight_block_with_model(weight_block, model, ql=None):
    
    ori_shape = weight_block.shape
    if ori_shape[-1] != model.input_size:
        weight_block = weight_block.reshape(ori_shape[0], -1, model.input_size)
        
    data = {}
    data['weight_blcok'] = weight_block.to(model.device)
    if ql is not None:
        # assert ql.shape[0] == ori_shape
        data['q_level'] = ql.reshape(ori_shape[0], 1).to(model.device)
        
    out = model(data)
    w_hat = out['x_hat'].reshape(ori_shape[0], -1)
    
    num_pixels = weight_block.numel()
    if isinstance(out["likelihoods"], dict):
        bpp_loss = sum(
            (torch.log(likelihoods).sum() / -math.log(2))
            for likelihoods in out["likelihoods"].values()
        ).item()
    else :
        bpp_loss = (torch.log(out["likelihoods"]).sum() / -math.log(2)).item()

        # out_enc = model.compress(data)
    # try:
    #     out_dec = model.decompress(out_enc["strings"][0], out_enc["shape"], data["q_level"])
    # except:
    #     out_dec = model.decompress(out_enc["strings"][0], out_enc["shape"])
    
    # for s in out_enc["strings"]:
    #     bpp += len(s[0]) * 8.0

    return w_hat, num_pixels, bpp_loss
    
def pseudo_compress_tensor(w, model, direction, bs, q_level, hess_eigen):
    if direction == 'col':
        w = w.T
    ori_shape = w.shape
    
    if args.bundle:
        w = w.reshape(ori_shape[0], -1, model.input_size)  # (row, col) --> (row, -1, inputsize)
    else :
        w = w.reshape(-1, model.input_size)
        
    w_hat = torch.zeros(w.shape, dtype=w.dtype, device=w.device)

    num_pixels = 0
    bpp_loss = 0
    bpp = 0

    if q_level is not None:
        q_level = q_level.reshape(-1, )

    if hess_eigen is not None:
        R = hess_eigen['eigenvectors'].size(0)
        hess_eigen = hess_eigen['eigenvectors'].to(device)
        hess_eigen = hess_eigen.reshape(1, R, -1, model.input_size).repeat(bs, 1, 1, 1)

    for start_idx in range(0, w.shape[0], bs):
        end_idx = min(start_idx + bs, w.shape[0])
        batch = w[start_idx:end_idx]
        
        ql = None
        if q_level is not None:
            ql = q_level[start_idx:end_idx]

        x_hat, n_pixels, bpp_loss_ = compress_weight_block_with_model(batch, model, ql)
        
        w_hat[start_idx:end_idx] = x_hat
        num_pixels += n_pixels
        bpp_loss += bpp_loss_

    w_hat = w_hat.reshape(ori_shape).cpu()

    if direction == 'col':
        w_hat = w_hat.T
    
    return {'w_hat': w_hat,
            'bpp_loss_sum': bpp_loss,
            'num_pixels': num_pixels,
            'bpp': bpp}

def pseudo_compress_tensor_gptq(W, model, direction, bs, q_level, H):
    
    assert direction == 'col'
    if q_level is not None:
        q_level = q_level.reshape(-1, )
    
    W = W.to(device)
    Losses = torch.zeros_like(W, device=W.device)
    Q = torch.zeros_like(W, device=W.device)
    assert torch.isfinite(H).all()
    
    H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    H = torch.linalg.cholesky(H, upper=True)
    Hinv = H
    assert torch.isfinite(H).all()
    
    rows = W.shape[0]
    columns = W.shape[1]
    
    num_pixels = 0
    bpp_loss = 0
    bpp = 0

    for i1 in range(0, columns, bs):
        i2 = min(i1 + bs, columns)
        count = i2 - i1
        
        ql = None
        if q_level is not None:
            ql = q_level[i1:i2]

        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1, device=W1.device)
        Err1 = torch.zeros_like(W1, device=W1.device)
        Losses1 = torch.zeros_like(W1, device=W1.device)
        Hinv1 = Hinv[i1:i2, i1:i2]

        for i in range(count):
            w = W1[:, i]
            d = Hinv1[i, i]
            
            assert w.size(-1) == rows
            if args.bundle:
                w_reshape = w.reshape(1, -1, model.input_size)  # (row, col) --> (row, -1, inputsize)
            else :
                w_reshape = w.reshape(-1, model.input_size)

            ql_ = None if ql is None else ql[i]
            x_hat, n_pixels, bpp_loss_ = compress_weight_block_with_model(w_reshape, model, ql)
            
            q = x_hat.flatten()
            num_pixels += n_pixels
            bpp_loss += bpp_loss_

            Q1[:, i] = q
            Losses1[:, i] = (w - q)**2 / d**2

            err1 = (w - q) / d
            assert torch.isfinite(err1).all()
            W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
            Err1[:, i] = err1

        Q[:, i1:i2] = Q1
        Losses[:, i1:i2] = Losses1 / 2

        W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

    return {'w_hat': Q.cpu(),
            'bpp_loss_sum': bpp_loss,
            'num_pixels': num_pixels,
            'bpp': bpp}

def pseudo_compress_tensor_ldlq(Wr, model, direction, bs, q_level, H):
    assert direction == 'col'
    
    L, D = block_LDL(H, 128)
    
    '''
    want eta = (Wr - hatWr) @ L
    want hatWr + eta = Wr + (Wr - hatWr) @ (L - I)
    want hatWr = Q( Wr + (Wr - hatWr) @ (L - I) )
    '''
    (m, n) = Wr.shape
    hatWr = torch.zeros_like(Wr, dtype=Wr.dtype, device=Wr.device)

    num_pixels = 0
    bpp_loss = 0
    bpp = 0

    for k in reversed(range(n)):
        WXWX = Wr[:, k:k+1] + (Wr[:, k+1:] - hatWr[:, k+1:]) @ L[k+1:, k:k+1]
    
        ql_ = None if q_level is None else q_level[i]
        x_hat, n_pixels, bpp_loss_ = compress_weight_block_with_model(WXWX.reshape(1, -1, model.input_size), model, ql)
        
        q = x_hat.flatten()
        hatWr[:, k] = q
        num_pixels += n_pixels
        bpp_loss += bpp_loss_

    for ie in range(args.quip_tune_iters):
        assert Error
        num_pixels = 0
        bpp_loss = 0
        bpp = 0
        
        for k in reversed(range(n)):
            WXWX = hatWr[:, k:k+1] + (Wr - hatWr) @ H[:, k:k+1] @ torch.linalg.inv(H[k:k+1, k:k+1])
            
            ql_ = None if q_level is None else q_level[i]
            x_hat, n_pixels, bpp_loss_ = compress_weight_block_with_model(WXWX.reshape(1, -1, model.input_size), model, ql)
            
            q = x_hat.flatten()
            hatWr[:, k] = q
            
            if ie == args.quip_tune_iters - 1:
                num_pixels += n_pixels
                bpp_loss += bpp_loss_

    return {'w_hat': hatWr.cpu(),
            'bpp_loss_sum': bpp_loss,
            'num_pixels': num_pixels,
            'bpp': bpp}


def block_LDL(H, b, check_nan=True):
    n = H.shape[0]
    assert (n % b == 0)
    m = n // b
    try:
        L = torch.linalg.cholesky(H)
    except:
        return None
    DL = torch.diagonal(L.reshape(m, b, m, b), dim1=0, dim2=2).permute(2, 0, 1)
    D = (DL @ DL.permute(0, 2, 1)).cpu()
    DL = torch.linalg.inv(DL)
    L = L.view(n, m, b)
    for i in range(m):
        L[:, i, :] = L[:, i, :] @ DL[i, :, :]

    if check_nan and L.isnan().any():
        return None

    L = L.reshape(n, n)
    return (L, D.to(DL.device))

def pseudo_compress_model(model, comp_model, direction, bs, args):
    
    mse_total = 0
    n_total = 0
    total_bpp_loss = 0
    total_num_pixels = 0
    bpp = 0
    
    comp_model.to(device)
    comp_model.update()
    
    in_ch_rank = None
    if args.ql is not None:
        assert args.direction == 'col'
        in_ch_rank = torch.load(args.ql, weights_only=False)
    
    hesseigen = None
    if args.hesseigen is not None:
        assert args.direction == 'row'
        hesseigen = torch.load(args.hesseigen, weights_only=False)

    hess = None
    if args.hess is not None:
        assert args.direction == 'row'
        hess = torch.load(args.hess, weights_only=False)
    
    with torch.no_grad():
        # layers = model.model.layers
        layers = get_blocks(model)
        for i in tqdm(range(len(layers)), desc="pseudo compress quantization..."):
            named_linears = get_named_linears(layers[i])
            
            quip_hess = None
            if args.quip_hess is not None:
                quip_hess = {}
                quip_hess['qkv'] = torch.load(f'{args.quip_hess}/{i}_qkv.pt', weights_only=False)
                quip_hess['o'] = torch.load(f'{args.quip_hess}/{i}_o.pt', weights_only=False)
                quip_hess['up'] = torch.load(f'{args.quip_hess}/{i}_up.pt', weights_only=False)
                quip_hess['down'] = torch.load(f'{args.quip_hess}/{i}_down.pt', weights_only=False)
            
            for n, m in named_linears.items():
                in_ch_rank_weight = None
                if in_ch_rank is not None:
                    in_ch_rank_weight = in_ch_rank[i][n]
                
                hesseigen_weight = None
                if hesseigen is not None:
                    hesseigen_weight = hesseigen[i][n]
                 
                if hess is not None:
                    H = hess[i][n].to(device)
                    n_h = H.shape[0]
                    H = regularize_H(H, n_h, args.sigma_reg)
                    
                H_flat = None
                if quip_hess is not None:
                    if 'q_proj' in n or 'k_proj' in n or 'v_proj' in n:
                        H_flat = quip_hess['qkv']
                    elif 'o_proj' in n:
                        H_flat = quip_hess['o']
                    elif 'up_proj' in n or 'gate' in n:
                        H_flat = quip_hess['up']
                    elif 'down_proj' in n:
                        H_flat = quip_hess['down']
                    
                    H = flat_to_sym(H_flat['flatH'], H_flat['n']).to(device)
                    mu = H_flat['mu'].to(device)
                    H.add_(mu[None, :] * mu[:, None])
                    n_h = H_flat['n']
                    H = regularize_H(H, n_h, args.sigma_reg)
                
                
                W = m.weight.data.detach().to(device)
                if args.diag_scale:
                    Wr = W
                    H = H.to(device)
                    Hr = H / H.abs().max()
                    diagH = torch.diag(Hr)
                    scaleWH = diagH.sqrt().to(torch.float32)
                    scaleWH = scaleWH.clamp(min=1e-8)
                    Wr = Wr * scaleWH[None, :]
                    Hr = Hr / scaleWH[None, :]
                    Hr = Hr / scaleWH[:, None]
                    W = Wr
                    H = Hr
                
                if args.gptq:
                    out = pseudo_compress_tensor_gptq(W, comp_model, direction, bs, in_ch_rank_weight, H)                
                elif args.ldlq:
                    out = pseudo_compress_tensor_ldlq(W, comp_model, direction, bs, in_ch_rank_weight, H)
                else:
                    out = pseudo_compress_tensor(W, comp_model, direction, bs, in_ch_rank_weight, hesseigen_weight)         
                    
                
                W_hat = out['w_hat']
                if args.diag_scale:
                    W_hat = W_hat.to(device)
                    W_hat = W_hat / scaleWH[None, :]
                    W_hat = W_hat.to('cpu')
                
                mse = ((m.weight.data - W_hat)**2).sum().item()
                
                m.weight.data = W_hat
                
                num = m.weight.data.numel()
                mse_total += mse
                n_total += num
                
                total_bpp_loss += out['bpp_loss_sum']
                total_num_pixels += out['num_pixels']
                # bpp += out['bpp']

                # print(f"layer{i}_{n} | mse: {mse/num/std**2}, bpp_loss: {out['bpp_loss']}, bpp: {out['bpp']}")
                logging.info(f"layer{i}_{n} | mse: {mse/num/std**2}, bpp_loss: {out['bpp_loss_sum']/out['num_pixels']}, bpp: {out['bpp']}")
                
        mse_total = mse_total / n_total / std **2 
        total_bpp_loss /= total_num_pixels
        # bpp /= total_num_pixels
    
    # print(f'#### Total | mse: {mse_total}, bpp_loss: {bpp_loss}, bpp: {bpp} ####')
    logging.info(f'#### Total | mse: {mse_total}, bpp_loss: {total_bpp_loss}, bpp: {bpp} ####')
    
    return {'mse': mse_total,
            'bpp_loss': total_bpp_loss,
            'bpp': bpp}

def get_model_weight_stats(model, args, size):
    
    if args.diag_scale == True:
        with open('/home/jgryu/Weight_compression/Wparam_dataset/block_pt/meta-llama--Llama-2-7b-hf/scaled_sig0.001_row_4096_dataset_stats.json', 'r') as f:
            data_stats = json.load(f)
        mean = torch.tensor(data_stats['train']['mean'])
        std = torch.tensor(data_stats['train']['std'])
    else:
        dataset_stats = {}
        weights = []
        layers = get_blocks(model)
        for i in tqdm(range(len(layers)), desc="calculating model weight mean & std"):
            named_linears = get_named_linears(layers[i])
            for n, m in named_linears.items():
                w = m.weight.data.detach()
                if args.direction == 'col':
                    w = w.T    
                w = w.reshape(-1, size)
                weights.append(w)
        
        weights = torch.cat(weights, dim = 0)
        
        mean = weights.mean(0)
        std = weights.std(0)
        # mean = weights.mean()
        # std = weights.std()
        
    
    return mean, std

if __name__ == "__main__":

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    
    net = AutoModelForCausalLM.from_pretrained(args.lm_model_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(args.lm_model_path, local_files_only=True)

    config = os.path.join(os.path.dirname(args.comp_model_path), 'config.json')
    with open(config, 'r', encoding='utf-8') as file:
        config = json.load(file)
    config = Config(**config)
    
    shift, scale = get_model_weight_stats(net, args, config.input_size)    
    
    save_directory = (
        f"{args.save_path}/{args.lm_model_path.split('/')[-1]}/{os.path.join(*args.comp_model_path.split('/')[-3:])}"
    )
    log_file = save_directory.replace('_reconstructed',  '_eval')
    os.makedirs(log_file, mode=0o777 ,exist_ok=True)
    setup_logging(log_file + f'/{(args.direction).upper()}_compression_log.txt')

    if config.architecture == 'nwc_ql' and not hasattr(config, "Q"):
        config.Q = 4
        
    comp_model = get_model(config.architecture, config, scale=scale.to(device), shift=shift.to(device))      
    ckpt = torch.load(args.comp_model_path, weights_only=False)
    
    if 'scale' in ckpt["state_dict"]:
        del ckpt["state_dict"]['scale']
    if 'shift' in ckpt["state_dict"]:
        del ckpt["state_dict"]['shift']
    
    comp_model.load_state_dict(ckpt["state_dict"])
   
    result = pseudo_compress_model(net, comp_model, args.direction, args.batch_size, args)

    save_directory += f"/{(args.direction).upper()}_MSE{round(result['mse'], 5)}_bpploss{round(result['bpp_loss'], 4)}_bpp{round(result['bpp'], 4)}"

    net = net.to(dtype=torch.bfloat16)
    
    if args.no_save == False:
        print(f'## Strart saving {save_directory}')
        net.save_pretrained(save_directory)
        tokenizer.save_pretrained(save_directory)
        print('## End saving')
        
    # from eval_ppl import main
    # import random
    # args.hf_path = save_directory
    # args.no_use_cuda_graph = True
    
    # torch.set_grad_enabled(False)
    # random.seed(args.seed)
    # torch.random.manual_seed(args.seed)
    # main(args)

    
    # python ../../model_lm_eval/eval_ppl.py \
    #     --hf_path $pretrain_path \
    #     --seqlen 2048 \
    #     --no_use_cuda_graph | tee -a "$log_path"