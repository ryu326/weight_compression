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
)
from torch.utils.data import DataLoader
import logging

notebook_dir = os.path.dirname(os.path.abspath("__file__"))
project_root = os.path.abspath(os.path.join(notebook_dir, ".."))

std = 0.012528747320175171

if project_root not in sys.path:
    sys.path.append(project_root)

import models
from models import get_model

def parse_args():
    parser = argparse.ArgumentParser(description="Reconstruct model using specified configuration.")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--direction", type=str, default='row')
    parser.add_argument("--batch_size", type=int, default=32768)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--no_save", action='store_true', default = False)
    return parser.parse_args()

args = parse_args()

class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)

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

def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}

def pseudo_compress_tensor(w, model, direction, bs):
    if direction == 'col':
        w = w.T
    
    ori_shape = w.shape
    
    if args.dataset == 'seq':
        w = w.reshape(-1, 128, model.input_size)  # ( -1, -1) --> (-1, size, size)
    else :
        w = w.reshape(-1, model.input_size)  # ( -1, -1) --> (-1, size, size)
    w_hat = torch.zeros(w.shape, dtype=w.dtype, device=w.device)

    num_pixels = 0
    bpp_loss = 0
    bpp = 0

    for start_idx in range(0, w.shape[0], bs):
        end_idx = min(start_idx + bs, w.shape[0])
        batch = w[start_idx:end_idx]

        data = {}
        data['weight_block'] = batch.cuda()
        out = model(data)
        x_hat = out["x_hat"].cpu()

        w_hat[start_idx:end_idx] = x_hat
        
        num_pixels += batch.numel()
        if isinstance(out["likelihoods"], dict):
            bpp_loss += sum(
                (torch.log(likelihoods).sum() / -math.log(2))
                for likelihoods in out["likelihoods"].values()
            ).item()
        else :
            bpp_loss += (torch.log(out["likelihoods"]).sum() / -math.log(2)).item()

        # out_enc = model.compress(data)
        # try:
        #     out_dec = model.decompress(out_enc["strings"][0], out_enc["shape"], data["q_level"])
        # except:
        #     out_dec = model.decompress(out_enc["strings"][0], out_enc["shape"])
        
        # for s in out_enc["strings"]:
        #     bpp += len(s[0]) * 8.0

    w_hat = w_hat.reshape(ori_shape).cpu()

    if direction == 'col':
        w_hat = w_hat.T
        
    # bpp_loss /= num_pixels
    # bpp /= num_pixels
    
    return {'w_hat': w_hat,
            'bpp_loss_sum': bpp_loss,
            'num_pixels': num_pixels,
            'bpp': bpp}

def pseudo_compress_model(model, comp_model, direction, bs):
    
    mse_total = 0
    n_total = 0
    total_bpp_loss = 0
    total_num_pixels = 0
    bpp = 0
    
    comp_model.to(device)
    comp_model.update()
    
    with torch.no_grad():
        layers = model.model.layers
        for i in tqdm(range(len(layers)), desc="pseudo compress quantization..."):
            named_linears = get_named_linears(layers[i])
            for n, m in named_linears.items():
                out = pseudo_compress_tensor(m.weight.data, comp_model, direction, bs)                
                mse = ((m.weight.data - out['w_hat'])**2).sum().item()
                
                m.weight.data = out['w_hat']
                
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

if __name__ == "__main__":

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    import models
    config = os.path.join(os.path.dirname(args.model_path), 'config.json')
    with open(config, 'r', encoding='utf-8') as file:
        config = json.load(file)
    config = Config(**config)

    log_file = (
        f"/home/jgryu/Weight_compression/model_eval/{args.save_path}/{os.path.join(*args.model_path.split('/')[-3:])}"
    )
    os.makedirs(log_file, mode=0o777 ,exist_ok=True)
    setup_logging(log_file + f'/{(args.direction).upper()}_compression_log.txt')

    if args.dataset == 'seq':
        scale=torch.zeros(128, config.input_size)
        shift=torch.zeros(128, config.input_size)
    else:
        scale=torch.zeros(config.input_size)
        shift=torch.zeros(config.input_size)
        
    comp_model = get_model(config.architecture, config, scale=scale, shift=shift)      

    # if args.dataset == 'seq':
    #     comp_model = models.SimpleVAECompressionModel(
    #     input_size=config['input_size'],
    #     dim_encoder=config['dim_encoder'],
    #     n_resblock=config['n_resblock'],
    #     M = config['input_size'],
    #     scale=torch.zeros(128, config['input_size']),
    #     shift=torch.zeros(128, config['input_size'])
    # )
    # else :    
    #     comp_model = models.SimpleVAECompressionModel(
    #         input_size=config['input_size'],
    #         dim_encoder=config['dim_encoder'],
    #         n_resblock=config['n_resblock'],
    #         M = config['input_size'],
    #         scale=torch.zeros(config['input_size']),
    #         shift=torch.zeros(config['input_size'])
    #     )

    ckpt = torch.load(args.model_path)
    comp_model.load_state_dict(ckpt["state_dict"])
    
    # input_mag = torch.load('/home/jgryu/weight_compression/wparam_dataset/calib_data/layer_inputs_channelwise_mag.pt', weights_only=False)    

    cache_directory = "/home/jgryu/Weight_compression/Wparam_dataset_v0/model_zoo/huggingface"
    ckpt_path = latest_version_path(cache_directory, "meta-llama/Meta-Llama-3-8B")
    net = AutoModelForCausalLM.from_pretrained(ckpt_path, local_files_only=True)
    ckpt_path = "/home/jgryu/Weight_compression/model_cache/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920"
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, local_files_only=True)

    result = pseudo_compress_model(net, comp_model, args.direction, args.batch_size)

    save_directory = (
        f"/home/jgryu/Weight_compression/model_reconstructed/{args.save_path}/{os.path.join(*args.model_path.split('/')[-3:])}"
    )
    save_directory += f"/{(args.direction).upper()}_MSE{round(result['mse'], 5)}_bpploss{round(result['bpp_loss'], 4)}_bpp{round(result['bpp'], 4)}"

    net = net.to(dtype=torch.bfloat16)
    
    if args.no_save == False:
        print(f'## Strart saving {save_directory}')
        net.save_pretrained(save_directory)
        tokenizer.save_pretrained(save_directory)
        print('## End saving')