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
    parser.add_argument("--lm_model_path", type=str)
    parser.add_argument("--comp_model_path", type=str)
    parser.add_argument("--direction", type=str, default='row')
    parser.add_argument("--batch_size", type=int, default=32768)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--no_save", action='store_true', default = False)
    parser.add_argument("--bundle", action='store_true', default = False)
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

def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}

def pseudo_compress_tensor(w, model, direction, bs):
    if direction == 'col':
        w = w.T
    
    ori_shape = w.shape
    
    if args.bundle:
        w = w.reshape(ori_shape[0], 128, model.input_size)  # (row, col) --> (row, -1, inputsize)
    else :
        w = w.reshape(-1, model.input_size)
        
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

def get_model_stats(model, args):
    dataset_stats = {}
    
    weights = []
    for i in tqdm(range(len(layers)), desc="calculating model weight mean & std"):
        named_linears = get_named_linears(layers[i])
        for n, m in named_linears.items():
            w = m.weight.data.detach()
            if args.direction == 'col':
                w = w.T    
            w = w.reshape(-1, size)
            weights.append(w)
            
    mean = weights.mean(0)
    std = weights.std(0)
    
    return mean, std

if __name__ == "__main__":

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

 # input_mag = torch.load('/home/jgryu/weight_compression/wparam_dataset/calib_data/layer_inputs_channelwise_mag.pt', weights_only=False)    

    net = AutoModelForCausalLM.from_pretrained(args.lm_model_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(args.lm_model_path, local_files_only=True)

    shift, scale = get_model_weight_stats(net, args)    

    import models
    config = os.path.join(os.path.dirname(args.comp_model_path), 'config.json')
    with open(config, 'r', encoding='utf-8') as file:
        config = json.load(file)
    config = Config(**config)

    save_directory = (
        f"{args.save_path}/{args.lm_model_path.split('/')[-1]}/{os.path.join(*args.comp_model_path.split('/')[-3:])}"
    )
    log_file = save_directory.replace('_reconstructed',  '_eval')
    os.makedirs(log_file, mode=0o777 ,exist_ok=True)
    setup_logging(log_file + f'/{(args.direction).upper()}_compression_log.txt')

    # if args.dataset == 'seq':
    #     scale=torch.zeros(128, config.input_size)
    #     shift=torch.zeros(128, config.input_size)
    # else:
    #     scale=torch.zeros(config.input_size)
    #     shift=torch.zeros(config.input_size)
        
    comp_model = get_model(config.architecture, config, scale=scale, shift=shift)      

    ckpt = torch.load(args.comp_model_path)
    
    if 'scale' in ckpt["state_dict"]:
        del ckpt["state_dict"]['scale']
    if 'shift' in ckpt["state_dict"]:
        del ckpt["state_dict"]['shift']
    
    comp_model.load_state_dict(ckpt["state_dict"])
   
    result = pseudo_compress_model(net, comp_model, args.direction, args.batch_size)

    save_directory += f"/{(args.direction).upper()}_MSE{round(result['mse'], 5)}_bpploss{round(result['bpp_loss'], 4)}_bpp{round(result['bpp'], 4)}"

    net = net.to(dtype=torch.bfloat16)
    
    if args.no_save == False:
        print(f'## Strart saving {save_directory}')
        net.save_pretrained(save_directory)
        tokenizer.save_pretrained(save_directory)
        print('## End saving')
        
        
        
        
    # python ../../model_lm_eval/eval_ppl.py \
    #     --hf_path $pretrain_path \
    #     --seqlen 2048 \
    #     --no_use_cuda_graph | tee -a "$log_path"