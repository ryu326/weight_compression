import argparse
import copy
import datetime
import gc
import math
import os
import time

from tqdm import tqdm

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

import glog
import torch
import torch.multiprocessing as mp
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_attn_mask_utils import \
    _prepare_4d_causal_attention_mask

from lib import codebook, utils
from lib.algo import finetune, quip
from lib.utils.unsafe_import import model_from_hf_path

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_cpu_threads', default=8, type=int)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--devset_size', default=64, type=int)
parser.add_argument('--ctx_size', default=4096, type=int)
parser.add_argument('--sample_proc', default=1, type=int)
parser.add_argument('--base_model', type=str)
parser.add_argument('--hf_path', type=str)
parser.add_argument('--ckpt_path', type=str)
parser.add_argument('--output_ckpt_path', type=str)


def save_fn(quant_model, args):
    ct = 0
    for j in range(len(quant_model.model.layers)):
        layer = quant_model.model.layers[j]
        utils.save_susv(layer.self_attn.qkv_proj,
                        f'{args.ckpt_path}/{ct}_qkv.pt')
        utils.save_susv(layer.self_attn.o_proj,
                        f'{args.ckpt_path}/{ct}_o.pt')
        utils.save_susv(layer.mlp.upgate_proj,
                        f'{args.ckpt_path}/{ct}_up.pt')
        utils.save_susv(layer.mlp.down_proj,
                        f'{args.ckpt_path}/{ct}_down.pt')
        torch.save(
            {
                'input_layernorm':
                layer.input_layernorm.weight,
                'post_attention_layernorm':
                layer.post_attention_layernorm.weight,
            }, f'{args.ckpt_path}/{ct}_layernorm.pt')
        glog.info(f'wrote layer {ct}')
        ct += 1
    torch.save(
        {
            'lm_head': quant_model.lm_head.weight,
            'norm': quant_model.model.norm.weight,
        }, f'{args.ckpt_path}/lmhead.pt')


def llama_arg_fn(output, args, kwargs):
    return (output[0], *args[1:]), kwargs


def get_emb(args, kwargs):
    return args[0]


def main(args):
    torch.set_grad_enabled(False)

    quant_model = model_from_hf_path(args.hf_path,
                                    use_cuda_graph=False,
                                    use_flash_attn=False,
                                    device_map=None)[0].cpu()
    ckpt = torch.load(args.output_ckpt_path, map_location='cpu')
    best_sd = ckpt['model_state_dict']
    
    # --- [시작] 요청하신 확인 코드 ---
    print("\n" + "="*60)
    print("Verifying Qidxs keys in Model (quant_model) vs. Checkpoint (best_sd)")
    print("="*60)
    
    model_keys = set(quant_model.state_dict().keys())
    ckpt_keys = set(best_sd.keys())
    
    # 모델의 레이어 수를 가져옵니다.
    num_layers = len(quant_model.model.layers)
    modules_to_check = [
        "self_attn.qkv_proj", 
        "self_attn.o_proj", 
        "mlp.upgate_proj", 
        "mlp.down_proj"
    ]

    model_sd = quant_model.state_dict() # 모델의 state_dict를 한 번만 가져옵니다.
    num_layers = len(quant_model.model.layers)
    modules_to_check = [
        "self_attn.qkv_proj", 
        "self_attn.o_proj", 
        "mlp.upgate_proj", 
        "mlp.down_proj"
    ]

    all_tensors_equal = True
    for i in range(num_layers):
        # print(f"\n--- Checking Layer {i} ---")
        for mod_name in modules_to_check:
            model_key = f"model.layers.{i}.{mod_name}.Qidxs"
            ckpt_key = f"model.layers.{i}.{mod_name}.Qidxs_0"           
            model_tensor = model_sd[model_key]
            ckpt_tensor = best_sd[ckpt_key]            
            assert torch.equal(model_tensor, ckpt_tensor)

    # --- [종료] 요청하신 확인 코드 ---
    

    new_best_sd = {}
    for key, value in best_sd.items():
        if key.endswith(".Qidxs_0"):
            new_key = key.replace(".Qidxs_0", ".Qidxs")
            new_best_sd[new_key] = value
        else:
            new_best_sd[key] = value
    
    import ipdb; ipdb.set_trace()
    # quant_model.load_state_dict(new_best_sd)
    quant_model.load_state_dict(new_best_sd, strict=False)
    save_fn(quant_model, args)
    

if __name__ == '__main__':
    mp.set_start_method('spawn')
    mp.set_sharing_strategy('file_system')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    main(args)
