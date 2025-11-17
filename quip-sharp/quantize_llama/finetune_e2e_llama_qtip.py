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
from accelerate import infer_auto_device_map, init_empty_weights
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_attn_mask_utils import \
    _prepare_4d_causal_attention_mask

from lib import codebook, utils
from lib.algo import finetune
from lib.linear.fused_quantized_linear import FusedQuantizedLinear
from lib.linear.quantized_linear import QuantizedLinear
# from lib.utils.unsafe_import import model_from_hf_path
import transformers, accelerate
from model.llama import LlamaForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_cpu_threads', default=8, type=int)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--devset_size', default=64, type=int)
parser.add_argument('--ctx_size', default=4096, type=int)
parser.add_argument('--sample_proc', default=1, type=int)
parser.add_argument('--base_model', type=str)
parser.add_argument('--hf_path', type=str)
parser.add_argument('--hf_output_path', type=str) ## ryu
parser.add_argument('--ft_lr', default=1e-5, type=float)
parser.add_argument('--ft_bs', default=8, type=int)
parser.add_argument('--ft_update_freq', default=1, type=int)
parser.add_argument('--ft_epochs', default=1, type=int)
parser.add_argument('--ft_valid_freq', default=1, type=int)
parser.add_argument('--ft_valid_size', default=128, type=float)
parser.add_argument('--ft_early_stop', default=3, type=int)
parser.add_argument('--ft_train_mode', action='store_true')
parser.add_argument('--ft_grad_ckpt', action='store_true')
parser.add_argument('--ft_nshards', default=-1, type=int)
parser.add_argument('--resume_ckpt', type=str)
parser.add_argument('--ckpt_path', type=str)



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


def model_from_hf_path(path, max_mem_ratio=0.7, device_map=None):

    # AutoConfig fails to read name_or_path correctly
    bad_config = transformers.AutoConfig.from_pretrained(path)
    is_quantized = hasattr(bad_config, 'quip_params')
    model_type = bad_config.model_type
    if is_quantized:
        if model_type == 'llama':
            model_str = transformers.LlamaConfig.from_pretrained(
                path)._name_or_path
            model_cls = LlamaForCausalLM
        else:
            raise Exception
    else:
        model_cls = transformers.AutoModelForCausalLM
        model_str = path

    if device_map is None:
        mmap = {
            i: f"{torch.cuda.mem_get_info(i)[1]*max_mem_ratio/(1 << 30)}GiB"
            for i in range(torch.cuda.device_count())
        }
        model = model_cls.from_pretrained(path,
                                          torch_dtype='auto',
                                          low_cpu_mem_usage=True,
                                          attn_implementation='sdpa')
        device_map = accelerate.infer_auto_device_map(
            model,
            no_split_module_classes=['LlamaDecoderLayer'],
            max_memory=mmap)
    model = model_cls.from_pretrained(path,
                                      torch_dtype='auto',
                                      low_cpu_mem_usage=True,
                                      attn_implementation='sdpa',
                                      device_map=device_map)

    return model, model_str


def llama_arg_fn(output, args, kwargs):
    return (output[0], *args[1:]), kwargs


def get_emb(args, kwargs):
    return args[0]


def main(args):
    torch.set_grad_enabled(False)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    devset = utils.sample_rp1t(tokenizer, args.devset_size, args.ctx_size,
                               args.sample_proc)

    with init_empty_weights():
        orig_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype='auto',
            device_map='sequential',
            low_cpu_mem_usage=True)

    start_dev = max(orig_model.hf_device_map.values()) + 1
    end_dev = torch.cuda.device_count()
    fake_dev_map = {
        'model.embed_tokens': start_dev,
        'model.rotary_emb': start_dev,
        'model.norm': end_dev - 1,
        'lm_head': end_dev - 1
    }
    per_dev = math.ceil(
        (len(orig_model.model.layers) + 4) / (end_dev - start_dev))
    for i in range(len(orig_model.model.layers)):
        fake_dev_map[f'model.layers.{i}'] = (i + 2) // per_dev + start_dev

    orig_dtype = orig_model.model.embed_tokens.weight.dtype
    print(orig_dtype)
    print(fake_dev_map)
    del orig_model  # remanifest in eval process
    utils.clean()

    quant_model = model_from_hf_path(args.hf_path,
                                     device_map=fake_dev_map)[0].float()

    for name, module in quant_model.named_modules():
            # module.SU = nn.Parameter(module.SU.float(), requires_grad=True)
            # module.SV = nn.Parameter(module.SV.float(), requires_grad=True)
            module.grad_ckpt = args.ft_grad_ckpt
            module.train_mode = args.ft_train_mode

                
    utils.clean()
    with torch.enable_grad():
        finetune.finetune_susv_e2e_qtip(quant_model, start_dev, devset, orig_dtype,
                                   args)
    try:
        quant_model = quant_model.to(orig_dtype)
        quant_model.config._name_or_path = args.base_model
    except Exception as e:
        print(f"Error when converting quant_model to orig_dtype: {e}")
        pass
    quant_model.save_pretrained(args.hf_output_path, safe_serialization=True)
    
    
    try:
        save_fn(quant_model, args)
    except:
        new_best_sd = {}
        best_sd = quant_model.state_dict()
        for key, value in best_sd.items():
            if key.endswith(".Qidxs_0"):
                new_key = key.replace(".Qidxs_0", ".Qidxs")
                new_best_sd[new_key] = value
            else:
                new_best_sd[key] = value
        quant_model.load_state_dict(new_best_sd, strict=False)
        save_fn(quant_model, args)
        
if __name__ == '__main__':
    mp.set_start_method('spawn')
    mp.set_sharing_strategy('file_system')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    main(args)