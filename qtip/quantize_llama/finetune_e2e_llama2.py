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
from lib.linear import QuantizedLinear
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
parser.add_argument('--hf_output_path', type=str)
parser.add_argument('--ft_lr', default=1e-5, type=float)
parser.add_argument('--ft_bs', default=2, type=int)
parser.add_argument('--ft_update_freq', default=2, type=int)
parser.add_argument('--ft_epochs', default=1, type=int)
parser.add_argument('--ft_valid_freq', default=1, type=int)
parser.add_argument('--ft_valid_size', default=128, type=float)
parser.add_argument('--ft_early_stop', default=3, type=int)
parser.add_argument('--ft_train_lut', action='store_true')
parser.add_argument('--ft_prefetch_trellis', action='store_true')
parser.add_argument('--ft_grad_ckpt', action='store_true')


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

    orig_model = AutoModelForCausalLM.from_pretrained(args.base_model,
                                                      torch_dtype='auto',
                                                      device_map='auto',
                                                      low_cpu_mem_usage=True)
    orig_logits = utils.calculate_logits(orig_model, devset, args.batch_size)
    orig_logits = orig_logits[:, :-1].contiguous().softmax(dim=-1).float()

    del orig_model
    utils.clean()

    quant_model = model_from_hf_path(args.hf_path,
                                     device_map=None)[0].float().cpu()
    emb = quant_model.model.embed_tokens(devset)
    position_ids = torch.arange(args.ctx_size, dtype=torch.int32)[None, :] + \
        torch.zeros(args.ft_bs, args.ctx_size, dtype=torch.int32)
    attention_mask = _prepare_4d_causal_attention_mask(
        None, (args.ft_bs, args.ctx_size), emb[:args.ft_bs], 0)

    # construct shards
    nshards = torch.cuda.device_count()
    nlayers = len(quant_model.model.layers)
    shards = [nn.ModuleList([]) for _ in range(nshards)]
    for i in range(nshards):
        for j in range(int(nlayers * i / nshards),
                       int(nlayers * (i + 1) / nshards)):
            shards[i].append(quant_model.model.layers[j])
        shards[i] = {'device': i, 'arg_fn': llama_arg_fn, 'shard': shards[i]}
    output_layer = {
        'layer': nn.Sequential(quant_model.model.norm, quant_model.lm_head),
        'fn': get_emb
    }

    shard_model = utils.ShardTransformer(shards, output_layer,
                                         args.ft_grad_ckpt, args.ft_train_mode)
    shard_model.manifest(emb[:args.ft_bs],
                         position_ids=position_ids,
                         attention_mask=attention_mask)

    for name, module in quant_model.named_modules():
        if isinstance(module, QuantizedLinear):
            module.SU = nn.Parameter(module.SU.float(), requires_grad=True)
            module.SV = nn.Parameter(module.SV.float(), requires_grad=True)
            if module.tlut is not None and args.ft_train_lut:
                module.tlut.requires_grad = True
            if args.ft_train_lut:
                module.mode = 'train-recons'
                glog.info('overriding ft_prefetch_trellis')
            elif args.ft_prefetch_trellis:
                module.mode = 'train-fixW'
            else:
                module.mode = 'eval'
            module.grad_ckpt = args.ft_grad_ckpt
    utils.clean()
    with torch.enable_grad():
        finetune.finetune_susv_e2e(quant_model, start_dev, devset, orig_dtype,
                                   args)

    for name, module in quant_model.named_modules():
        if isinstance(module, QuantizedLinear):
            del module.codebook_class
            if args.ft_train_lut and not module.has_kernel:
                module.trellis = module.packed_trellis
    quant_model = quant_model.to(orig_dtype)
    quant_model.config._name_or_path = args.base_model
    quant_model.save_pretrained(args.hf_output_path, safe_serialization=True)


if __name__ == '__main__':
    mp.set_start_method('spawn')
    mp.set_sharing_strategy('file_system')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    main(args)
