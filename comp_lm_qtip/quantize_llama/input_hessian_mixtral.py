import argparse
import datetime
import os
import random
from copy import deepcopy

from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import numpy
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedTokenizerFast, MixtralForCausalLM)
from transformers.modeling_attn_mask_utils import \
    _prepare_4d_causal_attention_mask

from lib import utils

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=2, type=int)
parser.add_argument('--large_batch_size', default=512, type=int)
parser.add_argument('--devset_size', default=8192, type=int)
parser.add_argument('--ctx_size', default=4096, type=int)
parser.add_argument('--base_model',
                    default='meta-llama/Llama-2-70b-hf',
                    type=str)
parser.add_argument('--save_path', default='hessians/llama2_70b', type=str)
parser.add_argument('--sample_proc', default=32, type=int)


def main(args):
    print("loading model...")
    print("loaded model!")
    gpu_id = int(os.environ["LOCAL_RANK"])
    # tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token

    print("loading dataset...")
    devset = utils.sample_rp1t_concat(tokenizer,
                                      args.devset_size,
                                      args.ctx_size,
                                      nproc=args.sample_proc)
    devset = torch.split(devset, args.large_batch_size)
    for lbi in range(len(devset)):
        model = MixtralForCausalLM.from_pretrained(args.base_model,
                                                     torch_dtype="auto",
                                                     low_cpu_mem_usage=True)
        print(f'processing split {lbi}')
        dev_emb = model.model.embed_tokens(devset[lbi].view(
            -1, args.batch_size, args.ctx_size))

        print("loaded dataset!")

        position_ids = torch.arange(args.ctx_size, dtype=torch.int64)[None, :] + \
            torch.zeros(args.batch_size, args.ctx_size, dtype=torch.int64)
        if hasattr(model.config, 'sliding_window'):
            attention_mask = _prepare_4d_causal_attention_mask(
                None, (args.batch_size, args.ctx_size),
                dev_emb[0],
                0,
                sliding_window=model.config.sliding_window)
        else:
            attention_mask = _prepare_4d_causal_attention_mask(
                None, (args.batch_size, args.ctx_size), dev_emb[0], 0)

        position_ids = position_ids.cuda()
        attention_mask = attention_mask.cuda()
        transformer_layer_index = 0
        while len(model.model.layers) > 0:
            print(gpu_id, 1)
            layer = model.model.layers[0]
            layer = layer.cuda()
            save_pfx = f'/dev/shm/{transformer_layer_index}'
            done_qkv = utils.register_input_H_hook(layer.self_attn.q_proj,
                                                   f'{save_pfx}_qkv', gpu_id)
            done_o = utils.register_input_H_hook(layer.self_attn.o_proj,
                                                 f'{save_pfx}_o', gpu_id)
            done_gate = utils.register_input_H_hook(layer.block_sparse_moe.gate,
                                                    f'{save_pfx}_gate', gpu_id)
            done_experts_up = []
            done_experts_down = []

            for expert_idx, expert_layer in enumerate(layer.block_sparse_moe.experts):
                # w3 (up_proj와 동일)
                done_w3 = utils.register_input_H_hook(
                    expert_layer.w3,
                    f'{save_pfx}_expert{expert_idx}_w3', gpu_id
                )
                done_experts_up.append(done_w3)

                # w2 (down_proj와 동일)
                done_w2 = utils.register_input_H_hook(
                    expert_layer.w2,
                    f'{save_pfx}_expert{expert_idx}_w2', gpu_id
                )
                done_experts_down.append(done_w2)
                
            for di in range(len(dev_emb)):
                tmp_input = dev_emb[di].cuda()
                ##
                # position_embeddings = model.model.rotary_emb(dev_emb[di].cuda(), position_ids)
                dev_emb[di] = layer(dev_emb[di].cuda(),
                                    position_ids=position_ids,
                                    attention_mask=attention_mask,
                                    use_cache=False,
                                    # position_embeddings=position_embeddings, ##
                                    output_attentions=False)[0].cpu()
                tmp_input = tmp_input.cpu()
                del tmp_input
                utils.clean()
                
                
            layer = layer.cpu()
            del layer, model.model.layers[0]
            utils.clean()
            
            hook_fns = [
                ('qkv', done_qkv),
                ('o', done_o),
                ('gate', done_gate)
            ]
            for expert_idx in range(len(done_experts_up)):
                hook_fns.append( (f'expert{expert_idx}_w3', done_experts_up[expert_idx]) )
                hook_fns.append( (f'expert{expert_idx}_w2', done_experts_down[expert_idx]) )

            for key, fn in hook_fns:
                fn()
                utils.clean()
                
            dist.barrier()
            
            # 3. 재구성된 리스트를 기반으로 데이터 집계 및 저장
            if gpu_id == 0:
                for key, fn in hook_fns: # fn_dict 대신 hook_fns 사용
                    save_path = f"{args.save_path}/{transformer_layer_index}_{key}.pt" # key가 동적으로 생성됨
                    if os.path.exists(save_path):
                        data = torch.load(save_path,
                                          map_location=torch.device('cpu'))
                        data['flatH'] = data['flatH'].to(
                            torch.float64) * data['ct']
                    else:
                        data = None
                    gi = 0
                    gi_path = f"/dev/shm/{transformer_layer_index}_{key}_{gi}.pt" # key가 동적으로 생성됨
                    while os.path.exists(gi_path):
                        print(gi_path)
                        d2 = torch.load(gi_path,
                                        map_location=torch.device('cpu'))
                        if data is not None:
                            data['flatH'] += utils.sym_to_flat(d2['H'])
                            data['ct'] += d2['ct']
                            del d2
                            utils.clean()
                        else:
                            data = d2
                            data['flatH'] = utils.sym_to_flat(data['H'])
                            del data['H']
                        os.remove(gi_path)
                        gi += 1
                        gi_path = f"/dev/shm/{transformer_layer_index}_{key}_{gi}.pt"
                    data['flatH'] /= data['ct']
                    data['flatH'] = data['flatH'].float()
                    torch.save(data, save_path)
                    del data
                    utils.clean()

            dist.barrier()

            print(f"done processing layer {transformer_layer_index}")
            transformer_layer_index += 1

        del position_ids, attention_mask

        del dev_emb
        utils.clean()
        del model
        utils.clean()


if __name__ == "__main__":
    #mp.set_start_method('spawn')
    torch.set_grad_enabled(False)
    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    dist.init_process_group(backend="nccl")
    gpu_id = int(os.environ["LOCAL_RANK"])
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)
    torch.manual_seed(gpu_id)
    random.seed(gpu_id)
    numpy.random.seed(gpu_id)

    main(args)

    dist.destroy_process_group()
