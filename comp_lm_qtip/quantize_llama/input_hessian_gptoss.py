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

# [수정] 커스텀 모델 클래스 처리를 위한 import
from transformers import (AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast)

try:
    from transformers import GptOssForCausalLM
except ImportError:
    # 명시적으로 import가 안 되면 AutoModel로 대체하되 아래에서 trust_remote_code=True 사용
    GptOssForCausalLM = None

from transformers.modeling_attn_mask_utils import \
    _prepare_4d_causal_attention_mask

from lib import utils

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=2, type=int)
parser.add_argument('--large_batch_size', default=512, type=int)
parser.add_argument('--devset_size', default=8192, type=int)
parser.add_argument('--ctx_size', default=4096, type=int)
parser.add_argument('--base_model', default='', type=str)
parser.add_argument('--save_path', default='hessians/gptoss_moe', type=str)
parser.add_argument('--sample_proc', default=32, type=int)


def main(args):
    print("loading model...")
    gpu_id = int(os.environ["LOCAL_RANK"])
    
    # 토크나이저 (trust_remote_code 옵션 추가 권장)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("loaded model definition!")

    print("loading dataset...")
    devset = utils.sample_rp1t_concat(tokenizer,
                                      args.devset_size,
                                      args.ctx_size,
                                      nproc=args.sample_proc)
    devset = torch.split(devset, args.large_batch_size)

    for lbi in range(len(devset)):
        # [수정] 모델 로딩: trust_remote_code=True 필수
        try:
            model = GptOssForCausalLM.from_pretrained(args.base_model,
                                                        torch_dtype="auto",
                                                        low_cpu_mem_usage=True,
                                                        trust_remote_code=True)
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(args.base_model,
                                                        torch_dtype="auto",
                                                        low_cpu_mem_usage=True,
                                                        trust_remote_code=True)
            
        print(f'processing split {lbi}')
        dev_emb = model.model.embed_tokens(devset[lbi].view(
            -1, args.batch_size, args.ctx_size))

        print("loaded dataset!")

        position_ids = torch.arange(args.ctx_size, dtype=torch.int64)[None, :] + \
            torch.zeros(args.batch_size, args.ctx_size, dtype=torch.int64)
        
        # Sliding window 처리
        if hasattr(model.config, 'sliding_window') and model.config.sliding_window is not None:
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
            
            # --- [요청사항 반영] Q, O만 유지 (K, V 제거) ---
            done_qkv = utils.register_input_H_hook(layer.self_attn.q_proj, f'{save_pfx}_qkv', gpu_id)
            done_o = utils.register_input_H_hook(layer.self_attn.o_proj, f'{save_pfx}_o', gpu_id)
            
            # --- MLP Hooks (GptOss 구조에 맞춤) ---
            # 1. Router
            done_router = utils.register_input_H_hook(layer.mlp.router, f'{save_pfx}_router', gpu_id)
            
            # 2. Experts (Fused)
            # GptOssExperts는 개별 expert 리스트가 아닌 통짜 모듈이므로 전체 입력 수집
            done_experts = utils.register_input_H_hook(layer.mlp.experts, f'{save_pfx}_experts_fused', gpu_id)

            for di in range(len(dev_emb)):
                tmp_input = dev_emb[di].cuda()
                
                # GptOss의 Forward 방식에 맞게 position_embeddings 계산 후 전달
                position_embeddings = model.model.rotary_emb(tmp_input, position_ids)
                
                layer_out = layer(
                    tmp_input,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    position_embeddings=position_embeddings
                )
                
                if isinstance(layer_out, tuple):
                    dev_emb[di] = layer_out[0].cpu()
                else:
                    dev_emb[di] = layer_out.cpu()

                tmp_input = tmp_input.cpu()
                del tmp_input
                utils.clean()
                
            layer = layer.cpu()
            del layer, model.model.layers[0]
            utils.clean()
            
            # --- [요청사항 반영] Hook 리스트에서 k, v 제거 ---
            hook_fns = [
                ('q', done_qkv),
                ('o', done_o),
                ('router', done_router),
                ('experts_fused', done_experts) 
            ]

            for key, fn in hook_fns:
                fn()
                utils.clean()
                
            dist.barrier()
            
            if gpu_id == 0:
                for key, fn in hook_fns:
                    save_path_file = f"{args.save_path}/{transformer_layer_index}_{key}.pt"
                    if os.path.exists(save_path_file):
                        data = torch.load(save_path_file, map_location=torch.device('cpu'))
                        data['flatH'] = data['flatH'].to(torch.float64) * data['ct']
                    else:
                        data = None
                    gi = 0
                    gi_path = f"/dev/shm/{transformer_layer_index}_{key}_{gi}.pt"
                    while os.path.exists(gi_path):
                        d2 = torch.load(gi_path, map_location=torch.device('cpu'))
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
                    
                    if data is not None:
                        data['flatH'] /= data['ct']
                        data['flatH'] = data['flatH'].float()
                        torch.save(data, save_path_file)
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