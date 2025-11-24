import argparse
import datetime
import os
import random
import types
from copy import deepcopy

from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import numpy
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from transformers import (AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast)

try:
    from transformers import GptOssForCausalLM
except ImportError:
    GptOssForCausalLM = AutoModelForCausalLM

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

# --- [수정된 부분] 중복 저장 방지를 위한 패치 함수 ---
def patched_expert_forward(self, hidden_states: torch.Tensor, router_indices=None, routing_weights=None) -> torch.Tensor:
    batch_size = hidden_states.shape[0]
    num_experts = routing_weights.shape[1]
    
    # (seq_len, hidden) 형태로 폄
    hidden_states = hidden_states.reshape(-1, self.hidden_size)

    # --- [최적화 핵심] ---
    # .repeat() 하기 전에 Hook을 통과시킵니다.
    # 이렇게 하면 모든 Expert에 들어가는 동일한 입력을 한 번만 저장합니다.
    # (수학적으로 Covariance Matrix의 스케일만 달라지고 분포 정보는 동일함)
    self.hook_gate_in(hidden_states)
    
    # 실제 연산을 위해 데이터 복사 (Inference Logic)
    hidden_states = hidden_states.repeat(num_experts, 1)
    hidden_states = hidden_states.view(num_experts, -1, self.hidden_size)
    
    # 2. Gate Up Projection (Fused)
    # 이미 위에서 hook을 통과했으므로 여기선 바로 연산
    gate_up = torch.bmm(hidden_states, self.gate_up_proj) + self.gate_up_proj_bias[..., None, :]
    gate, up = gate_up[..., ::2], gate_up[..., 1::2]
    
    # Activation
    gate = gate.clamp(min=None, max=self.limit)
    up = up.clamp(min=-self.limit, max=self.limit)
    glu = gate * torch.sigmoid(gate * self.alpha)
    
    intermediate = (up + 1) * glu
    
    # --- [유지] down_proj 입력 캡처 ---
    # 여기 값은 Expert마다 다르므로 (Weights가 달라서), Flatten된 전체를 다 저장해야 정확함
    intermediate = self.hook_down_in(intermediate)

    # 3. Down Projection
    next_states = torch.bmm(intermediate, self.down_proj)
    next_states = next_states + self.down_proj_bias[..., None, :]
    
    # 4. Output Reshape & Routing
    next_states = next_states.view(num_experts, batch_size, -1, self.hidden_size)
    next_states = next_states * routing_weights.transpose(0, 1).view(num_experts, batch_size, -1)[..., None]
    next_states = next_states.sum(dim=0)
    
    return next_states

def main(args):
    print("loading model...")
    gpu_id = int(os.environ["LOCAL_RANK"])
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("loading dataset...")
    devset = utils.sample_rp1t_concat(tokenizer, args.devset_size, args.ctx_size, nproc=args.sample_proc)
    devset = torch.split(devset, args.large_batch_size)

    for lbi in range(len(devset)):
        try:
            model = GptOssForCausalLM.from_pretrained(args.base_model, torch_dtype="auto", low_cpu_mem_usage=True, trust_remote_code=True)
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype="auto", low_cpu_mem_usage=True, trust_remote_code=True)
            
        print(f'processing split {lbi}')
        dev_emb = model.model.embed_tokens(devset[lbi].view(-1, args.batch_size, args.ctx_size))

        position_ids = torch.arange(args.ctx_size, dtype=torch.int64)[None, :] + torch.zeros(args.batch_size, args.ctx_size, dtype=torch.int64)
        
        if hasattr(model.config, 'sliding_window') and model.config.sliding_window is not None:
            attention_mask = _prepare_4d_causal_attention_mask(None, (args.batch_size, args.ctx_size), dev_emb[0], 0, sliding_window=model.config.sliding_window)
        else:
            attention_mask = _prepare_4d_causal_attention_mask(None, (args.batch_size, args.ctx_size), dev_emb[0], 0)

        position_ids = position_ids.cuda()
        attention_mask = attention_mask.cuda()
        
        transformer_layer_index = 0
        
        while len(model.model.layers) > 0:
            print(gpu_id, 1)
            layer = model.model.layers[0]
            layer = layer.cuda()
            save_pfx = f'/dev/shm/{transformer_layer_index}'
            
            # 1. Self Attention Hooks (Q, O only)
            done_qkv = utils.register_input_H_hook(layer.self_attn.q_proj, f'{save_pfx}_qkv', gpu_id)
            done_o = utils.register_input_H_hook(layer.self_attn.o_proj, f'{save_pfx}_o', gpu_id)
            
            # 2. Router Hook
            done_router = utils.register_input_H_hook(layer.mlp.router, f'{save_pfx}_router', gpu_id)
            
            # 3. Experts Hook (Monkey Patching)
            experts_module = layer.mlp.experts
            experts_module.hook_gate_in = torch.nn.Identity()
            experts_module.hook_down_in = torch.nn.Identity()
            
            # Hook 등록
            done_experts_gate = utils.register_input_H_hook(experts_module.hook_gate_in, f'{save_pfx}_experts_gate_up', gpu_id)
            done_experts_down = utils.register_input_H_hook(experts_module.hook_down_in, f'{save_pfx}_experts_down', gpu_id)
            
            # 메서드 교체
            experts_module.forward = types.MethodType(patched_expert_forward, experts_module)

            for di in range(len(dev_emb)):
                tmp_input = dev_emb[di].cuda()
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
            
            hook_fns = [
                ('q', done_qkv),
                ('o', done_o),
                ('router', done_router),
                ('experts_gate_up', done_experts_gate),
                ('experts_down', done_experts_down)
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