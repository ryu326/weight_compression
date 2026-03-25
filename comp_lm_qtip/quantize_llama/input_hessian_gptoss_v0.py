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
                          PreTrainedTokenizerFast, Qwen3MoeForCausalLM, GptOssForCausalLM)
from transformers.modeling_attn_mask_utils import \
    _prepare_4d_causal_attention_mask
from model.gptoss_standard_moe import GptOssForCausalLM as GptOssForCausalLM_standard

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

def transfer_weights(orig_model, std_model):
    """
    원본 GptOss 모델의 가중치를 리팩토링된 Standard MoE 모델로 이식합니다.
    """
    print("🔄 가중치 변환 및 이식 시작...")
    
    orig_state_dict = orig_model.state_dict()
    std_state_dict = std_model.state_dict()
    
    # 변경된 state_dict를 담을 딕셔너리
    new_state_dict = {}

    # GptOss Config 정보 (Expert 수 등)
    num_experts = orig_model.config.num_local_experts
    
    for key, value in orig_state_dict.items():
        # 1. Router 변환 (weight/bias -> gate.weight/gate.bias)
        if "router.weight" in key:
            # 예: model.layers.0.mlp.router.weight -> model.layers.0.mlp.router.gate.weight
            new_key = key.replace("router.weight", "router.gate.weight")
            new_state_dict[new_key] = value
            
        elif "router.bias" in key:
            new_key = key.replace("router.bias", "router.gate.bias")
            new_state_dict[new_key] = value
            
        # 2. Experts 변환 (3D Tensor -> ModuleList of Linear)
        elif "experts.gate_up_proj" in key:
            # key: ...mlp.experts.gate_up_proj (3D Tensor)
            # value shape: [num_experts, hidden, 2*inter]
            
            # Bias인지 Weight인지 확인
            if "gate_up_proj_bias" in key: # Bias 처리
                base_key = key.replace("experts.gate_up_proj_bias", "experts.experts") # ...mlp.experts.experts
                for i in range(num_experts):
                    # target: ...mlp.experts.experts.0.gate_up_proj.bias
                    target_key = f"{base_key}.{i}.gate_up_proj.bias"
                    new_state_dict[target_key] = value[i]
            else: # Weight 처리
                base_key = key.replace("experts.gate_up_proj", "experts.experts") # ...mlp.experts.experts
                for i in range(num_experts):
                    # target: ...mlp.experts.experts.0.gate_up_proj.weight
                    target_key = f"{base_key}.{i}.gate_up_proj.weight"
                    new_state_dict[target_key] = value[i].t()

        elif "experts.down_proj" in key:
            # key: ...mlp.experts.down_proj (3D Tensor)
            # value shape: [num_experts, inter, hidden]
            
            if "down_proj_bias" in key: # Bias 처리
                # value shape: [num_experts, hidden]
                base_key = key.replace("experts.down_proj_bias", "experts.experts")
                for i in range(num_experts):
                    target_key = f"{base_key}.{i}.down_proj.bias"
                    new_state_dict[target_key] = value[i]
            else: # Weight 처리
                # value shape: [num_experts, inter, hidden]
                base_key = key.replace("experts.down_proj", "experts.experts")
                for i in range(num_experts):
                    target_key = f"{base_key}.{i}.down_proj.weight"
                    
                    # ⚠️ 중요: nn.Linear Weight Transpose
                    new_state_dict[target_key] = value[i].t()
        # 3. 그 외 나머지 (Attention, Norm, Embeddings 등) - 그대로 복사
        else:
            new_state_dict[key] = value

    # strict=True로 해서 키가 하나라도 안 맞으면 에러가 나게 검증
    std_model.load_state_dict(new_state_dict, strict=True)
    print("✅ 가중치 이식 완료!")


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

    cache_directory = '/home/jgryu/.cache/huggingface/hub'
    orig_model = GptOssForCausalLM.from_pretrained(
        args.base_model, 
        cache_dir=cache_directory, 
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True
        # attn_implementation='sdpa',
    )
                                    
    devset = torch.split(devset, args.large_batch_size)
    for lbi in range(len(devset)):
        # --- [수정 1] MixtralForCausalLM -> Qwen3MoeForCausalLM ---
        # model = Qwen3MoeForCausalLM.from_pretrained(args.base_model,
        #                                              torch_dtype="auto",
        #                                              low_cpu_mem_usage=True)

        model = GptOssForCausalLM_standard.from_pretrained(
            args.base_model, 
            cache_dir=cache_directory, 
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )
        transfer_weights(orig_model, model) 

        print(f'processing split {lbi}')
        dev_emb = model.model.embed_tokens(devset[lbi].view(
            -1, args.batch_size, args.ctx_size))

        print("loaded dataset!")

        position_ids = torch.arange(args.ctx_size, dtype=torch.int64)[None, :] + \
            torch.zeros(args.batch_size, args.ctx_size, dtype=torch.int64)
        
        # Qwen 모델도 sliding_window를 지원하므로 이 로직은 유효합니다.
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
            
            # 1. Self Attention Hooks (GptOssAttention 구조에 맞춤)
            # q_proj, k_proj, v_proj가 보통 같은 입력을 받으므로 q_proj에만 걸어도 Input X는 확보됨
            done_qkv = utils.register_input_H_hook(layer.self_attn.q_proj,
                                                   f'{save_pfx}_qkv', gpu_id)
            done_o = utils.register_input_H_hook(layer.self_attn.o_proj,
                                                 f'{save_pfx}_o', gpu_id)
            
            # 2. Router Hook 수정
            # GptOssMLP -> self.router (GptOssTopKRouter) -> self.gate (Linear)
            done_gate = utils.register_input_H_hook(layer.mlp.router.gate,
                                                    f'{save_pfx}_gate', gpu_id)
            
            done_experts_gate_up = []
            done_experts_down = []

            # 3. Experts Loop 수정
            # GptOssMLP -> self.experts (GptOssExperts) -> self.experts (ModuleList)
            # Qwen/Mixtral과 달리 expert 객체 자체가 리스트가 아니라, 내부 멤버변수가 리스트임
            for expert_idx, expert_layer in enumerate(layer.mlp.experts.experts):
                
                # [수정 요청 반영] Gate/Up Fused Layer
                # GptOssLayer -> self.gate_up_proj (Linear)
                done_gate_up = utils.register_input_H_hook(
                    expert_layer.gate_up_proj,
                    f'{save_pfx}_expert{expert_idx}_gate_up', gpu_id
                )
                done_experts_gate_up.append(done_gate_up)

                # Down Projection
                # GptOssLayer -> self.down_proj (Linear)
                done_down = utils.register_input_H_hook(
                    expert_layer.down_proj,
                    f'{save_pfx}_expert{expert_idx}_down', gpu_id
                )
                done_experts_down.append(done_down)
                
            for di in range(len(dev_emb)):
                tmp_input = dev_emb[di].cuda()
                
                # 순전파 호출
                position_embeddings = model.model.rotary_emb(dev_emb[di].cuda(), position_ids)                
                dev_emb[di] = layer(dev_emb[di].cuda(),
                                    position_ids=position_ids,
                                    attention_mask=attention_mask,
                                    use_cache=False,
                                    position_embeddings=position_embeddings,
                                    output_attentions=False)[0].cpu()
                tmp_input = tmp_input.cpu()
                del tmp_input
                utils.clean()
                
            layer = layer.cpu()
            del layer, model.model.layers[0]
            utils.clean()
            
            # 4. 훅 리스트 생성 (키 이름 변경: w3/w2 -> gate_up/down)
            hook_fns = [
                ('qkv', done_qkv),
                ('o', done_o),
                ('gate', done_gate)
            ]
            
            for expert_idx in range(len(done_experts_gate_up)):
                # Fused Gate+Up
                hook_fns.append( (f'expert{expert_idx}_gate_up', done_experts_gate_up[expert_idx]) )
                # Down
                hook_fns.append( (f'expert{expert_idx}_down', done_experts_down[expert_idx]) )

            for key, fn in hook_fns:
                fn()
                utils.clean()
                
            dist.barrier()
            
            # 5. 데이터 집계 및 저장 (로직 동일)
            if gpu_id == 0:
                for key, fn in hook_fns:
                    save_path = f"{args.save_path}/{transformer_layer_index}_{key}.pt"
                    # ... (이하 저장 로직 기존과 동일) ...
                    if os.path.exists(save_path):
                        data = torch.load(save_path, map_location=torch.device('cpu'))
                        data['flatH'] = data['flatH'].to(torch.float64) * data['ct']
                    else:
                        data = None
                    gi = 0
                    gi_path = f"/dev/shm/{transformer_layer_index}_{key}_{gi}.pt"
                    while os.path.exists(gi_path):
                        # print(gi_path)
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