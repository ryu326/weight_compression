import argparse
import os
import time

import glog

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

import torch
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_attn_mask_utils import \
    _prepare_4d_causal_attention_mask

from lib import utils
from lib.algo import finetune_mixtral # 리팩토링된 워커 모듈 사용
from lib.codebook import bitshift
from operator import attrgetter

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_cpu_threads', default=8, type=int)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--devset_size', default=384, type=int)
parser.add_argument('--ctx_size', default=4096, type=int)
parser.add_argument('--save_path', type=str)
parser.add_argument('--in_hess_path', type=str)
parser.add_argument('--base_model', type=str)
parser.add_argument('--sigma_reg', default=1e-2, type=float)
parser.add_argument('--sigma_reg2', default=1e-2, type=float)
parser.add_argument('--scale_override', default=-1, type=float)
parser.add_argument('--codebook', type=str)
parser.add_argument('--use_fp64', action='store_true')
parser.add_argument('--no_use_buffered', action='store_true')
parser.add_argument('--sample_proc', default=1, type=int)
parser.add_argument('--lowmem_ldlq', action='store_true')
parser.add_argument('--ft_lr', default=3e-6, type=float)
parser.add_argument('--ft_bs', default=4, type=int)
parser.add_argument('--ft_update_freq', default=1, type=int)
parser.add_argument('--ft_epochs', default=5, type=int)
parser.add_argument('--ft_valid_freq', default=1, type=int)
parser.add_argument('--ft_valid_size', default=128, type=float)
parser.add_argument('--ft_early_stop', default=5, type=int)
parser.add_argument('--ft_grad_ckpt', action='store_true')
parser.add_argument('--td_x', default=16, type=int)
parser.add_argument('--td_y', default=16, type=int)
parser.add_argument('--L', default=16, type=int)
parser.add_argument('--K', default=2, type=int)
parser.add_argument('--V', default=2, type=int)
parser.add_argument('--tlut_bits', default=0, type=int)
parser.add_argument('--decode_mode', default='lut', type=str)
parser.add_argument('--ft_train_lut', action='store_true')
parser.add_argument('--split_for_tp', action='store_true')
parser.add_argument('--tp_rank', default=8, type=int)
parser.add_argument('--skip_list', default=None, type=str)


def check_exist_mixtral(idx, args, model_config):
    """Mixtral 레이어에 필요한 모든 파일이 존재하는지 확인"""
    suffix = ['q', 'k', 'v', 'o', 'layernorm']
    suffix.append('gate')
    if hasattr(model_config, 'num_local_experts'):
        # num_experts = model_config.num_local_experts
        num_experts = getattr(model_config, 'num_local_experts', getattr(model_config, 'num_experts', 0))
        
        for i in range(num_experts):
            suffix.append(f'expert{i}_w1')
            suffix.append(f'expert{i}_w2')
            suffix.append(f'expert{i}_w3')
    else:
        glog.warning("model_config에 num_local_experts가 없습니다.")
        return False

    for s in suffix:
        test = f'{args.save_path}/{idx}_{s}.pt'
        if not os.path.exists(test):
            return False
    return True


def quantize_moe_decoder(layer, idx, cb, args, device, pre_orig_emb,
                         orig_emb, model_config, skip_list, attention_mask, rotary_emb):
    
    if check_exist_mixtral(idx, args, model_config):
        glog.info(f"Layer {idx}의 파일이 이미 존재하므로 스킵합니다.")
        return

    if skip_list is None:
        skip_list = []
        
    quant_order = []
    
    # 1. Attention Layers (공통)
    # 대부분의 모델이 q_proj, k_proj, v_proj, o_proj 이름을 사용함
    for thing in [('self_attn.v_proj', 'v', 'qkv', 'v', 'col'),
                  ('self_attn.q_proj', 'q', 'qkv', 'q', 'col'),
                  ('self_attn.k_proj', 'k', 'qkv', 'k', 'col'),
                  ('self_attn.o_proj', 'o', 'o', 'o', 'row')]:
        if f'{idx}_{thing[1]}' not in skip_list:
            quant_order.append(thing)
        else:
            attrgetter(thing[0])(layer).weight.requires_grad = False
            print(f'skipping {idx}_{thing[1]}')

    # 2. MoE Layers (모델 타입별 분기)
    model_type = model_config.model_type.lower()
    is_qwen = 'qwen' in model_type
    
    if is_qwen:
        # Qwen: layer.mlp.experts.{i}.[gate_proj, up_proj, down_proj]
        gate_module_name = 'mlp.gate'
        expert_prefix = 'mlp.experts'
        # Qwen 실제 레이어 이름 매핑 (w1: gate, w3: up, w2: down)
        layer_name_map = {'w1': 'gate_proj', 'w3': 'up_proj', 'w2': 'down_proj'}
    else:
        # Mixtral: layer.block_sparse_moe.experts.{i}.[w1, w3, w2]
        gate_module_name = 'block_sparse_moe.gate'
        expert_prefix = 'block_sparse_moe.experts'
        # Mixtral 실제 레이어 이름 매핑
        layer_name_map = {'w1': 'w1', 'w3': 'w3', 'w2': 'w2'}

    # (1) Router Gate
    gate_thing = (gate_module_name, 'gate', 'gate', 'gate', 'col')
    if f'{idx}_{gate_thing[1]}' not in skip_list:
        quant_order.append(gate_thing)
    else:
        attrgetter(gate_thing[0])(layer).weight.requires_grad = False
        print(f'skipping {idx}_{gate_thing[1]}')
        
# (3) Experts Loop (간소화됨)
    # 처리할 레이어 목록: (논리적 이름, Hessian 소스 이름, RCP 타입)
    # w1(gate)은 w3(up)의 Hessian을 공유합니다.
    expert_layers_config = [
        ('w1', 'w3', 'col'), # w1 uses w3 hessian
        ('w3', 'w3', 'col'),
        ('w2', 'w2', 'row')
    ]

    # num_experts = model_config.num_local_experts
    num_experts = getattr(model_config, 'num_local_experts', getattr(model_config, 'num_experts', 0))
    for i in range(num_experts):
        for logical_name, hess_src, rcp in expert_layers_config:
            
            # 실제 레이어 속성 이름 (예: gate_proj 또는 w1)
            attr_name = layer_name_map[logical_name]
            # 전체 모듈 경로 (예: mlp.experts.0.gate_proj)
            full_module_name = f"{expert_prefix}.{i}.{attr_name}"
            
            # 저장 키 및 Hessian 키 (예: expert0_w1, expert0_w3)
            save_key = f"expert{i}_{logical_name}"
            hess_key = f"expert{i}_{hess_src}"
            
            # (linear_attr, name, in_hess_name, out_hess_name, rcp)
            item = (full_module_name, save_key, hess_key, save_key, rcp)
            
            if f'{idx}_{save_key}' not in skip_list:
                quant_order.append(item)
            else:
                attrgetter(full_module_name)(layer).weight.requires_grad = False
                print(f'skipping {idx}_{save_key}')
    # 워커 함수 호출
    finetune_mixtral.quantize_finetune_decoder_layer(layer, quant_order, idx, cb, args,
                                             device, pre_orig_emb, orig_emb, attention_mask, rotary_emb)
    
    torch.save(
        {
            'input_layernorm': layer.input_layernorm.weight,
            'post_attention_layernorm': layer.post_attention_layernorm.weight,
        }, f'{args.save_path}/{idx}_layernorm.pt')

def main(args):
    if args.skip_list is not None:
        args.skip_list = args.skip_list.split(',')
        
    dtype_ = torch.float64 if args.use_fp64 else torch.float32

    cb = bitshift.bitshift_codebook(L=args.L,
                                    K=args.K,
                                    V=args.V,
                                    tlut_bits=args.tlut_bits,
                                    decode_mode=args.decode_mode)
    
    # [통합] AutoModelForCausalLM 사용
    # MixtralForCausalLM, Qwen3MoeForCausalLM 등을 자동으로 로드
    model = AutoModelForCausalLM.from_pretrained(args.base_model,
                                                 torch_dtype='auto',
                                                 low_cpu_mem_usage=True,
                                                 trust_remote_code=True) # Qwen 등 최신 모델을 위해 권장

    glog.info(f"Loaded model type: {model.config.model_type}")

    # save configs
    all_config = {'quant_args': args, 'model_config': model.config}
    quip_params = {
        'codebook': args.codebook,
        'codebook_version': cb.version,
        'L': args.L,
        'K': args.K,
        'V': args.V,
        'tlut_bits': args.tlut_bits,
        'decode_mode': args.decode_mode,
        'td_x': args.td_x,
        'td_y': args.td_y,
        'split_for_tp': args.split_for_tp,
        'skip_list': args.skip_list,
    }
    # Qwen 등 일부 config는 dict 업데이트가 안될 수 있으므로 예외처리나 setattr 사용
    if hasattr(model.config, 'quip_params'):
        model.config.quip_params = quip_params
    else:
        # 딕셔너리 형태로 저장
        all_config['model_config'].__dict__.update({'quip_params': quip_params})

    torch.save(all_config, os.path.join(args.save_path, 'config.pt'))

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    glog.info('loaded model and tokenizer')

    devset = utils.sample_rp1t(tokenizer, args.devset_size, args.ctx_size,
                               args.sample_proc)
    glog.info('loaded dataset and devset')

    nproc = torch.cuda.device_count()
    orig_emb_cache = [model.model.embed_tokens(devset)]

    for _ in range(nproc):
        orig_emb_cache.append(
            torch.zeros(orig_emb_cache[0].shape,
                        dtype=orig_emb_cache[0].dtype,
                        device=orig_emb_cache[0].device))

    position_ids = torch.arange(args.ctx_size, dtype=torch.int32)[None, :] + \
        torch.zeros(args.batch_size, args.ctx_size, dtype=torch.int32)
    
    sliding_window = getattr(model.config, 'sliding_window', None)
    if sliding_window is None:
        glog.warning("Sliding window config가 없습니다. Causal mask를 사용합니다.")
    
    # Qwen 등 일부 모델은 sliding_window가 config에 있어도 None일 수 있음
    attention_mask = _prepare_4d_causal_attention_mask(
        None, (args.batch_size, args.ctx_size),
        orig_emb_cache[0][:args.batch_size], 0,
        sliding_window=sliding_window)

    cur_device = 0
    proc_list = [None for _ in range(nproc)]
    for i in range(len(model.model.layers)):
        glog.info(f'layer {i} gpu {cur_device}')
        if proc_list[cur_device] is not None:
            proc_list[cur_device][0].join()
            model.model.layers[proc_list[cur_device][1]] = None
            utils.clean()
            if cur_device == 0:
                orig_emb_cache[0].copy_(orig_emb_cache[-1])
        if cur_device + 1 < nproc and proc_list[cur_device + 1] is not None:
            proc_list[cur_device + 1][0].join()
        utils.clean()
        st = time.time()
        position_ids = position_ids.to(cur_device)
        attention_mask = attention_mask.to(cur_device)
        model.model.layers[i].to(cur_device)
        
        # --- [수정] Forward 인자 준비 (모델별 분기) ---
        layer_kwargs = {
            "hidden_states": None, # 아래 루프에서 채움
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "use_cache": False,
            "output_attentions": False,
        }

        # Qwen 모델인 경우에만 position_embeddings 계산 및 추가
        rotary_emb = None
        if "qwen" in all_config['model_config'].model_type.lower():
            # rotary_emb 호출 (Qwen 전용)
            position_embeddings = model.model.rotary_emb(
                orig_emb_cache[cur_device][0:1].to(cur_device), 
                position_ids
            )
            layer_kwargs["position_embeddings"] = position_embeddings
            rotary_emb = model.model.rotary_emb

        # ------------------------------------------------

        if args.ft_epochs > 0:
            for j in range(args.devset_size // args.batch_size):
                utils.clean()
                
                # 입력 데이터 준비
                input_feat = orig_emb_cache[cur_device][args.batch_size * j : args.batch_size * (j + 1)].to(cur_device)
                layer_kwargs["hidden_states"] = input_feat # hidden_states 설정

                orig_emb_cache[cur_device + 1][args.batch_size * j : args.batch_size * (j + 1)] = \
                    model.model.layers[i](**layer_kwargs)[0].cpu()    
        # ## qwen, Not for mixtral
        # cos, sin = model.model.rotary_emb(
        #     orig_emb_cache[cur_device][0:1].to(cur_device), # x (샘플 하나만 보내도 됨)
        #     position_ids # position_ids
        # )        
        # position_embeddings = (cos, sin)
        
        # if args.ft_epochs > 0:
        #     for j in range(args.devset_size // args.batch_size):
        #         utils.clean()
        #         orig_emb_cache[cur_device + 1][args.batch_size * j : args.batch_size * (j + 1)] = \
        #             model.model.layers[i](
        #                 orig_emb_cache[cur_device][args.batch_size * j : args.batch_size * (j + 1)].to(cur_device),
        #                 position_ids=position_ids,
        #                 attention_mask=attention_mask,
        #                 use_cache=False,
        #                 output_attentions=False,
        #                 position_embeddings=position_embeddings)[0].cpu()
        else:
            orig_emb_cache[cur_device + 1] = orig_emb_cache[cur_device]
        model.model.layers[i].cpu()
        position_ids = position_ids.cpu()
        attention_mask = attention_mask.cpu()
        utils.clean()
        glog.info('computed original embedding for layer {} in {}s'.format(i, time.time() - st))

        # [통합] quantize_moe_decoder 호출
        proc_list[cur_device] = (mp.Process(target=quantize_moe_decoder,
                                            args=(
                                                model.model.layers[i],
                                                i,
                                                cb,
                                                args,
                                                cur_device,
                                                orig_emb_cache[cur_device],
                                                orig_emb_cache[cur_device + 1],
                                                all_config['model_config'],
                                                args.skip_list,
                                                attention_mask[:args.ft_bs].cpu(), # Slicing 적용, 
                                                rotary_emb
                                            )), i)
        proc_list[cur_device][0].start()

        cur_device = (cur_device + 1) % nproc

    for p in proc_list:
        p[0].join()

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    mp.set_start_method('spawn')
    mp.set_sharing_strategy('file_system')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.save_path, exist_ok=True)
    main(args)
