import argparse
import os
import time

import glog

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

import torch
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer, MixtralForCausalLM
from transformers.modeling_attn_mask_utils import \
    _prepare_4d_causal_attention_mask

from lib import utils
from lib.algo import finetune_mixtral
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


# [교체]
def check_exist_mixtral(idx, args, model_config):
    """Mixtral 레이어에 필요한 모든 파일이 존재하는지 확인"""
    suffix = ['q', 'k', 'v', 'o', 'layernorm']
    suffix.append('gate')
    if hasattr(model_config, 'num_local_experts'):
        num_experts = model_config.num_local_experts
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


# [교체]
def quantize_mixtral_decoder(layer, idx, cb, args, device, pre_orig_emb,
                             orig_emb, model_config, skip_list, attention_mask):
    
    if check_exist_mixtral(idx, args, model_config):
        glog.info(f"Layer {idx}의 파일이 이미 존재하므로 스킵합니다.")
        return

    if skip_list is None:
        skip_list = []
        
    quant_order = []
    
    for thing in [('self_attn.v_proj', 'v', 'qkv', 'v', 'col'),
                  ('self_attn.q_proj', 'q', 'qkv', 'q', 'col'),
                  ('self_attn.k_proj', 'k', 'qkv', 'k', 'col'),
                  ('self_attn.o_proj', 'o', 'o', 'o', 'row')]:
        if f'{idx}_{thing[1]}' not in skip_list:
            quant_order.append(thing)
        else:
            attrgetter(thing[0])(layer).weight.requires_grad = False
            print(f'skipping {idx}_{thing[1]}')

    
    # 2. MoE 라우터 게이트
    # (layer_name, save_name, in_hess_file, out_hess_file, type)
    # 이전 input_hessian 스크립트에서 'gate'로 저장했습니다.
    gate_thing = ('block_sparse_moe.gate', 'gate', 'gate', 'gate', 'col')
    if f'{idx}_{gate_thing[1]}' not in skip_list:
        quant_order.append(gate_thing)
    else:
        attrgetter(gate_thing[0])(layer).weight.requires_grad = False
        print(f'skipping {idx}_{gate_thing[1]}')
    # attrgetter('block_sparse_moe.gate')(layer).weight.requires_grad = False

    # 3. MoE 전문가 레이어 (동적 생성)
    num_experts = model_config.num_local_experts
    for i in range(num_experts):
        
        # w1 (Llama의 gate_proj에 해당)
        # Hessian: Llama의 gate_proj가 up_proj의 Hessian을 썼듯이,
        # w1은 w3의 Hessian(expert{i}_w3)을 사용합니다.
        w1_name = f'block_sparse_moe.experts.{i}.w1'
        w1_save = f'expert{i}_w1'
        w1_hess = f'expert{i}_w3' # w3의 Hessian 사용
        w1_thing = (w1_name, w1_save, w1_hess, w1_save, 'col')
        
        if f'{idx}_{w1_save}' not in skip_list:
            quant_order.append(w1_thing)
        else:
            attrgetter(w1_name)(layer).weight.requires_grad = False
            print(f'skipping {idx}_{w1_save}')

        # w3 (Llama의 up_proj에 해당)
        # Hessian: 이전 스크립트에서 'expert{i}_w3'로 저장했습니다.
        w3_name = f'block_sparse_moe.experts.{i}.w3'
        w3_save = f'expert{i}_w3'
        w3_hess = f'expert{i}_w3' # 자신의 Hessian 사용
        w3_thing = (w3_name, w3_save, w3_hess, w3_save, 'col')
        
        if f'{idx}_{w3_save}' not in skip_list:
            quant_order.append(w3_thing)
        else:
            attrgetter(w3_name)(layer).weight.requires_grad = False
            print(f'skipping {idx}_{w3_save}')

        # w2 (Llama의 down_proj에 해당)
        # Hessian: 이전 스크립트에서 'expert{i}_w2'로 저장했습니다.
        w2_name = f'block_sparse_moe.experts.{i}.w2'
        w2_save = f'expert{i}_w2'
        w2_hess = f'expert{i}_w2' # 자신의 Hessian 사용
        w2_thing = (w2_name, w2_save, w2_hess, w2_save, 'row')
        
        if f'{idx}_{w2_save}' not in skip_list:
            quant_order.append(w2_thing)
        else:
            attrgetter(w2_name)(layer).weight.requires_grad = False
            print(f'skipping {idx}_{w2_save}')
            
    finetune_mixtral.quantize_finetune_decoder_layer(layer, quant_order, idx, cb, args,
                                             device, pre_orig_emb, orig_emb, attention_mask)
    
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
    
    model = MixtralForCausalLM.from_pretrained(args.base_model,
                                               torch_dtype='auto',
                                               low_cpu_mem_usage=True)

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
    all_config['model_config'].update({'quip_params': quip_params})
    torch.save(all_config, os.path.join(args.save_path, 'config.pt'))

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    glog.info('loaded model')

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
    
    # --- [수정] 슬라이딩 윈도우 어텐션 마스크 생성 ---
    sliding_window = model.config.sliding_window
    if sliding_window is None:
        glog.warning("Sliding window config가 없습니다. Causal mask를 사용합니다.")
        sliding_window = None # _prepare_4d... 함수는 None을 처리합니다.

    attention_mask = _prepare_4d_causal_attention_mask(
        None, (args.batch_size, args.ctx_size),
        orig_emb_cache[0][:args.batch_size], 0,
        sliding_window=sliding_window) # sliding_window 인자 추가
    # --- [수정 완료] ---

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
        ##
        if args.ft_epochs > 0:
            for j in range(args.devset_size // args.batch_size):
                utils.clean()
                # [확인] Llama/Mixtral 모두 position_ids를 받으므로 이 forward pass는 수정 불필요
                orig_emb_cache[cur_device + 1][args.batch_size * j : args.batch_size * (j + 1)] = \
                    model.model.layers[i](
                        orig_emb_cache[cur_device][args.batch_size * j : args.batch_size * (j + 1)].to(cur_device),
                        position_ids=position_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        output_attentions=False)[0].cpu()
        else:
            orig_emb_cache[cur_device + 1] = orig_emb_cache[cur_device]
        model.model.layers[i].cpu()
        position_ids = position_ids.cpu()
        attention_mask = attention_mask.cpu()
        utils.clean()
        glog.info('computed original embedding for layer {} in {}s'.format(i, time.time() - st))

        # --- [수정] quantize_llama_decoder -> quantize_mixtral_decoder ---
        proc_list[cur_device] = (mp.Process(target=quantize_mixtral_decoder, # 함수 이름 변경
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
                                                attention_mask[:args.ft_bs].cpu()
                                            )), i)
        proc_list[cur_device][0].start()
        # --- [수정 완료] ---

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
