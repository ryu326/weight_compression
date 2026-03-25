import argparse
import os
import time
import json
import glog
import torch
from torch import nn
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from tqdm import tqdm

try:
    from transformers import MixtralForCausalLM
except ImportError:
    MixtralForCausalLM = None

try:
    from transformers import Qwen3MoeForCausalLM
except ImportError:
    Qwen3MoeForCausalLM = None

try:
    from model.gptoss_standard_moe_v11 import GptOssForCausalLM as GptOssForCausalLM_v11
    from transformers import GptOssForCausalLM
except ImportError:
    GptOssForCausalLM = None

from lib import utils

from lib.utils.load_hf import model_from_hf_path_gptoss


parser = argparse.ArgumentParser()
parser.add_argument('--quantized_path', type=str, required=True, help='Path to the quantized checkpoint directory')
parser.add_argument('--hf_output_path', type=str, required=True, help='Path to save the restored HF model')
parser.add_argument('--base_model', type=str, required=True, help='Path or name of the base (original) model')
parser.add_argument('--skip_list', type=str, default='')



def get_What(saved_layer_data, comp_config):
    W_hat = saved_layer_data.get('W_hat')
    if W_hat is None:
        W_hat = utils.de_standardize_Wr(saved_layer_data['hatWr'], saved_layer_data['metadata'], comp_config)
    return W_hat

def load_proj_or_restore(module, attr_name, idx, layer_suffix, path_prefix, skip_list, comp_config, comp_result, orig_module = None):
    full_key = f'{idx}_{layer_suffix}'
    target_layer = getattr(module, attr_name)
    
    if full_key not in skip_list:
        filepath = f'{path_prefix}/{full_key}.pt'
        if not os.path.exists(filepath):
            glog.error(f"File not found: {filepath}")
            raise FileNotFoundError(filepath)
            
        saved = torch.load(filepath, map_location='cpu', weights_only=False)
        
        comp_result[filepath] = {k: v for k, v in saved.items() 
                if not isinstance(v, torch.Tensor) and k not in ['codes', 'metadata']}
        comp_result['bpp_loss'] += saved.get('bpp_loss_sum', 0)
        comp_result['num_pixels'] += saved.get('num_pixels', 0)
        comp_result['bpp'] += saved.get('bpp_sum', 0)
        
        W_hat = get_What(saved, comp_config)
        
        target_layer.weight.data.copy_(W_hat.to(target_layer.weight.dtype))
        
        if 'bias' in saved and saved['bias'] is not None:
             if target_layer.bias is not None:
                 target_layer.bias.data.copy_(saved['bias'].to(target_layer.bias.dtype))
                 
        if orig_module is not None:
            orig_layer = getattr(orig_module, attr_name)
            target_layer.bias.data.copy_(orig_layer.bias.data.to(target_layer.bias.dtype))   
    else:
        pass


# --- Main Logic ---

def main(args):
    with torch.no_grad():
        if not os.path.exists(args.quantized_path):
            raise FileNotFoundError(f"Quantized path not found: {args.quantized_path}")

        # 1. Config 로드
        saved_config = torch.load(os.path.join(args.quantized_path, 'config.pt'), weights_only=False)
        model_config = saved_config['model_config']
        comp_config = saved_config['quant_args']
        
            
        skip_list = args.skip_list.split(',') if args.skip_list else []
        glog.info(f'Skip list: {skip_list}')
        
        comp_result = {
            'bpp_loss': 0,
            'bpp': 0,
            'ppl': 0,
            'num_pixels': 0
        }
        try:
            with open(os.path.join(args.quantized_path, 'config.json'), 'r') as f:
                saved_config_json = json.load(f)
            comp_result['config'] = saved_config_json
        except Exception:
            pass
        
        glog.info(f"Model Config Loaded. Type: {model_config.model_type}")
        is_qwen = "qwen" in model_config.model_type.lower()
        is_gpt_oss = "gpt_oss" in model_config.model_type.lower()
        glog.info(f"Detected Model Architecture: {'GPT-OSS' if is_gpt_oss else ('Qwen MoE' if is_qwen else 'Mixtral/Other MoE')}")

        # 3. Base Model 로드 (원본 가중치 포함)
        # 커스텀 클래스 대신 AutoModel 사용. 원본 가중치를 로드한 상태에서 시작.
        glog.info(f"Loading base model from {args.base_model}...")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
        
        if is_gpt_oss:
            model = GptOssForCausalLM_v11.from_pretrained(
                args.base_model,
                torch_dtype='auto',
                low_cpu_mem_usage=True,
                trust_remote_code=True, # Qwen 등 일부 모델은 필요할 수 있음
                device_map="cpu" # 복원 중에는 CPU 사용 권장 (OOM 방지)
            )
            orig_model, _ = model_from_hf_path_gptoss(args.base_model, device_map='cpu',
                                                sep_rnorm = False, 
                                                gptoss_replace_version='v1.1')      
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.base_model,
                torch_dtype='auto',
                low_cpu_mem_usage=True,
                trust_remote_code=True, # Qwen 등 일부 모델은 필요할 수 있음
                device_map="cpu" # 복원 중에는 CPU 사용 권장 (OOM 방지)
            )

        # 4. LM Head & Global Norm 복원
        lmhead_path = f'{args.quantized_path}/lmhead.pt'
        if os.path.exists(lmhead_path):
            lmhead_data = torch.load(lmhead_path, map_location='cpu')
            model.lm_head.weight.data.copy_(lmhead_data['lm_head'].to(model.lm_head.weight.dtype))
            # HF 모델은 보통 model.norm 또는 model.model.norm에 위치
            norm_module = model.model.norm if hasattr(model.model, 'norm') else getattr(model, 'norm', None)
            if norm_module:
                norm_module.weight.data.copy_(lmhead_data['norm'].to(norm_module.weight.dtype))
            glog.info("Loaded LM Head and Final Norm")

        
        # Qwen vs Mixtral 구조 매핑 설정
        if is_qwen:
            # Qwen3/2 MoE: layer.mlp.gate, layer.mlp.experts (gate_proj, down_proj, up_proj)
            moe_attr = 'mlp' 
            # gate_attr = 'gate' # mlp.gate
            # 저장된 파일 suffix -> 모델 속성명 매핑
            expert_proj_map = {
                'w1': 'gate_proj',  # Qwen Gate
                'w2': 'down_proj',  # Qwen Down
                'w3': 'up_proj'     # Qwen Up
            }
        elif is_gpt_oss:
            # GPT-OSS
            moe_attr = 'mlp'
            # GPT-OSS는 expert_proj_map이 다름 (w1,w3 -> gate_up)
            expert_proj_map = {
                'gate_up': 'gate_up_proj', # saved_suffix : model_attr
                'down': 'down_proj'
            }
        else:
            # Mixtral: layer.block_sparse_moe.gate, layer.block_sparse_moe.experts (w1, w2, w3)
            moe_attr = 'block_sparse_moe'
            # gate_attr = 'gate'
            expert_proj_map = {
                'w1': 'w1',
                'w2': 'w2',
                'w3': 'w3'
            }

        num_layers = len(model.model.layers)
        pbar = tqdm(range(num_layers), desc="Restoring Layers")
        for ii in pbar:
            layer = model.model.layers[ii]
            
            # 5-1. LayerNorm 복원
            ln_path = f'{args.quantized_path}/{ii}_layernorm.pt'
            if os.path.exists(ln_path):
                ln_data = torch.load(ln_path, map_location='cpu')
                # load_layernorm(layer, ln_data)
                layer.input_layernorm.weight.copy_(ln_data['input_layernorm'].to(layer.input_layernorm.weight.dtype))
                layer.post_attention_layernorm.weight.copy_(ln_data['post_attention_layernorm'].to(layer.post_attention_layernorm.weight.dtype))

            # 5-2. Self Attention 복원 (공통)
            # module, attr_name, idx, suffix, path, skip, params
            load_proj_or_restore(layer.self_attn, 'q_proj', ii, 'q', args.quantized_path, skip_list, comp_config, comp_result)
            load_proj_or_restore(layer.self_attn, 'k_proj', ii, 'k', args.quantized_path, skip_list, comp_config, comp_result)
            load_proj_or_restore(layer.self_attn, 'v_proj', ii, 'v', args.quantized_path, skip_list, comp_config, comp_result)
            load_proj_or_restore(layer.self_attn, 'o_proj', ii, 'o', args.quantized_path, skip_list, comp_config, comp_result)

            # 5-3. MoE Block 복원
            if hasattr(layer, moe_attr):
                moe_block = getattr(layer, moe_attr)
                
                # GPT-OSS는 Router 모듈이 별도로 존재 (mlp.router.gate)
                if is_gpt_oss and hasattr(moe_block, 'router'):
                    orig_moe_block = getattr(orig_model.model.layers[ii], moe_attr)
                    load_proj_or_restore(moe_block.router, 'gate', ii, 'gate', 
                                    args.quantized_path, skip_list, comp_config, comp_result, orig_moe_block.router)
                else:
                    # Qwen/Mixtral은 MoE 블록 바로 아래에 gate 존재 (mlp.gate / block_sparse_moe.gate)
                    load_proj_or_restore(moe_block, 'gate', ii, 'gate', 
                                    args.quantized_path, skip_list, comp_config, comp_result)
                
                # --- [B] Experts 복원 ---
                # GPT-OSS는 experts 모듈 안에 다시 experts 리스트가 있음 (mlp.experts.experts)
                if is_gpt_oss:
                    experts_container = moe_block.experts
                    experts_list = experts_container.experts if hasattr(experts_container, 'experts') else experts_container
                    orig_experts_container = orig_moe_block.experts
                    orig_experts_list = orig_experts_container.experts if hasattr(orig_experts_container, 'experts') else orig_experts_container
                else:
                    experts_list = moe_block.experts
                
                num_experts = len(experts_list)
                
                for expert_idx in range(num_experts):
                    expert_module = experts_list[expert_idx]
                    orig_expert_module = orig_experts_list[expert_idx] if orig_experts_list is not None else None
                    
                    # 각 Expert 내부의 Projection 복원 (모델별 매핑 사용)
                    for suffix, proj_attr in expert_proj_map.items():
                        layer_suffix = f'expert{expert_idx}_{suffix}'
                        load_proj_or_restore(expert_module, proj_attr, ii, layer_suffix, 
                                            args.quantized_path, skip_list, comp_config, comp_result, orig_expert_module)

            if (ii + 1) % 1 == 0:
                glog.info(f'Loaded layer {ii + 1}/{num_layers}')

    # 6. 저장
    
    if comp_result['num_pixels'] > 0:
        comp_result['bpp_loss'] = comp_result['bpp_loss'] / comp_result['num_pixels']
        comp_result['bpp'] = comp_result['bpp'] / comp_result['num_pixels']
    
    if isinstance(comp_result['bpp_loss'], torch.Tensor):
        comp_result['bpp_loss'] = comp_result['bpp_loss'].item()
    if isinstance(comp_result['bpp'], torch.Tensor):
        comp_result['bpp'] = comp_result['bpp'].item()
        
    file_path = f'{args.hf_output_path}_result.json'
    if os.path.exists(file_path):
        os.rename(file_path, f'{args.hf_output_path}_result_.json')
    with open(file_path, 'w') as f:
        json.dump(comp_result, f, indent=2)

    glog.info(f'Saving restored model to {args.hf_output_path}...')
    model.save_pretrained(args.hf_output_path, safe_serialization=True)
    tokenizer.save_pretrained(args.hf_output_path)
    glog.info('Done.')

if __name__ == '__main__':
    torch.manual_seed(0)
    args = parser.parse_args()
    main(args)