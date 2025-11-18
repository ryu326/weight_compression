import argparse
import os
import time
import json
import glog
import torch
from torch import nn
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

try:
    from transformers import MixtralForCausalLM
except ImportError:
    MixtralForCausalLM = None

try:
    from transformers import Qwen3MoeForCausalLM
except ImportError:
    Qwen3MoeForCausalLM = None

from lib import utils

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--quantized_path', type=str)
parser.add_argument('--hf_output_path', type=str)
parser.add_argument('--skip_list', type=str, default='')
parser.add_argument('--use_codes', action='store_true')
parser.add_argument('--W_key', type=str, default='')
parser.add_argument('--sep_rnorm', action='store_true')


def get_model_specific_structure(model_type, layer, layer_idx):
    """
    모델 타입에 따라 로딩해야 할 가중치 리스트를 반환합니다.
    반환 형식: list of (save_key_suffix, parent_module, proj_attr_name)
    
    save_key_suffix: 파일명에 사용된 접미사 (예: 'q', 'expert0_w1')
    parent_module: 해당 레이어를 포함하는 부모 모듈 객체
    proj_attr_name: 부모 모듈 내의 속성 이름 (예: 'q_proj', 'gate_proj')
    """
    # 1. Self Attention (모든 모델 공통)
    target_modules = [
        ('q', layer.self_attn, 'q_proj'),
        ('k', layer.self_attn, 'k_proj'),
        ('v', layer.self_attn, 'v_proj'),
        ('o', layer.self_attn, 'o_proj'),
    ]

    model_type = model_type.lower()

    # 2. MLP / MoE Structure (모델별 분기)
    if 'mixtral' in model_type:
        # --- Mixtral Structure ---
        # Router
        target_modules.append(('gate', layer.block_sparse_moe, 'gate')) 
        
        # Experts
        # 저장 파일 형식: {layer}_expert{i}_{w1|w2|w3}.pt
        # Mixtral 실제 이름: experts[i].w1, w2, w3
        num_experts = len(layer.block_sparse_moe.experts)
        for i in range(num_experts):
            expert = layer.block_sparse_moe.experts[i]
            target_modules.append((f'expert{i}_w1', expert, 'w1')) 
            target_modules.append((f'expert{i}_w3', expert, 'w3')) 
            target_modules.append((f'expert{i}_w2', expert, 'w2')) 

    elif 'qwen' in model_type:
        # --- Qwen MoE Structure ---
        # Qwen2/3 MoE: mlp.gate, mlp.experts
        mlp = layer.mlp
        
        # Router
        target_modules.append(('gate', mlp, 'gate')) 

        # Experts
        # 저장 파일 형식: {layer}_expert{i}_{w1|w2|w3}.pt (통일됨)
        # Qwen 실제 이름: gate_proj(w1), up_proj(w3), down_proj(w2)
        
        if hasattr(mlp, 'experts'):
            num_experts = len(mlp.experts)
            for i in range(num_experts):
                expert = mlp.experts[i]
                # Mapping: 저장된 w1 -> 모델의 gate_proj
                target_modules.append((f'expert{i}_w1', expert, 'gate_proj'))
                # Mapping: 저장된 w3 -> 모델의 up_proj
                target_modules.append((f'expert{i}_w3', expert, 'up_proj'))
                # Mapping: 저장된 w2 -> 모델의 down_proj
                target_modules.append((f'expert{i}_w2', expert, 'down_proj'))
    
    # (Llama 등 Dense 모델은 요청에 따라 제외됨)

    return target_modules


def main(args):
    assert os.path.exists(args.quantized_path)
    saved_config = torch.load(os.path.join(args.quantized_path, 'config.pt'), weights_only=False)
    model_config = saved_config['model_config']
    comp_config = saved_config['quant_args']
    
    # 모델 타입 확인
    model_type = getattr(model_config, 'model_type', 'llama')
    glog.info(f"Detected Model Type: {model_type}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_config._name_or_path, trust_remote_code=True)
    
    # 모델 클래스 선택
    model_cls = AutoModelForCausalLM
    
    if args.sep_rnorm:
        # sep_rnorm 사용 시 커스텀 클래스 우선 사용
        if 'mixtral' in model_type and MixtralForCausalLM:
            model_cls = MixtralForCausalLM
        elif 'qwen' in model_type and Qwen3MoeForCausalLM:
            model_cls = Qwen3MoeForCausalLM

    glog.info(f"Loading model with class: {model_cls.__name__}")

    model = model_cls.from_pretrained(model_config._name_or_path,
                                      torch_dtype='auto',
                                      low_cpu_mem_usage=True,
                                      config=model_config,
                                      trust_remote_code=True)

    orig_model = AutoModelForCausalLM.from_pretrained(model_config._name_or_path,
                                           torch_dtype='auto',
                                           low_cpu_mem_usage=True,
                                           config=model_config,
                                           trust_remote_code=True)

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

    cpu = torch.device('cpu')
    
    # LM Head 로딩
    if os.path.exists(f'{args.quantized_path}/lmhead.pt'):
        lmhead_data = torch.load(f'{args.quantized_path}/lmhead.pt',
                                 map_location=cpu, weights_only=False)
        
        # 모델 구조에 따라 lm_head 위치가 다를 수 있음 (보통 model.lm_head)
        if hasattr(model, 'lm_head'):
            model.lm_head.weight.copy_(lmhead_data['lm_head'].to(model.lm_head.weight.dtype))
        
        # Norm 로딩 (model.model.norm 또는 model.norm)
        target_norm = getattr(model.model, 'norm', getattr(model, 'norm', None))
        if target_norm is not None:
            target_norm.weight.copy_(lmhead_data['norm'].to(target_norm.weight.dtype))
            
        glog.info("Loaded LM Head and Norm")
    else:
        glog.info("lmhead.pt not found. Keeping original heads/norms.")

    # Layers 순회
    layers = model.model.layers
    
    for ii in range(len(layers)):
        layer = layers[ii]

        # LayerNorm 로딩
        if os.path.exists(f'{args.quantized_path}/{ii}_layernorm.pt'):
            ln_data = torch.load(f'{args.quantized_path}/{ii}_layernorm.pt',
                                 map_location=cpu, weights_only=False)
            layer.input_layernorm.weight.copy_(ln_data['input_layernorm'].to(
                layer.input_layernorm.weight.dtype))
            layer.post_attention_layernorm.weight.copy_(
                ln_data['post_attention_layernorm'].to(
                    layer.post_attention_layernorm.weight.dtype))
        
        # --- 모델 타입에 따른 타겟 모듈 가져오기 ---
        target_modules = get_model_specific_structure(model_type, layer, ii)
        
        for name_suffix, submodule, proj_name in target_modules:
            skip_key = f'{ii}_{name_suffix}' # 예: 0_q, 0_expert0_w1
            
            # Skip 리스트에 없고 파일이 존재하면 로드
            if skip_key not in skip_list:
                file_path = f'{args.quantized_path}/{skip_key}.pt'
                
                if not os.path.exists(file_path):
                    continue

                saved_layer = torch.load(file_path, map_location=cpu, weights_only=False)
                proj_layer = getattr(submodule, proj_name)
                
                # 결과 통계 수집
                if isinstance(saved_layer, dict):
                    comp_result[f'{skip_key}.pt'] = {k: v for k, v in saved_layer.items() 
                                                    if not isinstance(v, torch.Tensor) and k not in ['codes', 'metadata']}
                    comp_result['bpp_loss'] += saved_layer.get('bpp_loss_sum' + args.W_key, 0)
                    comp_result['num_pixels'] += saved_layer.get('num_pixels', 0)
                    comp_result['bpp'] += saved_layer.get('bpp_sum', 0)
                
                # 가중치 복원
                if args.sep_rnorm:
                    try:
                        # 커스텀 CompLinear 레이어인 경우
                        proj_layer.Wr.copy_(saved_layer['hatWr'])
                        proj_layer.row_norm.copy_(saved_layer['metadata']['row_std'])
                    except Exception:
                        # 일반 Linear 레이어인데 sep_rnorm 옵션이 켜진 경우 (Fallback)
                        if hasattr(proj_layer, 'Wr'):
                             proj_layer.Wr.copy_(saved_layer['W_hat'])
                             proj_layer.row_norm = nn.Parameter(saved_layer['row_norm'])
                        else:
                             W_hat = saved_layer.get('W_hat' + args.W_key)
                             if W_hat is None: 
                                 W_hat = utils.de_standardize_Wr(saved_layer['hatWr'], saved_layer['metadata'], comp_config)
                             proj_layer.weight.copy_(W_hat.to(proj_layer.weight.dtype))
                else:
                    W_hat = saved_layer.get('W_hat' + args.W_key)
                    if W_hat is None:
                        # W_hat이 없으면 압축된 hatWr와 metadata로 복원
                        W_hat = utils.de_standardize_Wr(saved_layer['hatWr'], saved_layer['metadata'], comp_config)
                    
                    # 복원된 가중치 복사
                    proj_layer.weight.copy_(W_hat.to(proj_layer.weight.dtype))
                    
                    # [중요] Bias 복사 (요청 반영)
                    if 'bias' in saved_layer and saved_layer['bias'] is not None:
                        if proj_layer.bias is not None:
                            proj_layer.bias.data.copy_(saved_layer['bias'].to(proj_layer.bias.dtype))
                        else:
                            # 원래 bias가 없던 레이어라면 Parameter 생성
                            proj_layer.bias = nn.Parameter(saved_layer['bias'].to(proj_layer.weight.dtype))

            else:
                glog.info(f'### skipping {skip_key} ###')
                # Skip된 경우 원본 가중치 유지 (이미 로드된 orig_model이나 model 상태 유지)
                # 여기서는 model이 이미 원본 가중치로 초기화되어 있거나, 
                # 이전 단계에서 로드되었으므로 별도 조치가 필요 없으나,
                # 만약 명시적 복구가 필요하다면 orig_model에서 복사해야 함.
                pass

        glog.info(f'loaded layer {ii}')
        
    if comp_result['num_pixels'] > 0:
        comp_result['bpp_loss'] = comp_result['bpp_loss'] / comp_result['num_pixels']
        comp_result['bpp'] = comp_result['bpp'] / comp_result['num_pixels']
    
    # 텐서를 스칼라로 변환
    if isinstance(comp_result['bpp_loss'], torch.Tensor):
        comp_result['bpp_loss'] = comp_result['bpp_loss'].item()
    if isinstance(comp_result['bpp'], torch.Tensor):
        comp_result['bpp'] = comp_result['bpp'].item()

    glog.info(f'saving model to {args.hf_output_path}...')
    model.save_pretrained(args.hf_output_path, safe_serialization=True)
    tokenizer.save_pretrained(args.hf_output_path)
    del model
    
    file_path = f'{args.hf_output_path}_result.json'
    if os.path.exists(file_path):
        os.rename(file_path, f'{args.hf_output_path}_result_.json')
    with open(file_path, 'w') as f:
        json.dump(comp_result, f, indent=2)

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    args = parser.parse_args()
    main(args)