import argparse
import os
import torch
import glog
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from lib import utils
from lib.codebook import bitshift
from lib.linear.quantized_linear import QuantizedLinear
from tqdm import tqdm

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--quantized_path', type=str, required=True, help='Path to the quantized checkpoint directory')
parser.add_argument('--hf_output_path', type=str, required=True, help='Path to save the restored HF model')
parser.add_argument('--base_model', type=str, required=True, help='Path or name of the base (original) model')

# --- Helper Functions ---

def has_kernel(decode_mode, L, K, V, tlut_bits, td_x, td_y):
    if decode_mode != 'quantlut_sym': return False
    if L != 16: return False
    if V != 2: return False
    if K < 2 or K > 4: return False
    if tlut_bits != 9: return False
    if td_x != 16 or td_y != 16: return False
    return True

def initialize_codebook(quant_layer):
    assert not hasattr(quant_layer, 'built_codebook_class') or not quant_layer.built_codebook_class
    quant_layer.codebook_class = bitshift.BitshiftLinear(
        quant_layer.td_x, quant_layer.td_y, quant_layer.L,
        quant_layer.K, quant_layer.V, quant_layer.tlut_bits,
        quant_layer.decode_mode, dtype=quant_layer.dtype,
        tlut=quant_layer.tlut, has_kernel=quant_layer.has_kernel
    )
    rcp = quant_layer.rcp.item()
    del quant_layer.rcp
    quant_layer.rcp = rcp
    quant_layer.built_codebook_class = True

def get_What(quip_params, orig_layer_weight, saved_layer_data, layer_name):
    """
    양자화된 데이터를 기반으로 복원된 가중치(W_hat)를 계산하여 반환합니다.
    layer_name이 'gate'인 경우 td_x를 // 2로 처리합니다.
    """
    td_x = quip_params['td_x']
    
    # [수정됨] gate 레이어에 대한 예외 처리
    if layer_name == 'gate':
        td_x = td_x // 2

    td_y = quip_params['td_y']
    L = quip_params['L']
    K = quip_params['K']
    V = quip_params['V']
    tlut_bits = quip_params['tlut_bits']
    decode_mode = quip_params['decode_mode']
    
    # 임시 QuantizedLinear 생성
    quant_layer = QuantizedLinear(orig_layer_weight.shape[1],
                    orig_layer_weight.shape[0],
                    td_x, td_y, L, K, V, tlut_bits, decode_mode,
                    dtype=orig_layer_weight.dtype,
                    bias=True)
    
    quant_layer.mode = 'train-fixW'
    quant_layer.to('cuda') # 계산은 GPU에서 수행
    utils.unpack_quip(quant_layer, saved_layer_data)
    
    quant_layer.has_kernel = has_kernel(decode_mode, L, K, V, tlut_bits, td_x, td_y)
    initialize_codebook(quant_layer)
    
    quant_layer.codebook_class.cache_hatW(quant_layer.trellis, quant_layer.had_left,
                                       quant_layer.had_right, quant_layer.K_left,
                                       quant_layer.K_right, len(quant_layer.SV),
                                       len(quant_layer.SU), quant_layer.rcp,
                                       quant_layer.tp_rank)
    
    hatW = quant_layer.codebook_class.hatW
    SU = quant_layer.SU
    SV = quant_layer.SV
    scale = quant_layer.codebook_class.scale

    target_dtype = hatW.dtype     
    SV = SV.to(target_dtype)
    SU = SU.to(target_dtype)

    W_reconstructed = torch.diag(SV * scale) @ hatW @ torch.diag(SU)

    return W_reconstructed

# def load_layernorm(layer, ln_data):
#     """
#     저장된 LayerNorm 데이터를 실제 모델의 LayerNorm 속성에 매핑하여 로드합니다.
#     Qwen/Mixtral/LLaMA 등 모델마다 속성명이 다를 수 있음을 고려합니다.
#     """
#     # 매핑 정의: {저장된_키: [가능한_모델_속성명_후보]}
#     # 보통 QUIP 저장 포맷은 layer_norm1, layer_norm2를 사용
#     mappings = [
#         ('layer_norm1', ['input_layernorm', 'input_norm', 'attention_norm']),
#         ('layer_norm2', ['post_attention_layernorm', 'post_attention_norm', 'ffn_norm'])
#     ]

#     for key, attr_candidates in mappings:
#         if key in ln_data:
#             src_tensor = ln_data[key]
#             # 모델에서 해당 속성을 찾아서 복사
#             for attr in attr_candidates:
#                 if hasattr(layer, attr):
#                     target_ln = getattr(layer, attr)
#                     target_ln.weight.data.copy_(src_tensor.to(target_ln.weight.dtype))
#                     break

def load_proj_or_restore(module, attr_name, idx, layer_suffix, path_prefix, skip_list, quip_params):
    full_key = f'{idx}_{layer_suffix}'
    target_layer = getattr(module, attr_name)
    
    if full_key not in skip_list:
        filepath = f'{path_prefix}/{full_key}.pt'
        if not os.path.exists(filepath):
            glog.error(f"File not found: {filepath}")
            raise FileNotFoundError(filepath)
            
        saved = torch.load(filepath, map_location='cpu', weights_only=False)
        
        W_hat = get_What(quip_params, target_layer.weight.data, saved, layer_name=layer_suffix)
        
        target_layer.weight.data.copy_(W_hat.to(target_layer.weight.dtype))
        
        if 'bias' in saved and saved['bias'] is not None:
             if target_layer.bias is not None:
                 target_layer.bias.data.copy_(saved['bias'].to(target_layer.bias.dtype))
    else:
        pass


# --- Main Logic ---

def main(args):
    if not os.path.exists(args.quantized_path):
        raise FileNotFoundError(f"Quantized path not found: {args.quantized_path}")

    # 1. Config 로드
    saved_config_path = os.path.join(args.quantized_path, 'config.pt')
    saved_config_data = torch.load(saved_config_path, map_location='cpu')
    model_config = saved_config_data['model_config']
    
    # skip_list 초기화
    if model_config.quip_params.get('skip_list') is None:
        model_config.quip_params['skip_list'] = []
    
    glog.info(f"Model Config Loaded. Type: {model_config.model_type}")

    # 2. 모델 타입 감지 및 구조 설정
    is_qwen = "qwen" in model_config.model_type.lower()
    glog.info(f"Detected Model Architecture: {'Qwen MoE' if is_qwen else 'Mixtral/Other MoE'}")

    # 3. Base Model 로드 (원본 가중치 포함)
    # 커스텀 클래스 대신 AutoModel 사용. 원본 가중치를 로드한 상태에서 시작.
    glog.info(f"Loading base model from {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
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

    # 5. Layer별 복원 루프
    quip_params = model_config.quip_params
    
    # Qwen vs Mixtral 구조 매핑 설정
    if is_qwen:
        # Qwen3/2 MoE: layer.mlp.gate, layer.mlp.experts (gate_proj, down_proj, up_proj)
        moe_attr = 'mlp' 
        gate_attr = 'gate' # mlp.gate
        # 저장된 파일 suffix -> 모델 속성명 매핑
        expert_proj_map = {
            'w1': 'gate_proj',  # Qwen Gate
            'w2': 'down_proj',  # Qwen Down
            'w3': 'up_proj'     # Qwen Up
        }
    else:
        # Mixtral: layer.block_sparse_moe.gate, layer.block_sparse_moe.experts (w1, w2, w3)
        moe_attr = 'block_sparse_moe'
        gate_attr = 'gate'
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
        load_proj_or_restore(layer.self_attn, 'q_proj', ii, 'q', args.quantized_path, quip_params['skip_list'], quip_params)
        load_proj_or_restore(layer.self_attn, 'k_proj', ii, 'k', args.quantized_path, quip_params['skip_list'], quip_params)
        load_proj_or_restore(layer.self_attn, 'v_proj', ii, 'v', args.quantized_path, quip_params['skip_list'], quip_params)
        load_proj_or_restore(layer.self_attn, 'o_proj', ii, 'o', args.quantized_path, quip_params['skip_list'], quip_params)

        # 5-3. MoE Block 복원
        if hasattr(layer, moe_attr):
            moe_block = getattr(layer, moe_attr)
            
            # Gate (Router) 복원
            load_proj_or_restore(moe_block, gate_attr, ii, 'gate', args.quantized_path, quip_params['skip_list'], quip_params)
            
            # Experts 복원
            # experts 리스트 접근
            experts_list = moe_block.experts
            num_experts = len(experts_list)
            
            for expert_idx in range(num_experts):
                expert_module = experts_list[expert_idx]
                
                # 각 Expert 내부의 Projection 복원 (w1, w2, w3)
                for suffix, proj_attr in expert_proj_map.items():
                    layer_suffix = f'expert{expert_idx}_{suffix}'
                    load_proj_or_restore(expert_module, proj_attr, ii, layer_suffix, 
                                         args.quantized_path, quip_params['skip_list'], quip_params)

        if (ii + 1) % 1 == 0:
            glog.info(f'Loaded layer {ii + 1}/{num_layers}')

    # 6. 저장
    glog.info(f'Saving restored model to {args.hf_output_path}...')
    model.save_pretrained(args.hf_output_path, safe_serialization=True)
    tokenizer.save_pretrained(args.hf_output_path)
    glog.info('Done.')

if __name__ == '__main__':
    torch.manual_seed(0)
    args = parser.parse_args()
    main(args)