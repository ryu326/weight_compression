import argparse
import os
import time

import glog
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from lib import codebook, utils
from lib.utils.unsafe_import import model_from_hf_path
# from model.mixtral import MixtralForCausalLM
# from model.qwen3moe import Qwen3MoeForCausalLM
# from transformers import MixtralForCausalLM as OrigMixtral
try:
    from model.mixtral import MixtralForCausalLM
    from transformers import MixtralForCausalLM as OrigMixtral
except ImportError:
    OrigMixtral = None
    MixtralForCausalLM = None
    
try:
    from model.qwen3moe import Qwen3MoeForCausalLM
    from transformers import Qwen3MoeForCausalLM as OrigQwen3Moe
except ImportError:
    OrigQwen3Moe = None
    Qwen3MoeForCausalLM = None


torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--quantized_path', type=str)
parser.add_argument('--hf_output_path', type=str)
parser.add_argument('--base_model', type=str)


def load_or_copy_quip(parent_module, attr_name, original_module,
                        layer_name_suffix, layer_idx, args, model_config, cpu):
    """
    디스크에서 양자화된 레이어를 로드하고 압축을 풀거나, 
    skip_list에 있으면 원본 레이어로 교체합니다.
    """
    layer_key = f'{layer_idx}_{layer_name_suffix}'
    target_module = getattr(parent_module, attr_name)
    
    if layer_key not in model_config.quip_params['skip_list']:
        quantized_file_path = f'{args.quantized_path}/{layer_key}.pt'
        
        if os.path.exists(quantized_file_path):
            saved_layer = torch.load(quantized_file_path, map_location=cpu)
            
            # 1. 가중치(Weight) 언패킹
            utils.unpack_quip(target_module, saved_layer)
            
            # 2. [추가] 바이어스(Bias) 복사
            # saved_layer에 bias가 있고, 타겟 모듈도 bias를 가지고 있다면 복사
            if 'bias' in saved_layer and saved_layer['bias'] is not None:
                if target_module.bias is not None:
                    target_module.bias.data.copy_(saved_layer['bias'].to(target_module.bias.dtype))
                else:
                    # 원래 바이어스가 없던 레이어에 바이어스가 생기는 경우는 드물지만,
                    # 만약 구조가 바뀌었다면 Parameter로 생성해줘야 함 (보통은 위 if문에서 처리됨)
                    # target_module.bias = torch.nn.Parameter(saved_layer['bias'].to(target_module.weight.dtype))
                    raise
                    
            glog.info(f"Loaded quantized layer: {layer_key}")
        else:
            # 파일이 없으면 에러 발생 (요청사항)
            glog.error(f"Quantized file missing: {quantized_file_path}")
            raise FileNotFoundError(f"Quantized file not found: {quantized_file_path}")
    else:
        # skip_list에 있으면 모듈 자체를 원본으로 교체
        glog.info(f"Skipping {layer_key}, replacing with original module.")
        setattr(parent_module, attr_name, original_module)
# --- [추가 완료] ---


def main(args):
    assert os.path.exists(args.quantized_path)
    saved_config = torch.load(os.path.join(args.quantized_path, 'config.pt'))
    model_config = saved_config['model_config']
    glog.info(f"Model Config: {model_config}")
    
    # 모델 타입 감지
    is_qwen = "qwen" in model_config.model_type.lower()    
    glog.info(f"Detected Model Type: {'Qwen MoE' if is_qwen else 'Mixtral MoE'}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if is_qwen:
        CustomModelClass = Qwen3MoeForCausalLM
        OrigModelClass = OrigQwen3Moe # 또는 OrigQwen2Moe
    else:
        CustomModelClass = MixtralForCausalLM
        OrigModelClass = OrigMixtral

    # [통합] AutoModel 사용 (Qwen3, Mixtral 모두 지원)
    glog.info(f"Loading target model from {args.base_model}...")
    model = CustomModelClass.from_pretrained(args.base_model,
                                               torch_dtype='auto',
                                               low_cpu_mem_usage=True,
                                               config=model_config)
    
    glog.info(f"Loading original model from {args.base_model} for fallback...")
    orig_model = OrigModelClass.from_pretrained(args.base_model,
                                             torch_dtype='auto',
                                             low_cpu_mem_usage=True,
                                             config=model_config,
                                             trust_remote_code=True)

    if model_config.quip_params['skip_list'] is None:
        model_config.quip_params['skip_list'] = []
    
    cpu = torch.device('cpu')
    
    # LM Head & Norm 로드
    if os.path.exists(f'{args.quantized_path}/lmhead.pt'):
        lmhead_data = torch.load(f'{args.quantized_path}/lmhead.pt', map_location=cpu)
        model.lm_head.weight.copy_(lmhead_data['lm_head'].to(model.lm_head.weight.dtype))
        model.model.norm.weight.copy_(lmhead_data['norm'].to(model.model.norm.weight.dtype))
        glog.info("Loaded LM Head and Norm")

    for ii in range(len(model.model.layers)):
        layer = model.model.layers[ii]
        orig_layer = orig_model.model.layers[ii] # 원본 레이어 참조

        # LayerNorm 로드
        if os.path.exists(f'{args.quantized_path}/{ii}_layernorm.pt'):
            ln_data = torch.load(f'{args.quantized_path}/{ii}_layernorm.pt', map_location=cpu)
            layer.input_layernorm.weight.copy_(ln_data['input_layernorm'].to(layer.input_layernorm.weight.dtype))
            layer.post_attention_layernorm.weight.copy_(ln_data['post_attention_layernorm'].to(layer.post_attention_layernorm.weight.dtype))

        # --- 1. 어텐션 레이어 (공통) ---
        load_or_copy_quip(layer.self_attn, 'q_proj', orig_layer.self_attn.q_proj, 'q', ii, args, model_config, cpu)
        load_or_copy_quip(layer.self_attn, 'k_proj', orig_layer.self_attn.k_proj, 'k', ii, args, model_config, cpu)
        load_or_copy_quip(layer.self_attn, 'v_proj', orig_layer.self_attn.v_proj, 'v', ii, args, model_config, cpu)
        load_or_copy_quip(layer.self_attn, 'o_proj', orig_layer.self_attn.o_proj, 'o', ii, args, model_config, cpu)

        # --- 2. MoE 레이어 (분기 처리) ---
        if is_qwen:
            # Qwen: layer.mlp.gate, layer.mlp.experts
            moe_block = layer.mlp
            orig_moe_block = orig_layer.mlp
            gate_attr_name = 'gate' # mlp.gate
            
            # 레이어 이름 매핑 (저장된 파일 suffix -> 실제 속성 이름)
            # 저장시: expert{i}_w1 (gate_proj), expert{i}_w2 (down_proj), expert{i}_w3 (up_proj)
            expert_map = {
                'w1': 'gate_proj',
                'w2': 'down_proj',
                'w3': 'up_proj'
            }
        else:
            # Mixtral: layer.block_sparse_moe.gate, layer.block_sparse_moe.experts
            moe_block = layer.block_sparse_moe
            orig_moe_block = orig_layer.block_sparse_moe
            gate_attr_name = 'gate'
            
            expert_map = {
                'w1': 'w1',
                'w2': 'w2',
                'w3': 'w3'
            }

        # 2-1. Gate 로드
        load_or_copy_quip(moe_block, gate_attr_name, getattr(orig_moe_block, gate_attr_name), 
                          'gate', ii, args, model_config, cpu)
            
        # 2-2. Experts 로드
        # Mixtral은 num_local_experts, Qwen은 num_experts일 수 있음 (안전하게 처리)
        num_experts = getattr(model_config, 'num_local_experts', getattr(model_config, 'num_experts', 0))
        
        for expert_idx in range(num_experts):
            quant_expert_module = moe_block.experts[expert_idx]
            orig_expert_module = orig_moe_block.experts[expert_idx]

            # w1, w2, w3 순회하며 로드
            for suffix, attr_name in expert_map.items():
                load_or_copy_quip(quant_expert_module, attr_name, getattr(orig_expert_module, attr_name),
                                  f'expert{expert_idx}_{suffix}', ii, args, model_config, cpu)
        
        glog.info(f'loaded layer {ii}')
            
    glog.info(f'saving model to {args.hf_output_path}...')
    model.save_pretrained(args.hf_output_path, safe_serialization=True)
    tokenizer.save_pretrained(args.hf_output_path) # 토크나이저도 같이 저장

    del model
    
    # 로드 테스트
    glog.info('Testing load from saved path...')
    model, _ = model_from_hf_path(args.hf_output_path)
    glog.info('successfully loaded hfized model')


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    args = parser.parse_args()
    main(args)