import argparse
import os
import time

import glog
import torch
from transformers import AutoTokenizer

from lib import codebook, utils
from lib.utils.unsafe_import import model_from_hf_path

# --- [수정 1] Llama -> Mixtral ---
# 사용자 정의 모델 클래스를 Mixtral로 변경
from model.mixtral import MixtralForCausalLM
# 원본 Hugging Face 모델도 Mixtral로 변경
from transformers import MixtralForCausalLM as OrigMixtral
# --- [수정 1 완료] ---

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--quantized_path', type=str)
parser.add_argument('--hf_output_path', type=str)
## ryu
parser.add_argument('--base_model', type=str)


# --- [추가] 헬퍼 함수 ---
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
            utils.unpack_quip(target_module, saved_layer)
        else:
            raise
            # glog.warning(f"Quantized file not found: {quantized_file_path}. Copying original weights.")
            # # 파일이 없으면 원본 가중치로 대체
            # target_module.weight.copy_(original_module.weight)
            # if original_module.bias is not None and target_module.bias is not None:
            #     target_module.bias.copy_(original_module.bias)
    else:
        # skip_list에 있으면 모듈 자체를 원본으로 교체
        glog.info(f"Skipping {layer_key}, replacing with original module.")
        setattr(parent_module, attr_name, original_module)
# --- [추가 완료] ---


def main(args):
    assert os.path.exists(args.quantized_path)
    saved_config = torch.load(os.path.join(args.quantized_path, 'config.pt'))
    model_config = saved_config['model_config']
    glog.info(model_config)
    fused = model_config.quip_params.get('fused', True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model) # ryu가 수정한 base_model 사용

    # --- [수정 2] Llama -> Mixtral ---
    # LlamaForCausalLM -> MixtralForCausalLM
    model = MixtralForCausalLM.from_pretrained(args.base_model,
                                               torch_dtype='auto',
                                               low_cpu_mem_usage=True,
                                               config=model_config)
    
    # OrigLlama -> OrigMixtral
    orig_model = OrigMixtral.from_pretrained(args.base_model,
                                             torch_dtype='auto',
                                             low_cpu_mem_usage=True,
                                             config=model_config)
    # --- [수정 2 완료] ---

    if model_config.quip_params['skip_list'] is None:
        model_config.quip_params['skip_list'] = []
    
    cpu = torch.device('cpu')
    if os.path.exists(f'{args.quantized_path}/lmhead.pt'):
        lmhead_data = torch.load(f'{args.quantized_path}/lmhead.pt', map_location=cpu)
        model.lm_head.weight.copy_(lmhead_data['lm_head'].to(model.lm_head.weight.dtype))
        model.model.norm.weight.copy_(lmhead_data['norm'].to(model.model.norm.weight.dtype))

    for ii in range(len(model.model.layers)):
        layer = model.model.layers[ii]
        orig_layer = orig_model.model.layers[ii] # 원본 레이어 참조

        if os.path.exists(f'{args.quantized_path}/{ii}_layernorm.pt'):
            ln_data = torch.load(f'{args.quantized_path}/{ii}_layernorm.pt',
                                 map_location=cpu)
            layer.input_layernorm.weight.copy_(ln_data['input_layernorm'].to(
                layer.input_layernorm.weight.dtype))
            layer.post_attention_layernorm.weight.copy_(
                ln_data['post_attention_layernorm'].to(
                    layer.post_attention_layernorm.weight.dtype))

        # --- [수정 3] 헬퍼 함수를 사용하도록 리팩토링 ---
        
        # 1. 어텐션 레이어
        load_or_copy_quip(layer.self_attn, 'q_proj', orig_layer.self_attn.q_proj,
                            'q', ii, args, model_config, cpu)
        load_or_copy_quip(layer.self_attn, 'k_proj', orig_layer.self_attn.k_proj,
                            'k', ii, args, model_config, cpu)
        load_or_copy_quip(layer.self_attn, 'v_proj', orig_layer.self_attn.v_proj,
                            'v', ii, args, model_config, cpu)
        load_or_copy_quip(layer.self_attn, 'o_proj', orig_layer.self_attn.o_proj,
                            'o', ii, args, model_config, cpu)

        # 2. MoE 라우터 게이트
        load_or_copy_quip(layer.block_sparse_moe, 'gate', orig_layer.block_sparse_moe.gate,
                            'gate', ii, args, model_config, cpu)
            
        # 3. 모든 전문가(Expert) 로드
        num_experts = model_config.num_local_experts
        for expert_idx in range(num_experts):
            quant_expert_module = layer.block_sparse_moe.experts[expert_idx]
            orig_expert_module = orig_layer.block_sparse_moe.experts[expert_idx]

            load_or_copy_quip(quant_expert_module, 'w1', orig_expert_module.w1,
                                f'expert{expert_idx}_w1', ii, args, model_config, cpu)
            load_or_copy_quip(quant_expert_module, 'w2', orig_expert_module.w2,
                                f'expert{expert_idx}_w2', ii, args, model_config, cpu)
            load_or_copy_quip(quant_expert_module, 'w3', orig_expert_module.w3,
                                f'expert{expert_idx}_w3', ii, args, model_config, cpu)
        
        # --- [리팩토링 완료] ---
        # --- [수정 3 완료] ---

        glog.info(f'loaded layer {ii}')
            
    glog.info(f'saving model...')
    model.save_pretrained(args.hf_output_path, safe_serialization=True)

    del model

    model, _ = model_from_hf_path(args.hf_output_path)

    glog.info('successfully loaded hfized model')


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    args = parser.parse_args()
    main(args)