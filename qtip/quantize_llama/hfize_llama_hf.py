import argparse
import os
import torch
import glog
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from lib import utils
from lib.codebook import bitshift
from lib.linear.quantized_linear import QuantizedLinear
from tqdm import tqdm
import time

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--quantized_path', type=str, required=True, help='Path to the quantized checkpoint directory')
parser.add_argument('--hf_output_path', type=str, required=True, help='Path to save the restored HF model')
parser.add_argument('--base_model', type=str, required=True, help='Path or name of the base (original) model')

# --- Helper Functions ---

def get_text_model(model):
    return model.language_model if hasattr(model, 'language_model') else model.model


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
    LLaMA 수정: MoE Router용 예외 처리(td_x // 2)를 제거했습니다.
    """
    td_x = quip_params['td_x']
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
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    # start_time = time.time()
    quant_layer.codebook_class.cache_hatW(quant_layer.trellis, quant_layer.had_left,
                                       quant_layer.had_right, quant_layer.K_left,
                                       quant_layer.K_right, len(quant_layer.SV),
                                       len(quant_layer.SU), quant_layer.rcp,
                                       quant_layer.tp_rank)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # glog.info(f"{layer_name} Total Compression Time: {elapsed_time*1000:.4f} ms")
    
    hatW = quant_layer.codebook_class.hatW
    SU = quant_layer.SU
    SV = quant_layer.SV
    scale = quant_layer.codebook_class.scale

    target_dtype = hatW.dtype     
    SV = SV.to(target_dtype)
    SU = SU.to(target_dtype)

    W_reconstructed = torch.diag(SV * scale) @ hatW @ torch.diag(SU)

    return W_reconstructed

def load_proj_or_restore(module, attr_name, idx, layer_suffix, path_prefix, skip_list, quip_params):
    full_key = f'{idx}_{layer_suffix}'
    
    # module 내에 해당 속성(attr_name)이 있는지 확인
    if not hasattr(module, attr_name):
        glog.warning(f"Attribute {attr_name} not found in layer {idx}. Skipping.")
        return

    target_layer = getattr(module, attr_name)
    
    if full_key not in skip_list:
        filepath = f'{path_prefix}/{full_key}.pt'
        if not os.path.exists(filepath):
            glog.error(f"File not found: {filepath}")
            raise FileNotFoundError(filepath)
            
        saved = torch.load(filepath, map_location='cpu', weights_only=False)
        
        # layer_name 인자는 get_What 내부 로직용이지만, 수정된 버전에서는 단순 로깅/참조용입니다.
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
    
    if model_config.quip_params.get('skip_list') is None:
        model_config.quip_params['skip_list'] = []
    
    glog.info(f"Model Config Loaded. Type: {model_config.model_type}")

    # 2. Base Model 로드
    glog.info(f"Loading base model from {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype='auto',
        low_cpu_mem_usage=True,
        trust_remote_code=True, 
        device_map="cpu" # 복원 중 OOM 방지를 위해 CPU 유지
    )
    text_model = get_text_model(model)

    # 3. LM Head & Global Norm 복원
    lmhead_path = f'{args.quantized_path}/lmhead.pt'
    if os.path.exists(lmhead_path):
        lmhead_data = torch.load(lmhead_path, map_location='cpu')
        model.lm_head.weight.data.copy_(lmhead_data['lm_head'].to(model.lm_head.weight.dtype))
        
        # LLaMA/Gemma3 text model norm
        norm_module = text_model.norm if hasattr(text_model, 'norm') else getattr(model, 'norm', None)
        if norm_module:
             norm_module.weight.data.copy_(lmhead_data['norm'].to(norm_module.weight.dtype))
        glog.info("Loaded LM Head and Final Norm")

    # 4. Layer별 복원 루프 (LLaMA Dense Structure)
    quip_params = model_config.quip_params
    num_layers = len(text_model.layers)
    
    pbar = tqdm(range(num_layers), desc="Restoring Layers")
    for ii in pbar:
        layer = text_model.layers[ii]
        
        # 4-1. LayerNorm 복원
        ln_path = f'{args.quantized_path}/{ii}_layernorm.pt'
        if os.path.exists(ln_path):
            ln_data = torch.load(ln_path, map_location='cpu')
            # LLaMA: input_layernorm, post_attention_layernorm
            if hasattr(layer, 'input_layernorm'):
                layer.input_layernorm.weight.copy_(ln_data['input_layernorm'].to(layer.input_layernorm.weight.dtype))
            if hasattr(layer, 'post_attention_layernorm'):
                layer.post_attention_layernorm.weight.copy_(ln_data['post_attention_layernorm'].to(layer.post_attention_layernorm.weight.dtype))
            if hasattr(layer, 'pre_feedforward_layernorm') and 'pre_feedforward_layernorm' in ln_data:
                layer.pre_feedforward_layernorm.weight.copy_(
                    ln_data['pre_feedforward_layernorm'].to(layer.pre_feedforward_layernorm.weight.dtype))
            if hasattr(layer, 'post_feedforward_layernorm') and 'post_feedforward_layernorm' in ln_data:
                layer.post_feedforward_layernorm.weight.copy_(
                    ln_data['post_feedforward_layernorm'].to(layer.post_feedforward_layernorm.weight.dtype))

        # 4-2. Self Attention 복원
        # 일반적으로 QuIP 체크포인트에서 suffix는 q, k, v, o 로 저장됨
        load_proj_or_restore(layer.self_attn, 'q_proj', ii, 'q', args.quantized_path, quip_params['skip_list'], quip_params)
        load_proj_or_restore(layer.self_attn, 'k_proj', ii, 'k', args.quantized_path, quip_params['skip_list'], quip_params)
        load_proj_or_restore(layer.self_attn, 'v_proj', ii, 'v', args.quantized_path, quip_params['skip_list'], quip_params)
        load_proj_or_restore(layer.self_attn, 'o_proj', ii, 'o', args.quantized_path, quip_params['skip_list'], quip_params)

        # 4-3. MLP 복원 (LLaMA: gate_proj, up_proj, down_proj)
        # 일반적으로 QuIP 체크포인트에서 Dense MLP는 gate, up, down (혹은 w1, w3, w2) suffix 사용
        # 여기서는 가장 표준적인 gate, up, down suffix를 가정합니다.
        if hasattr(layer, 'mlp'):
            load_proj_or_restore(layer.mlp, 'gate_proj', ii, 'gate', args.quantized_path, quip_params['skip_list'], quip_params)
            load_proj_or_restore(layer.mlp, 'up_proj',   ii, 'up',   args.quantized_path, quip_params['skip_list'], quip_params)
            load_proj_or_restore(layer.mlp, 'down_proj', ii, 'down', args.quantized_path, quip_params['skip_list'], quip_params)

        if (ii + 1) % 1 == 0:
            glog.info(f'Loaded layer {ii + 1}/{num_layers}')

    # 5. 저장
    glog.info(f'Saving restored model to {args.hf_output_path}...')
    model.save_pretrained(args.hf_output_path, safe_serialization=True)
    tokenizer.save_pretrained(args.hf_output_path)
    glog.info('Done.')

if __name__ == '__main__':
    torch.manual_seed(0)
    args = parser.parse_args()
    main(args)
