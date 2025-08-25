import argparse
import os
import time
import json 

import glog
import torch
from transformers import AutoTokenizer

# BitshiftLinear 클래스를 가져오기 위해 임포트 추가
from lib.codebook import bitshift
from lib import codebook, utils
from lib.utils.unsafe_import import model_from_hf_path
from model.llama import LlamaForCausalLM
from transformers import LlamaForCausalLM as OrigLlama

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--quantized_path', type=str)
parser.add_argument('--hf_output_path', type=str)
## ryu
parser.add_argument('--base_model', type=str)
parser.add_argument('--output_path', type=str)

# torch.compile 관련 오류 방지를 위해 dynamo 비활성화
torch._dynamo.disable()

def get_and_calc_mse(quant_layer, orig_layer_weight):
    # 1. cache_hatW를 호출하여 중간 단계의 hatW (하다마드 변환까지 적용된)를 계산합니다.
    quant_layer.codebook_class.cache_hatW(quant_layer.trellis, quant_layer.had_left,
                                       quant_layer.had_right, quant_layer.K_left,
                                       quant_layer.K_right, len(quant_layer.SV),
                                       len(quant_layer.SU), quant_layer.rcp,
                                       quant_layer.tp_rank)
    hatW = quant_layer.codebook_class.hatW

    # 2. 최종 가중치 복원을 위해 SU, SV, scale 스케일링 값들을 가져옵니다.
    SU = quant_layer.SU
    SV = quant_layer.SV
    # import ipdb; ipdb.set_trace()
    
    scale = quant_layer.codebook_class.scale

    # 3. 최종 가중치 W_reconstructed를 복원합니다.
    # 이 가중치가 원본 가중치 W와 동일한 스케일을 갖습니다.
    # W_reconstructed = diag(SV * scale) @ hatW @ diag(SU)
    # W_reconstructed = (SV * scale).unsqueeze(1) * hatW * SU.unsqueeze(0)
    W_reconstructed = torch.diag(SV * scale) @ hatW @ torch.diag(SU)

    # 4. 원본 가중치(W_orig)와 완전히 복원된 가중치(W_reconstructed) 사이의 MSE를 계산합니다.
    W_orig = orig_layer_weight
    mse = torch.mean((W_orig - W_reconstructed) ** 2).item()
    
    return mse


def initialize_codebook(quant_layer):
    """
    forward pass를 호출하지 않고 codebook_class를 수동으로 초기화합니다.
    QuantizedLinear 모듈의 forward 메서드에 있는 초기화 로직을 모방합니다.
    """
    if not hasattr(quant_layer, 'built_codebook_class') or not quant_layer.built_codebook_class:
        quant_layer.codebook_class = bitshift.BitshiftLinear(
            quant_layer.td_x, quant_layer.td_y, quant_layer.L,
            quant_layer.K, quant_layer.V, quant_layer.tlut_bits,
            quant_layer.decode_mode, dtype=quant_layer.dtype,
            tlut=quant_layer.tlut, has_kernel=quant_layer.has_kernel
        )
        # rcp 텐서를 파이썬 float 값으로 변환 (사용자 제공 코드 기준)
        rcp = quant_layer.rcp.item()
        del quant_layer.rcp
        quant_layer.rcp = rcp
            
        quant_layer.built_codebook_class = True


def main(args):
    assert os.path.exists(args.quantized_path)
    saved_config = torch.load(os.path.join(args.quantized_path, 'config.pt'))
    model_config = saved_config['model_config']
    # glog.info(model_config)
    fused = model_config.quip_params.get('fused', True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # glog.info(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_config._name_or_path)

    # --- *** 수정된 부분 (1) *** ---
    # 모델을 CPU에 먼저 로드하여 VRAM을 절약합니다.
    # glog.info("Loading models to CPU...")
    model = LlamaForCausalLM.from_pretrained(args.base_model,
                                        torch_dtype='auto',
                                        low_cpu_mem_usage=True,
                                        config=model_config)
    
    orig_model = OrigLlama.from_pretrained(args.base_model,
                                        torch_dtype='auto',
                                        low_cpu_mem_usage=True,
                                        config=model_config)

    comp_result = {}

    if model_config.quip_params['skip_list'] is None:
        model_config.quip_params['skip_list'] = []
    
    # lm_head와 norm 레이어는 CPU에서 처리합니다.
    if os.path.exists(f'{args.quantized_path}/lmhead.pt'):
        lmhead_data = torch.load(f'{args.quantized_path}/lmhead.pt',
                                 map_location='cpu')
        model.lm_head.weight.copy_(lmhead_data['lm_head'].to(
            model.lm_head.weight.dtype))
        model.model.norm.weight.copy_(lmhead_data['norm'].to(
            model.model.norm.weight.dtype))

    for ii in range(len(model.model.layers)):
        # --- *** 수정된 부분 (2) *** ---
        # 현재 레이어만 GPU로 이동합니다.
        # glog.info(f"Moving layer {ii} to {device} for processing...")
        model.model.layers[ii] = model.model.layers[ii].to(device)
        orig_model.model.layers[ii] = orig_model.model.layers[ii].to(device)
        layer = model.model.layers[ii]
        orig_layer = orig_model.model.layers[ii]

        if os.path.exists(f'{args.quantized_path}/{ii}_layernorm.pt'):
            # layernorm 데이터는 GPU에 있는 레이어에 직접 복사합니다.
            ln_data = torch.load(f'{args.quantized_path}/{ii}_layernorm.pt',
                                 map_location=device)
            layer.input_layernorm.weight.copy_(ln_data['input_layernorm'].to(
                layer.input_layernorm.weight.dtype))
            layer.post_attention_layernorm.weight.copy_(
                ln_data['post_attention_layernorm'].to(
                    layer.post_attention_layernorm.weight.dtype))

        layer_map = {
            'q': (layer.self_attn, 'q_proj', orig_layer.self_attn),
            'k': (layer.self_attn, 'k_proj', orig_layer.self_attn),
            'v': (layer.self_attn, 'v_proj', orig_layer.self_attn),
            'o': (layer.self_attn, 'o_proj', orig_layer.self_attn),
            'up': (layer.mlp, 'up_proj', orig_layer.mlp),
            'gate': (layer.mlp, 'gate_proj', orig_layer.mlp),
            'down': (layer.mlp, 'down_proj', orig_layer.mlp),
        }

        for name, (sub_module, attr_name, orig_sub_module) in layer_map.items():
            layer_key = f'{ii}_{name}'
            
            if layer_key not in model_config.quip_params['skip_list']:
                # --- *** 수정된 부분 (3) *** ---
                # 레이어 데이터를 CPU가 아닌 GPU로 바로 로드합니다.
                saved_layer_data = torch.load(f'{args.quantized_path}/{layer_key}.pt',
                                         map_location=device)
                
                proj_layer = getattr(sub_module, attr_name)
                orig_proj_layer = getattr(orig_sub_module, attr_name)
                
                utils.unpack_quip(proj_layer, saved_layer_data)
                
                proj_layer.has_kernel = False
                initialize_codebook(proj_layer)
                
                mse = get_and_calc_mse(proj_layer, orig_proj_layer.weight.data)
                
                comp_result[f'{layer_key}.pt'] = {'mse': mse, 'num_pixels': orig_proj_layer.weight.data.numel()}
                glog.info(f'{layer_key}, {mse}')
                # print(layer_key, mse)
            else:
                orig_proj_layer = getattr(orig_sub_module, attr_name)
                setattr(sub_module, attr_name, orig_proj_layer)
        
        # --- *** 수정된 부분 (4) *** ---
        # 처리가 끝난 레이어를 다시 CPU로 보내고 GPU 캐시를 비웁니다.
        # glog.info(f"Moving layer {ii} back to CPU to free VRAM...")
        model.model.layers[ii] = model.model.layers[ii].to('cpu')
        orig_model.model.layers[ii] = orig_model.model.layers[ii].to('cpu')
        torch.cuda.empty_cache()
        # del model.model.layers[ii], orig_model.model.layers[ii]
        # glog.info(f'loaded and processed layer {ii}')

    mse_file_path = f'{args.output_path}_MSE.json'
    with open(mse_file_path, 'w') as f:
        json.dump(comp_result, f, indent=2)
    glog.info(f"MSE results saved to {mse_file_path}")
            
    # 아래 부분은 사용하지 않는 것으로 보여 주석 처리 유지합니다.
    # glog.info(f'saving model...')
    # model.save_pretrained(args.hf_output_path, safe_serialization=True)
    # del model
    # model, _ = model_from_hf_path(args.hf_output_path)
    # glog.info('successfully loaded hfized model')


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    args = parser.parse_args()
    main(args)