import argparse
import os
import glog
import torch
# from transformers import CLIPModel as OriCLIP
from transformers import LlavaForConditionalGeneration
import json

from lib.codebook import bitshift
from lib import utils
from lib.utils.unsafe_import import model_from_hf_path

from lib.linear.quantized_linear import QuantizedLinear

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--quantized_path', type=str)
parser.add_argument('--hf_output_path', type=str)
parser.add_argument('--base_model', type=str)
parser.add_argument('--skip_list', type=str)


def has_kernel(decode_mode, L, K, V, tlut_bits, td_x, td_y):
    if decode_mode != 'quantlut_sym':
        return False
    if L != 16:
        return False
    if V != 2:
        return False
    if K < 2 or K > 4:
        return False
    if tlut_bits != 9:
        return False
    if td_x != 16 or td_y != 16:
        return False
    return True


def get_What(quip_params, orig_layer_weight, saved_layer_data):

    td_x = quip_params['td_x']
    td_y = quip_params['td_y']
    L = quip_params['L']
    K = quip_params['K']
    V = quip_params['V']
    tlut_bits = quip_params['tlut_bits']
    decode_mode = quip_params['decode_mode']
    
    quant_layer = QuantizedLinear(orig_layer_weight.shape[1],
                    orig_layer_weight.shape[0],
                    td_x,
                    td_y,
                    L,
                    K,
                    V,
                    tlut_bits,
                    decode_mode,
                    dtype=orig_layer_weight.dtype,
                    bias=True)
    
    quant_layer.mode = 'train-fixW'
    
    quant_layer.to('cuda')
    utils.unpack_quip(quant_layer, saved_layer_data)
    
    quant_layer.has_kernel = has_kernel(
        decode_mode, L, K, V, tlut_bits, td_x, td_y
    )
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

    # 3. 최종 가중치 W_reconstructed를 복원합니다.
    # 이 가중치가 원본 가중치 W와 동일한 스케일을 갖습니다.
    # W_reconstructed = diag(SV * scale) @ hatW @ diag(SU)
    # W_reconstructed = (SV * scale).unsqueeze(1) * hatW * SU.unsqueeze(0)
    W_reconstructed = torch.diag(SV * scale) @ hatW @ torch.diag(SU)

    # # 4. 원본 가중치(W_orig)와 완전히 복원된 가중치(W_reconstructed) 사이의 MSE를 계산합니다.
    # W_orig = orig_layer_weight
    # mse = torch.mean((W_orig - W_reconstructed) ** 2).item()
    return W_reconstructed


def initialize_codebook(quant_layer):
    # if not hasattr(quant_layer, 'built_codebook_class') or not quant_layer.built_codebook_class:
    assert not hasattr(quant_layer, 'built_codebook_class') or not quant_layer.built_codebook_class
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
        
        
def load_layernorm(layer, ln_data):
    if hasattr(layer, 'layer_norm1'):
        layer.layer_norm1.weight.copy_(ln_data['layer_norm1'].to(layer.layer_norm1.weight.dtype))
    if hasattr(layer, 'layer_norm2'):
        layer.layer_norm2.weight.copy_(ln_data['layer_norm2'].to(layer.layer_norm2.weight.dtype))

def load_proj_or_restore(module, key, idx, name, orig_module, path_prefix, skip_list, quip_params):
    full_key = f'{idx}_{name}'
    if full_key not in skip_list:
        saved = torch.load(f'{path_prefix}/{full_key}.pt', map_location='cpu', weights_only=False)
    
        proj_layer = getattr(module, key)
        
        W_hat = get_What(quip_params, proj_layer.weight.data, saved) 
        
        proj_layer.weight.copy_(W_hat.to(proj_layer.weight.dtype))        

    else:
        setattr(module, name, getattr(orig_module, key))
        raise


def main(args):
    assert os.path.exists(args.quantized_path)
    saved_config = torch.load(os.path.join(args.quantized_path, 'config.pt'), weights_only=False)
    model_config = saved_config['model_config']
    quip_params = model_config.quip_params
    
    
    model = LlavaForConditionalGeneration.from_pretrained(args.base_model)
    orig_model = LlavaForConditionalGeneration.from_pretrained(args.base_model)

    skip_list = args.skip_list.split(',') if args.skip_list else []
    comp_result = {}
    comp_result['bpp_loss'] = 0
    comp_result['ppl'] = 0
    comp_result['num_pixels'] = 0
    try:
        with open(os.path.join(args.quantized_path, 'config.json'), 'r') as f:
            saved_config = json.load(f)
        comp_result['config'] = saved_config
    except Exception as e:
        print(f"Failed to load config: {e}")

    cpu = torch.device('cpu')

    def load_clip_block(prefix, layers, orig_layers):
        for i in range(len(layers)):
            layer = layers[i]
            orig = orig_layers[i]
            glog.info(f'Loading {prefix} layer {i}')
            ln_path = f'{args.quantized_path}/{prefix}{i}_layernorm.pt'
            # if os.path.exists(ln_path):
                # ln_data = torch.load(ln_path, map_location=cpu, weights_only=False)
                # load_layernorm(layer, ln_data)

            load_proj_or_restore(layer.self_attn, 'q_proj', f'{prefix}{i}', 'q', orig.self_attn, args.quantized_path, skip_list, quip_params)
            load_proj_or_restore(layer.self_attn, 'k_proj', f'{prefix}{i}', 'k', orig.self_attn, args.quantized_path, skip_list, quip_params)
            load_proj_or_restore(layer.self_attn, 'v_proj', f'{prefix}{i}', 'v', orig.self_attn, args.quantized_path, skip_list, quip_params)
            load_proj_or_restore(layer.self_attn, 'o_proj', f'{prefix}{i}', 'o', orig.self_attn, args.quantized_path, skip_list, quip_params)
            load_proj_or_restore(layer.mlp, 'up_proj', f'{prefix}{i}', 'up', orig.mlp, args.quantized_path, skip_list, quip_params)
            load_proj_or_restore(layer.mlp, 'down_proj', f'{prefix}{i}', 'down', orig.mlp, args.quantized_path, skip_list, quip_params)
            load_proj_or_restore(layer.mlp, 'gate_proj', f'{prefix}{i}', 'gate', orig.mlp, args.quantized_path, skip_list, quip_params)

    # Load both text and vision branches
    load_clip_block('', model.language_model.model.layers, orig_model.language_model.model.layers)

    glog.info(f'Saving model to {args.hf_output_path}...')
    model.save_pretrained(args.hf_output_path, safe_serialization=True)
    del model

    # if comp_result['num_pixels'] > 0:
    #     comp_result['bpp_loss'] = comp_result['bpp_loss'] / comp_result['num_pixels']
    #     # comp_result['bpp'] = comp_result['bpp'] / comp_result['num_pixels']
    
    # if isinstance(comp_result['bpp_loss'], torch.Tensor):
    #     comp_result['bpp_loss'] = comp_result['bpp_loss'].item()
    # # if isinstance(comp_result['bpp'], torch.Tensor):
    # #     comp_result['bpp'] = comp_result['bpp'].item()

    # model, _ = model_from_hf_path(args.hf_output_path)
    # glog.info('Successfully loaded hfized model')

    # file_path = f'{args.hf_output_path}_result.json'
    # if os.path.exists(file_path):
    #     os.rename(file_path, f'{args.hf_output_path}_result_.json')
    # with open(file_path, 'w') as f:
    #     json.dump(comp_result, f, indent=2)
        
if __name__ == '__main__':
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    args = parser.parse_args()
    main(args)
