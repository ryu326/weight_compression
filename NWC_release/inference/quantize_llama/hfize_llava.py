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


def get_What(comp_config, saved_layer_data):
    
    W_hat = saved_layer_data['W_hat']
    if W_hat == None:
        W_hat = utils.de_standardize_Wr(saved_layer_data['hatWr'], saved_layer_data['metadata'], comp_config)
    
    return W_hat

def load_layernorm(layer, ln_data):
    if hasattr(layer, 'layer_norm1'):
        layer.layer_norm1.weight.copy_(ln_data['layer_norm1'].to(layer.layer_norm1.weight.dtype))
    if hasattr(layer, 'layer_norm2'):
        layer.layer_norm2.weight.copy_(ln_data['layer_norm2'].to(layer.layer_norm2.weight.dtype))

def load_proj_or_restore(module, key, idx, name, orig_module, path_prefix, skip_list, comp_config, comp_result):
    full_key = f'{idx}_{name}'
    if full_key not in skip_list:
        saved = torch.load(f'{path_prefix}/{full_key}.pt', map_location='cpu', weights_only=False)
        proj_layer = getattr(module, key)
        W_hat = get_What(comp_config, saved) 
        proj_layer.weight.copy_(W_hat.to(proj_layer.weight.dtype))      
        
        comp_result[f'{full_key}.pt'] = {k: v for k, v in saved.items() if not isinstance(v, torch.Tensor) and k != 'codes' and k != 'metadata'}
        comp_result['bpp_loss'] += saved['bpp_loss_sum']
        comp_result['num_pixels'] += saved['num_pixels']
        comp_result['bpp'] += saved['bpp_sum']
    else:
        setattr(module, name, getattr(orig_module, key))
        raise

def main(args):
    assert os.path.exists(args.quantized_path)
    saved_config = torch.load(os.path.join(args.quantized_path, 'config.pt'), weights_only=False)
    model_config = saved_config['model_config']
    comp_config = saved_config['quant_args']
    # glog.info(model_config)
    
    
    model = LlavaForConditionalGeneration.from_pretrained(args.base_model)
    orig_model = LlavaForConditionalGeneration.from_pretrained(args.base_model)

    skip_list = args.skip_list.split(',') if args.skip_list else []
    comp_result = {
        'bpp_loss': 0,
        'bpp': 0,
        'ppl': 0,
        'num_pixels': 0
    }    
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

            load_proj_or_restore(layer.self_attn, 'q_proj', f'{prefix}{i}', 'q', orig.self_attn, args.quantized_path, skip_list, comp_config, comp_result)
            load_proj_or_restore(layer.self_attn, 'k_proj', f'{prefix}{i}', 'k', orig.self_attn, args.quantized_path, skip_list, comp_config, comp_result)
            load_proj_or_restore(layer.self_attn, 'v_proj', f'{prefix}{i}', 'v', orig.self_attn, args.quantized_path, skip_list, comp_config, comp_result)
            load_proj_or_restore(layer.self_attn, 'o_proj', f'{prefix}{i}', 'o', orig.self_attn, args.quantized_path, skip_list, comp_config, comp_result)
            load_proj_or_restore(layer.mlp, 'up_proj', f'{prefix}{i}', 'up', orig.mlp, args.quantized_path, skip_list, comp_config, comp_result)
            load_proj_or_restore(layer.mlp, 'down_proj', f'{prefix}{i}', 'down', orig.mlp, args.quantized_path, skip_list, comp_config, comp_result)
            load_proj_or_restore(layer.mlp, 'gate_proj', f'{prefix}{i}', 'gate', orig.mlp, args.quantized_path, skip_list, comp_config, comp_result)

    # Load both text and vision branches
    load_clip_block('', model.language_model.model.layers, orig_model.language_model.model.layers)

    glog.info(f'Saving model to {args.hf_output_path}...')
    model.save_pretrained(args.hf_output_path, safe_serialization=True)
    del model

    if comp_result['num_pixels'] > 0:
        comp_result['bpp_loss'] = comp_result['bpp_loss'] / comp_result['num_pixels']
        comp_result['bpp'] = comp_result['bpp'] / comp_result['num_pixels']
    
    if isinstance(comp_result['bpp_loss'], torch.Tensor):
        comp_result['bpp_loss'] = comp_result['bpp_loss'].item()
    if isinstance(comp_result['bpp'], torch.Tensor):
        comp_result['bpp'] = comp_result['bpp'].item()

    # model, _ = model_from_hf_path(args.hf_output_path)
    # glog.info('Successfully loaded hfized model')

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