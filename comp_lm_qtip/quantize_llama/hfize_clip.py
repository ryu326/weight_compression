import argparse
import os
import glog
import torch
from transformers import CLIPModel as OriCLIP
import json

from lib import utils
from lib.utils.unsafe_import import model_from_hf_path

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--quantized_path', type=str)
parser.add_argument('--hf_output_path', type=str)
parser.add_argument('--base_model', type=str)
parser.add_argument('--skip_list', type=str)


def load_layernorm(layer, ln_data):
    if hasattr(layer, 'layer_norm1'):
        layer.layer_norm1.weight.copy_(ln_data['layer_norm1'].to(layer.layer_norm1.weight.dtype))
    if hasattr(layer, 'layer_norm2'):
        layer.layer_norm2.weight.copy_(ln_data['layer_norm2'].to(layer.layer_norm2.weight.dtype))

def load_proj_or_restore(module, key, idx, name, orig_module, path_prefix, skip_list, comp_result):
    full_key = f'{idx}_{name}'
    if full_key not in skip_list:
        saved = torch.load(f'{path_prefix}/{full_key}.pt', map_location='cpu', weights_only=False)
        # utils.unpack_quip(getattr(module, key), saved)
        proj_layer = getattr(module, key)
        proj_layer.weight.copy_(saved['W_hat'].to(proj_layer.weight.dtype))       
        
        comp_result[f'{idx}_{name}.pt'] = {k:v for k, v in saved.items() if not isinstance(v, torch.Tensor)}
        comp_result['bpp_loss'] += saved['bpp_loss_sum']
        comp_result['num_pixels'] += saved['num_pixels']
    else:
        setattr(module, name, getattr(orig_module, key))
        raise


def main(args):
    assert os.path.exists(args.quantized_path)
    saved_config = torch.load(os.path.join(args.quantized_path, 'config.pt'), weights_only=False)
    model_config = saved_config['model_config']
    glog.info(model_config)

    model = OriCLIP.from_pretrained(args.base_model,
                                      torch_dtype='auto',
                                      low_cpu_mem_usage=True,
                                      config=model_config)
    orig_model = OriCLIP.from_pretrained(args.base_model,
                                           torch_dtype='auto',
                                           low_cpu_mem_usage=True,
                                           config=model_config)

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
            ln_path = f'{args.quantized_path}/{prefix}_{i}_layernorm.pt'
            if os.path.exists(ln_path):
                ln_data = torch.load(ln_path, map_location=cpu, weights_only=False)
                load_layernorm(layer, ln_data)

            load_proj_or_restore(layer.self_attn, 'q_proj', f'{prefix}_{i}', 'q', orig.self_attn, args.quantized_path, skip_list, comp_result)
            load_proj_or_restore(layer.self_attn, 'k_proj', f'{prefix}_{i}', 'k', orig.self_attn, args.quantized_path, skip_list, comp_result)
            load_proj_or_restore(layer.self_attn, 'v_proj', f'{prefix}_{i}', 'v', orig.self_attn, args.quantized_path, skip_list, comp_result)
            load_proj_or_restore(layer.self_attn, 'out_proj', f'{prefix}_{i}', 'o', orig.self_attn, args.quantized_path, skip_list, comp_result)

            load_proj_or_restore(layer.mlp, 'fc1', f'{prefix}_{i}', 'fc1', orig.mlp, args.quantized_path, skip_list, comp_result)
            load_proj_or_restore(layer.mlp, 'fc2', f'{prefix}_{i}', 'fc2', orig.mlp, args.quantized_path, skip_list, comp_result)

    # Load both text and vision branches
    load_clip_block('text', model.text_model.encoder.layers, orig_model.text_model.encoder.layers)
    load_clip_block('vision', model.vision_model.encoder.layers, orig_model.vision_model.encoder.layers)

    glog.info(f'Saving model to {args.hf_output_path}...')
    model.save_pretrained(args.hf_output_path, safe_serialization=True)

    del model
    # model, _ = model_from_hf_path(args.hf_output_path)
    # glog.info('Successfully loaded hfized model')

    with open(f'{args.hf_output_path}_result.json', 'w') as f:
        comp_result['bpp_loss'] /= comp_result['num_pixels']
        json.dump(comp_result, f, indent=2)
        
if __name__ == '__main__':
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    args = parser.parse_args()
    main(args)
