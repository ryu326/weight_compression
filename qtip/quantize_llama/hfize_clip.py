import argparse
import os
import glog
import torch
from transformers import CLIPModel as OriCLIP
from model.clip import CLIPModel


from lib import utils
from lib.utils.unsafe_import import model_from_hf_path

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--quantized_path', type=str)
parser.add_argument('--hf_output_path', type=str)
parser.add_argument('--base_model', type=str)


def load_layernorm(layer, ln_data):
    if hasattr(layer, 'layer_norm1'):
        layer.layer_norm1.weight.copy_(ln_data['layer_norm1'].to(layer.layer_norm1.weight.dtype))
        layer.layer_norm1.bias.copy_(ln_data['layer_norm1'].to(layer.layer_norm1.bias.dtype))
    if hasattr(layer, 'layer_norm2'):
        layer.layer_norm2.weight.copy_(ln_data['layer_norm2'].to(layer.layer_norm2.weight.dtype))
        layer.layer_norm2.bias.copy_(ln_data['layer_norm2'].to(layer.layer_norm2.bias.dtype))


def load_proj_or_restore(module, key, idx, name, orig_module, path_prefix, skip_list):
    full_key = f'{idx}_{name}'
    if full_key not in skip_list:
        saved = torch.load(f'{path_prefix}/{full_key}.pt', map_location='cpu')
        utils.unpack_quip(getattr(module, key), saved)
        getattr(module, key).bias.data.copy_(getattr(orig_module, key).bias.data)
    else:
        setattr(module, key, getattr(orig_module, key))


def main(args):
    assert os.path.exists(args.quantized_path)
    saved_config = torch.load(os.path.join(args.quantized_path, 'config.pt'))
    model_config = saved_config['model_config']
    glog.info(model_config)

    # skip_list = model_config.get('quip_params', {}).get('skip_list', [])
    # if skip_list is None:
    skip_list = []
    # import ipdb; ipdb.set_trace()

    model = CLIPModel.from_pretrained(args.base_model,
                                      torch_dtype='auto',
                                      low_cpu_mem_usage=True,
                                      config=model_config)
    orig_model = OriCLIP.from_pretrained(args.base_model,
                                           torch_dtype='auto',
                                           low_cpu_mem_usage=True,
                                           config=model_config)

    cpu = torch.device('cpu')

    def load_clip_block(prefix, layers, orig_layers):
        for i in range(len(layers)):
            layer = layers[i]
            orig = orig_layers[i]
            glog.info(f'Loading {prefix} layer {i}')
            ln_path = f'{args.quantized_path}/{prefix}_{i}_layernorm.pt'
            # if os.path.exists(ln_path):
            #     ln_data = torch.load(ln_path, map_location=cpu)
            #     load_layernorm(layer, ln_data)

            load_proj_or_restore(layer.self_attn, 'q_proj', f'{prefix}_{i}', 'q', orig.self_attn, args.quantized_path, skip_list)
            load_proj_or_restore(layer.self_attn, 'k_proj', f'{prefix}_{i}', 'k', orig.self_attn, args.quantized_path, skip_list)
            load_proj_or_restore(layer.self_attn, 'v_proj', f'{prefix}_{i}', 'v', orig.self_attn, args.quantized_path, skip_list)
            load_proj_or_restore(layer.self_attn, 'out_proj', f'{prefix}_{i}', 'o', orig.self_attn, args.quantized_path, skip_list)

            load_proj_or_restore(layer.mlp, 'fc1', f'{prefix}_{i}', 'fc1', orig.mlp, args.quantized_path, skip_list)
            load_proj_or_restore(layer.mlp, 'fc2', f'{prefix}_{i}', 'fc2', orig.mlp, args.quantized_path, skip_list)

    # Load both text and vision branches
    load_clip_block('text', model.text_model.encoder.layers, orig_model.text_model.encoder.layers)
    load_clip_block('vision', model.vision_model.encoder.layers, orig_model.vision_model.encoder.layers)

        # === 복사 도우미 함수들 ===
    def copy_tensor_attr(module, attr):
        if hasattr(module, attr):
            glog.info(f'Copying tensor attribute {attr}')
            getattr(model, attr).data.copy_(getattr(orig_model, attr).data)

    def copy_submodule(model_sub, orig_sub, name):
        glog.info(f'Copying submodule: {name}')
        model_sub.load_state_dict(orig_sub.state_dict())

    # === logit_scale (tensor) 복사 ===
    if hasattr(model, 'logit_scale') and hasattr(orig_model, 'logit_scale'):
        glog.info('Copying logit_scale')
        model.logit_scale.data.copy_(orig_model.logit_scale.data)

    # === visual_projection & text_projection ===
    copy_submodule(model.visual_projection, orig_model.visual_projection, 'visual_projection')
    copy_submodule(model.text_projection, orig_model.text_projection, 'text_projection')

    # === text_model.embeddings ===
    copy_submodule(model.text_model.embeddings.token_embedding, orig_model.text_model.embeddings.token_embedding, 'text token_embedding')
    copy_submodule(model.text_model.embeddings.position_embedding, orig_model.text_model.embeddings.position_embedding, 'text position_embedding')

    # === text_model.final_layer_norm ===
    copy_submodule(model.text_model.final_layer_norm, orig_model.text_model.final_layer_norm, 'text final_layer_norm')

    # === vision_model.embeddings ===
    copy_submodule(model.vision_model.embeddings.patch_embedding, orig_model.vision_model.embeddings.patch_embedding, 'vision patch_embedding')
    copy_submodule(model.vision_model.embeddings.position_embedding, orig_model.vision_model.embeddings.position_embedding, 'vision position_embedding')

    # === vision_model.pre_layrnorm (오타 주의: pre_layernorm이 맞는지 확인 필요) ===
    if hasattr(model.vision_model, 'pre_layernorm이'):
        copy_submodule(model.vision_model.pre_layernorm이, orig_model.vision_model.pre_layernorm이, 'vision pre_layernorm이')

    # === vision_model.post_layernorm ===
    copy_submodule(model.vision_model.post_layernorm, orig_model.vision_model.post_layernorm, 'vision post_layernorm')


    glog.info(f'Saving model to {args.hf_output_path}...')
    model.save_pretrained(args.hf_output_path, safe_serialization=True)

    del model
    # model, _ = model_from_hf_path(args.hf_output_path)
    # glog.info('Successfully loaded hfized model')


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    args = parser.parse_args()
    main(args)
