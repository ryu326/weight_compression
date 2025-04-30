import torch
from transformers import CLIPModel, LlamaForCausalLM, AutoTokenizer
import torch.nn as nn

def rtn_quantize(weight: torch.Tensor, num_bits: int, quant_type: str = 'per_tensor', group_size: int = 0):
    assert quant_type in ['per_tensor', 'per_channel', 'group'], "quant_type must be one of ['per_tensor', 'per_channel', 'group']"
    device = weight.device
    weight = weight.to('cuda:1')

    qmin = -(2 ** (num_bits - 1))
    qmax = (2 ** (num_bits - 1)) - 1

    if quant_type == 'per_tensor':
        scale = weight.abs().max() / qmax
        quant = torch.round(weight / scale).clamp(qmin, qmax)
        return (quant * scale).to(device)

    elif quant_type == 'per_channel':
        # out_features x in_features → channel-wise: dim=1
        scale = weight.abs().amax(dim=1, keepdim=True) / qmax  # (out_features, 1)
        quant = torch.round(weight / scale).clamp(qmin, qmax)
        return (quant * scale).to(device)

    elif quant_type == 'group':
        assert group_size > 0 and weight.dim() == 2, "Group quantization requires valid group_size and 2D weight"
        out_dim, in_dim = weight.shape
        quant_weight = torch.empty_like(weight)

        for i in range(0, in_dim, group_size):
            group_slice = weight[:, i:i + group_size]
            scale = group_slice.abs().amax(dim=1, keepdim=True) / qmax  # (out_features, 1)
            quant = torch.round(group_slice / scale).clamp(qmin, qmax)
            quant_weight[:, i:i + group_size] = quant * scale

        return quant_weight.to(device)


def quantize_attn_mlp_block_clip(layer: nn.Module, num_bits: int, quant_type='per_tensor', group_size=0):
    for name in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
        proj = getattr(layer.self_attn, name)
        proj.weight.data = rtn_quantize(proj.weight.data, num_bits, quant_type, group_size)

    for name in ['fc1', 'fc2']:
        fc = getattr(layer.mlp, name)
        fc.weight.data = rtn_quantize(fc.weight.data, num_bits, quant_type, group_size)


def apply_rtn_to_clip_model(model: CLIPModel, num_bits=4, quant_type='per_tensor', group_size=0):
    print(f"Quantizing text_model.encoder.layers...")
    for i, layer in enumerate(model.text_model.encoder.layers):
        quantize_attn_mlp_block_clip(layer, num_bits, quant_type, group_size)

    print(f"Quantizing vision_model.encoder.layers...")
    for i, layer in enumerate(model.vision_model.encoder.layers):
        quantize_attn_mlp_block_clip(layer, num_bits, quant_type, group_size)


def quantize_attn_mlp_block_llama(layer: nn.Module, num_bits: int, quant_type='per_tensor', group_size=0):
    for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
        proj = getattr(layer.self_attn, name)
        proj.weight.data = rtn_quantize(proj.weight.data, num_bits, quant_type, group_size)

    for name in ['up_proj', 'gate_proj', 'down_proj']:
        fc = getattr(layer.mlp, name)
        fc.weight.data = rtn_quantize(fc.weight.data, num_bits, quant_type, group_size)

def apply_rtn_to_llama_model(model: LlamaForCausalLM, num_bits=4, quant_type='per_tensor', group_size=0):
    for i, layer in enumerate(model.model.layers):
        quantize_attn_mlp_block_llama(layer, num_bits, quant_type, group_size)

if __name__ == '__main__':

    # 선택: 'per_tensor', 'per_channel', 'group'
    quant_type = 'group'
    group_size = 128  # group일 경우만 의미 있음
    tokenizer = AutoTokenizer.from_pretrained("../Wparam_dataset/hf_model/meta-llama--Meta-Llama-3-8B")
    
    for b in range(4, 10):
        # model = CLIPModel.from_pretrained("../Wparam_dataset/hf_model/openai--clip-vit-large-patch14", torch_dtype=torch.float32)
        # model = LlamaForCausalLM.from_pretrained("../Wparam_dataset/hf_model/meta-llama--Meta-Llama-3-8B", torch_dtype=torch.float32)
        # model.eval()
        
        # apply_rtn_to_clip_model(model, num_bits=b, quant_type=quant_type, group_size=group_size)
        # apply_rtn_to_llama_model(model, num_bits=b, quant_type=quant_type, group_size=group_size)
        # model.save_pretrained(f'../hf_model_comp/RTN/meta-llama--Meta-Llama-3-8B_W{b}g{group_size}')
        # tokenizer.save_pretrained(f'../hf_model_comp/RTN/meta-llama--Meta-Llama-3-8B_W{b}g{group_size}')
        tokenizer.save_pretrained(f'../hf_model_comp/awq/meta-llama--Meta-Llama-3-8B/w{b}-g128-fake-quantized')