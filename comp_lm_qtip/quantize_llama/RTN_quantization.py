import torch
from transformers import CLIPModel, LlamaForCausalLM, AutoTokenizer
import torch.nn as nn
import argparse
import os

def rtn_quantize(weight: torch.Tensor, num_bits: int, quant_type: str = 'per_tensor', group_size: int = 0):
    assert quant_type in ['per_tensor', 'per_channel', 'group'], "quant_type must be one of ['per_tensor', 'per_channel', 'group']"
    device = weight.device
    weight = weight.to('cuda')

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

def main():
    """메인 실행 함수: 인자를 파싱하고 양자화 프로세스를 실행합니다."""
    parser = argparse.ArgumentParser(description="Apply RTN quantization to Hugging Face models.")
    
    parser.add_argument('--model_type', type=str, choices=['llama', 'clip'], default='llama', 
                        help="Type of the model to quantize ('llama' or 'clip').")
    parser.add_argument('--model_path', type=str, required=True, 
                        help="Path to the pretrained model directory.")
    parser.add_argument('--output_path', type=str, default=None,
                        help="Path to save the quantized model and tokenizer.")
    parser.add_argument('--num_bits', type=int, default=4,
                        help="Number of bits for quantization (e.g., 4, 8).")
    parser.add_argument('--quant_type', type=str, default='group', choices=['per_tensor', 'per_channel', 'group'],
                        help="Quantization type ('per_tensor', 'per_channel', 'group').")
    parser.add_argument('--group_size', type=int, default=128,
                        help="Group size for group quantization. Only used if quant_type is 'group'.")

    args = parser.parse_args()

    # 'group' 양자화 선택 시 group_size 유효성 검사
    if args.quant_type == 'group' and args.group_size <= 0:
        raise ValueError("Group size must be a positive integer for 'group' quantization.")

    print(f"--- Starting Quantization ---")
    print(f"Model Type: {args.model_type}")
    print(f"Model Path: {args.model_path}")
    print(f"Output Path: {args.output_path}")
    print(f"Number of Bits: {args.num_bits}")
    print(f"Quantization Type: {args.quant_type}")
    if args.quant_type == 'group':
        print(f"Group Size: {args.group_size}")
    print("-----------------------------")

    # 모델 및 토크나이저 로드
    print(f"Loading model and tokenizer from {args.model_path}...")
    
    # FP32로 모델을 로드하여 정확한 양자화 수행
    model_kwargs = {'torch_dtype': torch.float32}
    
    if args.model_type == 'llama':
        model = LlamaForCausalLM.from_pretrained(args.model_path, **model_kwargs)
        apply_func = apply_rtn_to_llama_model
    elif args.model_type == 'clip':
        model = CLIPModel.from_pretrained(args.model_path, **model_kwargs)
        apply_func = apply_rtn_to_clip_model
    else:
        # argparse의 choices 옵션 덕분에 이 부분은 실행될 가능성이 낮습니다.
        raise ValueError(f"Unsupported model type: {args.model_type}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model.eval() # 추론 모드로 설정

    # 양자화 적용
    print(f"Applying RTN quantization (W{args.num_bits}, type={args.quant_type})...")
    apply_func(model, num_bits=args.num_bits, quant_type=args.quant_type, group_size=args.group_size)
    print("Quantization complete.")

    # 양자화된 모델 및 토크나이저 저장
    args.output_path = args.output_path or f'../hf_model_comp/awq/meta-llama--Meta-Llama-3-8B/w{args.num_bits}-g128-fake-quantized'

    print(f"Saving quantized model and tokenizer to {args.output_path}...")
    os.makedirs(args.output_path, exist_ok=True)
    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    print("Successfully saved.")


if __name__ == '__main__':
    main()

# if __name__ == '__main__':

#     # 선택: 'per_tensor', 'per_channel', 'group'
#     quant_type = 'group'
#     group_size = 128  # group일 경우만 의미 있음
#     # tokenizer = AutoTokenizer.from_pretrained("../Wparam_dataset/hf_model/meta-llama--Meta-Llama-3-8B")
#     tokenizer = AutoTokenizer.from_pretrained("../Wparam_dataset/hf_model/meta-llama--Meta-Llama-3-8B")
    
#     for b in range(4, 10):
#         # model = CLIPModel.from_pretrained("../Wparam_dataset/hf_model/openai--clip-vit-large-patch14", torch_dtype=torch.float32)
#         # model = LlamaForCausalLM.from_pretrained("../Wparam_dataset/hf_model/meta-llama--Meta-Llama-3-8B", torch_dtype=torch.float32)
#         # model.eval()
        
#         # apply_rtn_to_clip_model(model, num_bits=b, quant_type=quant_type, group_size=group_size)
#         # apply_rtn_to_llama_model(model, num_bits=b, quant_type=quant_type, group_size=group_size)
#         # model.save_pretrained(f'../hf_model_comp/RTN/meta-llama--Meta-Llama-3-8B_W{b}g{group_size}')
#         # tokenizer.save_pretrained(f'../hf_model_comp/RTN/meta-llama--Meta-Llama-3-8B_W{b}g{group_size}')
#         tokenizer.save_pretrained(f'../hf_model_comp/awq/meta-llama--Meta-Llama-3-8B/w{b}-g128-fake-quantized')