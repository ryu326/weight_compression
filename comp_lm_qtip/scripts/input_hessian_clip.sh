export CUDA_VISIBLE_DEVICES=2,3
export HF_HOME=/workspace/hf_cache/huggingface_nwc

# torchrun --nproc_per_node=4 -m quantize_llama.input_hessian_clip \
#     --batch_size 2 --devset_size 2 \
#     --base_model openai/clip-vit-large-patch14 \
#     --save_path ../Wparam_dataset/quip_hess/clip-vit-large-patch14_test

# python -m quantize_llama.input_hessian_clip \
#     --batch_size 1024 --devset_size 2048 \
#     --base_model openai/clip-vit-large-patch14 \
#     --save_path ../Wparam_dataset/quip_hess/clip-vit-large-patch14_2048

# python -m quantize_llama.input_hessian_clip \
#     --batch_size 1024 --devset_size 512 \
#     --base_model google/siglip-base-patch16-224 \
#     --save_path ../Wparam_dataset/quip_hess/siglip-base-patch16-224_512


# torchrun --nproc_per_node=4 -m quantize_llama.input_hessian_llava \
#     --batch_size 2 --devset_size 512 \
#     --base_model /workspace/Weight_compression/Wparam_dataset/hf_model/llava-hf--llava-1.5-7b-hf \
#     --save_path ../Wparam_dataset/quip_hess/llava-hf--llava-1.5-7b-hf_512

torchrun --nproc_per_node=2 -m quantize_llama.input_hessian_dinov2 \
    --batch_size 128 --devset_size 512 \
    --base_model /workspace/Weight_compression/Wparam_dataset/hf_model/facebook--dinov2-large-imagenet1k-1-layer \
    --save_path ../Wparam_dataset/quip_hess/llava-hf--dinov2-large-imagenet1k-1-layer_cc512

torchrun --nproc_per_node=2 -m quantize_llama.input_hessian_dinov2 \
    --batch_size 128 --devset_size 512 \
    --base_model /workspace/Weight_compression/Wparam_dataset/hf_model/facebook--dinov2-base-imagenet1k-1-layer \
    --save_path ../Wparam_dataset/quip_hess/llava-hf--dinov2-base-imagenet1k-1-layer_cc512

    # --base_model google/siglip2-base-patch16-224 \ 
    # --base_model /workspace/Weight_compression/Wparam_dataset/hf_model/google--siglip2-base-patch16-224 \
