export CUDA_VISIBLE_DEVICES=0
# torchrun --nproc_per_node=4 -m quantize_llama.input_hessian_clip \
#     --batch_size 2 --devset_size 2 \
#     --base_model openai/clip-vit-large-patch14 \
#     --save_path ../Wparam_dataset/quip_hess/clip-vit-large-patch14_test
python -m quantize_llama.input_hessian_clip \
    --batch_size 1024 --devset_size 2048 \
    --base_model openai/clip-vit-large-patch14 \
    --save_path ../Wparam_dataset/quip_hess/clip-vit-large-patch14_2048