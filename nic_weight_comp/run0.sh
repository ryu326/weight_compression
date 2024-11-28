for d in 256 32; do
    CUDA_VISIBLE_DEVICES=0 python test_image_pretrain.py \
        --dataset_dir /workspace/jgryu/Weight_compression/Wparam_dataset/image_shape_wp/llama-2-7b_mlp_train_json/${d}_${d} \
        --normalize
done

for d in 256 32; do
    CUDA_VISIBLE_DEVICES=0 python test_image_pretrain.py \
        --dataset_dir /workspace/jgryu/Weight_compression/Wparam_dataset/image_shape_wp/llama-2-7b_attn_train_json/${d}_${d} \
        --normalize
done

# for d in 256 32 64 512 1024; do
#     CUDA_VISIBLE_DEVICES=0 python test_image_pretrain.py \
#         --dataset_dir /workspace/jgryu/Weight_compression/Wparam_dataset/image_shape_wp/meta-llama-3-8b_mlp_val_json/${d}_${d}
# done

# for d in 256 32 64 512 1024; do
#     CUDA_VISIBLE_DEVICES=1 python test_image_pretrain.py \
#         --dataset_dir /workspace/jgryu/Weight_compression/Wparam_dataset/image_shape_wp/meta-llama-3-8b_mlp_val_json/${d}_${d} \
#         --normalize
# done

# for d in 256 32 64 512 1024; do
#     CUDA_VISIBLE_DEVICES=2 python test_image_pretrain.py \
#         --dataset_dir /workspace/jgryu/Weight_compression/Wparam_dataset/image_shape_wp/meta-llama-3-8b_mlp_val_json/${d}_${d} \
#         --normalize \
#         --imagelize
# done