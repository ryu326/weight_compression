
for d in 1024; do
    CUDA_VISIBLE_DEVICES=2 python test_image_pretrain.py \
        --dataset_dir /workspace/jgryu/Weight_compression/Wparam_dataset/image_shape_wp/llama-2-7b_mlp_train_json/${d}_${d} \
        --normalize
done

for d in 1024; do
    CUDA_VISIBLE_DEVICES=2 python test_image_pretrain.py \
        --dataset_dir /workspace/jgryu/Weight_compression/Wparam_dataset/image_shape_wp/llama-2-7b_attn_train_json/${d}_${d} \
        --normalize
done
