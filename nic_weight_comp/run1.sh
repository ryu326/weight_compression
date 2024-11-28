for d in 64 512; do
    CUDA_VISIBLE_DEVICES=1 python test_image_pretrain.py \
        --dataset_dir /workspace/jgryu/Weight_compression/Wparam_dataset/image_shape_wp/llama-2-7b_mlp_train_json/${d}_${d} \
        --normalize
done

for d in 64 512; do
    CUDA_VISIBLE_DEVICES=1 python test_image_pretrain.py \
        --dataset_dir /workspace/jgryu/Weight_compression/Wparam_dataset/image_shape_wp/llama-2-7b_attn_train_json/${d}_${d} \
        --normalize
done

