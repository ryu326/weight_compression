CUDA_VISIBLE_DEVICES=0 taskset -c 0-7 python -u train.py --dist_port 6044 \
    --dataset_path ../Wparam_dataset/dataset_per_row/meta-llama/Meta-Llama-3-8B/mlp_attn_16_row_dataset.pt \
    --architecture vqvae \
    --save_dir vqvae \
    --input_size 16 \
    --dim_embeddings 16 \
    --K 8 \
    --P 10 \
    --dim_encoder 512 \
    --batch_size 2048


# CUDA_VISIBLE_DEVICES=0 taskset -c 0-7 python -u train.py --dist_port 6044 \
#     --dataset_path ../Wparam_dataset/dataset_per_row/meta-llama/Meta-Llama-3-8B/mlp_attn_16_row_dataset.pt \
#     --architecture vq_seedlm \
#     --save_dir vqseedlm_v101 \
#     --input_size 16 \
#     --n_embeddings 512 \
#     --P 8 \
#     --dim_encoder 512 \
#     --batch_size 1024
    