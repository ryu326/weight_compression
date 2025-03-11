# CUDA_VISIBLE_DEVICES=2 taskset -c 16-23 python -u train_calib_mag.py --dist_port 6044 \
CUDA_VISIBLE_DEVICES=2 taskset -c 16-23 python -u train_calib_mag_scale_input.py --dist_port 6044 \
    --architecture vqvae_idx \
    --save_dir vqvae_idx_scale_input \
    --iter 1500000 \
    --input_size 16 \
    --dim_embeddings 16 \
    --K 8 \
    --P 12 \
    --dim_encoder 512 \
    --batch_size 2048 \
    --loss nmse \
    --block_direction col

# CUDA_VISIBLE_DEVICES=2 taskset -c 16-23 python -u train_calib_mag.py --dist_port 6044 \
#     --architecture vqvae_idx \
#     --save_dir vqvae_idx \
#     --iter 1500000 \
#     --input_size 16 \
#     --dim_embeddings 16 \
#     --K 8 \
#     --P 12 \
#     --dim_encoder 512 \
#     --batch_size 2048 \
#     --loss nmse \
#     --block_direction col