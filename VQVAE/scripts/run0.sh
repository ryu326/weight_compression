CUDA_VISIBLE_DEVICES=0 taskset -c 0-7 python -u train_nwc.py --dist_port 6044 \
    --architecture nwc_ql \
    --save_dir nwc_ql \
    --iter 200000 \
    --input_size 16 \
    --M 16 \
    --dim_encoder 512 \
    --batch_size 1024 \
    --loss rdloss_ql \
    --block_direction col \
    --lmbda 100000 \
    --dataset block_seq_ql_random \


#################################################
    # --architecture nwc_hp \
    # --save_dir nwc_hp \
    # --iter 200000 \
    # --input_size 16 \
    # --M 16 \
    # --N 16 \
    # --dim_encoder 512 \
    # --batch_size 2048 \
    # --loss rdloss \
    # --block_direction row \
    # --lmbda 30000 \
    # --dataset block_seq

    # --architecture nwc \
    # --save_dir nwc \
    # --iter 200000 \
    # --input_size 16 \
    # --M 16 \
    # --dim_encoder 512 \
    # --batch_size 2048 \
    # --loss rdloss \
    # --block_direction row \
    # --lmbda 50 \
    # --dataset gaussian_seq

# CUDA_VISIBLE_DEVICES=0 taskset -c 0-7 python -u train_nwc.py --dist_port 6044 \
#     --architecture nwc_ql \
#     --save_dir nwc_ql \
#     --iter 200000 \
#     --input_size 16 \
#     --M 16 \
#     --dim_encoder 512 \
#     --batch_size 2048 \
#     --loss rdloss_ql \
#     --block_direction col \
#     --lmbda 50 \
#     --dataset block_seq_ql_random \

# CUDA_VISIBLE_DEVICES=1 taskset -c 8-15 python -u train_nwc.py --dist_port 6044 \
#     --architecture nwc_ql \
#     --save_dir nwc_ql \
#     --iter 1500000 \
#     --input_size 16 \
#     --M 16 \
#     --dim_encoder 512 \
#     --batch_size 2048 \
#     --loss rdloss_ql \
#     --block_direction col \
#     --lmbda 100 \
#     --dataset block_seq_ql_random \

# CUDA_VISIBLE_DEVICES=0 taskset -c 0-7 python -u train_nwc.py --dist_port 6044 \
#     --architecture nwc \
#     --save_dir nwc \
#     --iter 1500000 \
#     --input_size 512 \
#     --dim_encoder 512 \
#     --M 512 \
#     --batch_size 4096 \
#     --loss rdloss \
#     --block_direction col \
#     --lmbda 50 \
#     --dataset block


# CUDA_VISIBLE_DEVICES=0 taskset -c 0-7 python -u train_nwc.py --dist_port 6044 \
# #     --architecture nwc \
# #     --save_dir nwc \
# #     --iter 1500000 \
# #     --input_size 16 \
# #     --dim_encoder 512 \
# #     --M 16 \
# #     --batch_size 4096 \
# #     --loss rdloss \
# #     --block_direction col \
# #     --lmbda 100 \
# #     --dataset block

# CUDA_VISIBLE_DEVICES=0 taskset -c 0-7 python -u train_nwc.py --dist_port 6044 \
#     --architecture nwc \
#     --save_dir nwc \
#     --iter 1500000 \
#     --input_size 16 \
#     --dim_encoder 512 \
#     --M 16 \
#     --batch_size 4096 \
#     --loss rdloss \
#     --block_direction row \
#     --lmbda 50 \
#     --dataset gaussian



# CUDA_VISIBLE_DEVICES=0 taskset -c 0-7 python -u train_nwc.py --dist_port 6044 \
#     --architecture nwc_ql \
#     --save_dir nwc_ql \
#     --iter 1500000 \
#     --input_size 16 \
#     --M 16 \
#     --dim_encoder 512 \
#     --batch_size 4096 \
#     --loss rdloss_ql \
#     --block_direction col \
#     --lmbda 100 \
#     --dataset block_ql_random \

# CUDA_VISIBLE_DEVICES=3 taskset -c 24-31 python -u train_nwc.py --dist_port 6044 \
#     --architecture nwc \
#     --save_dir nwc \
#     --iter 200000 \
#     --input_size 16 \
#     --M 16 \
#     --dim_encoder 512 \
#     --batch_size 2048 \
#     --loss rdloss \
#     --block_direction row \
#     --lmbda 30000 \
#     --dataset block_seq

# CUDA_VISIBLE_DEVICES=0 taskset -c 0-7 python -u train.py --dist_port 6044 \
#     --architecture vqvae_idx_mag \
#     --save_dir vqvae_idx_mag \
#     --iter 1500000 \
#     --input_size 16 \
#     --dim_embeddings 2 \
#     --K 6 \
#     --P 8 \
#     --dim_encoder 512 \
#     --batch_size 4096 \
#     --loss nmse \
#     --block_direction row \
#     --dataset block_mag


# vqvae_qlike
# CUDA_VISIBLE_DEVICES=0 taskset -c 0-7 python -u train.py --dist_port 6044 \
#     --architecture vqvae \
#     --save_dir test \
#     --iter 2 \
#     --input_size 16 \
#     --dim_embeddings 2 \
#     --K 6 \
#     --P 8 \
#     --dim_encoder 512 \
#     --batch_size 4096 \
#     --loss nmse \
#     --block_direction row \
#     --dataset block

# CUDA_VISIBLE_DEVICES=0 taskset -c 0-7 python -u train.py --dist_port 6044 \
#     --architecture vqvae_scale \
#     --save_dir vqvae_random_scale \
#     --iter 1500000 \
#     --input_size 16 \
#     --dim_embeddings 16 \
#     --K 8 \
#     --P 6 \
#     --dim_encoder 512 \
#     --batch_size 256 \
#     --loss nmse \
#     --block_direction col \
#     --dataset vector_random_scale

# CUDA_VISIBLE_DEVICES=0 taskset -c 0-7 python -u train.py --dist_port 6044 \
#     --architecture vqvae_mag \
#     --save_dir vqvae_mag_vec \
#     --iter 1500000 \
#     --input_size 16 \
#     --dim_embeddings 16 \
#     --K 8 \
#     --P 6 \
#     --dim_encoder 512 \
#     --batch_size 256 \
#     --loss smse \
#     --block_direction row \
#     --vector

# CUDA_VISIBLE_DEVICES=0 taskset -c 0-7 python -u train.py --dist_port 6044 \
#     --architecture vqvae_mag \
#     --save_dir vqvae_mag \
#     --iter 1500000 \
#     --input_size 16 \
#     --dim_embeddings 16 \
#     --K 8 \
#     --P 6 \
#     --dim_encoder 512 \
#     --batch_size 2048 \
#     --loss smse \
#     --block_direction col