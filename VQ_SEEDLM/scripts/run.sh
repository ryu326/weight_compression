nohup bash scripts/run0.sh > ./training_logs/run0.log 2>&1 &
nohup bash scripts/run1.sh > ./training_logs/run1.log 2>&1 &
nohup bash scripts/run2.sh > ./training_logs/run2.log 2>&1 &
nohup bash scripts/run3.sh > ./training_logs/run3.log 2>&1 &


# CUDA_VISIBLE_DEVICES=2 taskset -c 16-23 python -u train_calib_mag.py --dist_port 6044 \
#     --architecture vqvae_idx \
#     --save_dir test \
#     --iter 1500000 \
#     --input_size 16 \
#     --dim_embeddings 16 \
#     --K 8 \
#     --P 12 \
#     --dim_encoder 512 \
#     --batch_size 2048 \
#     --loss smse \
#     --block_direction col