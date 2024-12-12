CUDA_VISIBLE_DEVICES=3 taskset -c 24-31 python -u train.py --dist_port 6044 \
    --dataset_path ../Wparam_dataset/dataset_per_row/meta-llama/Meta-Llama-3-8B/mlp_16_row_dataset.pt \
    --input_size 16 \
    --n_embeddings 512 \
    --P 32 \
    --dim_encoder 64 \
    --batch_size 512