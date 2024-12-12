CUDA_VISIBLE_DEVICES=2 taskset -c 16-23 python -u train.py --dist_port 6044 \
    --dataset_path ../Wparam_dataset/dataset_per_row/meta-llama/Meta-Llama-3-8B/mlp_16_row_dataset.pt \
    --input_size 16 \
    --n_embeddings 512 \
    --P 4 \
    --dim_encoder 256 \
    --batch_size 512