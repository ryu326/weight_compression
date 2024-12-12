CUDA_VISIBLE_DEVICES=0 taskset -c 0-7 python -u train.py --dist_port 6044 \
    --dataset_path ../Wparam_dataset/dataset_per_row/meta-llama/Meta-Llama-3-8B/mlp_16_row_dataset.pt \
    --input_size 16 \
    --n_embeddings 256 \
    --P 4 \
    --dim_encoder 256 \
    --batch_size 512
    