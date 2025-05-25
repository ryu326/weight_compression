lmbda=10000
CUDA_VISIBLE_DEVICES=3 taskset -c 24-31 python -u train_nwc.py \
    --architecture nwc_ql \
    --dataset_path ../Wparam_dataset/block_pt/llama8b+7b/droplast_modelwise_norm2_col_1024.pt \
    --dataset block_seq_ql_random \
    --iter 200000 \
    --input_size 16 \
    --M 16 \
    --dim_encoder 512 \
    --batch_size 2048 \
    --loss rdloss_ql --Q 4 \
    --lmbda $lmbda