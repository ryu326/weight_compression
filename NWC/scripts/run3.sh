lmbda=10000
CUDA_VISIBLE_DEVICES=3 taskset -c 24-31 python -u train_nwc.py \
    --architecture nwc_ql \
    --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/col_1024_gaussian_padding.pt \
    --dataset block_seq_ql_random \
    --iter 200000 \
    --run_name no_lnorm \
    --input_size 16 \
    --M 16 \
    --Q 4 \
    --no_layernorm \
    --dim_encoder 512 \
    --batch_size 2048 \
    --loss rdloss_ql \
    --lmbda $lmbda