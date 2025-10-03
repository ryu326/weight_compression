(lmbdas=(30 50 100 300 1000 10000)
for lmbda in "${lmbdas[@]}"; do
    echo "=== Running with λ=${lmbda} ==="
    CUDA_VISIBLE_DEVICES=0 python -u train_nwc.py \
        --architecture nwc_ql \
        --dataset_path ../dataset/block_pt/meta-llama--Meta-Llama-3-8B/col_1024.pt \
        --dataset block_seq_ql_random \
        --iter 200000 \
        --input_size 16 \
        --M 16 \
        --Q 4 \
        --dim_encoder 512 \
        --batch_size 2048 \
        --loss rdloss_ql \
        --lmbda $lmbda
done