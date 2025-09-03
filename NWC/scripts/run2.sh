lmbdas=(1000 300)
for lmbda in "${lmbdas[@]}"; do
    echo "=== Running with λ=${lmbda} ==="
    CUDA_VISIBLE_DEVICES=2 taskset -c 32-47 python -u train_nwc.py \
        --architecture nwc_ql \
        --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/col_1024_gaussian_padding.pt \
        --dataset block_seq_ql_random \
        --run_name debug_module \
        --iter 200000 \
        --input_size 16 \
        --M 16 \
        --dim_encoder 512 \
        --batch_size 2048 \
        --loss rdloss_ql --Q 4 \
        --lmbda $lmbda
done
  