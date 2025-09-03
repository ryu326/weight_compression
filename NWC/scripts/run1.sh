lmbdas=(300 100)
for lmbda in "${lmbdas[@]}"; do
    echo "=== Running with λ=${lmbda} ==="
    CUDA_VISIBLE_DEVICES=1 taskset -c 16-31 python -u train_nwc.py \
        --architecture nwc_scale_cond \
        --dataset_path /workspace/Weight_compression/Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/scaleH_sig0.0001_std_rnormed_with_col_std_lidx_row_1024.pt \
        --dataset block_seq_scale_cond \
        --iter 200000 \
        --input_size 128 \
        --run_name aug_scale \
        --M 256 \
        --n_resblock 4 \
        --dim_encoder 1024 \
        --batch_size 2048 \
        --loss rdloss \
        --lmbda $lmbda \
        --aug_scale --aug_scale_p 0.1
done