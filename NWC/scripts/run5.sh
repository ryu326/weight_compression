lmbda=30
min=5
max=10000
CUDA_VISIBLE_DEVICES=5 taskset -c 0-63 python -u train_nwc.py \
    --architecture nwc_scale_cond \
    --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/scaleH_sig0.0001_std_rnormed_with_col_std_lidx_row_1024.pt \
    --dataset block_seq_scale_cond_uniform \
    --uniform_scale_max 31.6 \
    --iter 200000 \
    --input_size 128 \
    --M 256 \
    --n_resblock 4 \
    --dim_encoder 1024 \
    --batch_size 2048 \
    --loss rdloss \
    --lmbda $lmbda