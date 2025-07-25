lmbda=300
min=1
max=10000
CUDA_VISIBLE_DEVICES=2 taskset -c 32-47 python -u train_nwc.py \
    --architecture nwc_scale_cond \
    --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/scaleH_sig0.0001_std_rnormed_with_col_std_lidx_row_1024.pt \
    --dataset block_seq_scale_cond \
    --iter 200000 \
    --input_size 16 \
    --M 256 \
    --n_resblock 4 \
    --dim_encoder 512 \
    --batch_size 2048 \
    --loss rdloss \
    --lmbda $lmbda