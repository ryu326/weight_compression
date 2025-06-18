lmbda=50
min=5
max=10000
CUDA_VISIBLE_DEVICES=3 taskset -c 24-31 python -u train_nwc.py \
    --architecture nwc_qmap2 \
    --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/col_1024_gaussian_padding.pt \
    --dataset block_seq_qmap \
    --iter 200000 \
    --input_size 16 \
    --M 17 \
    --dim_encoder 512 \
    --batch_size 2048 \
    --loss rdloss_qmap2 \
    --lmbda $lmbda