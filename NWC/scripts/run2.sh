lmbda=100
min=1
max=10000
CUDA_VISIBLE_DEVICES=2 taskset -c 16-23 python -u train_nwc.py \
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