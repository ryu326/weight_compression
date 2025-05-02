lmbda=1000
CUDA_VISIBLE_DEVICES=2 taskset -c 16-23 python -u train_nwc.py \
    --architecture nwc_ql \
    --dataset_path ../Wparam_dataset/block_pt/meta-llama--Llama-2-7b-hf/col_1024.pt \
    --pretrained_path ./checkpoint/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt/M16/lmbda${lmbda}_*/best_loss_model_*.pth.tar \
    --run_name llama8b_c1024_7b \
    --dataset block_seq_ql_random \
    --iter 20000 \
    --input_size 16 \
    --M 16 \
    --dim_encoder 512 \
    --batch_size 2048 \
    --loss rdloss_ql --Q 4 \
    --lmbda $lmbda