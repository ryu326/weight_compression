lmbda=10000
CUDA_VISIBLE_DEVICES=3 taskset -c 24-31 python -u train_nwc.py \
    --architecture nwc_ql \
    --dataset_path ../Wparam_dataset/block_pt/meta-llama--Llama-2-7b-hf/droplast_col_1024.pt \
    --pretrained_path ./checkpoint/nwc_ql/block_seq_ql_random_scaler_gaussian__llama8b_col_1024.pt/rdloss_ql_size16_encdim512_M16_Q4_R0_m0_batch_size2048_total_iter200000_lr0.0001_seed100/lmbda${lmbda}_*/best_loss_model_*.pth.tar \
    --run_name gaussian_7b_droplast \
    --dataset block_seq_ql_random \
    --iter 20000 \
    --input_size 16 \
    --M 16 \
    --dim_encoder 512 \
    --batch_size 2048 \
    --loss rdloss_ql --Q 4 \
    --lmbda $lmbda