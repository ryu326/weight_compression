CUDA_VISIBLE_DEVICES=0 taskset -c 0-7 python -u train_nwc.py --dist_port 6044 \
    --architecture nwc_ql \
    --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/col_1024_gaussian_padding.pt \
    --dataset block_seq_ql_random \
    --iter 200000 \
    --input_size 16 \
    --M 16 \
    --Q 4 \
    --dim_encoder 512 \
    --batch_size 2048 \
    --loss rdloss_ql \
    --lmbda 50
    
    
###############
    # --architecture nwc \
    # --dataset_path ../Wparam_dataset/block_pt/meta-llama--Llama-2-7b-hf/scaled_sig0.001_row_4096.pt \
    # --dataset block_seq_scalar_mean \
    # --iter 200000 \
    # --input_size 16 \
    # --M 16 \
    # --dim_encoder 512 \
    # --batch_size 1024 \
    # --loss rdloss \
    # --lmbda 50

    # --architecture nwc \
    # --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/adapt_4096_eigen.pt \
    # --dataset block_seq \
    # --iter 200000 \
    # --input_size 16 \
    # --M 16 \
    # --save_dir eigenblock \
    # --dim_encoder 512 \
    # --batch_size 2048 \
    # --loss rdloss \
    # --lmbda 50

    # --architecture nwc_ql \
    # --dataset_path ../Wparam_dataset/block_pt/facebook--opt-6.7b/adapt_4096.pt \
    # --dataset block_seq_ql_random \
    # --iter 200000 \
    # --input_size 16 \
    # --M 16 \
    # --Q 4 \
    # --dim_encoder 512 \
    # --batch_size 1024 \
    # --loss rdloss_ql \
    # --lmbda 50

    # --architecture nwc_hess \
    # --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/adapt_4096.pt \
    # --dataset block_seq_hesseigen \
    # --iter 200000 \
    # --input_size 16 \
    # --M 16 \
    # --R 10 \
    # --m 2 \
    # --dim_encoder 512 \
    # --batch_size 256 \
    # --loss proxy_hess \
    # --lmbda 50