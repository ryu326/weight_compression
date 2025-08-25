lmbdas=(20 1000 300 )  # 원하는 λ 값 리스트

for lmbda in "${lmbdas[@]}"; do
    echo "=== Running with λ=${lmbda} ==="
    CUDA_VISIBLE_DEVICES=0,1,2,3 taskset -c 0-15 \
    torchrun --standalone --nproc_per_node=4 train_nwc_ddp.py \
        --architecture nwc_scale_cond_ltc \
        --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/scaleH_sig0.0001_std_rnormed_with_col_std_lidx_row_1024.pt \
        --dataset block_seq_scale_cond \
        --iter 20000 \
        --input_size 16 \
        --ltc_N 256 \
        --run_name N256 \
        --M 24 \
        --n_resblock 4 \
        --dim_encoder 512 \
        --batch_size 128 \
        --global_batch_size 512 \
        --loss rdloss \
        --lmbda $lmbda

        # --architecture nwc_ql_ltc \
        # --run_name N256 \
        # --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/col_1024_gaussian_padding.pt \
        # --dataset block_seq_ql_random \
        # --iter 20000 \
        # --input_size 16 \
        # --ltc_N 256 \
        # --M 16 \
        # --Q 4 \
        # --dim_encoder 512 \
        # --batch_size 256 \
        # --global_batch_size 1024 \
        # --loss rdloss_ql \
        # --lmbda $lmbda
done
        # --architecture nwc_scale_cond_ltc \
        # --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/scaleH_sig0.0001_std_rnormed_with_col_std_lidx_row_1024.pt \
        # --dataset block_seq_scale_cond \
        # --iter 100000 \
        # --input_size 128 \
        # --ltc_N 256 \
        # --M 144 \
        # --n_resblock 4 \
        # --dim_encoder 1024 \
        # --batch_size 256 \
        # --global_batch_size 768 \
        # --loss rdloss \
        # --lmbda $lmbda

        # --architecture nwc_ql_ltc \
        # --run_name N256 \
        # --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/col_1024_gaussian_padding.pt \
        # --dataset block_seq_ql_random \
        # --iter 100000 \
        # --input_size 16 \
        # --ltc_N 256 \
        # --M 16 \
        # --Q 4 \
        # --dim_encoder 512 \
        # --batch_size 256 \
        # --global_batch_size 1024 \
        # --loss rdloss_ql \
        # --lmbda $lmbda