# lmbdas=(30 1000 50 100)
(
lmbdas=(1000 50)
for lmbda in "${lmbdas[@]}"; do
    echo "=== Running with λ=${lmbda} ==="
    CUDA_VISIBLE_DEVICES=0 taskset -c 0-15 python -u train_nwc.py \
        --architecture nwc_scale_cond --loss rdloss \
        --dataset_path /workspace/Weight_compression/Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/scaleH_sig0.0001_std_rnormed_with_col_std_lidx_row_1024.pt \
        --dataset block_seq_scale_cond --iter 200000 --batch_size 2048 \
        --input_size 128 --M 256 --n_resblock 4 --dim_encoder 1024 \
        --lmbda $lmbda --run_name aug_scale --aug_scale --aug_scale_p 0.1        
done
) > ./logs/run0.log 2>&1 &

(
lmbdas=(300 100)
for lmbda in "${lmbdas[@]}"; do
    echo "=== Running with λ=${lmbda} ==="
    CUDA_VISIBLE_DEVICES=1 taskset -c 16-31 python -u train_nwc.py \
        --architecture nwc_scale_cond --loss rdloss \
        --dataset_path /workspace/Weight_compression/Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/scaleH_sig0.0001_std_rnormed_with_col_std_lidx_row_1024.pt \
        --dataset block_seq_scale_cond --iter 200000 --batch_size 2048 \
        --input_size 128 --M 256 --n_resblock 4 --dim_encoder 1024 \
        --lmbda $lmbda --run_name aug_scale --aug_scale --aug_scale_p 0.1        
done
) > ./logs/run1.log 2>&1 &


(lmbdas=(1000 50)
for lmbda in "${lmbdas[@]}"; do
    echo "=== Running with λ=${lmbda} ==="
    CUDA_VISIBLE_DEVICES=2 taskset -c 32-47 python -u train_nwc.py \
        --architecture nwc_scale_cond --loss rdloss \
        --dataset_path /workspace/Weight_compression/Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/scaleH_sig0.0001_std_rnormed_with_col_std_lidx_row_1024.pt \
        --dataset block_seq_scale_cond_uniform --iter 200000 --batch_size 2048 \
        --input_size 128 --M 256 --n_resblock 4 --dim_encoder 1024 \
        --lmbda $lmbda \
        --uniform_scale_max 31.6 \
        --run_name debug      
done
) > ./logs/run2.log 2>&1 &

(lmbdas=(300 100)
for lmbda in "${lmbdas[@]}"; do
    echo "=== Running with λ=${lmbda} ==="
    CUDA_VISIBLE_DEVICES=3 taskset -c 48-63 python -u train_nwc.py \
        --architecture nwc_scale_cond --loss rdloss \
        --dataset_path /workspace/Weight_compression/Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/scaleH_sig0.0001_std_rnormed_with_col_std_lidx_row_1024.pt \
        --dataset block_seq_scale_cond_uniform31.6 --iter 200000 --batch_size 2048 \
        --input_size 128 --M 256 --n_resblock 4 --dim_encoder 1024 \
        --lmbda $lmbda \
        --uniform_scale_max 31.6 \
        --run_name debug      
done
) > ./logs/run3.log 2>&1 &


        # --architecture nwc_scale_cond \
        # --dataset_path "/workspace/Weight_compression/Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/row_1024_rnormed_scale_cond(col_std).pt" \
        # --dataset block_seq_scale_cond \
        # --iter 200000 \
        # --input_size 128 \
        # --M 256 \
        # --n_resblock 4 \
        # --dim_encoder 1024 \
        # --batch_size 2048 \
        # --loss rdloss \
        # --lmbda $lmbda

        # --architecture nwc_scale_cond \
        # --dataset_path "/workspace/Weight_compression/Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/row_1024_whiten_scale_cond(col_std).pt" \
        # --dataset block_seq_scale_cond \
        # --iter 200000 \
        # --input_size 128 \
        # --M 256 \
        # --n_resblock 4 \
        # --dim_encoder 1024 \
        # --batch_size 2048 \
        # --loss rdloss \
        # --lmbda $lmbda

        # --architecture nwc_scale_cond \
        # --dataset_path "../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/row_1024_rnormed_scale_cond(scaleWH).pt" \
        # --dataset block_seq_scale_cond_uniform \
        # --uniform_scale_max 50 \
        # --normalize \
        # --iter 200000 \
        # --input_size 128 \
        # --M 256 \
        # --n_resblock 4 \
        # --dim_encoder 1024 \
        # --batch_size 2048 \
        # --loss rdloss \
        # --lmbda $lmbda

        # --architecture nwc_scale_cond \
        # --dataset_path "../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/row_1024_rnormed_scale_cond(scaleWH).pt" \
        # --dataset block_seq_scale_cond \
        # --normalize \
        # --iter 200000 \
        # --input_size 128 \
        # --M 256 \
        # --n_resblock 4 \
        # --dim_encoder 1024 \
        # --batch_size 2048 \
        # --loss rdloss \
        # --lmbda $lmbda

        # --architecture nwc_scale_cond \
        # --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/scaleH_sig0.0001_with_col_std_lidx_row_1024.pt \
        # --dataset block_seq_scale_cond \
        # --run_name no_rnorm \
        # --iter 200000 \
        # --input_size 128 \
        # --M 256 \
        # --n_resblock 4 \
        # --dim_encoder 1024 \
        # --batch_size 2048 \
        # --loss rdloss \
        # --lmbda $lmbda

        # --architecture nwc_scale_cond \
        # --dataset_path "../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/row_1024_rnormed_scale_cond(scaleWH).pt" \
        # --dataset block_seq_scale_cond_uniform \
        # --uniform_scale_max 50 \
        # --pre_normalize \
        # --iter 200000 \
        # --input_size 128 \
        # --M 256 \
        # --n_resblock 4 \
        # --dim_encoder 1024 \
        # --batch_size 2048 \
        # --loss rdloss \
        # --lmbda $lmbda

        # --architecture nwc_scale_cond \
        # --dataset_path "../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/row_1024_rnormed_scale_cond(scaleWH).pt" \
        # --dataset block_seq_scale_cond \
        # --pre_normalize \
        # --iter 200000 \
        # --input_size 128 \
        # --M 256 \
        # --n_resblock 4 \
        # --dim_encoder 1024 \
        # --batch_size 2048 \
        # --loss rdloss \
        # --lmbda $lmbda

        # --architecture nwc_scale_cond \
        # --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/scaleH_sig0.0001_std_rnormed_with_col_std_lidx_row_1024.pt \
        # --dataset block_seq_scale_cond \
        # --iter 100000 \
        # --input_size 128 \
        # --M 144 \
        # --n_resblock 4 \
        # --dim_encoder 1024 \
        # --batch_size 768 \
        # --loss rdloss \
        # --lmbda $lmbda


    # --architecture nwc_ql_ltc \
    # --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/col_1024_gaussian_padding.pt \
    # --dataset block_seq_ql_random \
    # --iter 2 \
    # --run_name test \
    # --input_size 16 \
    # --M 16 \
    # --Q 4 \
    # --dim_encoder 512 \
    # --batch_size 512 \
    # --loss rdloss_ql \
    # --run_name test \
    # --lmbda $lmbda


    # --architecture nwc_scale_cond \
    # --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/scaleH_sig0.0001_std_rnormed_with_col_std_lidx_row_1024.pt \
    # --dataset block_seq_scale_cond \
    # --iter 200000 \
    # --input_size 128 \
    # --M 256 \
    # --n_resblock 4 \
    # --dim_encoder 1024 \
    # --batch_size 2048 \
    # --loss rdloss \
    # --lmbda $lmbda
    

    # --architecture nwc_scale_cond \
    # --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/scaleH_sig0.0001_std_rnormed_with_col_std_lidx_row_1024.pt \
    # --dataset block_seq_scale_cond \
    # --iter 200000 \
    # --input_size 128 \
    # --M 256 \
    # --n_resblock 4 \
    # --dim_encoder 4096 \
    # --batch_size 2048 \
    # --loss rdloss \
    # --lmbda $lmbda
    
###############
    # --architecture nwc_scale_cond \
    # --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/scaleH_sig0.0001_std_rnormed_with_col_std_lidx_row_1024.pt \
    # --dataset block_seq_scale_cond \
    # --run_name use_hyper \
    # --use_hyper \
    # --iter 200000 \
    # --input_size 128 \
    # --M 256 \
    # --n_resblock 4 \
    # --dim_encoder 1024 \
    # --batch_size 2048 \
    # --loss rdloss \
    # --lmbda $lmbda

    # --architecture nwc_scale_cond \
    # --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/scaleH_sig0.0001_with_colrow_std_lidx_row_1024.pt \
    # --dataset block_seq_scale_cond_uniform \
    # --uniform_scale_max 7.3 \
    # --iter 200000 \
    # --input_size 16 \
    # --M 256 \
    # --n_resblock 4 \
    # --dim_encoder 512 \
    # --batch_size 2048 \
    # --loss rdloss \
    # --lmbda $lmbda

    # --architecture nwc_scale_cond \
    # --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/scaleH_sig0.0001_with_colrow_std_lidx_row_1024.pt \
    # --dataset block_seq_scale_cond \
    # --iter 200000 \
    # --input_size 16 \
    # --M 256 \
    # --n_resblock 4 \
    # --dim_encoder 512 \
    # --batch_size 2048 \
    # --loss rdloss \
    # --lmbda $lmbda

    # --architecture nwc_scale_cond \
    # --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/scaleH_sig0.0001_std_rnormed_with_col_std_lidx_row_1024.pt \
    # --dataset block_seq_scale_cond \
    # --iter 200000 \
    # --input_size 128 \
    # --M 256 \
    # --n_resblock 4 \
    # --dim_encoder 4096 \
    # --batch_size 2048 \
    # --loss rdloss \
    # --lmbda $lmbda

    # --architecture nwc_scale_cond \
    # --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/scaleH_sig0.0001_std_rnormed_with_col_std_lidx_row_1024.pt \
    # --dataset block_seq_scale_cond \
    # --iter 200000 \
    # --input_size 16 \
    # --M 256 \
    # --n_resblock 4 \
    # --dim_encoder 512 \
    # --batch_size 2048 \
    # --loss rdloss \
    # --lmbda $lmbda

    # --architecture nwc_scale_cond \
    # --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/scaleH_sig0.0001_std_rnormed_with_col_std_lidx_row_1024.pt \
    # --dataset block_seq_scale_cond_uniform \
    # --uniform_scale_max 31.6 \
    # --iter 200000 \
    # --input_size 128 \
    # --M 256 \
    # --n_resblock 4 \
    # --dim_encoder 1024 \
    # --batch_size 2048 \
    # --loss rdloss \
    # --lmbda $lmbda

    # --architecture nwc_scale_cond \
    # --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/scaleH_sig0.0001_std_rnormed_with_col_std_lidx_row_1024.pt \
    # --dataset block_seq_scale_cond \
    # --iter 200000 \
    # --input_size 128 \
    # --M 256 \
    # --n_resblock 4 \
    # --dim_encoder 1024 \
    # --batch_size 2048 \
    # --loss rdloss \
    # --lmbda $lmbda

    # --architecture nwc_ql \
    # --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/scaleHinv_sig0.0001_std_rnormed_lidx_row_1024.pt \
    # --dataset block_seq_ql_random_pos \
    # --iter 200000 \
    # --input_size 128 \
    # --M 256 \
    # --dim_encoder 1024 \
    # --batch_size 2048 \
    # --loss rdloss_ql --Q 4 \
    # --lmbda $lmbda

    # --architecture nwc_ql \
    # --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/scaleH_sig0.0001_std_rnormed_lidx_row_1024.pt \
    # --dataset block_seq_ql_random_pos \
    # --iter 200000 \
    # --input_size 128 \
    # --M 256 \
    # --dim_encoder 1024 \
    # --batch_size 2048 \
    # --loss rdloss_ql --Q 4 \
    # --lmbda $lmbda

    # --architecture nwc_qmap3 \
    # --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/col_1024_gaussian_padding.pt \
    # --dataset block_seq_qmap \
    # --iter 200000 \
    # --input_size 16 \
    # --M 16 \
    # --dim_encoder 512 \
    # --batch_size 2048 \
    # --loss rdloss_qmap2 \
    # --lmbda $lmbda

    # --architecture nwc_ql \
    # --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/col_1024_gaussian_padding.pt \
    # --dataset block_seq_ql_random \
    # --run_name M32_ \
    # --iter 200000 \
    # --input_size 16 \
    # --M 32 \
    # --Q 4 \
    # --dim_encoder 512 \
    # --batch_size 2048 \
    # --loss rdloss_ql \
    # --lmbda $lmbda

    # --architecture nwc_qmap2 \
    # --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/col_1024_gaussian_padding.pt \
    # --dataset block_seq_qmap \
    # --iter 200000 \
    # --input_size 16 \
    # --M 17 \
    # --dim_encoder 512 \
    # --batch_size 2048 \
    # --loss rdloss_qmap2 \
    # --lmbda $lmbda

    # --architecture nwc_ql \
    # --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/scaleH_sig0.0001_rnormed_row_1024.pt \
    # --dataset block_seq_ql_random \
    # --iter 200000 \
    # --input_size 128 \
    # --M 128 \
    # --dim_encoder 1024 \
    # --batch_size 2048 \
    # --loss rdloss_ql --Q 4 \
    # --lmbda $lmbda

    # --architecture nwc_ql \
    # --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/rnormed_col_1024.pt \
    # --dataset block_seq_ql_random \
    # --iter 200000 \
    # --input_size 128 \
    # --M 64 \
    # --dim_encoder 512 \
    # --batch_size 2048 \
    # --loss rdloss_ql --Q 4 \
    # --lmbda $lmbda

    # --architecture nwc_ql \
    # --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/scaleH_sig0.0001_rnormed_row_1024.pt \
    # --dataset block_seq_ql_random \
    # --iter 200000 \
    # --input_size 16 \
    # --M 16 \
    # --dim_encoder 512 \
    # --batch_size 2048 \
    # --loss rdloss_ql --Q 4 \
    # --lmbda $lmbda

    # --architecture nwc_qmap \
    # --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/col_1024_gaussian_padding.pt \
    # --dataset block_seq_qmap \
    # --iter 200000 \
    # --input_size 16 \
    # --M 17 \
    # --dim_encoder 512 \
    # --batch_size 2048 \
    # --loss rdloss_qmap2 \
    # --lmbda $lmbda

    # --architecture nwc_ql_ste \
    # --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/col_1024_gaussian_padding.pt \
    # --dataset block_seq_ql_random \
    # --iter 200000 \
    # --input_size 16 \
    # --M 16 \
    # --Q 4 \
    # --dim_encoder 512 \
    # --batch_size 2048 \
    # --loss rdloss_ql \
    # --lmbda $lmbda

    # --architecture nwc_qmap \
    # --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/col_1024_gaussian_padding.pt \
    # --dataset block_seq_qmap \
    # --iter 200000 \
    # --input_size 16 \
    # --M 17 \
    # --dim_encoder 512 \
    # --batch_size 2048 \
    # --loss rdloss_qmap \
    # --lmbda_min $min \
    # --lmbda_max $max

    # --architecture nwc_ql2 \
    # --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/col_1024_gaussian_padding.pt \
    # --dataset block_vec_ql_random \
    # --run_name learnable_scale_no_norm \
    # --iter 200000 \
    # --input_size 16 \
    # --M 16 \
    # --dim_encoder 512 \
    # --batch_size 2048 \
    # --loss rdloss_ql --Q 4 \
    # --lmbda $lmbda

    # --architecture nwc_ql \
    # --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/col_1024_gaussian_padding.pt \
    # --dataset block_seq_ql_random \
    # --run_name use_hyper \
    # --use_hyper \
    # --iter 200000 \
    # --input_size 16 \
    # --M 16 \
    # --dim_encoder 512 \
    # --batch_size 2048 \
    # --loss rdloss_ql --Q 4 \
    # --lmbda $lmbda

    # --architecture nwc_qmap \
    # --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/col_1024_gaussian_padding.pt \
    # --dataset block_seq_qmap \
    # --iter 200000 \
    # --input_size 16 \
    # --M 17 \
    # --dim_encoder 512 \
    # --batch_size 2048 \
    # --loss rdloss_qmap \
    # --lmbda_min $min \
    # --lmbda_max $max

    # --architecture nwc_ql_pe \
    # --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/col_1024_idx_ltype_stats.pt \
    # --dataset block_seq_ql_random_pos \
    # --iter 200000 \
    # --input_size 16 \
    # --M 16 \
    # --dim_encoder 512 \
    # --batch_size 2048 \
    # --loss rdloss_ql --Q 4 \
    # --lmbda $lmbda

    # --architecture nwc_ql \
    # --dataset_path ../Wparam_dataset/block_pt/llama8b+7b/droplast_modelwise_norm2_col_1024.pt \
    # --dataset block_seq_ql_random \
    # --iter 200000 \
    # --input_size 16 \
    # --M 16 \
    # --dim_encoder 512 \
    # --batch_size 2048 \
    # --loss rdloss_ql --Q 4 \
    # --lmbda $lmbda

    # --architecture nwc_ql \
    # --dataset_path ../Wparam_dataset/block_pt/llama_8b_7b/droplast_col_1024.pt \
    # --dataset block_seq_ql_random \
    # --iter 200000 \
    # --input_size 16 \
    # --M 16 \
    # --dim_encoder 512 \
    # --batch_size 2048 \
    # --loss rdloss_ql --Q 4 \
    # --lmbda $lmbda


    # --architecture nwc_ql \
    # --dataset_path ../Wparam_dataset/block_pt/meta-llama--Llama-2-7b-hf/droplast_col_1024.pt \
    # --dataset block_seq_ql_random \
    # --iter 200000 \
    # --input_size 16 \
    # --M 16 \
    # --dim_encoder 512 \
    # --batch_size 2048 \
    # --loss rdloss_ql --Q 4 \
    # --lmbda $lmbda

    # --architecture nwc_ql \
    # --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/col_1024_gaussian_padding.pt \
    # --pretrained_path ./checkpoint/nwc_ql/block_seq_ql_random_scaler_gaussian__llama8b_col_1024.pt/rdloss_ql_size16_encdim512_M16_Q4_R0_m0_batch_size2048_total_iter200000_lr0.0001_seed100/lmbda${lmbda}_*/best_loss_model_*.pth.tar \
    # --run_name gaussian_8b \
    # --dataset block_seq_ql_random \
    # --iter 20000 \
    # --input_size 16 \
    # --M 16 \
    # --dim_encoder 512 \
    # --batch_size 2048 \
    # --loss rdloss_ql --Q 4 \
    # --lmbda $lmbda

    # --architecture nwc_ql \
    # --dataset_path ../Wparam_dataset/block_pt/meta-llama--Llama-2-7b-hf/droplast_col_1024.pt \
    # --pretrained_path ./checkpoint/nwc_ql/block_seq_ql_random_scaler_gaussian__llama8b_col_1024.pt/rdloss_ql_size16_encdim512_M16_Q4_R0_m0_batch_size2048_total_iter200000_lr0.0001_seed100/lmbda${lmbda}_*/best_loss_model_*.pth.tar \
    # --run_name gaussian_7b_droplast \
    # --dataset block_seq_ql_random \
    # --iter 20000 \
    # --input_size 16 \
    # --M 16 \
    # --dim_encoder 512 \
    # --batch_size 2048 \
    # --loss rdloss_ql --Q 4 \
    # --lmbda $lmbda

    # --architecture nwc_ql \
    # --dataset_path ../Wparam_dataset/block_pt/gaussian/llama8b_col_1024.pt \
    # --dataset block_seq_ql_random \
    # --iter 200000 \
    # --input_size 16 \
    # --M 16 \
    # --Q 4 \
    # --dim_encoder 512 \
    # --batch_size 2048 \
    # --loss rdloss_ql \
    # --lmbda $lmbda

    # --architecture nwc_ql \
    # --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/col_1024_gaussian_padding.pt \
    # --dataset block_seq_ql_random \
    # --iter 200000 \
    # --run_name no_lnorm \
    # --input_size 16 \
    # --M 16 \
    # --Q 4 \
    # --no_layernorm \
    # --dim_encoder 512 \
    # --batch_size 2048 \
    # --loss rdloss_ql \
    # --lmbda $lmbda

    # --architecture nwc_ql2 \
    # --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/col_1024_gaussian_padding.pt \
    # --dataset block_vec_ql_random \
    # --iter 2 \
    # --save_dir test \
    # --input_size 16 \
    # --M 16 \
    # --dim_encoder 512 \
    # --batch_size 2048 \
    # --loss rdloss_ql --Q 4 \
    # --lmbda $lmbda

    # --architecture nwc_ql \
    # --dataset_path ../Wparam_dataset/block_pt/meta-llama--Llama-2-7b-hf/droplast_col_1024.pt \
    # --pretrained_path ./checkpoint/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt/M16/lmbda${lmbda}_*/best_loss_model_*.pth.tar \
    # --run_name 8b_7b_droplast \
    # --dataset block_seq_ql_random \
    # --iter 20000 \
    # --input_size 16 \
    # --M 16 \
    # --dim_encoder 512 \
    # --batch_size 2048 \
    # --loss rdloss_ql --Q 4 \
    # --lmbda $lmbda

    # --architecture nwc_ql \
    # --dataset_path ../Wparam_dataset/block_pt/meta-llama--Llama-2-7b-hf/col_1024.pt \
    # --pretrained_path ./checkpoint/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__shuffled_col_1024.pt/rdloss_ql_size16_encdim512_M16_Q4_R0_m0_batch_size2048_total_iter200000_lr0.0001_seed100/lmbda${lmbda}_*/best_loss_model_*.pth.tar \
    # --run_name 8b_shuffle_7b \
    # --dataset block_seq_ql_random \
    # --iter 20000 \
    # --input_size 16 \
    # --M 16 \
    # --dim_encoder 512 \
    # --batch_size 2048 \
    # --loss rdloss_ql --Q 4 \
    # --lmbda $lmbda

    # --architecture nwc_ql \
    # --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/shuffled_col_1024.pt \
    # --dataset block_seq_ql_random \
    # --iter 200000 \
    # --input_size 16 \
    # --M 16 \
    # --dim_encoder 512 \
    # --batch_size 2048 \
    # --loss rdloss_ql --Q 4 \
    # --lmbda $lmbda

    # --architecture nwc_ql \
    # --dataset_path ../Wparam_dataset/block_pt/meta-llama--Llama-2-7b-hf/col_1024.pt \
    # --pretrained_path ./checkpoint/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt/M16/lmbda${lmbda}_*/best_loss_model_*.pth.tar \
    # --run_name llama8b_c1024_7b \
    # --dataset block_seq_ql_random \
    # --iter 20000 \
    # --input_size 16 \
    # --M 16 \
    # --dim_encoder 512 \
    # --batch_size 2048 \
    # --loss rdloss_ql --Q 4 \
    # --lmbda $lmbda

    # --architecture nwc_ql \
    # --dataset_path ../Wparam_dataset/block_pt/openai--clip-vit-large-patch14/vision_text_col_256.pt \
    # --pretrained_path ./checkpoint/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt/M16/lmbda${lmbda}_*/best_loss_model_*.pth.tar \
    # --run_name clip_llama8b_col1024_pretrained \
    # --dataset block_seq_ql_random \
    # --iter 100000 \
    # --input_size 16 \
    # --M 16 \
    # --dim_encoder 512 \
    # --batch_size 4096 \
    # --loss rdloss_ql --Q 4 \
    # --lmbda $lmbda

    # --architecture nwc_ql \
    # --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/col_1024_gaussian_padding.pt \
    # --dataset block_seq_ql_random \
    # --iter 200000 \
    # --input_size 16 \
    # --M 32 \
    # --Q 4 \
    # --dim_encoder 512 \
    # --batch_size 2048 \
    # --loss rdloss_ql \
    # --lmbda 1000

    # --architecture nwc_ql \
    # --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/col_1024_colwise_normed.pt \
    # --dataset block_seq_ql_random \
    # --iter 200000 \
    # --input_size 16 \
    # --M 64 \
    # --Q 4 \
    # --dim_encoder 512 \
    # --batch_size 1024 \
    # --loss rdloss_ql \
    # --lmbda 30
    

    # --architecture nwc_ql \
    # --dataset_path ../Wparam_dataset/block_pt/openai--clip-vit-large-patch14/vision_text_col_256.pt \
    # --dataset block_seq_ql_random \
    # --iter 100000 \
    # --input_size 16 \
    # --M 16 \
    # --dim_encoder 512 \
    # --batch_size 4096 \
    # --loss rdloss_ql --Q 4 \
    # --lmbda 100

    # --architecture nwc \
    # --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/scaled3_RHT_sig1e-06_col_1024.pt \
    # --dataset block_seq \
    # --iter 200000 \
    # --input_size 16 \
    # --M 16 \
    # --dim_encoder 512 \
    # --batch_size 2048 \
    # --loss rdloss \
    # --lmbda 30

    # --architecture nwc_ql_cdt \
    # --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/col_1024_layerwise_stats.pt \
    # --dataset block_seq_ql_random_lstats \
    # --iter 200000 \
    # --input_size 16 \
    # --M 16 \
    # --Q 4 \
    # --C 7 \
    # --dim_encoder 512 \
    # --batch_size 2048 \
    # --loss rdloss_ql \
    # --lmbda 1000

    # --architecture nwc_ql_bn \
    # --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/col_1024_gaussian_padding.pt \
    # --dataset block_seq \
    # --iter 200000 \
    # --input_size 16 \
    # --M 16 \
    # --Q 4 \
    # --dim_encoder 512 \
    # --batch_size 2048 \
    # --loss rdloss_ql \
    # --lmbda 1000

    # --architecture nwc \
    # --dataset_path ../Wparam_dataset/block_pt/meta-llama--Llama-2-7b-hf/scaled3_RHT_sig0.0001_col_4096.pt \
    # --dataset block_seq \
    # --iter 200000 \
    # --input_size 16 \
    # --M 16 \
    # --dim_encoder 512 \
    # --batch_size 1024 \
    # --loss rdloss \
    # --lmbda 10

    # --architecture nwc_ql \
    # --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/col_1024_gaussian_padding.pt \
    # --dataset block_seq_ql_random \
    # --iter 200000 \
    # --input_size 16 \
    # --M 16 \
    # --Q 4 \
    # --dim_encoder 512 \
    # --batch_size 2048 \
    # --loss rdloss_ql \
    # --lmbda 10

    # --architecture nwc_ql \
    # --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/col_1024_gaussian_padding.pt \
    # --dataset block_seq_ql_random \
    # --iter 200000 \
    # --input_size 16 \
    # --M 16 \
    # --Q 4 \
    # --dim_encoder 512 \
    # --batch_size 2048 \
    # --loss rdloss_ql \
    # --lmbda 50

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