lmbda=50
CUDA_VISIBLE_DEVICES=0 taskset -c 0-7 python -u train_nwc.py \
    --architecture nwc_ql \
    --dataset_path ../Wparam_dataset/block_pt/llama8b+7b/droplast_modelwise_norm2_col_1024.pt \
    --dataset block_seq_ql_random \
    --iter 200000 \
    --input_size 16 \
    --M 16 \
    --dim_encoder 512 \
    --batch_size 2048 \
    --loss rdloss_ql --Q 4 \
    --lmbda $lmbda

    
    
###############
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