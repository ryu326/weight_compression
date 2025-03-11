base_dirs=(
    # '../model_reconstructed/nwc_ql/block_seq_ql_random_col_16'
    '../model_reconstructed/nwc/block_seq_row_16'
    # '../model_reconstructed/nwc_hp'
    # '../model_reconstructed/nwc/gaussian_seq_row_16/lmbda30000_rdloss_encdim512_M16_batch_size2048_total_iter200000_lr0.0001_seed100'
    # '../model_reconstructed/nwc_hp/block_seq_row_16/lmbda50_rdloss_encdim512_M16_batch_size2048_total_iter200000_lr0.0001_seed100'
    # '../model_reconstructed/nwc_ql/block_seq_ql_random_col_16/lmbda10000_rdloss_ql_encdim512_M16_batch_size2048_total_iter200000_lr0.0001_seed100/best_loss_model_loss_10.96029_bpp_6.2788_MSE_0.0004_total_iter_140000.pth.tar/COL_MSE_0.00051_bpploss5.94951'
)
# export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=3

# base_dir 배열을 반복
for base_dir in "${base_dirs[@]}"
do
    # pretrain_paths=($(find "$base_dir" -type f -name "tokenizer.json" -exec dirname {} \; | sort -u))
    pretrain_paths=($(find "$base_dir" -type f -name "tokenizer.json" -exec dirname {} \; | sort -u | grep 'ROWCOL'))
    for pretrain_path in "${pretrain_paths[@]}"
    do
        log_path=$(echo "$pretrain_path" | sed 's|model_reconstructed|model_eval|')_quip_result.txt

        log_dir=$(dirname "$log_path")
        mkdir -p "$log_dir"

        echo "Running evaluation for directory: $pretrain_path"

        python ../quip-sharp/eval/eval_ppl.py \
            --hf_path $pretrain_path \
            --seqlen 2048 \
            --no_use_cuda_graph | tee -a "$log_path"
    done
done

# base_dir='../model_reconstructed/nwc_ql_test'
# base_dir='../model_reconstructed/nwc/gaussian_row_16/lmbda50_rdloss_encdim512_M16_batch_size4096_total_iter1500000_lr0.0001_seed100'
# base_dir='../model_reconstructed/nwc/block_col_128'
# base_dir='../model_reconstructed/nwc/block_seq_row_16/lmbda200_rdloss_encdim512_M16_batch_size2048_total_iter200000_lr0.0001_seed100'
# base_dir='../model_reconstructed/nwc/block_row_16/lmbda50_rdloss_encdim512_M16_batch_size4096_total_iter1500000_lr0.0001_seed100'
# base_dir='../model_reconstructed/nwc_ql/block_seq_ql_random_col_16'
# base_dir='../model_reconstructed/nwc/gaussian_seq_row_16'

# # pretrain_paths=($(find "$base_dir" -type f -name "*.pth.tar*" -exec dirname {} \; | sort -u))
# # pretrain_paths=($(find "$base_dir" -type d))
# pretrain_paths=($(find "$base_dir" -type f -name "config.json" -exec dirname {} \; | sort -u))


# export CUDA_VISIBLE_DEVICES=0,1,2,3

# # 경로 배열을 반복
# for pretrain_path in "${pretrain_paths[@]}"
# do
#     log_path=$(echo "$pretrain_path" | sed 's|model_reconstructed|model_eval|')_quip_result.txt

#     log_dir=$(dirname "$log_path")
#     mkdir -p "$log_dir"

#     echo "Running evaluation for directory: $pretrain_path"

#     python ../quip-sharp/eval/eval_ppl.py \
#         --hf_path $pretrain_path \
#         --seqlen 2048 \
#         --no_use_cuda_graph | tee -a "$log_path"

# done

# echo "Evaluation completed for all directories and tasks."

