# CUDA_VISIBLE_DEVICES=0,1,2,3 python eval_ppl.py \
#     --seqlen 2048 \
#     --hf_path ../../model_cache/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920 \
#     --no_use_cuda_graph


base_dir='../../model_cache_reconstructed/awq'
base_dir='../../model_cache_reconstructed/vqvae/vqvae_corrected_scale/mlp_attn_16_row_dataset.pt'
base_dir='../../model_cache_reconstructed/vqvae_idx/row_v2/per_row_16_calib'
base_dir='../../model_reconstructed/rtn'
base_dir='../../model_reconstructed/vqvae_qlike/row_16_calib/bpp5.0_size16_nmse_ne32_de1_K5_P16_encdim512_batch_size4096_total_iter1500000_lr0.0001_seed100'
base_dir='../../model_reconstructed/vqvae_idx_mag/row_16_calib_col_recon'
base_dir='../../model_reconstructed/vqvae_idx_mag/row_16_calib'
base_dir='../../model_reconstructed/nwc/block_row_16'
base_dir='../../model_reconstructed/nwc/block_col_128'
# base_dir='../../model_reconstructed/nwc_ql/block_ql_col_16/lmbda1000_rdloss_ql_encdim512_batch_size4096_total_iter1500000_lr0.0001_seed100'
base_dir='../../model_reconstructed/nwc/block_col_128/lmbda100_rdloss_encdim512_batch_size4096_total_iter1500000_lr0.0001_seed100/best_loss_model_loss_4.49518_bpp_4.14475_MSE_0.00727_total_iter_950000.pth.tar_MSE_0.00729_bpploss3.77784'
base_dir='../../model_reconstructed/nwc_ql_random/block_ql_random_col_128'
base_dir='../../model_reconstructed/nwc/row_recon_block_row_16'
base_dir='../../model_reconstructed/vqvae_qlike/col_recon_row_16_calib'
base_dir='../../model_reconstructed/nwc_ql_random/block_ql_random_col_128/lmbda200_rdloss_ql_encdim512_batch_size4096_total_iter1500000_lr0.0001_seed100/best_loss_model_loss_4.96425_bpp_5.75975_MSE_0.00427_total_iter_190000.pth.tar_MSE_0.01025_bpploss3.58498'
base_dir='../../model_reconstructed/nwc/col_recon_block_col_128'
base_dir='../../model_reconstructed/nwc/row_recon_block_col_128'
pretrain_paths=($(find "$base_dir" -type d))

export CUDA_VISIBLE_DEVICES=0,1,2,3

# 경로 배열을 반복
for pretrain_path in "${pretrain_paths[@]}"
do
    # 로그 파일 경로 설정
    log_path="${pretrain_path}_quip_result.txt"

    # 로그 디렉토리 생성
    log_dir=$(dirname "$log_path")
    mkdir -p "$log_dir"


    echo "Running evaluation for directory: $pretrain_path"

    # 평가 실행 및 로그 저장
    python  eval_ppl.py \
        --hf_path $pretrain_path \
        --seqlen 2048 \
        --no_use_cuda_graph | tee -a "$log_path"

done

echo "Evaluation completed for all directories and tasks."