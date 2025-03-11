#!/bin/bash

# uniform_noise 디렉토리에서 모든 하위 디렉토리 경로를 가져오기
# base_dir="../model_cache_reconstructed/uniform_noise/exp_magnitude/r0.1"
# base_dir="../model_cache_reconstructed/vq_seedlm_/mlp_16_row_dataset.pt_v101"
# base_dir='../model_cache_reconstructed/vqvae_part2/mlp_attn_16_row_dataset.pt'
# base_dir='../model_cache_reconstructed/vqvae_idx_v1/per_row_16_calib'
# base_dir='../model_cache_reconstructed/vqvae_corrected_scale/mlp_attn_16_row_dataset.pt'
# base_dir='../model_cache_reconstructed/seedlm/bpp4.0_C8_P3_K16'
# base_dir='../model_cache_reconstructed/vqvae_idx_v2/per_row_16_calib'
# base_dir='/home/jgryu/Weight_compression/model_cache_reconstructed/vqvae_idx_col_scale_input/per_col_16_calib'
# base_dir='/home/jgryu/Weight_compression/model_cache_reconstructed/vqvae_idx_v2_recent/per_row_16_calib'
# base_dir='/home/jgryu/Weight_compression/model_cache_reconstructed/vqvae_idx_v2_random_all/per_row_16_calib'
base_dir='../model_reconstructed/vqvae_mag/col_16_calib'
base_dir='/home/jgryu/Weight_compression/model_reconstructed/vqvae_mag_zeroinput'
base_dir='/home/jgryu/Weight_compression/model_cache_reconstructed/vqvae_idx_col_keep_random/r0.1_per_col_16_calib'
base_dir='/home/jgryu/Weight_compression/model_cache_reconstructed/vqvae_idx/row_salient/r0.1_per_row_16_calib'
base_dir='../model_reconstructed/vqvae_mag_vec/row_16_calib'
base_dir='/home/jgryu/Weight_compression/model_reconstructed/vqvae_mag_vec_random/row_16_calib'
base_dir='/home/jgryu/Weight_compression/model_cache_reconstructed/vqvae_idx/col_random_idx/per_col_16_calib'
pretrain_paths=($(find "$base_dir" -type d))
# 평가 태스크 설정
tasks=("wikitext")

# GPU 설정
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 경로 배열을 반복
for pretrain_path in "${pretrain_paths[@]}"
do
    # 로그 파일 경로 설정
    log_path="${pretrain_path}_result.txt"

    # 로그 디렉토리 생성
    log_dir=$(dirname "$log_path")
    mkdir -p "$log_dir"

    # 태스크별로 평가 실행
    for task in "${tasks[@]}"
    do
        echo "Running evaluation for directory: $pretrain_path on task: $task"

        # 평가 실행 및 로그 저장
        lm_eval --model hf \
            --model_args pretrained=$pretrain_path,parallelize=True \
            --tasks $task \
            --batch_size 1 | tee -a "$log_path"
    done

done

echo "Evaluation completed for all directories and tasks."


# export CUDA_VISIBLE_DEVICES=0,1,2,3 
# lm_eval --model hf \
#     --model_args pretrained=/home/jgryu/Weight_compression/model_cache_reconstructed/vqvae_idx_v1/per_row_16_calib/bpp8.0_size16_smse_neNone_de16_K8_P16_encdim512_batch_size2048_total_iter2000000_lr0.0001_seed100/best_mse_model_MSE_0.00016_total_iter_1900000.pth.tar,parallelize=True \
#     --tasks wikitext \
#     --batch_size 1 | tee -a "/home/jgryu/Weight_compression/model_cache_reconstructed/vqvae_idx_v1/per_row_16_calib/bpp8.0_size16_smse_neNone_de16_K8_P16_encdim512_batch_size2048_total_iter2000000_lr0.0001_seed100/best_mse_model_MSE_0.00016_total_iter_1900000.pth.tar_result"

# #!/bin/bash

# # 사전 학습 모델 경로 배열
# declare -a pretrain_paths=(
#     # "../model_cache_reconstructed/vq_seedlm_/mlp_16_row_dataset.pt/size16_ne256_P4_batch_size512_total_iter2000000_lr0.0001_seed100/best_mse_model_MSE_0.28962_total_iter_1250000.pth.tar"
#     # "../model_cache_reconstructed/vq_seedlm_/mlp_16_row_dataset.pt/size16_ne512_P4_batch_size512_total_iter2000000_lr0.0001_seed100/best_mse_model_MSE_0.11122_total_iter_2000000.pth.tar"
#     # "../model_cache_reconstructed/vq_seedlm_/mlp_16_row_dataset.pt/size16_ne512_P32_batch_size512_total_iter2000000_lr0.0001_seed100/best_mse_model_MSE_0.0_total_iter_1750000.pth.tar"
#     # "../model_cache_reconstructed/vq_seedlm_/mlp_16_row_dataset.pt/size16_ne512_P16_batch_size512_total_iter2000000_lr0.0001_seed100/best_mse_model_MSE_0.00191_total_iter_2000000.pth.tar"
#     # "../model_cache_reconstructed/vq_seedlm_/mlp_16_row_dataset.pt/size16_ne512_P8_batch_size512_total_iter2000000_lr0.0001_seed100/best_mse_model_MSE_0.01937_total_iter_1700000.pth.tar"
#     # "../model_cache_reconstructed/vq_seedlm_/mlp_16_row_dataset.pt/size16_ne512_de256_P4_batch_size512_total_iter2000000_lr0.0001_seed100/best_mse_model_MSE_0.07395_total_iter_1850000.pth.tar"
#     "../model_cache_reconstructed/test/original_llama"
# )

# # 평가 태스크 설정
# # tasks=("wikitext" "arc_easy" "arc_challenge" "winogrande" "boolq" "hellaswag")
# tasks=("wikitext")

# # GPU 설정
# export CUDA_VISIBLE_DEVICES=0,1,2,3

# # 경로 배열을 반복
# for pretrain_path in "${pretrain_paths[@]}"
# do
#     # 로그 파일 경로 설정
#     log_path="${pretrain_path}_result.txt"

#     # 로그 디렉토리 생성
#     log_dir=$(dirname "$log_path")
#     mkdir -p "$log_dir"

#     # 태스크별로 평가 실행
#     for task in "${tasks[@]}"
#     do
#         echo "Running evaluation for model: $pretrain_path on task: $task"

#         # 평가 실행 및 로그 저장
#         lm_eval --model hf \
#             --model_args pretrained=$pretrain_path,parallelize=True \
#             --tasks $task \
#             --batch_size 1 | tee -a "$log_path"
#     done

# done

# echo "Evaluation completed for all models and tasks."