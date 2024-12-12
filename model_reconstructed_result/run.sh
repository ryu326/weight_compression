#!/bin/bash

# Pretrained 모델 경로
pretrain_path=/home/jgryu/Weight_compression/model_cache_reconstructed/vq_seedlm_/mlp_16_row_dataset.pt/size16_ne512_P4_batch_size512_total_iter2000000_lr0.0001_seed100/best_mse_model_MSE_0.11122_total_iter_2000000.pth.tar
# 로그 파일 경로 설정
log_path=${pretrain_path}_result.txt

# 로그 파일 디렉토리 생성
log_dir=$(dirname $log_path)
mkdir -p $log_dir

# lm-evaluation-harness 디렉토리를 PATH에 추가 (현재 세션에만 적용)
export PATH=$PATH:$(realpath ../lm-evaluation-harness)

# 실행 명령
# CUDA_VISIBLE_DEVICES=0,1,2,3 lm_eval --model hf \
#     --model_args pretrained=$pretrain_path,parallelize=True \
#     --tasks wikitext,arc_easy,arc_challenge,winogrande,boolq,hellaswag \
#     --batch_size 1 | tee $log_path
CUDA_VISIBLE_DEVICES=0,1,2,3 lm_eval --model hf \
    --model_args pretrained=$pretrain_path,parallelize=True \
    --tasks arc_easy \
    --batch_size 1 | tee $log_path