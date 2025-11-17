#!/bin/bash
set -e

# Hugging Face read token (필요 시 수정)
HF_TOKEN="YOUR_HF_READ_TOKEN"

# 첫 번째 모델 다운로드
python download_hf.py \
  --folder_path "/home/jgryu/workspace/weight_compression/quip_hess/Wparam_dataset/Llama-3.2-1B-Instruct-Hessians" \
  --repo_id "relaxml/Llama-3.2-1B-Instruct-Hessians" \

# 두 번째 모델 다운로드
python download_hf.py \
  --folder_path "/home/jgryu/workspace/weight_compression/quip_hess/Wparam_dataset/Llama-3.2-3B-Instruct-Hessians" \
  --repo_id "relaxml/Llama-3.2-3B-Instruct-Hessians" \
