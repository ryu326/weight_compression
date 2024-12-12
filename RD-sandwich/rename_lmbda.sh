#!/bin/bash

# 대상 디렉토리 설정
target_dir="/home/jgryu/Weight_compression/RD-sandwich/checkpoints_v3/llama3_8B_out-in-lier/attn_d256_zscore4_outlier/llama3-8B_d256_b1024_e150_lr1e-5_normalize"

# 디렉토리 내 모든 파일 처리
for file in "$target_dir"/*; do
  # 파일 이름에서 'lmbda=<integer>' 부분을 찾아 'lmbda=<float>'로 변경
  if [[ $file =~ lmbda=([0-9]+) ]]; then
    new_file=$(echo "$file" | sed -E 's/lmbda=([0-9]+)/lmbda=\1.0/')
    # 파일 이름 변경
    mv "$file" "$new_file"
    echo "Renamed: $file -> $new_file"
  fi
done

target_dir="/home/jgryu/Weight_compression/RD-sandwich/checkpoints_v3/llama3_8B_out-in-lier/mlp_d256_zscore4_outlier/llama3-8B_d256_b1024_e150_lr1e-5_normalize"

# 디렉토리 내 모든 파일 처리
for file in "$target_dir"/*; do
  # 파일 이름에서 'lmbda=<integer>' 부분을 찾아 'lmbda=<float>'로 변경
  if [[ $file =~ lmbda=([0-9]+) ]]; then
    new_file=$(echo "$file" | sed -E 's/lmbda=([0-9]+)/lmbda=\1.0/')
    # 파일 이름 변경
    mv "$file" "$new_file"
    echo "Renamed: $file -> $new_file"
  fi
done