#!/bin/bash

# 원본 디렉토리와 대상 디렉토리 설정
SRC_DIR="../model_reconstructed"
DEST_DIR="../model_eval"

# 파일 이동
find "$SRC_DIR" -type f -name "*.txt" | while read -r file; do
    # 원본 파일의 상대 경로를 계산
    REL_PATH="${file#$SRC_DIR/}"
    # 대상 디렉토리 경로 생성
    DEST_PATH="$DEST_DIR/$REL_PATH"
    # 대상 디렉토리가 존재하지 않으면 생성
    mkdir -p "$(dirname "$DEST_PATH")"
    # 파일 이동
    mv "$file" "$DEST_PATH"
done
