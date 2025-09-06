#!/bin/bash

# ==============================================================================
# Llama-2-7b-hf 모델에 대해 다양한 비트 수로 RTN 양자화를 실행하는 스크립트
# ==============================================================================

# --- 설정 (Configuration) ---

export CUDA_VISIBLE_DEVICES=3

# 모델 및 경로 설정
MODEL_TYPE="llama"
MODEL_NAME="meta-llama--Llama-2-7b-hf"
MODEL_PATH="/workspace/Weight_compression/Wparam_dataset/hf_model/${MODEL_NAME}"
BASE_OUTPUT_DIR="/workspace/Weight_compression/hf_model_comp/RTN"

# 양자화 파라미터 설정
QUANT_TYPE="group"
GROUP_SIZE=128

# 양자화를 실행할 비트 리스트
BITS_TO_RUN=(3 4 5 6 7 8)


# 스크립트 시작 메시지
echo "🚀 Starting RTN quantization process for ${MODEL_NAME}"
echo "=========================================================="

# 지정된 비트 리스트를 순회하며 양자화 실행
for bit in "${BITS_TO_RUN[@]}"; do
    
    # 결과물을 저장할 최종 경로 생성
    OUTPUT_PATH="${BASE_OUTPUT_DIR}/${MODEL_NAME}_W${bit}g${GROUP_SIZE}"
    
    echo "🔄 Running quantization for ${bit}-bit..."
    echo "   - Model: ${MODEL_PATH}"
    echo "   - Output: ${OUTPUT_PATH}"
    
    # 파이썬 스크립트 실행
    python ${PYTHON_SCRIPT} \
        --model_type ${MODEL_TYPE} \
        --model_path ${MODEL_PATH} \
        --output_path ${OUTPUT_PATH} \
        --num_bits ${bit} \
        --quant_type ${QUANT_TYPE} \
        --group_size ${GROUP_SIZE}
        
    # 실행 결과 확인
    if [ $? -eq 0 ]; then
        echo "✅ Successfully completed ${bit}-bit quantization."
    else
        echo "❌ Error during ${bit}-bit quantization. Aborting."
        exit 1
    fi
    
    echo "----------------------------------------------------------"

done

echo "🎉 All quantization tasks are complete."
echo "=========================================================="