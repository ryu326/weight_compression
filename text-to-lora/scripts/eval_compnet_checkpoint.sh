#!/bin/bash

export HF_HOME=/workspace/hf_cache/huggingface_nwc

GPU_LIST=(0 1 2 3)

paths=(
    /workspace/Weight_compression/text-to-lora/train_outputs/compnet_recon/compnet/v4_ld5.0_20250912-042046_Q1ExEe7d/comp_model.pt
    /workspace/Weight_compression/text-to-lora/train_outputs/compnet_recon/compnet/v4_ld10.0_20250911-174317_AF5XXlWL/comp_model.pt
    /workspace/Weight_compression/text-to-lora/train_outputs/compnet_recon/compnet/v4_ld10.0_20250912-041747_ocPnhU1u/comp_model.pt
    /workspace/Weight_compression/text-to-lora/train_outputs/compnet_recon/compnet/v4_ld25.0_20250912-041748_wQ5aPJzv/comp_model.pt
    /workspace/Weight_compression/text-to-lora/train_outputs/compnet_recon/compnet/v4_ld30.0_20250911-174318_smSnX89f/comp_model.pt
    /workspace/Weight_compression/text-to-lora/train_outputs/compnet_recon/compnet/v4_ld50_20250910-093437_m1ygdR2L/comp_model.pt
    /workspace/Weight_compression/text-to-lora/train_outputs/compnet_recon/compnet/v4_ld50.0_20250912-041642_UUEZtyox/comp_model.pt
    /workspace/Weight_compression/text-to-lora/train_outputs/compnet_recon/compnet/v4_ld100_/comp_model.pt
)

LOG_DIR="./logs"

# --- 스크립트 실행 (Execution) ---
mkdir -p "$LOG_DIR"
NUM_GPUS=${#GPU_LIST[@]}

echo "총 ${#paths[@]}개의 체크포인트를 ${NUM_GPUS}개의 GPU에서 병렬로 평가합니다."
echo "로그는 '${LOG_DIR}' 디렉토리에 저장됩니다."
echo "-------------------------------------------------"

# paths 배열의 인덱스(0, 1, 2...)를 순회합니다.
for i in "${!paths[@]}"; do
    gpu_id=${GPU_LIST[$((i % NUM_GPUS))]}
    
    path=${paths[$i]}
    log_name="${LOG_DIR}/eval_$((i + 1)).log"
    echo ">> 시작: 작업 $((i + 1)) (GPU: ${gpu_id}) -> 로그: ${log_name}"
    (
        export CUDA_VISIBLE_DEVICES=$gpu_id
        uv run python scripts/eval_compnet_checkpoint.py \
            --checkpoint_path "$path" \
            --full_eval
    ) > $log_name 2>&1 &

    # GPU 개수만큼의 작업(한 배치)이 모두 실행되었다면, 해당 작업들이 끝날 때까지 대기합니다.
    if (((i + 1) % NUM_GPUS == 0)); then
        echo "--- GPU 배치가 모두 실행되었습니다. 현재 작업들이 끝날 때까지 대기합니다... ---"
        wait
    fi
done

# 마지막에 남은 작업들이 (배치를 채우지 못한 경우) 모두 끝날 때까지 최종 대기합니다.
wait
echo "-------------------------------------------------"
echo "✅ 모든 평가 작업이 완료되었습니다."


# export CUDA_VISIBLE_DEVICES=0
# export HF_HOME=/workspace/hf_cache/huggingface_nwc
# lmbda=100
# uv run python scripts/eval_compnet_checkpoint.py \
#     --checkpoint_path /workspace/Weight_compression/text-to-lora/train_outputs/compnet_recon/compnet/v4_ld${lmbda}_*/comp_model.pt \
#     --full_eval > ./logs/${lmbda}.log