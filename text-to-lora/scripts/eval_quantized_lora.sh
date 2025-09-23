# --- 설정 변수 ---
# 환경 변수 설정
export HF_HOME=/workspace/hf_cache/huggingface_nwc
SAVE_PATH="/workspace/Weight_compression/text-to-lora/train_outputs/qunat_lora/group128"
MODE="group"              
LOG_DIR="./logs"             

MODE="group"
GPU_LIST=(0 1 2 3)
TASKS=(1 2 3 4 5 6 7 8)


mkdir -p "$LOG_DIR"
NUM_GPUS=${#GPU_LIST[@]}

echo "로그 파일은 '${LOG_DIR}' 디렉토리에 저장됩니다."
echo "-------------------------------------------------"

for i in "${!TASKS[@]}"; do
    gpu_id=${GPU_LIST[$((i % NUM_GPUS))]}
    bit_val=${TASKS[$i]}
    
    log_file="${LOG_DIR}/eval_bit_${bit_val}.log"

    echo ">> 시작: bit=${bit_val}  GPU=${gpu_id} (로그: ${log_file})"

    (
        CUDA_VISIBLE_DEVICES=$gpu_id uv run python scripts/eval_quantized_lora.py \
            --save_dir "$SAVE_PATH" \
            --mode "$MODE" \
            --group 128 \
            --full_eval \
            --bit "$bit_val"
    ) > "$log_file" 2>&1 &

    if (((i + 1) % NUM_GPUS == 0)); then
        echo "--- GPU 배치 완료. 현재 작업들이 끝날 때까지 대기합니다... ---"
        wait
    fi
done

wait
echo "✅ 모든 작업이 완료되었습니다. '${LOG_DIR}' 디렉토리에서 로그를 확인하세요."





# for b in 1 2 3 4 5 6 7 8
# uv run python scripts/eval_quantized_lora.py \
#     --save_dir "$SAVE_PATH" \
#     --mode "$MODE" \
#     --group 128 \
#     --full_eval \
#     --bit $b


# 고정할 인자 값 설정
# CHECKPOINT_PATH="/workspace/Weight_compression/text-to-lora/train_outputs/compnet_recon/compnet/quant23/comp_model.pt"
# CHECKPOINT_PATH="/workspace/Weight_compression/text-to-lora/train_outputs/qunat_lora/quant2/comp_model.pt"
# MODE="group"

# # {2..8}은 2 3 4 5 6 7 8 과 동일합니다.
# for bit in 2 3 3 4 5 6 7 8
# do
#     echo "=================================================="
#     echo "Running evaluation for bit=${bit}, mode=${MODE}"
#     echo "=================================================="

#     uv run python scripts/eval_quantized_lora.py \
#         --checkpoint_path "$CHECKPOINT_PATH" \
#         --bit "$bit" \
#         --mode "$MODE" \
#         --group 128 > ./logs/quant_group128_${bit}bit.log 2>&1
#         # --full_eval
# done

# nohup sh scripts/eval_quantized_lora.sh > ./logs/quant_eval.log 2>&1 &
# nohup sh scripts/eval_quantized_lora1.sh > ./logs/quant_eval1.log 2>&1 &
# nohup sh scripts/eval_quantized_lora2.sh > ./logs/quant_eval2.log 2>&1 &
# nohup sh scripts/eval_quantized_lora3.sh > ./logs/quant_eval3.log 2>&1 &