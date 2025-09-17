#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="mistralai/Mistral-7B-Instruct-v0.2"
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"
export HF_HOME=/workspace/hf_cache/huggingface_nwc

######################
# GPU 0 작업들
######################
lora_dirs=(
    # "/workspace/Weight_compression/text-to-lora/train_outputs/sft/oracle_lora/['arc_challenge']_20250830-140200_m1TCFSDD"
    # "/workspace/Weight_compression/text-to-lora/train_outputs/sft/oracle_lora/['arc_easy']_20250830-134527_0Wh3Cchk"
    "/workspace/Weight_compression/text-to-lora/train_outputs/sft/oracle_lora/['boolq']_20250830-093642_N5GJ5X8i"
    # "/workspace/Weight_compression/text-to-lora/train_outputs/sft/oracle_lora/['gsm8k']_20250830-145103_9gb0HMns"
    "/workspace/Weight_compression/text-to-lora/train_outputs/sft/oracle_lora/['hellaswag']_20250830-105259_OiEWNa27"
    # "/workspace/Weight_compression/text-to-lora/train_outputs/sft/oracle_lora/['hellaswag']_20250905-001943_rdDMTFMa"
)
tasks=(boolq hellaswag)

for i in "${!lora_dirs[@]}"; do
  export CUDA_VISIBLE_DEVICES=0
  log_file="${LOG_DIR}/eval_oracle_lora_${tasks[$i]}_${i}.log"
  echo "[GPU0] Launching task=${tasks[$i]} | log=$log_file"
  uv run python scripts/run_eval.py \
      --model-dir "$MODEL_DIR" \
      --lora-dirs "${lora_dirs[$i]}" \
      --save-results --save-to-base-model-dir \
      --tasks "${tasks[$i]}"
done

######################
# GPU 1 작업들
######################
MODEL_DIR="mistralai/Mistral-7B-Instruct-v0.2"
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"
export HF_HOME=/workspace/hf_cache/huggingface_nwc

lora_dirs=(
    "/workspace/Weight_compression/text-to-lora/train_outputs/sft/oracle_lora/['mbpp']_20250830-154501_edGGjsED"
    "/workspace/Weight_compression/text-to-lora/train_outputs/sft/oracle_lora/['openbookqa']_20250830-142304_HHAt8nQx"
    "/workspace/Weight_compression/text-to-lora/train_outputs/sft/oracle_lora/['piqa']_20250830-105239_4Pdp2TEg"
    "/workspace/Weight_compression/text-to-lora/train_outputs/sft/oracle_lora/['piqa']_20250904-233252_7o4aDXNc"
    "/workspace/Weight_compression/text-to-lora/train_outputs/sft/oracle_lora/['winogrande']_20250830-103652_Xd5wcOgY"
)
tasks=(mbpp openbookqa piqa piqa winogrande)

for i in "${!lora_dirs[@]}"; do
  export CUDA_VISIBLE_DEVICES=1
  log_file="${LOG_DIR}/eval_oracle_lora_${tasks[$i]}_${i}.log"
  echo "[GPU1] Launching task=${tasks[$i]} | log=$log_file"
  uv run python scripts/run_eval.py \
      --model-dir "$MODEL_DIR" \
      --lora-dirs "${lora_dirs[$i]}" \
      --save-results --save-to-base-model-dir \
      --tasks "${tasks[$i]}"
done



export HF_HOME=/workspace/hf_cache/huggingface_nwc
export CUDA_VISIBLE_DEVICES=1
uv run python scripts/run_eval.py \
    --model-dir mistralai/Mistral-7B-Instruct-v0.2 \
    --lora-dirs "/workspace/Weight_compression/text-to-lora/train_outputs/sft/oracle_lora/['winogrande']_20250830-103652_Xd5wcOgY" \
    --save-results --save-to-base-model-dir --tasks winogrande > ./logs/eval_oracle_lora_winogrande.log