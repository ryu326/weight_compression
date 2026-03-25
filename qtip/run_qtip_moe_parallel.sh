#!/bin/bash

##########################################################################
##                           SCRIPT CONTROLS                            ##
##########################################################################

# --- 환경 변수 및 경로 설정 ---
export HF_HOME=/home/jgryu/.cache/huggingface
# export HF_HOME=/workspace/Weight_compression/hf_cache/
export WANDB_SILENT=true

CKPT="../hf_model_comp/qtip/ckpt"
HF="../hf_model_comp/qtip/hf"
LOG="./log"
RES="../hf_model_comp_results_v2/qtip"

mkdir -p $CKPT $HF $LOG $RES

# --- 모델 및 실험 설정 ---
MODELS_TO_RUN=(
    # "mixtral"
    # "qwen3moe"
    "gptoss"
)

EXP_TYPES_TO_RUN=(
    # "ft1"
    "noft"
)

# K값 목록 (병렬화 대상)
K_VALUES=(3 4 5 6)

declare -A MODEL_PATHS
MODEL_PATHS=(
    ["mixtral"]="mistralai/Mixtral-8x7B-v0.1"
    ["qwen3moe"]="/workspace/Weight_compression/Wparam_dataset/hf_model/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
    ["gptoss"]="openai/gpt-oss-20b"
)

declare -A HESS_PATHS
HESS_PATHS=(
    ["mixtral"]="/home/jgryu/workspace/weight_compression/Wparam_dataset/quip_hess/Mixtral-8x7B-v0.1_1024"
    ["qwen3moe"]="../Wparam_dataset/quip_hess/Qwen3-30B-A3B"
    ["gptoss"]="/home/jgryu/workspace/weight_compression/Wparam_dataset/quip_hess/gpt-oss-20b_1024"
)

##########################################################################
##                       PARALLEL EXECUTION FUNCTION                    ##
##########################################################################

run_pipeline() {
    local gpu_id="$1"
    local model_key="$2"
    local exp_type="$3"
    local K="$4"

    local base_model="${MODEL_PATHS[$model_key]}"
    local hess_path="${HESS_PATHS[$model_key]}"
    
    # Epoch 설정
    local ft_epochs=0
    if [ "$exp_type" == "ft1" ]; then
        ft_epochs=5
    fi

    # 경로 및 이름 설정
    local NAME="${model_key}/${exp_type}/${K}bit"
    local SAVE_PATH="$CKPT/$NAME"
    local LOG_FILE="${LOG}/${NAME}.log"
    local HF_PATH="$HF/${NAME}_hf_form"

    mkdir -p "$SAVE_PATH"
    mkdir -p "$(dirname "$LOG_FILE")"

    # 서브쉘 내에서 실행 (환경변수 격리)
    (
        export CUDA_VISIBLE_DEVICES="${gpu_id}"
        
        echo "[$model_key | $exp_type | K=$K] Started on GPU $gpu_id. Logging to $LOG_FILE"

        # ---------------------------------------------------------
        # 1. Quantize (주석 처리됨 - 원본 유지)
        # ---------------------------------------------------------
        # echo "### [Stage: Quantize | K=$K] ###" | tee "$LOG_FILE"
        # python -m quantize_llama.quantize_finetune_moe \
        #     --save_path $SAVE_PATH \
        #     --codebook bitshift \
        #     --base_model $base_model \
        #     --in_hess_path $hess_path \
        #     --scale_override 0.9 \
        #     --ft_epochs $ft_epochs \
        #     --td_x 16 --td_y 16 --L 16 --K $K --V 2 \
        #     --batch_size 4 --ft_bs 1 \
        #     --decode_mode quantlut_sym --tlut_bits 9 2>&1 | tee -a "$LOG_FILE"

        # ---------------------------------------------------------
        # 2. Hfize
        # ---------------------------------------------------------
        if [ "$K" -ge 1 ]; then
            echo "### [Stage: Hfize | K=$K] ###" | tee -a "$LOG_FILE"
            python -m quantize_llama.hfize_moe_hf \
                --quantized_path "$SAVE_PATH" \
                --hf_output_path "$HF_PATH" \
                --base_model "$base_model" > "$LOG_FILE" 2>&1
        fi

        # ---------------------------------------------------------
        # 3. Eval Setup
        # ---------------------------------------------------------
        local MANIFEST_FLAG=""
        if [ "$K" -ge 1 ]; then
            MANIFEST_FLAG="--manifest_model"
        fi

        # ---------------------------------------------------------
        # 4. Eval Zero-shot
        # ---------------------------------------------------------

        echo "### [Stage: Eval PPL | K=$K] ###" | tee -a "$LOG_FILE"
        python -m eval.eval_ppl \
            --hf_path "${HF_PATH}" \
            --output_path "${RES}/${NAME}" \
            --seqlen 2048 \
            $MANIFEST_FLAG  >> "$LOG_FILE" 2>&1

        # echo "### [Stage: Eval Zero-shot | K=$K] ###" | tee -a "$LOG_FILE"
        # python -m eval.eval_zeroshot \
        #     --tasks arc_challenge,arc_easy,boolq,piqa,winogrande,hellaswag,mmlu \
        #     --hf_path "${HF_PATH}" \
        #     --output_path "${RES}/${NAME}_common_mmlu" \
        #     $MANIFEST_FLAG 2>&1 > "$LOG_FILE" 2>&1

        # ---------------------------------------------------------
        # 5. Cleanup
        # ---------------------------------------------------------
        if [ "$HF_PATH" != "$HF" ]; then
            echo "Cleaning up temporary files for $NAME" | tee -a "$LOG_FILE"
            rm -rf "$HF_PATH"
        fi

        echo "[$model_key | $exp_type | K=$K] Finished."
    )
}

##########################################################################
##                        MAIN EXECUTION LOOP                           ##
##########################################################################

TARGET_GPUS=(0 1 2 3 4 5)
# 배열의 크기 (사용할 GPU 개수) 자동 계산
NUM_GPUS=${#TARGET_GPUS[@]}

echo "Running on ${NUM_GPUS} GPUs: ${TARGET_GPUS[*]}"

pids=()
job_counter=0

for model_key in "${MODELS_TO_RUN[@]}"; do
    for exp_type in "${EXP_TYPES_TO_RUN[@]}"; do
        
        echo "============================================================"
        echo " Scheduling: Model=[$model_key] Exp=[$exp_type]"
        echo "============================================================"

        for K in "${K_VALUES[@]}"; do
            
            array_index=$((job_counter % NUM_GPUS))
            gpu_id=${TARGET_GPUS[$array_index]}
            
            echo "Assigning Job [K=$K] to GPU ID: $gpu_id"

            run_pipeline "$gpu_id" "$model_key" "$exp_type" "$K" &

            pids+=($!)
            
            ((job_counter++))
            # [안전 장치] GPU 개수만큼 작업이 실행되면 잠시 대기 (배치 실행)
            # 메모리 부족(OOM) 방지를 위해, GPU 개수만큼 프로세스가 뜨면 완료될 때까지 기다렸다가 다음 배치를 실행합니다.
            # 모든 작업을 한 번에 다 띄우고 싶다면 아래 if문을 주석 처리하세요.
            if [ "$((job_counter % NUM_GPUS))" -eq 0 ]; then
               echo "Waiting for current batch on GPUs ${TARGET_GPUS[*]} to finish..."
               wait
            fi

        done
    done
done

echo "All jobs launched. Waiting for any remaining completion..."

# 남은 백그라운드 작업 대기
for pid in "${pids[@]}"; do
    wait "${pid}" 2>/dev/null
done

echo "All experiments completed."