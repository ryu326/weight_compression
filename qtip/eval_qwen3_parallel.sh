#!/bin/bash
##########################################################################
##                          SCRIPT CONTROLS                             ##
##########################################################################

# 실험을 수행할 모델 목록
MODELS_TO_RUN=(
    # "mixtral"
    "qwen3moe"
)

# 각 모델에 대해 수행할 실험 타입 목록
EXP_TYPES_TO_RUN=(
    # "noft"
    "ft1"
)

# 병렬 실행 설정 (K값과 GPU 매핑)
# 예: K=2는 GPU 4번, K=3은 GPU 5번 사용
K_VALUES=(2 3 4 5 6)
GPU_LIST=(2 3 4 5 6)

CKPT="../hf_model_comp/qtip/ckpt"
HF="../hf_model_comp/qtip/hf"
LOG="./log"
RES="../hf_model_comp_results/qtip"

# --- 환경 변수 설정 ---
# GPU는 루프 내부에서 개별 할당하므로 여기서는 제거하거나 주석 처리
# export CUDA_VISIBLE_DEVICES=4,5 
export WANDB_SILENT=true
export HF_HOME=/workspace/Weight_compression/hf_cache/

##########################################################################
##                          MODEL CONFIGURATION                         ##
##########################################################################

declare -A MODEL_PATHS
MODEL_PATHS=(
    ["mixtral"]="mistralai/Mixtral-8x7B-v0.1"
    ["qwen3moe"]="/workspace/Weight_compression/Wparam_dataset/hf_model/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
)

declare -A HESS_PATHS
HESS_PATHS=(
    ["mixtral"]="/home/jgryu/workspace/weight_compression/Wparam_dataset/quip_hess/Mixtral-8x7B-v0.1_256"
    ["qwen3moe"]="../Wparam_dataset/quip_hess/Qwen3-30B-A3B"
)

##########################################################################
##                        MAIN EXECUTION LOGIC                          ##
##########################################################################

mkdir -p $CKPT $HF $LOG $RES

for model_key in "${MODELS_TO_RUN[@]}"; do
    base_model=${MODEL_PATHS[$model_key]}
    HESS=${HESS_PATHS[$model_key]}

    echo "============================================================"
    echo "                 STARTING MODEL: [$model_key]"
    echo "============================================================"

    # 2. 중간 루프: 설정된 모든 실험 타입을 순회
    for exp_type in "${EXP_TYPES_TO_RUN[@]}"; do
        
        ft_epochs=0
        if [ "$exp_type" == "ft1" ]; then
            ft_epochs=5
        fi

        echo "------------------------------------------------------------"
        echo "           Running Experiment Type: [$exp_type]"
        echo "------------------------------------------------------------"

        # 3. 내부 루프: K값에 따라 병렬 실행
        for i in "${!K_VALUES[@]}"; do
            K=${K_VALUES[$i]}
            gpu_id=${GPU_LIST[$i]}

            # 백그라운드 프로세스 시작
            (
                export CUDA_VISIBLE_DEVICES=$gpu_id
                
                NAME="${model_key}/${exp_type}/${K}bit"
                SAVE_PATH="$CKPT/$NAME"
                LOG_FILE="${LOG}/${NAME}_eval.log"
                HF_PATH="$HF/$NAME"

                mkdir -p $SAVE_PATH
                mkdir -p $(dirname "$LOG_FILE")

                echo ">>> [Start] K=$K on GPU $gpu_id | Model: $model_key"

                # -------------------------------------------------------
                # [Stage: Quantize] (Commented out as requested)
                # -------------------------------------------------------
                # echo "### [Stage: Quantize | K=$K] ###" >> $LOG_FILE
                # python -m quantize_llama.quantize_finetune_moe \
                #     --save_path $SAVE_PATH \
                #     --codebook bitshift \
                #     --base_model $base_model \
                #     --in_hess_path $HESS \
                #     --scale_override 0.9 \
                #     --ft_epochs $ft_epochs \
                #     --td_x 16 --td_y 16 --L 16 --K $K --V 2 \
                #     --decode_mode quantlut_sym --tlut_bits 9 >> $LOG_FILE 2>&1

                # -------------------------------------------------------
                # [Stage: Hfize] (Commented out as requested)
                # -------------------------------------------------------
                # echo "### [Stage: Hfize | K=$K] ###" >> $LOG_FILE
                # python -m quantize_llama.hfize_moe \
                #     --quantized_path $SAVE_PATH \
                #     --hf_output_path $HF_PATH \
                #     --base_model $base_model >> $LOG_FILE 2>&1

                MANIFEST_FLAG=""
                if [ "$K" -ge 1 ]; then
                    MANIFEST_FLAG="--manifest_model"
                fi

                # -------------------------------------------------------
                # [Stage: Eval PPL]
                # -------------------------------------------------------
                echo "### [Stage: Eval PPL | K=$K] ###" > "$LOG_FILE"
                python -m eval.eval_ppl \
                    --hf_path "${HF_PATH}" \
                    --output_path "${RES}/${NAME}" \
                    --seqlen 2048 \
                    $MANIFEST_FLAG >> "$LOG_FILE" 2>&1

                # -------------------------------------------------------
                # [Stage: Eval Zero-shot]
                # -------------------------------------------------------
                echo "### [Stage: Eval Zero-shot | K=$K] ###" > "$LOG_FILE"
                python -m eval.eval_zeroshot \
                    --tasks arc_challenge,arc_easy,boolq,piqa,winogrande,hellaswag,mmlu \
                    --batch_size 1 \
                    --hf_path "${HF_PATH}" \
                    --output_path "${RES}/${NAME}_common_mmlu" \
                    $MANIFEST_FLAG >> "$LOG_FILE" 2>&1

                echo ">>> [Done] K=$K on GPU $gpu_id"
            ) &
        done
        
        # 모든 K 작업이 끝날 때까지 대기 후 다음 실험 타입으로 이동
        wait
    done
done

echo "All experiments completed."