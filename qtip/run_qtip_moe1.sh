#!/bin/bash
##########################################################################
##                           SCRIPT CONTROLS                            ##
##########################################################################

# 실험을 수행할 모델 목록 (아래 'MODEL CONFIGURATION'에 정의된 이름 사용)
MODELS_TO_RUN=(
    # "mixtral"
    "qwen3moe"
)

# 각 모델에 대해 수행할 실험 타입 목록 ('ft', 'noft')
EXP_TYPES_TO_RUN=(
    # "noft"
    "ft1"
)

CKPT="../hf_model_comp/qtip/ckpt"
HF="../hf_model_comp/qtip/hf"
LOG="./log"
RES="../hf_model_comp_results/qtip"

# --- 환경 변수 설정 ---
export CUDA_VISIBLE_DEVICES=2,3,4
export WANDB_SILENT=true
# export HF_HOME=/home/jgryu/.cache/huggingface
export HF_HOME=/workspace/Weight_compression/hf_cache/

##########################################################################
##                         MODEL CONFIGURATION                          ##
##                  (사용할 모델과 경로를 여기에 정의)                    ##
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

        for K in 5 6; do
            NAME="${model_key}/${exp_type}/${K}bit"
            SAVE_PATH="$CKPT/$NAME"
            LOG_FILE="${LOG}/${NAME}_eval3.log"
            HF_PATH="$HF/${NAME}_hf_form"
            # PTH_PATH="../complexity_test/8b_qtip_${exp_type}_${K}bit.pth"

            mkdir -p $SAVE_PATH
            mkdir -p $(dirname "$LOG_FILE")

            # echo "### [Stage: Quantize | K=$K] ###" | tee $LOG_FILE
            # python -m quantize_llama.quantize_finetune_moe \
            #     --save_path $SAVE_PATH \
            #     --codebook bitshift \
            #     --base_model $base_model \
            #     --in_hess_path $HESS \
            #     --scale_override 0.9 \
            #     --ft_epochs $ft_epochs \
            #     --td_x 16 --td_y 16 --L 16 --K $K --V 2 \
            #     --decode_mode quantlut_sym --tlut_bits 9 2>&1 | tee -a $LOG_FILE

            # if [ "$K" -ge 1 ]; then
            #     echo "### [Stage: Hfize | K=$K] ###" | tee -a $LOG_FILE
            #     python -m quantize_llama.hfize_moe_hf \
            #         --quantized_path $SAVE_PATH \
            #         --hf_output_path $HF_PATH \
            #         --base_model $base_model 2>&1 | tee -a $OGL_FILE
            # fi
            # echo "### [Stage: Hfize | K=$K] ###" | tee -a $LOG_FILE
            # python -m quantize_llama.hfize_moe \
            #     --quantized_path $SAVE_PATH \
            #     --hf_output_path $HF_PATH \
            #     --base_model $base_model 2>&1 | tee -a $OGL_FILE

            # echo "### [Stage: Hfize | K=$K] ###" | tee -a $LOG_FILE
            # python -m quantize_llama.hfize_moe_pth \
            #     --quantized_path $SAVE_PATH \
            #     --hf_output_path $HF_PATH \
            #     --base_model $base_model \
                # 2>&1 | tee -a $LOG_FILE

            MANIFEST_FLAG=""
            if [ "$K" -ge 1 ]; then
                MANIFEST_FLAG="--manifest_model"
            fi

            # echo "### [Stage: Eval PPL | K=$K] ###" | tee -a "$LOG_FILE"
            # python -m eval.eval_ppl \
            #     --hf_path "${HF_PATH}" \
            #     --output_path "${RES}/${NAME}" \
            #     --seqlen 2048 \
            #     $MANIFEST_FLAG 2>&1 | tee -a "$LOG_FILE"
                # --max_mem_ratio 0.2 \


            echo "### [Stage: Eval Zero-shot | K=$K] ###" | tee -a "$LOG_FILE"
            python -m eval.eval_zeroshot \
                --tasks arc_challenge,arc_easy,boolq,piqa,winogrande,hellaswag,mmlu \
                --batch_size 1 \
                --hf_path "${HF_PATH}" \
                --output_path "${RES}/${NAME}_common_mmlu" \
                $MANIFEST_FLAG 2>&1 | tee -a "$LOG_FILE"

            # if [ "$HF_PATH" != "$HF" ]; then
            #     echo "Cleaning up temporary files for $SAVE_NAME"
            #     rm -rf "$HF_PATH"
            # fi
        done
    done
done