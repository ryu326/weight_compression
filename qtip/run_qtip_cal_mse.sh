#!/bin/bash
##########################################################################
##                           SCRIPT CONTROLS                            ##
##########################################################################

# 실험을 수행할 모델 목록 (아래 'MODEL CONFIGURATION'에 정의된 이름 사용)
MODELS_TO_RUN=(
    "llama3_8b"
    # "llama2_7b"
    # "llama3.2_3b"
    # "llama2_13b"
    # "llama3.2_1b_inst"
    # "llama3.2_3b_inst"
)

# 각 모델에 대해 수행할 실험 타입 목록 ('ft', 'noft')
EXP_TYPES_TO_RUN=(
    "ft1"
    # "noft"
)

CKPT="../hf_model_comp/qtip/ckpt"
HF="../hf_model_comp/qtip/hf"
LOG="./log"
RES="../hf_model_comp_results/qtip"

# --- 환경 변수 설정 ---
export CUDA_VISIBLE_DEVICES=4,5,6,7
export WANDB_SILENT=true
export TRANSFORMERS_NO_TORCHVISION=1
export HF_HOME=/home/jgryu/.cache/huggingface

##########################################################################
##                         MODEL CONFIGURATION                          ##
##                  (사용할 모델과 경로를 여기에 정의)                    ##
##########################################################################

declare -A MODEL_PATHS
MODEL_PATHS=(
    ["llama2_7b"]="../Wparam_dataset/hf_model/meta-llama--Llama-2-7b-hf"
    ["llama2_13b"]="../Wparam_dataset/hf_model/meta-llama--Llama-2-13b-hf"
    ["llama3_8b"]="../Wparam_dataset/hf_model/meta-llama--Meta-Llama-3-8B"
    ["vicuna_7b"]="../Wparam_dataset/hf_model/lmsys--vicuna-7b-v1.5"
    ["llama3.2_3b"]="../Wparam_dataset/hf_model/meta-llama--Llama-3.2-3B"
    ["llama3.2_3b"]="../Wparam_dataset/hf_model/meta-llama--Llama-3.2-3B"
    ["llama3.2_3b"]="../Wparam_dataset/hf_model/meta-llama--Llama-3.2-3B"
    ["llama3.2_3b_inst"]="../Wparam_dataset/hf_model/meta-llama--Llama-3.2-3B-Instruct"
    ["llama3.2_1b_inst"]="../Wparam_dataset/hf_model/meta-llama--Llama-3.2-1B-Instruct"
)

declare -A HESS_PATHS
HESS_PATHS=(
    ["llama2_7b"]="../Wparam_dataset/quip_hess/Hessians-Llama-2-7b-6144"
    ["llama2_13b"]="../Wparam_dataset/quip_hess/Hessians-Llama-2-13b-6144"
    ["llama3_8b"]="../Wparam_dataset/quip_hess/llama3_8b_6144"
    ["vicuna_7b"]="../Wparam_dataset/quip_hess/lmsys--vicuna-7b-v1.5_256"
    ["llama3.2_3b"]="../Wparam_dataset/quip_hess/meta-llama--Llama-3.2-3B-256"
    ["llama3.2_3b_inst"]="../Wparam_dataset/quip_hess/Llama-3.2-3B-Instruct-Hessians"
    ["llama3.2_1b_inst"]="../Wparam_dataset/quip_hess/Llama-3.2-1B-Instruct-Hessians"
)


# 필요한 디렉토리 생성
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

        for K in 2 3 4 5 6; do
            NAME="${model_key}/${exp_type}/${K}bit"
            SAVE_PATH="$CKPT/$NAME"
            LOG_FILE="${LOG}/${NAME}.log"
            HF_PATH="$HF/$NAME"

            python -m quantize_llama.hfize_llama_cal_mse \
                --quantized_path $SAVE_PATH \
                --hf_output_path $HF_PATH \
                --output_path ${RES}/${NAME} \
                --base_model $base_model 2>&1 | tee -a $LOG_FILE
        done
    done
done

echo "############################################################"
echo "##               All experiments finished.                ##"
echo "############################################################"