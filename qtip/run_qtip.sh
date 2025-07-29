#!/bin/bash
##########################################################################
##                           SCRIPT CONTROLS                            ##
##########################################################################

# 실험을 수행할 모델 목록 (아래 'MODEL CONFIGURATION'에 정의된 이름 사용)
MODELS_TO_RUN=(
    "llama3_8b"
    "llama2_7b"
    "llama2_13b"
    # "vicuna_7b"
)

# 각 모델에 대해 수행할 실험 타입 목록 ('ft', 'noft')
EXP_TYPES_TO_RUN=(
    "ft1"
    "noft"
)

CKPT="../hf_model_comp/qtip/ckpt"
HF="../hf_model_comp/qtip/hf"
LOG="./log"
RES="../hf_model_comp_results/qtip"

# --- 환경 변수 설정 ---
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_SILENT=true

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
)

declare -A HESS_PATHS
HESS_PATHS=(
    ["llama2_7b"]="../Wparam_dataset/quip_hess/Hessians-Llama-2-7b-6144"
    ["llama2_13b"]="../Wparam_dataset/quip_hess/Hessians-Llama-2-13b-6144"
    ["llama3_8b"]="../Wparam_dataset/quip_hess/llama3_8b_6144"
    ["vicuna_7b"]="../Wparam_dataset/quip_hess/lmsys--vicuna-7b-v1.5_256"
)

# [원본 보존] 헤시안 계산 스크립트 (필요시 주석 해제하여 사용)
# torchrun --nproc_per_node=4 -m quantize_llama.input_hessian_llama \
#     --batch_size 8 --devset_size 256 \
#     --base_model /workspace/Weight_compression/Wparam_dataset/hf_model/lmsys--vicuna-7b-v1.5 \
#     --save_path ../Wparam_dataset/quip_hess/lmsys--vicuna-7b-v1.5_256


##########################################################################
##                        MAIN EXECUTION LOGIC                          ##
##########################################################################

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

        # 3. 내부 루프: K 값을 순회 (2, 3, 4 bit)
        for K in 5 6; do
            NAME="${model_key}/${exp_type}/${K}bit"
            SAVE_PATH="$CKPT/$NAME"
            LOG_FILE="${LOG}/${NAME}.log"
            HF_PATH="$HF/$NAME"

            mkdir -p $SAVE_PATH
            mkdir -p $(dirname "$LOG_FILE")

            echo "### [Stage: Quantize | K=$K] ###" | tee $LOG_FILE
            # python -m quantize_llama.quantize_finetune_llama \
            #     --save_path $SAVE_PATH \
            #     --codebook bitshift \
            #     --base_model $base_model \
            #     --in_hess_path $HESS \
            #     --scale_override 0.9 \
            #     --ft_epochs $ft_epochs \
            #     --td_x 16 --td_y 16 --L 16 --K $K --V 2 \
            #     --decode_mode quantlut_sym --tlut_bits 9 2>&1 | tee -a $LOG_FILE

            echo "### [Stage: Hfize | K=$K] ###" | tee -a $LOG_FILE
            python -m quantize_llama.hfize_llama \
                --quantized_path $SAVE_PATH \
                --hf_output_path $HF_PATH \
                --base_model $base_model 2>&1 | tee -a $LOG_FILE

            # echo "### [Stage: Eval PPL | K=$K] ###" | tee -a $LOG_FILE
            # python -m eval.eval_ppl \
            #     --hf_path ${HF_PATH} \
            #     --output_path ${RES}/${NAME} \
            #     --seqlen 2048  2>&1 | tee -a $LOG_FILE

            echo "### [Stage: Eval Zero-shot | K=$K] ###" | tee -a $LOG_FILE
            python -m eval.eval_zeroshot \
                --tasks mmlu \
                --batch_size 8  --hf_path ${HF_PATH} \
                --output_path ${RES}/${NAME}_mmlu 2>&1 | tee -a $LOG_FILE

                # --tasks arc_challenge,arc_easy,boolq,piqa,winogrande \

            if [ "$HF_PATH" != "$HF" ]; then
                echo "Cleaning up temporary files for $SAVE_NAME"
                rm -rf "$HF_PATH"
            fi
        done
    done
done


# [원본 보존] e2e 실험 블록 (필요시 주석 해제하여 사용)
# echo "============================================================"
# echo "          STARTING End-to-End Finetuning Experiments"
# echo "============================================================"
# model_key="vicuna_7b" 
# base_model=${MODEL_PATHS[$model_key]}
# for K in 2 3 4; do
#     # e2e는 ft로 생성된 모델을 기반으로 함
#     FT_NAME="${model_key}/ft_${K}bit"
#     E2E_NAME="${model_key}/e2e_${K}bit"
#
#     FT_HF_PATH="$HF/$FT_NAME"
#     E2E_HF_PATH="$HF/$E2E_NAME"
#     LOG_FILE="${LOG}/${E2E_NAME}.log"
#
#     mkdir -p $(dirname "$LOG_FILE")
#
#     echo "### [Stage: End-to-End Finetuning | K=$K] ###" | tee $LOG_FILE
#     python -m quantize_llama.finetune_e2e_llama --base_model $base_model \
#         --hf_path $FT_HF_PATH --devset_size 640 --ft_valid_size 128 \
#         --ft_epochs 4 --ft_update_freq 4 --ft_bs 1 --ctx_size 4096 \
#         --start_dev 2 \
#         --ft_train_lut --hf_output_path $E2E_HF_PATH 2>&1 | tee -a $LOG_FILE 
#
#     echo "### [Stage: Eval PPL (e2e) | K=$K] ###" | tee -a $LOG_FILE
#     python -m eval.eval_ppl \
#         --hf_path $E2E_HF_PATH \
#         --output_path ${RES}/${E2E_NAME} \
#         --seqlen 2048  2>&1 | tee -a $LOG_FILE
#
#     echo "### [Stage: Eval Zero-shot (e2e) | K=$K] ###" | tee -a $LOG_FILE
#     python -m eval.eval_zeroshot --tasks arc_challenge,arc_easy,boolq,piqa,winogrande \
#         --batch_size 8  --hf_path $E2E_HF_PATH \
#         --output_path ${RES}/${E2E_NAME} 2>&1 | tee -a $LOG_FILE
# done


echo "############################################################"
echo "##               All experiments finished.                ##"
echo "############################################################"