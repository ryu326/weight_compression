#!/bin/bash
##########################################################################
##                           SCRIPT CONTROLS                            ##
##########################################################################

# 실험을 수행할 모델 목록 (아래 'MODEL CONFIGURATION'에 정의된 이름 사용)
MODELS_TO_RUN=(
    # "llama3_8b"
    # "llama2_7b"
    # "llama3.2_3b"
    "llama2_13b"
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
export CUDA_VISIBLE_DEVICES=2,3
export WANDB_SILENT=true
export TRANSFORMERS_NO_TORCHVISION=1
export HF_HOME=/workspace/hf_cache/huggingface_qtip

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

        for K in 5 6; do
            NAME="${model_key}/${exp_type}/${K}bit"
            SAVE_PATH="$CKPT/$NAME"
            LOG_FILE="${LOG}/${NAME}.log"
            HF_PATH="$HF/$NAME"

            mkdir -p $SAVE_PATH
            mkdir -p $(dirname "$LOG_FILE")

            # echo "### [Stage: Quantize | K=$K] ###" | tee $LOG_FILE
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

            # python -m quantize_llama.hfize_llama_cal_mse \
            #     --quantized_path $SAVE_PATH \
            #     --hf_output_path $HF_PATH \
            #     --output_path ${RES}/${NAME} \
            #     --base_model $base_model 2>&1 | tee -a $LOG_FILE

            MANIFEST_FLAG=""
            # if [ "$K" -ge 5 ]; then
            #     MANIFEST_FLAG="--manifest_model"
            # fi

            echo "### [Stage: Eval PPL | K=$K] ###" | tee -a "$LOG_FILE"
            python -m eval.eval_ppl \
                --hf_path "${HF_PATH}" \
                --output_path "${RES}/${NAME}" \
                --seqlen 2048 \
                $MANIFEST_FLAG 2>&1 | tee -a "$LOG_FILE"

            # echo "### [Stage: Eval Zero-shot | K=$K] ###" | tee -a "$LOG_FILE"
            # python -m eval.eval_zeroshot \
            #     --tasks arc_challenge,arc_easy,boolq,piqa,winogrande,hellaswag,mmlu \
            #     --batch_size 1 \
            #     --hf_path "${HF_PATH}" \
            #     --output_path "${RES}/${NAME}_common_mmlu" \
            #     $MANIFEST_FLAG 2>&1 | tee -a "$LOG_FILE"


            # python -m eval.eval_zeroshot \
            #     --tasks mmlu \
            #     --batch_size 8  --hf_path ${HF_PATH} \
            #     --output_path ${RES}/${NAME}_mmlu 2>&1 | tee -a $LOG_FILE


            # RESULT_FILE="${RES}/${NAME}_zeroshot_results.json"
            # if [ ! -f "${RESULT_FILE}" ]; then
            #     echo "결과 파일이 존재하지 않습니다. 제로샷 평가를 시작합니다..."
            #     python -m eval.eval_zeroshot \
            #         --tasks arc_challenge,arc_easy,boolq,piqa,winogrande \
            #         --batch_size 8 \
            #         --hf_path ${HF_PATH} \
            #         --manifest_model \
            #         --output_path ${RES}/${NAME} | tee -a $LOG_FILE
            # else
            #     # 파일이 이미 존재하면 메시지를 출력하고 넘어감
            #     echo "이미 결과 파일(${RESULT_FILE})이 존재하므로 평가를 건너뜁니다."
            # fi

            # RESULT_FILE="${RES}/${NAME}_mmlu_zeroshot_results.json"
            # if [ ! -f "${RESULT_FILE}" ]; then
            #     echo "결과 파일이 존재하지 않습니다. 제로샷 평가를 시작합니다..."
            #     python -m eval.eval_zeroshot \
            #         --tasks mmlu \
            #         --batch_size 8 \
            #         --hf_path ${HF_PATH} \
            #         --manifest_model \
            #         --output_path ${RES}/${NAME}_mmlu | tee -a $LOG_FILE
            # else
            #     # 파일이 이미 존재하면 메시지를 출력하고 넘어감
            #     echo "이미 결과 파일(${RESULT_FILE})이 존재하므로 평가를 건너뜁니다."
            # fi

            # RESULT_FILE="${RES}/${NAME}_2_zeroshot_results.json"
            # if [ ! -f "${RESULT_FILE}" ]; then
            #     echo "결과 파일이 존재하지 않습니다. 제로샷 평가를 시작합니다..."
            #     python -m eval.eval_zeroshot \
            #         --tasks hellaswag,openbookqa,mathqa,sciq,pubmedqa \
            #         --batch_size 8 \
            #         --hf_path ${HF_PATH} \
            #         --manifest_model \
            #         --output_path ${RES}/${NAME}_2 | tee -a $LOG_FILE
            # else
            #     # 파일이 이미 존재하면 메시지를 출력하고 넘어감
            #     echo "이미 결과 파일(${RESULT_FILE})이 존재하므로 평가를 건너뜁니다."
            # fi

            # python -m eval.eval_zeroshot \
            #     --tasks hellaswag,openbookqa,mathqa,sciq,pubmedqa \
            #     --batch_size 4  \
            #     --hf_path ${HF_PATH} \
            #     --output_path ${RES}/${NAME}_2 | tee -a $LOG_FILE

                # --tasks arc_challenge,arc_easy,boolq,piqa,winogrande \

            if [ "$HF_PATH" != "$HF" ]; then
                echo "Cleaning up temporary files for $SAVE_NAME"
                rm -rf "$HF_PATH"
            fi
        done
    done
done

echo "############################################################"
echo "##               All experiments finished.                ##"
echo "############################################################"


#!/bin/bash
##########################################################################
##                           SCRIPT CONTROLS                            ##
##########################################################################

# # 실험을 수행할 모델 목록 (아래 'MODEL CONFIGURATION'에 정의된 이름 사용)
# MODELS_TO_RUN=(
#     # "llama3_8b"
#     # "llama2_7b"
#     # "llama3.2_3b"
#     "llama2_13b"
#     # "llama3.2_1b_inst"
#     # "llama3.2_3b_inst"
# )

# # 각 모델에 대해 수행할 실험 타입 목록 ('ft', 'noft')
# EXP_TYPES_TO_RUN=(
#     "ft1"
#     # "noft"
# )

# CKPT="../hf_model_comp/qtip/ckpt"
# HF="../hf_model_comp/qtip/hf"
# LOG="./log"
# RES="../hf_model_comp_results/qtip"

# # --- 환경 변수 설정 ---
# export CUDA_VISIBLE_DEVICES=3
# export WANDB_SILENT=true
# export TRANSFORMERS_NO_TORCHVISION=1
# export HF_HOME=/workspace/hf_cache/huggingface_qtip

# ##########################################################################
# ##                         MODEL CONFIGURATION                          ##
# ##                  (사용할 모델과 경로를 여기에 정의)                    ##
# ##########################################################################

# declare -A MODEL_PATHS
# MODEL_PATHS=(
#     ["llama2_7b"]="../Wparam_dataset/hf_model/meta-llama--Llama-2-7b-hf"
#     ["llama2_13b"]="../Wparam_dataset/hf_model/meta-llama--Llama-2-13b-hf"
#     ["llama3_8b"]="../Wparam_dataset/hf_model/meta-llama--Meta-Llama-3-8B"
#     ["vicuna_7b"]="../Wparam_dataset/hf_model/lmsys--vicuna-7b-v1.5"
#     ["llama3.2_3b"]="../Wparam_dataset/hf_model/meta-llama--Llama-3.2-3B"
#     ["llama3.2_3b"]="../Wparam_dataset/hf_model/meta-llama--Llama-3.2-3B"
#     ["llama3.2_3b"]="../Wparam_dataset/hf_model/meta-llama--Llama-3.2-3B"
#     ["llama3.2_3b_inst"]="../Wparam_dataset/hf_model/meta-llama--Llama-3.2-3B-Instruct"
#     ["llama3.2_1b_inst"]="../Wparam_dataset/hf_model/meta-llama--Llama-3.2-1B-Instruct"
# )

# declare -A HESS_PATHS
# HESS_PATHS=(
#     ["llama2_7b"]="../Wparam_dataset/quip_hess/Hessians-Llama-2-7b-6144"
#     ["llama2_13b"]="../Wparam_dataset/quip_hess/Hessians-Llama-2-13b-6144"
#     ["llama3_8b"]="../Wparam_dataset/quip_hess/llama3_8b_6144"
#     ["vicuna_7b"]="../Wparam_dataset/quip_hess/lmsys--vicuna-7b-v1.5_256"
#     ["llama3.2_3b"]="../Wparam_dataset/quip_hess/meta-llama--Llama-3.2-3B-256"
#     ["llama3.2_3b_inst"]="../Wparam_dataset/quip_hess/Llama-3.2-3B-Instruct-Hessians"
#     ["llama3.2_1b_inst"]="../Wparam_dataset/quip_hess/Llama-3.2-1B-Instruct-Hessians"
# )

# # [원본 보존] 헤시안 계산 스크립트 (필요시 주석 해제하여 사용)
# # torchrun --nproc_per_node=4 -m quantize_llama.input_hessian_llama \
# #     --batch_size 8 --devset_size 256 \
# #     --base_model /workspace/Weight_compression/Wparam_dataset/hf_model/lmsys--vicuna-7b-v1.5 \
# #     --save_path ../Wparam_dataset/quip_hess/lmsys--vicuna-7b-v1.5_256


# ##########################################################################
# ##                        MAIN EXECUTION LOGIC                          ##
# ##########################################################################

# # 필요한 디렉토리 생성
# mkdir -p $CKPT $HF $LOG $RES

# for model_key in "${MODELS_TO_RUN[@]}"; do
#     base_model=${MODEL_PATHS[$model_key]}
#     HESS=${HESS_PATHS[$model_key]}

#     echo "============================================================"
#     echo "                 STARTING MODEL: [$model_key]"
#     echo "============================================================"

#     # 2. 중간 루프: 설정된 모든 실험 타입을 순회
#     for exp_type in "${EXP_TYPES_TO_RUN[@]}"; do
        
#         ft_epochs=0
#         if [ "$exp_type" == "ft1" ]; then
#             ft_epochs=5
#         fi

#         echo "------------------------------------------------------------"
#         echo "           Running Experiment Type: [$exp_type]"
#         echo "------------------------------------------------------------"

#         for K in 2 3 4 5 6; do
#             NAME="${model_key}/${exp_type}/${K}bit"
#             SAVE_PATH="$CKPT/$NAME"
#             LOG_FILE="${LOG}/${NAME}.log"
#             HF_PATH="$HF/$NAME"

#             mkdir -p $SAVE_PATH
#             mkdir -p $(dirname "$LOG_FILE")

#             # echo "### [Stage: Quantize | K=$K] ###" | tee $LOG_FILE
#             # python -m quantize_llama.quantize_finetune_llama \
#             #     --save_path $SAVE_PATH \
#             #     --codebook bitshift \
#             #     --base_model $base_model \
#             #     --in_hess_path $HESS \
#             #     --scale_override 0.9 \
#             #     --ft_epochs $ft_epochs \
#             #     --td_x 16 --td_y 16 --L 16 --K $K --V 2 \
#             #     --decode_mode quantlut_sym --tlut_bits 9 2>&1 | tee -a $LOG_FILE

#             echo "### [Stage: Hfize | K=$K] ###" | tee -a $LOG_FILE
#             python -m quantize_llama.hfize_llama \
#                 --quantized_path $SAVE_PATH \
#                 --hf_output_path $HF_PATH \
#                 --base_model $base_model 2>&1 | tee -a $LOG_FILE

#             # python -m quantize_llama.hfize_llama_cal_mse \
#             #     --quantized_path $SAVE_PATH \
#             #     --hf_output_path $HF_PATH \
#             #     --output_path ${RES}/${NAME} \
#             #     --base_model $base_model 2>&1 | tee -a $LOG_FILE

#             MANIFEST_FLAG=""
#             if [ "$K" -ge 5 ]; then
#                 MANIFEST_FLAG="--manifest_model"
#             fi

#             echo "### [Stage: Eval PPL | K=$K] ###" | tee -a "$LOG_FILE"
#             python -m eval.eval_ppl \
#                 --hf_path "${HF_PATH}" \
#                 --output_path "${RES}/${NAME}" \
#                 --seqlen 2048 \
#                 $MANIFEST_FLAG 2>&1 | tee -a "$LOG_FILE"

#             echo "### [Stage: Eval Zero-shot | K=$K] ###" | tee -a "$LOG_FILE"
#             python -m eval.eval_zeroshot \
#                 --tasks hellaswag,mmlu \
#                 --batch_size 1 \
#                 --hf_path "${HF_PATH}" \
#                 --output_path "${RES}/${NAME}_hs_mmlu" \
#                 $MANIFEST_FLAG 2>&1 | tee -a "$LOG_FILE"


#             # python -m eval.eval_zeroshot \
#             #     --tasks mmlu \
#             #     --batch_size 8  --hf_path ${HF_PATH} \
#             #     --output_path ${RES}/${NAME}_mmlu 2>&1 | tee -a $LOG_FILE


#             # RESULT_FILE="${RES}/${NAME}_zeroshot_results.json"
#             # if [ ! -f "${RESULT_FILE}" ]; then
#             #     echo "결과 파일이 존재하지 않습니다. 제로샷 평가를 시작합니다..."
#             #     python -m eval.eval_zeroshot \
#             #         --tasks arc_challenge,arc_easy,boolq,piqa,winogrande \
#             #         --batch_size 8 \
#             #         --hf_path ${HF_PATH} \
#             #         --manifest_model \
#             #         --output_path ${RES}/${NAME} | tee -a $LOG_FILE
#             # else
#             #     # 파일이 이미 존재하면 메시지를 출력하고 넘어감
#             #     echo "이미 결과 파일(${RESULT_FILE})이 존재하므로 평가를 건너뜁니다."
#             # fi

#             # RESULT_FILE="${RES}/${NAME}_mmlu_zeroshot_results.json"
#             # if [ ! -f "${RESULT_FILE}" ]; then
#             #     echo "결과 파일이 존재하지 않습니다. 제로샷 평가를 시작합니다..."
#             #     python -m eval.eval_zeroshot \
#             #         --tasks mmlu \
#             #         --batch_size 8 \
#             #         --hf_path ${HF_PATH} \
#             #         --manifest_model \
#             #         --output_path ${RES}/${NAME}_mmlu | tee -a $LOG_FILE
#             # else
#             #     # 파일이 이미 존재하면 메시지를 출력하고 넘어감
#             #     echo "이미 결과 파일(${RESULT_FILE})이 존재하므로 평가를 건너뜁니다."
#             # fi

#             # RESULT_FILE="${RES}/${NAME}_2_zeroshot_results.json"
#             # if [ ! -f "${RESULT_FILE}" ]; then
#             #     echo "결과 파일이 존재하지 않습니다. 제로샷 평가를 시작합니다..."
#             #     python -m eval.eval_zeroshot \
#             #         --tasks hellaswag,openbookqa,mathqa,sciq,pubmedqa \
#             #         --batch_size 8 \
#             #         --hf_path ${HF_PATH} \
#             #         --manifest_model \
#             #         --output_path ${RES}/${NAME}_2 | tee -a $LOG_FILE
#             # else
#             #     # 파일이 이미 존재하면 메시지를 출력하고 넘어감
#             #     echo "이미 결과 파일(${RESULT_FILE})이 존재하므로 평가를 건너뜁니다."
#             # fi

#             # python -m eval.eval_zeroshot \
#             #     --tasks hellaswag,openbookqa,mathqa,sciq,pubmedqa \
#             #     --batch_size 4  \
#             #     --hf_path ${HF_PATH} \
#             #     --output_path ${RES}/${NAME}_2 | tee -a $LOG_FILE

#                 # --tasks arc_challenge,arc_easy,boolq,piqa,winogrande \

#             if [ "$HF_PATH" != "$HF" ]; then
#                 echo "Cleaning up temporary files for $SAVE_NAME"
#                 rm -rf "$HF_PATH"
#             fi
#         done
#     done
# done

# echo "############################################################"
# echo "##               All experiments finished.                ##"
# echo "############################################################"