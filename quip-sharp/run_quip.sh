#!/bin/bash
##########################################################################
##                           SCRIPT CONTROLS                            ##
##########################################################################

# 실험을 수행할 모델 목록 (아래 'MODEL CONFIGURATION'에 정의된 이름 사용)
MODELS_TO_RUN=(
    # "llama3_8b"
    "llama2_7b"
    # "llama3.2_3b"
    "llama2_13b"
    "llama3.2_1b_inst"
    "llama3.2_3b_inst"
)

# 각 모델에 대해 수행할 실험 타입 목록 ('ft', 'noft')
EXP_TYPES_TO_RUN=(
    "ft1"
    "noft"
)


CKPT="../hf_model_comp/quip-sharp/ckpt"
HF="../hf_model_comp/quip-sharp/hf"
LOG="./log"
RES="../hf_model_comp_results/quip-sharp"

# --- 환경 변수 설정 ---
export CUDA_VISIBLE_DEVICES=2,3,5,6,7
export WANDB_SILENT=true

##########################################################################
##                         MODEL CONFIGURATION                          ##
##                  (사용할 모델과 경로를 여기에 정의)                    ##
##########################################################################
declare -A MODEL_PATHS
MODEL_PATHS=(
    # ["llama2_7b"]="../Wparam_dataset/hf_model/meta-llama--Llama-2-7b-hf"
    ["llama2_7b"]="meta-llama/Llama-2-7b-hf"
    # ["llama2_13b"]="../Wparam_dataset/hf_model/meta-llama--Llama-2-13b-hf"
    ["llama2_13b"]="meta-llama/Llama-2-13b-hf"
    ["llama3_8b"]="../Wparam_dataset/hf_model/meta-llama--Meta-Llama-3-8B"
    ["vicuna_7b"]="../Wparam_dataset/hf_model/lmsys--vicuna-7b-v1.5"
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

        for K in 2 3 4; do
            NAME="${model_key}/${exp_type}/${K}bit"
            # NAME="3_8b_ft1/3_8b_${K}bit"
            # SAVE_PATH="${CKPT}/3_8b_${K}bit_${exp_type}"
            SAVE_PATH="${CKPT}/${NAME}"
            LOG_FILE="${LOG}/${NAME}.log"
            HF_PATH="$HF/$NAME"

            mkdir -p $SAVE_PATH
            mkdir -p $(dirname "$LOG_FILE")

            if [ "$K" -eq 2 ]; then
                CODEBOOK="E8P12"
            elif [ "$K" -eq 3 ]; then
                CODEBOOK="E8P12RVQ3B"
            elif [ "$K" -eq 4 ]; then
                CODEBOOK="E8P12RVQ4B"
            fi

            echo "[Stage: Quantize with Finetuning] K=$K" | tee $LOG_FILE
            python -m quantize_llama.quantize_finetune_llama \
                --save_path $SAVE_PATH \
                --codebook $CODEBOOK \
                --scale_override 0.9 \
                --base_model $base_model \
                --hessian_path $HESS \
                --devset_size 384 \
                --ft_epochs 0 \
                --ft_valid_size 128 2>&1 | tee -a $LOG_FILE

            echo "[Stage: Convert to HF format] K=$K" | tee $LOG_FILE
            python -m quantize_llama.hfize_llama \
                --quantized_path $SAVE_PATH \
                --hf_output_path $HF_PATH 2>&1 | tee -a $LOG_FILE

            # echo "[Stage: End-to-End Finetuning] K=$K" | tee -a $LOG_FILE
            # python -m quantize_llama.finetune_e2e_llama \
            #     --base_model meta-llama/Llama-3.2-3B \
            #     --hf_path $HF_PATH \
            #     --devset_size 384 \
            #     --ft_valid_size 128 \
            #     --ft_epochs 8 \
            #     --ft_bs 1 \
            #     --ctx_size 4096 \
            #     --ft_update_freq 2 \
            #     --ft_train_mode \
            #     --batch_size 4 \
            #     --ckpt_path ${SAVE_PATH} 2>&1 | tee -a $LOG_FILE

            echo "### [Stage: Eval PPL | K=$K] ###" | tee -a $LOG_FILE
            python -m eval.eval_ppl \
                --hf_path ${HF_PATH} \
                --output_path ${RES}/${NAME} \
                --seqlen 2048  2>&1 | tee -a $LOG_FILE

            echo "### [Stage: Eval Zero-shot | K=$K] ###" | tee -a $LOG_FILE
            python -m eval.eval_zeroshot_ \
                --tasks arc_challenge,arc_easy,boolq,piqa,winogrande,hellaswag,mmlu \
                --batch_size 8  --hf_path ${HF_PATH} \
                --output_path ${RES}/${NAME}_common_mmlu 2>&1 | tee -a $LOG_FILE

            # echo "### [Stage: Eval Zero-shot | K=$K] ###" | tee -a $LOG_FILE
            # python -m eval.eval_zeroshot_ \
            #     --tasks mmlu \
            #     --batch_size 8  --hf_path ${HF_PATH} \
            #     --num_fewshot 5 \
            #     --output_path ${RES}/${NAME}_mmlu5shot 2>&1 | tee -a $LOG_FILE

            # echo "### [Stage: Eval Zero-shot | K=$K] ###" | tee -a $LOG_FILE
            # python -m eval.eval_zeroshot_ \
            #     --tasks gsm8k \
            #     --batch_size 8  --hf_path ${HF_PATH} \
            #     --num_fewshot 8 \
            #     --output_path ${RES}/${NAME}_gsm8k8shot 2>&1 | tee -a $LOG_FILE

            if [ "$HF_PATH" != "$HF" ] && [ "$exp_type" != "e2e" ]; then
                echo "Cleaning up temporary files for $NAME"
                rm -rf "$HF_PATH"
            fi
        done
    done
done

echo "############################################################"
echo "##               All experiments finished.                ##"
echo "############################################################"

# export CUDA_VISIBLE_DEVICES=0,1,2,3

# CKPT="../hf_model_comp/quip-sharp/ckpt"
# HF="../hf_model_comp/quip-sharp/hf"
# LOG="./log"

# HESS="/home/minkyu4506/weight_compression_dataset/llama3_8b_6144"
# # HESS="../Wparam_dataset/quip_hess/meta-llama--Llama-3.2-3B"
# # HESS="../Wparam_dataset/quip_hess/meta-llama--Llama-3.2-3B-256"

# mkdir -p $CKPT
# mkdir -p $LOG
# mkdir -p $HF

# for K in 
# do
#     echo "Running quantization for K=$K bits"

#     NAME="3_8b_ft1/3_8b_${K}bit"
#     SAVE_PATH="$CKPT/$NAME"
#     LOG_FILE="$LOG/$NAME"
#     HF_PATH="$HF/$NAME"

#     mkdir -p $SAVE_PATH
#     mkdir -p $HF_PATH

#     if [ "$K" -eq 2 ]; then
#         CODEBOOK="E8P12"
#     elif [ "$K" -eq 3 ]; then
#         CODEBOOK="E8P12RVQ3B"
#     elif [ "$K" -eq 4 ]; then
#         CODEBOOK="E8P12RVQ4B"
#     fi

#     echo "[Stage: Quantize with Finetuning] K=$K" | tee $LOG_FILE
#     python -m quantize_llama.quantize_finetune_llama \
#         --save_path $SAVE_PATH \
#         --codebook $CODEBOOK \
#         --scale_override 0.9 \
#         --base_model meta-llama/Llama-3.2-3B \
#         --hessian_path $HESS \
#         --devset_size 384 \
#         --ft_epochs 0 \
#         --ft_valid_size 128 2>&1 | tee -a $LOG_FILE

#     echo "[Stage: Convert to HF format] K=$K" | tee $LOG_FILE
#     python -m quantize_llama.hfize_llama \
#         --quantized_path $SAVE_PATH \
#         --hf_output_path $HF_PATH 2>&1 | tee -a $LOG_FILE

#     # echo "[Stage: End-to-End Finetuning] K=$K" | tee -a $LOG_FILE
#     # python -m quantize_llama.finetune_e2e_llama \
#     #     --base_model meta-llama/Llama-3.2-3B \
#     #     --hf_path $HF_PATH \
#     #     --devset_size 384 \
#     #     --ft_valid_size 128 \
#     #     --ft_epochs 8 \
#     #     --ft_bs 1 \
#     #     --ctx_size 4096 \
#     #     --ft_update_freq 2 \
#     #     --ft_train_mode \
#     #     --batch_size 4 \
#     #     --ckpt_path ${SAVE_PATH} 2>&1 | tee -a $LOG_FILE

#     echo "[Stage: Eval PPL] K=$K" | tee -a $LOG_FILE
#     python -m eval.eval_ppl \
#         --no_use_cuda_graph \
#         --hf_path $HF_PATH | tee -a ${HF_PATH}_ppl_result.txt

#     echo "[Stage: Eval Zero-shot] K=$K" | tee -a $LOG_FILE
#     python -m eval.eval_zeroshot \
#         --tasks arc_challenge,arc_easy,boolq,piqa,winogrande \
#         --batch_size 4 \
#         --output_path ${HF_PATH}_zeroshot_result.json \
#         --hf_path $HF_PATH 2>&1 | tee -a ${HF_PATH}_zeroshot_result.txt

#     #### No-Finetune 버전
#     # SAVE_PATH_no_ft="$CKPT/3_8b_${K}bit_no_ft"
#     # LOG_FILE_no_ft="$LOG/3_8b_${K}_no_ft.txt"
#     # HF_PATH_no_ft="$HF/3_8b_${K}bit_no_ft"

#     # echo "[Stage: Quantize (No Finetuning)] K=$K" | tee $LOG_FILE_no_ft
#     # python -m quantize_llama.quantize_finetune_llama \
#     #     --save_path $SAVE_PATH_no_ft \
#     #     --codebook $CODEBOOK \
#     #     --scale_override 0.9 \
#     #     --base_model meta-llama/Meta-Llama-3-8B \
#     #     --hessian_path $HESS \
#     #     --devset_size 384 \
#     #     --ft_epochs 0 \
#     #     --ft_valid_size 128 2>&1 | tee -a $LOG_FILE_no_ft

#     # echo "[Stage: Convert to HF format (No Finetuning)] K=$K" | tee -a $LOG_FILE_no_ft
#     # python -m quantize_llama.hfize_llama \
#     #     --quantized_path $SAVE_PATH_no_ft \
#     #     --hf_output_path $HF_PATH_no_ft 2>&1 | tee -a $LOG_FILE_no_ft

#     # echo "[Stage: Eval PPL (No Finetuning)] K=$K" | tee -a $LOG_FILE_no_ft
#     # python -m eval.eval_ppl \
#     #     --no_use_cuda_graph \
#     #     --hf_path $HF_PATH_no_ft 2>&1 | tee -a ${HF_PATH_no_ft}_ppl_result.txt

#     # echo "[Stage: Eval Zero-shot (No Finetuning)] K=$K" | tee -a $LOG_FILE_no_ft
#     # python -m eval.eval_zeroshot \
#     #     --tasks arc_challenge,arc_easy,boolq,piqa,winogrande \
#     #     --batch_size 4 \
#     #     --hf_path $HF_PATH_no_ft 2>&1 | tee -a ${HF_PATH_no_ft}_zeroshot_result.txt

# done
