#!/bin/bash
set -o pipefail  # <--- 이 줄 추가
set -x           # <--- 이 줄 추가


MODELS_TO_RUN=(
    # "llama3.2_1b_inst"
    # "llama3.2_3b_inst"
    "llama3_8b"
    # "llama2_7b"
    # "llama2_13b"
    # "llama3.2_3b"
)

EXP_TYPES_TO_RUN=(
    "ft1"
    # "noft"
)

CKPT="../hf_model_comp/quip-sharp/ckpt"
HF="../hf_model_comp/quip-sharp/hf"
LOG="./log"
RES="../hf_model_comp_results/quip-sharp"

export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
export WANDB_SILENT=true
export HF_HOME=/home/jgryu/.cache/huggingface

declare -A MODEL_PATHS

MODEL_PATHS=(
    ["llama2_7b"]="../Wparam_dataset/hf_model/meta-llama--Llama-2-7b-hf"
    # ["llama2_7b"]="meta-llama/Llama-2-7b-hf"
    ["llama2_13b"]="../Wparam_dataset/hf_model/meta-llama--Llama-2-13b-hf"
    # ["llama2_13b"]="meta-llama/Llama-2-13b-hf"
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

        for K in 4; do
            NAME="${model_key}/${exp_type}_e2e/${K}bit"
            CKPT_PATH="${CKPT}/${NAME}"
            HF_PATH="${HF}/${NAME}"
            
            OUT_NAME="${model_key}/${exp_type}_e2e_checkpoints/${K}bit"
            E2E_OUT_HF="${CKPT}/${OUT_NAME}"
            LOG_FILE="${LOG}/${NAME}.log"

            mkdir -p $CKPT_PATH
            mkdir -p $(dirname "$LOG_FILE")

            if [ "$K" -eq 2 ]; then
                CODEBOOK="E8P12"
            elif [ "$K" -eq 3 ]; then
                CODEBOOK="E8P12RVQ3B"
            elif [ "$K" -eq 4 ]; then
                CODEBOOK="E8P12RVQ4B"
            fi

            # echo "[Stage: Quantize with Finetuning] K=$K" | tee $LOG_FILE
            # python -m quantize_llama.quantize_finetune_llama \
            #     --save_path $CKPT_PATH \
            #     --codebook $CODEBOOK \
            #     --scale_override 0.9 \
            #     --base_model $base_model \
            #     --hessian_path $HESS \
            #     --devset_size 384 \
            #     --ft_epochs $ft_epochs \
            #     --ft_valid_size 128 2>&1 | tee $LOG_FILE

            # echo "[Stage: Backup Checkpoint] K=$K" | tee -a $LOG_FILE
            # echo "Copying ${CKPT_PATH} to ${CKPT_PATH}_ft1_only" | tee -a $LOG_FILE
            # cp -r "$CKPT_PATH" "${CKPT_PATH}_ft1_only" 2>&1 | tee -a $LOG_FILE

            echo "[Stage: Convert to HF format] K=$K" | tee -a $LOG_FILE
            python -m quantize_llama.hfize_llama \
                --quantized_path ${CKPT_PATH} \
                --hf_output_path $HF_PATH 2>&1 | tee -a $LOG_FILE

            echo "[Stage: End-to-End Finetuning] K=$K" | tee $LOG_FILE
            python -m quantize_llama.finetune_e2e_llama_qtip \
                --base_model $base_model \
                --hf_path $HF_PATH \
                --devset_size 384 \
                --ft_valid_size 128 \
                --ft_epochs 8 \
                --ft_bs 1 \
                --ctx_size 4096 \
                --ft_update_freq 2 \
                --batch_size 8 \
                --ckpt_path ${CKPT_PATH} \
                --hf_output_path $E2E_OUT_HF 2>&1 | tee -a $LOG_FILE


                # --resume_ckpt /home/jgryu/workspace/weight_compression/hf_model_comp/quip-sharp/ckpt/llama3_8b/hf_ft1_e2e/checkpoints/checkpoint_epoch_4.pt \

                # --ckpt_path ${CKPT_PATH} 2>&1 | tee -a $LOG_FILE
                # --ft_train_mode \

            # example
            # python -m quantize_llama.finetune_e2e_llama --base_model meta-llama/Llama-2-7b-hf --hf_path $HF/2_7b_2bit --devset_size 384 --ft_valid_size 128 --ft_epochs 8  --ft_bs 1 --ctx_size 4096 --ft_update_freq 2 --ft_train_mode --ckpt_path $CKPT/2_7b_2bit >> $LOG/2_7b_2bit 2>&1


            if [ "$HF_PATH" != "$HF" ] && [ "$exp_type" != "e2e" ]; then
                echo "Cleaning up temporary files for $NAME"
                rm -rf "$HF_PATH"
            fi

            echo "[Stage: Convert to HF format] K=$K" | tee -a $LOG_FILE
            python -m quantize_llama.hfize_llama \
                --quantized_path ${CKPT_PATH} \
                --hf_output_path $HF_PATH 2>&1 | tee -a $LOG_FILE

            echo "### [Stage: Eval PPL | K=$K] ###" | tee -a $LOG_FILE
            python -m eval.eval_ppl \
                --hf_path $HF_PATH \
                --output_path ${RES}/${NAME} \
                --no_use_cuda_graph \
                --seqlen 2048  2>&1 | tee -a $LOG_FILE

            echo "### [Stage: Eval Zero-shot | K=$K] ###" | tee -a $LOG_FILE
            python -m eval.eval_zeroshot_ \
                --tasks arc_challenge,arc_easy,boolq,piqa,winogrande,hellaswag,mmlu \
                --batch_size 8  --hf_path ${HF_PATH} \
                --output_path ${RES}/${NAME}_common_mmlu 2>&1 | tee -a $LOG_FILE

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



# python -m quantize_llama.hfize_llama \
#     --quantized_path /home/jgryu/workspace/weight_compression/hf_model_comp/quip-sharp/ckpt/llama3_8b/ft1_e2e_after/2bit \
#     --hf_output_path /home/jgryu/workspace/weight_compression/hf_model_comp/quip-sharp/hf/llama3_8b/ft1_e2e_after/2bit

# python -m quantize_llama.finetune_e2e_llama_save_ckpt \
#     --ckpt_path /home/jgryu/workspace/weight_compression/hf_model_comp/quip-sharp/ckpt/llama3_8b/ft1_e2e_after/2bit \
#     --output_ckpt_path /home/jgryu/workspace/weight_compression/hf_model_comp/quip-sharp/ckpt/llama3_8b/hf_ft1_e2e/2bit_checkpoints/checkpoint_epoch_7.pt \
#     --hf_path /home/jgryu/workspace/weight_compression/hf_model_comp/quip-sharp/hf/llama3_8b/ft1_e2e_after/2bit