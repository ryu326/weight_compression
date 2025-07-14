#!/bin/bash

#########################################################################
##                           MODEL CONFIGURATION                        ##
##      (수정된 부분: 여러 모델과 해당 Hessian 경로를 배열로 정의합니다)      ##
##########################################################################

# 실험할 모델 이름 배열
declare -a model_names=(
    "meta-llama--Llama-2-7b-hf"
    # "meta-llama--Meta-Llama-3-8B"
    "meta-llama--Llama-2-13b-hf"
)

# 각 모델에 해당하는 Hessian 데이터 경로 배열 (위 모델과 순서가 일치해야 합니다)
declare -a hess_paths=(
    "../Wparam_dataset/quip_hess/Hessians-Llama-2-7b-6144"
    # "../Wparam_dataset/quip_hess/llama3_8b_6144"
    "../Wparam_dataset/quip_hess/Hessians-Llama-2-13b-6144"
)

# NIC 모델을 선택합니다. 'tcm' 또는 'ftic' 중 하나를 선택할 수 있습니다.
nic_model="tcm"

##########################################################################
##                      EXPERIMENT PARAMETERS                           ##
##         (모든 실험에 공통으로 적용되는 설정을 여기에 둡니다)             ##
##########################################################################

# 여러 양자화 방법을 배열로 정의
path=(512)
norm_path=(256)
# lmbda_values=(0483)
lmbda_values=(0.05)

############################################
##              SCRIPT SETUP              ##
############################################
CKPT="../hf_model_comp/nic/ckpt"
HF="../hf_model_comp/nic/hf"
LOG="./log"
RES="../hf_model_comp_results/nic"

mkdir -p $CKPT
mkdir -p $HF
mkdir -p $LOG
mkdir -p $RES
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_SILENT=true

echo "========================================================================"
echo "            STARTING ALL EXPERIMENTS"
echo "========================================================================"


# -------------------------------------------------------------------------
# << 외부 루프: 위에 정의된 모든 모델을 순차적으로 실행 >>
# -------------------------------------------------------------------------
for i in "${!model_names[@]}"; do

    # 현재 루프에 해당하는 모델과 Hessian 경로를 설정합니다
    model_name="${model_names[$i]}"
    HESS="${hess_paths[$i]}"
    lm_model_path="../Wparam_dataset/hf_model/$model_name"

    echo "========================================================================"
    echo ">>> [STARTING] MODEL: $model_name"
    echo "========================================================================"

    # -------------------------------------------------------------------------
    # << 내부 루프: 각 모델에 대해 모든 파라미터 조합을 실행 >>
    # -------------------------------------------------------------------------
    for ps in "${path[@]}"; do
        for nps in "${norm_path[@]}"; do
            for lm in "${lmbda_values[@]}"; do

                exp_name="group64_patch512_lmbda${lm}"
                
                SAVE_NAME="${model_name}/${nic_model}/${exp_name}"
                nic_checkpoint_path="./nic_models/TCM/checkpoints/${lm}.pth.tar"
                
                echo "################## Running compression | ${exp_name} | ${model_name} ##################"
                mkdir -p "$(dirname "$LOG/$SAVE_NAME.log")"
                
                taskset -c 0-31 \
                python -m quantize_llama.quantize_finetune_llama --save_path "$CKPT/$SAVE_NAME" \
                    --base_model "$lm_model_path" \
                    --in_hess_path "$HESS" --devset_size 384 --ft_valid_size 128 --batch_size 8 \
                    --ft_epochs 0 \
                    --nic_model "$nic_model" \
                    --nic_checkpoint "$nic_checkpoint_path" \
                    --nic_patch_size "$ps" --nic_norm_patch_size "$nps" \
                    --quant_method group --group_sz 64 \
                    2>&1 | tee "$LOG/$SAVE_NAME.log"

                echo "################## Running hfize | ${exp_name} | ${model_name} ##################"
                python -m quantize_llama.hfize_llama --quantized_path "$CKPT/${SAVE_NAME}" \
                        --hf_output_path "$HF/$SAVE_NAME" 2>&1 | tee -a "$LOG/$SAVE_NAME.log"

                pretrain_path="$HF/$SAVE_NAME"
                echo "################## Running PPL evaluation | ${exp_name} | ${model_name} ##################"
                echo "Running evaluation for directory: $pretrain_path"
                python -m eval.eval_ppl_hf \
                    --hf_path "$pretrain_path" \
                    --seqlen 2048 \
                    --output_path "$RES/$SAVE_NAME" \
                    --dataset wikitext2,c4 \
                    --no_use_cuda_graph 2>&1 | tee -a "$LOG/$SAVE_NAME.log"

                # echo "################## Running benchmark evaluation | ${exp_name} | ${model_name} ##################"
                # python -m eval.eval_zeroshot_hf \
                #     --tasks arc_challenge,arc_easy,boolq,piqa,winogrande \
                #     --batch_size 2  \
                #     --hf_path "$pretrain_path" \
                #     --output_path "$RES/$SAVE_NAME"

                if [ "$pretrain_path" != "$HF" ]; then
                    echo "Cleaning up temporary files for $SAVE_NAME"
                    rm -rf "$pretrain_path"
                    # rm -rf "$CKPT/$SAVE_NAME" # 필요시 주석 해제하여 중간 체크포인트도 삭제
                fi
            done
        done
    done
    # -------------------------------------------------------------------------
    # << 내부 루프 종료 >>
    # -------------------------------------------------------------------------

    echo ">>> [FINISHED] MODEL: $model_name"
    echo "========================================================================"
    echo ""

done
# -------------------------------------------------------------------------
# << 외부 루프 종료 >>
# -------------------------------------------------------------------------


echo "========================================================================"
echo "                          All experiments finished."
echo "========================================================================"