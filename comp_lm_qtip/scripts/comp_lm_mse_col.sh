#!/bin/bash
# ##########################################################################
# ##                       EXPERIMENT CONFIGURATION                       ##
# ##########################################################################
comp_model_bases=(
    "dummy"
    "dummy"
)
quantize_flags=(
    "dummy"
    "dummy"
)
experiment_names=(
    "ql_ldlq128_rnorm_ft"
    # "scaleH_std_ldlq128_scale_cond(col)_ft/size128_encdim1024_M256"
)
##########################################################################
##                           MODEL CONFIGURATION                        ##
##########################################################################
model_names=(
    "meta-llama--Llama-2-7b-hf"
    # "meta-llama--Meta-Llama-3-8B"
    # "meta-llama--Llama-3.2-3B"
    # "meta-llama--Llama-2-13b-hf"
    # "meta-llama--Llama-2-70b-hf_"
)
hess_paths=(
    # "../Wparam_dataset/quip_hess/Hessians-Llama-2-7b-6144"
    "../Wparam_dataset/quip_hess/llama3_8b_6144"
    # "../Wparam_dataset/quip_hess/meta-llama--Llama-3.2-3B-256"
    # "../Wparam_dataset/quip_hess/Hessians-Llama-2-13b-6144"
    # "../Wparam_dataset/quip_hess/llama2_70b_relaxml_git/Hessians-Llama-2-70b-6144"
)
############################################
##              SCRIPT SETUP              ##
############################################
CKPT="../hf_model_comp/comp_qtip/ckpt"
HF="../hf_model_comp/comp_qtip/hf"
LOG="./log"
RES="../hf_model_comp_results"

mkdir -p $CKPT
mkdir -p $HF
mkdir -p $LOG
mkdir -p $RES
export CUDA_VISIBLE_DEVICES=0
export WANDB_SILENT=true
export HF_HOME=/workspace/hf_cache/huggingface_nwc
# export TRANSFORMERS_CACHE=$HF_HOME
# export HF_DATASETS_CACHE=$HF_HOME/datasets
# export HF_METRICS_CACHE=$HF_HOME/metrics

# 모든 실험에 공통으로 적용될 Lambda 값
# lmbda_values=(30 50 100 300 1000 10000)
# lmbda_values=(10 20 30 50 100 300 1000)
lmbda_values=(10 15 20 30 50 100 300 1000 10000)
# lmbda_values=(99 82 88 66 77 93 71)
##########################################################################
##                        MAIN EXECUTION LOOP                           ##
##########################################################################
# 1. 외부 루프: 설정된 모든 실험을 순회
for i in "${!experiment_names[@]}"; do
    # 현재 실험에 대한 변수 설정
    exp_name="${experiment_names[$i]}"
    comp_model_base="${comp_model_bases[$i]}"
    current_quantize_flags="${quantize_flags[$i]}"

    echo "========================================================================"
    echo "            STARTING EXPERIMENT SET: $exp_name"
    echo "========================================================================"

    # 2. 중간 루프: 설정된 모든 모델을 순회
    for j in "${!model_names[@]}"; do
        model_name="${model_names[$j]}"
        HESS="${hess_paths[$j]}"
        lm_model_path="../Wparam_dataset/hf_model/$model_name"

        echo "------------------------------------------------------------------------"
        echo "            MODEL: $model_name"
        echo "------------------------------------------------------------------------"
        echo "Using comp_model_base: $comp_model_base"
        echo "Using quantize flags: $current_quantize_flags"
        echo "Using Hessian path: $HESS"
        echo "------------------------------------------------------------------------"
        
        # 3. 내부 루프: 설정된 모든 Lambda 값을 순회
        for lmbda in "${lmbda_values[@]}"; do
            SAVE_NAME=${model_name}/${exp_name}/lmbda${lmbda}
            
            echo "################## Running hfize | lmbda=${lmbda} | Exp: ${exp_name} | Model: ${model_name} ##################"
            python -m quantize_llama.hfize_llama_cal_col_mse --quantized_path $CKPT/${SAVE_NAME} \
                    --hf_output_path $HF/${SAVE_NAME} \
                    2>&1 | tee -a $LOG/${SAVE_NAME}.log
        done
    done
done
