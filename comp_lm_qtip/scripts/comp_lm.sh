#!/bin/bash

# ##########################################################################
# ##                       EXPERIMENT CONFIGURATION                       ##
# ## -> 여기서 실행하고자 하는 실험들을 배열에 추가하거나 수정하세요.         ##
# ##########################################################################

# 각 실험에 대한 설명적인 이름 (SAVE_NAME의 일부로 사용됨)
experiment_names=(
    # "gaussian_padding_M16" # 예시: 새로운 실험 추가
    # "gaussian_padding_M32" # 예시: 새로운 실험 추가
    "scaleHinv_std_ldlq128/size128_encdim1024_M256"
    "scaleH_std_ldlq128/size128_encdim1024_M256"
    "scaleH_std_ldlq128/size128_encdim2048_M256"
)

# 위 experiment_names에 1:1로 매칭되는 comp_model_base 경로
comp_model_bases=(
    # "../NWC/checkpoint/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt/M16" # 예시
    # "../NWC/checkpoint/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt/M32__rdloss_ql_size16_encdim512_M32_Q4_R0_m0_batch_size2048_total_iter200000_lr0.0001_seed100" # 예시
    "../NWC/checkpoint/nwc_ql/block_seq_ql_random_pos_scaler_meta-llama--Meta-Llama-3-8B__scaleHinv_sig0.0001_std_rnormed_lidx_row_1024.pt/rdloss_ql_size128_encdim1024_M256_Q4_R0_m0_batch_size2048_total_iter200000_lr0.0001_seed100"
    "../NWC/checkpoint/nwc_ql/block_seq_ql_random_pos_scaler_meta-llama--Meta-Llama-3-8B__scaleH_sig0.0001_std_rnormed_lidx_row_1024.pt/rdloss_ql_size128_encdim1024_M256_Q4_R0_m0_batch_size2048_total_iter200000_lr0.0001_seed100"
    "../NWC/checkpoint/nwc_ql/block_seq_ql_random_pos_scaler_meta-llama--Meta-Llama-3-8B__scaleH_sig0.0001_std_rnormed_lidx_row_1024.pt/rdloss_ql_size128_encdim2048_M256_Q4_R0_m0_batch_size2048_total_iter200000_lr0.0001_seed100"
)

# 각 실험에 맞는 quantize_finetune_llama.py의 추가 인자들
# 각 설정은 하나의 문자열로 묶습니다.
quantize_flags=(
    # "--direction col" # 예시 (gaussian_padding_M16)
    # "--direction col" # 예시 (gaussian_padding_M32)
    "--direction row --scaleHinv --row_normalize --ldlq --comp_batch_size 128"
    "--direction row --scaleH --row_normalize --ldlq --comp_batch_size 128"
    "--direction row --scaleH --row_normalize --ldlq --comp_batch_size 128" 
)

##########################################################################
##                           MODEL CONFIGURATION                        ##
##         (모든 실험에 공통으로 적용되는 설정을 여기에 둡니다)             ##
##########################################################################

# model_name="meta-llama--Llama-2-7b-hf"
# HESS="../Wparam_dataset/quip_hess/Hessians-Llama-2-7b-6144"

model_name="meta-llama--Meta-Llama-3-8B"
HESS="../Wparam_dataset/quip_hess/llama3_8b_6144"

# model_name="meta-llama--Llama-2-13b-hf"
# HESS="../Wparam_dataset/quip_hess/Hessians-Llama-2-13b-6144"

############################################
##              SCRIPT SETUP              ##
############################################
lm_model_path="../Wparam_dataset/hf_model/$model_name"

CKPT="../hf_model_comp/comp_qtip/ckpt"
HF="../hf_model_comp/comp_qtip/hf"
LOG="./log"
RES="../hf_model_comp_results"

mkdir -p $CKPT
mkdir -p $HF
mkdir -p $LOG
mkdir -p $RES
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_SILENT=true

# 모든 실험에 공통으로 적용될 Lambda 및 QL 값
lmbda_values=(50 100)
ql_value=(0 1)

##########################################################################
##                        MAIN EXECUTION LOOP                           ##
##########################################################################
# 설정된 모든 실험을 순회하는 외부 루프]
for i in "${!experiment_names[@]}"; do
    # 현재 실험에 대한 변수 설정
    exp_name="${experiment_names[$i]}"
    comp_model_base="${comp_model_bases[$i]}"
    current_quantize_flags="${quantize_flags[$i]}"

    echo "========================================================================"
    echo "            STARTING EXPERIMENT: $exp_name"
    echo "========================================================================"
    echo "Using comp_model_base: $comp_model_base"
    echo "Using quantize flags: $current_quantize_flags"
    echo "========================================================================"

    for qlv in "${ql_value[@]}"; do
        for lmbda in "${lmbda_values[@]}"; do
            echo "################## Running compression | lmbda=${lmbda}, ql=${qlv} | Exp: ${exp_name} ##################"
            SAVE_NAME=${model_name}/${exp_name}/ql${qlv}_lmbda${lmbda}
            comp_model=$comp_model_base/lmbda${lmbda}_*/best_loss*.pth.tar
            mkdir -p $(dirname "$LOG/$SAVE_NAME.log")
            
            taskset -c 0-31 \
            python -m quantize_llama.quantize_finetune_llama --save_path $CKPT/$SAVE_NAME \
                --base_model $lm_model_path \
                --comp_model_path $comp_model \
                --in_hess_path $HESS --devset_size 384 --ft_valid_size 128 --batch_size 8 \
                --ft_epochs 0 \
                --ql_search_value ${qlv} \
                ${current_quantize_flags} \
                2>&1 | tee $LOG/$SAVE_NAME.log

            echo "################## Running hfize | lmbda=${lmbda}, ql=${qlv} | Exp: ${exp_name} ##################"
            python -m quantize_llama.hfize_llama --quantized_path $CKPT/${SAVE_NAME} \
                    --hf_output_path $HF/$SAVE_NAME 2>&1 | tee -a $LOG/$SAVE_NAME.log

            echo "################## Running PPL evaluation | lmbda=${lmbda}, ql=${qlv} | Exp: ${exp_name} ##################"
            pretrain_path=$HF/$SAVE_NAME
            mkdir -p "$log_dir"
            echo "Running evaluation for directory: $pretrain_path"
            python -m eval.eval_ppl_hf \
                --hf_path $pretrain_path \
                --seqlen 2048 \
                --output_path $RES/$SAVE_NAME \
                --no_use_cuda_graph 2>&1 | tee -a $LOG/$SAVE_NAME.log

            # echo "################## Running benchmark evaluation | lmbda=${lmbda}, ql=${qlv} | Exp: ${exp_name} ##################"
            # python -m eval.eval_zeroshot_hf \
            #     --tasks arc_challenge,arc_easy,boolq,piqa,winogrande \
            #     --batch_size 8  \
            #     --hf_path $pretrain_path \
            #     --output_path $RES/$SAVE_NAME

            if [ "$pretrain_path" != "$HF" ]; then
                echo "Cleaning up temporary files for $SAVE_NAME"
                rm -rf "$pretrain_path"
                rm -rf "$CKPT/$SAVE_NAME"
            fi
        done
    done
done

echo "========================================================================"
echo "                          All experiments finished."
echo "========================================================================"