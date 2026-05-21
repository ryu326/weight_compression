#!/bin/bash
# ##########################################################################
# ##                       EXPERIMENT CONFIGURATION                       ##
# ##########################################################################
comp_model_bases=(
    "../NWC/checkpoint/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt/M16"
    # "../NWC/checkpoint/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt/M16"
)
quantize_flags=(
    "--direction col --ql --Q 4 --normalization_search --ldlq --comp_batch_size 64 --ft_epochs 5"
    # "--direction col --ql --Q 4 --col_normalize --ldlq --comp_batch_size 64 --ft_epochs 5"
    # "--direction col --ql --Q 4 --row_normalize --ldlq --comp_batch_size 64 --ft_epochs 0"
)
experiment_names=(
    "ql_ldlq64_norm_search_ft"
    # "ql_ldlq64_rnorm_ft"
    # "ql_ldlq64_rnorm"
)

# [NEW] Define Lambda values specifically for each experiment above (Space separated strings)
# index 0 corresponds to "ql_ldlq64_rnorm_ft"
# index 1 corresponds to "ql_ldlq64_rnorm"
experiment_lmbdas=(
    "30 50 100 300 1000"        # Lambdas for the second experiment
)

##########################################################################
##                          MODEL CONFIGURATION                         ##
##########################################################################
model_names=(
    "mistralai/Mixtral-8x7B-v0.1"
    # "openai/gpt-oss-20b"
)
hess_paths=(
    "/home/jgryu/workspace/weight_compression/Wparam_dataset/quip_hess/Mixtral-8x7B-v0.1_1024"
    # "/home/jgryu/workspace/weight_compression/Wparam_dataset/quip_hess/gpt-oss-20b_1024"
)
############################################
##              SCRIPT SETUP              ##
############################################
CKPT="../hf_model_comp/comp_qtip/ckpt"
HF="../hf_model_comp/comp_qtip/hf"
LOG="./log"
RES="../hf_model_comp_results_v2"

mkdir -p $CKPT
mkdir -p $HF
mkdir -p $LOG
mkdir -p $RES
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export HF_HOME=/workspace/hf_cache/huggingface_nwc
export HF_HOME=/home/jgryu/.cache/huggingface

# [REMOVED] Global lmbda_values is replaced by experiment_lmbdas array above
# lmbda_values=(10000)

##########################################################################
##                         MAIN EXECUTION LOOP                          ##
##########################################################################
# 2. Middle Loop: Iterate over models
for j in "${!model_names[@]}"; do
    model_name="${model_names[$j]}"
    HESS="${hess_paths[$j]}"
    lm_model_path="$model_name"

    echo "------------------------------------------------------------------------"
    echo "            MODEL: $model_name"
    echo "------------------------------------------------------------------------"
    echo "Using Hessian path: $HESS"
    echo "------------------------------------------------------------------------"


    # 1. Outer Loop: Iterate over experiment configurations
    for i in "${!experiment_names[@]}"; do
        # Set variables for current experiment
        exp_name="${experiment_names[$i]}"
        comp_model_base="${comp_model_bases[$i]}"
        current_quantize_flags="${quantize_flags[$i]}"
        
        # [NEW] Extract the specific lambda list for this experiment
        current_lmbda_list="${experiment_lmbdas[$i]}"

        echo "========================================================================"
        echo "            STARTING EXPERIMENT SET: $exp_name"
        echo "            Lambdas to run: $current_lmbda_list"
        echo "========================================================================"
        
        # 3. Inner Loop: Iterate over the specific Lambda values for this experiment
        # We assume current_lmbda_list is a space-separated string (e.g. "1000 5000")
        for lmbda in $current_lmbda_list; do
            SAVE_NAME=${model_name}/${exp_name}/lmbda${lmbda}

            echo "################## Running compression | lmbda=${lmbda} | Exp: ${exp_name} | Model: ${model_name} ##################" 2>&1 | tee $LOG/$SAVE_NAME.log
            comp_model=$comp_model_base/lmbda${lmbda}_*/best_loss*.pth.tar
            mkdir -p $(dirname "$LOG/$SAVE_NAME.log")
            
            # Uncomment below to run compression
            # taskset -c 0-63 \
            # python -m quantize_llama.quantize_finetune_moe --save_path $CKPT/$SAVE_NAME \
            #     --base_model $lm_model_path \
            #     --comp_model_path $comp_model \
            #     --in_hess_path $HESS \
            #     --devset_size 384 --ft_valid_size 128 \
            #     --batch_size 4 --ft_bs 4 \
            #     ${current_quantize_flags} \
            #     2>&1 | tee -a $LOG/$SAVE_NAME.log
            
            echo "################## Running hfize | lmbda=${lmbda} | Exp: ${exp_name} | Model: ${model_name} ##################" 2>&1 | tee $LOG/${SAVE_NAME}_eval.log
            python -m quantize_llama.hfize_moe_hf --quantized_path $CKPT/${SAVE_NAME} \
                    --base_model $lm_model_path \
                    --hf_output_path $HF/${SAVE_NAME} \
                    2>&1 | tee -a $LOG/${SAVE_NAME}_eval.log
            hfize_exit_code=${PIPESTATUS[0]}
            if [ $hfize_exit_code -ne 0 ]; then
                echo "hfize failed with exit code $hfize_exit_code. Skipping eval: $SAVE_NAME" | tee -a "$LOG/${SAVE_NAME}_eval.log"
                continue
            fi


            echo "################## Running PPL evaluation | lmbda=${lmbda} | Exp: ${exp_name} | Model: ${model_name} ##################"
            echo "Running evaluation for directory: $HF/$SAVE_NAME"
            python -m eval.eval_ppl_hf \
                --hf_path $HF/${SAVE_NAME} \
                --seqlen 2048 \
                --output_path ${RES}/${SAVE_NAME} \
                --datasets wikitext2,c4 \
                --gptoss_replace_version standard \
                --no_use_cuda_graph 2>&1 | tee -a $LOG/${SAVE_NAME}_eval.log

            echo "################## Running benchmark evaluation | lmbda=${lmbda} | Exp: ${exp_name} | Model: ${model_name} ##################"
            python -m eval.eval_zeroshot_hf \
                --tasks arc_challenge,arc_easy,piqa,winogrande,boolq,hellaswag,mmlu \
                --hf_path $HF/$SAVE_NAME \
                --output_path $RES/${SAVE_NAME}_common_mmlu \
                --gptoss_replace_version standard \
                >> "$LOG/${SAVE_NAME}_eval.log" 2>&1 

            # 벤치마크 평가 결과 저장 (0이면 성공)
            eval_exit_code=${PIPESTATUS[0]}

            # ------------------------------------------------------------------
            # 삭제 로직 수정: 벤치마크가 성공(0)했을 때만 실행
            # ------------------------------------------------------------------
            if [ $eval_exit_code -eq 0 ]; then
                echo "Benchmark finished successfully. Starting cleanup..."

                # HF 모델 삭제
                if [ "$HF/$SAVE_NAME" != "$HF" ]; then
                    echo "Cleaning up temporary HF files: $HF/$SAVE_NAME"
                    rm -rf "$HF/$SAVE_NAME"
                    rm -rf "$CKPT/$SAVE_NAME"
                fi
            else
                echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                echo "CRITICAL: Benchmark failed with exit code $eval_exit_code."
                echo "Skipping cleanup to allow manual inspection: $SAVE_NAME"
                echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            fi
            # ------------------------------------------------------------------
        done
    done
done
