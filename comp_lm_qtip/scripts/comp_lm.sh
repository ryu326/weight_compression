#!/bin/bash
# ##########################################################################
# ##                       EXPERIMENT CONFIGURATION                       ##
# ##########################################################################
comp_model_bases=(
    # "../NWC/checkpoint/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt/M16"
    # '/workspace/Weight_compression/NWC/checkpoint/nwc_scale_cond/block_seq_scale_cond_scaler_meta-llama--Meta-Llama-3-8B__scaleH_sig0.0001_std_rnormed_with_col_std_lidx_row_1024.pt/rdloss_size128_encdim1024_M256_Q0_R0_m0_batch_size2048_total_iter200000_lr0.0001_seed100'
<<<<<<< HEAD
    '/workspace/Weight_compression/NWC/checkpoint/nwc_scale_cond/block_seq_scale_cond_scaler_meta-llama--Meta-Llama-3-8B__scaleH_sig0.0001_std_rnormed_with_col_std_lidx_row_1024.pt/aug_scale_p.2_M2_rdloss_size128_encdim1024_M256_Q0_R0_m0_batch_size2048_total_iter200000_lr0.0001_seed100'
    '/workspace/Weight_compression/NWC/checkpoint/nwc_scale_cond/block_seq_scale_cond_scaler_meta-llama--Meta-Llama-3-8B__scaleH_sig0.0001_std_rnormed_with_col_std_lidx_row_1024.pt/aug_scale_p.1_M4_rdloss_size128_encdim1024_M256_Q0_R0_m0_batch_size2048_total_iter200000_lr0.0001_seed100'
    '/workspace/Weight_compression/NWC/checkpoint/nwc_scale_cond/block_seq_scale_cond_scaler_meta-llama--Meta-Llama-3-8B__scaleH_sig0.0001_std_rnormed_with_col_std_lidx_row_1024.pt/aug_scale_p.2_M4_rdloss_size128_encdim1024_M256_Q0_R0_m0_batch_size2048_total_iter200000_lr0.0001_seed100'
=======
    # '/workspace/Weight_compression/NWC/checkpoint/nwc_scale_cond/block_seq_scale_cond_scaler_meta-llama--Llama-2-7b-hf__row_256_scaleH0.0001_rnormed_scale_cond(col_std).pt/rdloss_size128_encdim1024_M256_Q0_R0_m0_batch_size8192_total_iter200000_lr0.0001_seed100'
    # "/workspace/Weight_compression/NWC/checkpoint/nwc_ql_scale_cond/block_seq_scale_cond_scaler_meta-llama--Meta-Llama-3-8B__col_1024_scaleH0.0001_rnormed_scale_cond(col_std).pt/rdloss_size16_encdim512_M16_Q0_R0_m0_batch_size2048_total_iter200000_lr0.0001_seed100"
    "/workspace/Weight_compression/NWC/checkpoint/nwc_ql/block_seq_ql_random_scaler_synthetic__gaussian_llama8b_col_1024.pt/rdloss_ql_size16_encdim512_M16_Q4_R0_m0_batch_size2048_total_iter200000_lr0.0001_seed100"
>>>>>>> clip test
)
quantize_flags=(
    # "--direction col --scaleH --row_normalize --ldlq --comp_batch_size 128 --scale_cond"
    # "--direction row --scaleH --row_normalize --scale_cond --ldlq --comp_batch_size 128 --scale_cond_ub 31.6"
    # "--direction row --scaleH --row_normalize --scale_cond --ldlq --comp_batch_size 128"
    "--direction row --scaleH --row_normalize --scale_cond --ldlq --comp_batch_size 128"
    "--direction row --scaleH --row_normalize --scale_cond --ldlq --comp_batch_size 128"
    "--direction row --scaleH --row_normalize --scale_cond --ldlq --comp_batch_size 128"
    # "--direction row --scaleH --row_normalize --scale_cond --ldlq --comp_batch_size 128 --ft_epochs 5 --ft_metadata"
    # "--direction row --row_normalize --scale_cond --ldlq --comp_batch_size 128"
<<<<<<< HEAD
    # "--direction row --scaleH --row_normalize --ldlq --comp_batch_size 128 --scale_cond"
=======
>>>>>>> clip test
    # "--direction row --scaleH --row_normalize --ldlq --comp_batch_size 128 --scale_cond"
    # "--direction row --scaleH --row_normalize --ldlq --comp_batch_size 128 --scale_cond"
    "--direction col --ql --Q 4 --row_normalize"
    # "--direction col --ql --Q 4 --row_normalize --ldlq --comp_batch_size 128"
    # "--direction col --ql --Q 4 --row_normalize --ldlq --comp_batch_size 128"
    # "--direction col --ql --Q 4 --row_normalize --ldlq --comp_batch_size 32"
    # "--direction col --ql --Q 4 --row_normalize --ldlq --comp_batch_size 128 --ft_epochs 5"
    # "--direction col --ql --Q 4 --row_normalize --ft_epochs 5 --ft_rnorm"
    # "--direction col --ql --Q 4 --fp_iter --fp_iter_max 30"
    # "--direction col --ql --Q 4 --row_normalize --ldlq --comp_batch_size 128 --ft_epochs 5"
    # "--direction col --ql --Q 4 --row_normalize --ldlq --comp_batch_size 128 --ft_epochs 5 --ft_rnorm"
    # "--direction col --ql --Q 4 --row_normalize --ldlq --comp_batch_size 128 --ft_epochs 5 --ft_metadata"
    # "--direction col --ql --Q 4 --row_normalize --comp_batch_size 1024 --ft_epochs 5 --ft_y"
    # "--direction row --scaleH --row_normalize --ft_epochs 5 --ft_scale_cond0 --scale_cond0"
    # "--direction row --scaleHinv --row_normalize --ldlq --comp_batch_size 128"
    # "--direction row --scaleH --row_normalize --scale_cond --ldlq --comp_batch_size 128 --ft_epochs 5"
    # "--direction row --scaleH --row_normalize --scale_cond --ldlq --comp_batch_size 128 --lb_scaleH 1"
    # "--direction row --scaleH --row_normalize --scale_cond --ldlq --comp_batch_size 128"
    # "--direction row --scaleH --row_normalize --scale_cond --ldlq --comp_batch_size 128 --smooth_scaleH_alpha 0.5"
    # "--direction row --scaleH --row_normalize --ldlq --comp_batch_size 128 --scale_cond --scale_cond_test 0.01" 
    # "--direction row --scaleH --row_normalize --ldlq --comp_batch_size 128 --scale_cond2" 
    # "--direction row --scaleH --row_normalize --ldlq --comp_batch_size 128 --scale_cond2" 
    # "--direction row --scaleH --row_normalize --ldlq --comp_batch_size 128 --scale_cond2" 
    # "--direction row --scaleH --ldlq --comp_batch_size 128 --scale_cond3"
    # "--direction row --scaleH --row_normalize --comp_batch_size 2048 --scale_cond --fp_iter --fp_iter_max 10" 
    # "--direction col --ql --Q 4 --row_normalize"
    # "--direction col --ql --Q 4 --layer_normalize"
    # "dummy"
    # "dummy"
    # "--direction row --row_normalize --scale_cond0 --ldlq --comp_batch_size 128"
    # "--direction row --row_normalize --scale_cond0 --ldlq --comp_batch_size 128 --scale_cond_ub 50"
    # "--direction row --scale_cond0 --ldlq --comp_batch_size 128"
    # "--direction row --scale_cond0 --ldlq --comp_batch_size 128"
    # "--direction row --comp_batch_size 4096 --whiten --in_hess_eig_path /workspace/Weight_compression/Wparam_dataset/quip_hess_eig_reg0.0001/llama3_8b_6144"
)
experiment_names=(
    # "scaleH_rnorm_ldlq128_scale_cond(col_std)/(uniform31.6)size128_encdim1024_M256"
    # "scaleH_rnorm_ldlq128_scale_cond(col_std)/size128_encdim1024_M256"
<<<<<<< HEAD
    "scaleH_rnorm_ldlq128_scale_cond(col_std)/(aug_scale_p.2_M2)size128_encdim1024_M256"
    "scaleH_rnorm_ldlq128_scale_cond(col_std)/(aug_scale_p.1_M4)size128_encdim1024_M256"
    "scaleH_rnorm_ldlq128_scale_cond(col_std)/(aug_scale_p.2_M4)size128_encdim1024_M256"
    # "scale_cond(col)_rnorm/size128_encdim1024_M256"
    # "ql_compand_rnorm(trained_rnormed)"
    # "ql_compand_ldlq128_rnorm"
    # "ql_ldlq128_rnorm_debug"
    # "ql_fp_test"
    # "ql_rnorm2_ldlq128"
    # "scaleH_std_scale_cond(diagh)_ft_ft/size128_encdim2048_M256"
    # "ql_ldlq128_rnorm_ft(7b_droplast)"
    # "ql_ltc_ldlq128_rnorm"
    # "ql_ldlq128_rnorm_M32"
    # "ql_ldlq128_rnorm"
    # "ql_ldlq128_rnorm_ft_meta_ft_test"
    # "ql_ldlq128_rnorm_test"
    # "ql_y_ft_ft_test"
    # "scaleHinv_std_ldlq128/size128_encdim1024_M256"
    # "scaleH_std_ldlq128/size128_encdim1024_M256"
    # "scaleH_std_ldlq128/size128_encdim2048_M256"
    # "scaleH_std_scale_cond(col)/size128_encdim2048_M256_ft_meta_test"
    # "scaleH_std_ldlq128_scale_cond(col)_ft_scaleh_rnrom/size128_encdim2048_M256"
    # "scaleH_std_ldlq128_scale_cond(col)_uniform10/size128_encdim2048_M256"
    # "scaleH_std_ldlq128_scale_cond(col)_uniform31.6_ft_meta/size128_encdim1024_M256"
    # "scaleH_std_ldlq128_scale_cond(col)_uniform31.6/smooth_0.5_size128_encdim2048_M256"
    # "scaleH_std_ldlq128_scale_cond(col)_uniform31.6/smooth_0.1_size128_encdim2048_M256"
    # "scaleH_std_ldlq128_scale_cond(col)_uniform31.6/smooth_0.8_size128_encdim2048_M256"
    # "scaleH_std_ldlq128_scale_cond(col)_ft/size128_encdim1024_M256"
    # "scaleH_std_ldlq128_scale_cond(col)_ltc/size16_encdim512_M24"
    # "scaleH_std_ldlq128_scale_cond(col)/size128_encdim1024_M256_lb1"
    # "scaleH_std_ldlq128_scale_cond(colrow)/size16_encdim512_M256"
    # "test2_ql_rnorm"
    # "ql_rnorm_cnorm_trained"
    # "ql_lnorm_trained"
    # "scale_cond2_rnorm_ldldq128/(trained_rnormed)"
    # "scale_cond2_rnorm_ldldq128/(trained_rnormed)uniform50"
    # "scale_cond(col_std)_whiten/size128_encdim1024_M256"
=======
    # "scaleH_rnorm_ldlq128_scale_cond(col_std)/(8B_trained)size128_encdim1024_M256"
    # "ql_scale_cond_scaleH_rnorm_ldlq(col_std)"
    "ql_gaussian_rnorm"
>>>>>>> clip test
)
##########################################################################
##                           MODEL CONFIGURATION                        ##
##########################################################################
model_names=(
<<<<<<< HEAD
=======
    # "meta-llama--Llama-2-7b-hf"
>>>>>>> clip test
    "meta-llama--Meta-Llama-3-8B"
    # "meta-llama--Llama-2-7b-hf"
    # "meta-llama--Llama-3.2-3B"
    # "meta-llama--Llama-2-13b-hf"
    # "meta-llama--Llama-2-70b-hf_"
)
hess_paths=(
<<<<<<< HEAD
=======
    # "../Wparam_dataset/quip_hess/Hessians-Llama-2-7b-6144"
>>>>>>> clip test
    "../Wparam_dataset/quip_hess/llama3_8b_6144"
    "../Wparam_dataset/quip_hess/Hessians-Llama-2-7b-6144"
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
<<<<<<< HEAD
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
=======
export CUDA_VISIBLE_DEVICES=0
>>>>>>> clip test
export WANDB_SILENT=true
export HF_HOME=/workspace/hf_cache/huggingface_nwc
# export TRANSFORMERS_CACHE=$HF_HOME
# export HF_DATASETS_CACHE=$HF_HOME/datasets
# export HF_METRICS_CACHE=$HF_HOME/metrics

# 모든 실험에 공통으로 적용될 Lambda 값
# lmbda_values=(30 50 100 300 1000 10000)
# lmbda_values=(10 20 30 50 100 300 1000)
<<<<<<< HEAD
lmbda_values=(100 50)
=======
lmbda_values=(30 300 10000)
>>>>>>> clip test
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

            echo "################## Running compression | lmbda=${lmbda} | Exp: ${exp_name} | Model: ${model_name} ##################"
            comp_model=$comp_model_base/lmbda${lmbda}_*/best_loss*.pth.tar
            mkdir -p $(dirname "$LOG/$SAVE_NAME.log")
            
            taskset -c 0-63 \
            python -m quantize_llama.quantize_finetune_llama --save_path $CKPT/$SAVE_NAME \
                --base_model $lm_model_path \
                --comp_model_path $comp_model \
                --in_hess_path $HESS \
                --devset_size 384 --ft_valid_size 128 --batch_size 8 \
                ${current_quantize_flags} \
                2>&1 | tee $LOG/$SAVE_NAME.log

                # --devset_size 48 --ft_valid_size 16 --batch_size 1 \
                # --devset_size 384 --ft_valid_size 128 --batch_size 8 \

            
            echo "################## Running hfize | lmbda=${lmbda} | Exp: ${exp_name} | Model: ${model_name} ##################"
            python -m quantize_llama.hfize_llama --quantized_path $CKPT/${SAVE_NAME} \
                    --hf_output_path $HF/${SAVE_NAME} \
                    2>&1 | tee -a $LOG/${SAVE_NAME}.log

                    # --skip_list 1_down \
            # SAVE_NAME=${model_name}/${exp_name}/lmbda${lmbda}_skip1down


            echo "################## Running PPL evaluation | lmbda=${lmbda} | Exp: ${exp_name} | Model: ${model_name} ##################"
            echo "Running evaluation for directory: $HF/$SAVE_NAME"
            python -m eval.eval_ppl_hf \
                --hf_path $HF/${SAVE_NAME} \
                --seqlen 2048 \
                --output_path ${RES}/${SAVE_NAME} \
                --datasets wikitext2,c4 \
                --no_use_cuda_graph 2>&1 | tee -a $LOG/$SAVE_NAME.log

                # --datasets wikitext2,c4 \

<<<<<<< HEAD
            echo "################## Running benchmark evaluation | lmbda=${lmbda} | Exp: ${exp_name} | Model: ${model_name} ##################"
            python -m eval.eval_zeroshot_hf \
                --tasks arc_challenge,arc_easy,piqa,winogrande,boolq,mmlu \
                --batch_size 10 \
                --hf_path $HF/$SAVE_NAME \
                --output_path $RES/${SAVE_NAME}_zeroshot_mmlu \
                2>&1 | tee -a $LOG/$SAVE_NAME.log

=======
            # echo "################## Running benchmark evaluation | lmbda=${lmbda} | Exp: ${exp_name} | Model: ${model_name} ##################"
>>>>>>> clip test
            # python -m eval.eval_zeroshot_hf \
            #     --tasks arc_challenge,arc_easy,piqa,winogrande,boolq,hellaswag,mmlu \
            #     --batch_size 16 \
            #     --hf_path $HF/$SAVE_NAME \
            #     --output_path $RES/${SAVE_NAME} \
            #     2>&1 | tee -a $LOG/$SAVE_NAME.log

                # --tasks arc_challenge,arc_easy,piqa,winogrande,hellaswag,mmlu \
                # --tasks arc_challenge,arc_easy,boolq,piqa,winogrande \


            if [ "$HF/$SAVE_NAME" != "$HF" ]; then
                echo "Cleaning up temporary files for $SAVE_NAME"
                rm -rf "$HF/$SAVE_NAME"
                rm -rf "$CKPT/$SAVE_NAME"
            fi
        done
    done
done


# export CUDA_VISIBLE_DEVICES=0,1,2,3
# python -m eval.eval_zeroshot_hf \
#     --tasks mmlu \
#     --batch_size 8  \
#     --hf_path /workspace/Weight_compression/Wparam_dataset/hf_model/meta-llama--Meta-Llama-3-8B \
#     --output_path /workspace/Weight_compression/Wparam_dataset/hf_model/meta-llama--Meta-Llama-3-8B_mmlu

# python -m eval.eval_zeroshot_hf \
#     --tasks mmlu \
#     --batch_size 8  \
#     --hf_path /workspace/Weight_compression/Wparam_dataset/hf_model/meta-llama--Llama-2-7b-hf \
#     --output_path /workspace/Weight_compression/Wparam_dataset/hf_model/meta-llama--Llama-2-7b-hf_mmlu