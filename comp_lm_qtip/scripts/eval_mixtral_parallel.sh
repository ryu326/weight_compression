#!/bin/bash

# ---------------- CONFIGURATION ---------------- #
comp_model_base="../NWC/checkpoint/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt/M16"
quantize_flags="--direction col --ql --Q 4 --col_normalize --ldlq --comp_batch_size 128 --ft_epochs 5"
exp_name="ql_ldlq128_rnorm_ft"
model_name="mistralai/Mixtral-8x7B-v0.1"
hess_path="/home/jgryu/workspace/weight_compression/Wparam_dataset/quip_hess/Mixtral-8x7B-v0.1_256"
lm_model_path="$model_name"

CKPT="../hf_model_comp/comp_qtip/ckpt"
HF="../hf_model_comp/comp_qtip/hf"
LOG="./log"
RES="../hf_model_comp_results"

mkdir -p $CKPT $HF $LOG $RES

export HF_HOME=/home/jgryu/.cache/huggingface

# 매핑: lmbda 값과 사용할 GPU 번호
lmbda_values=(30 50 100 300 1000 10000)
gpu_list=(2 3 4 5 6 7)

# ---------------- EXECUTION ---------------- #

for i in "${!lmbda_values[@]}"; do
    lmbda=${lmbda_values[$i]}
    gpu_id=${gpu_list[$i]}

    (
        export CUDA_VISIBLE_DEVICES=$gpu_id
        
        SAVE_NAME="${model_name}/${exp_name}/lmbda${lmbda}"
        comp_model=$(ls $comp_model_base/lmbda${lmbda}_*/best_loss*.pth.tar | head -n 1)
        
        mkdir -p "$(dirname "$LOG/$SAVE_NAME.log")"

        echo "[Start] Lambda: $lmbda on GPU: $gpu_id | Save: $SAVE_NAME"

        # 1. Quantize & Finetune
        # python -m quantize_llama.quantize_finetune_moe \
        #     --save_path $CKPT/$SAVE_NAME \
        #     --base_model $lm_model_path \
        #     --comp_model_path "$comp_model" \
        #     --in_hess_path $hess_path \
        #     --devset_size 384 --ft_valid_size 128 --batch_size 8 \
        #     $quantize_flags \
        #     > "$LOG/$SAVE_NAME.log" 2>&1

        # 2. HF Conversion
        python -m quantize_llama.hfize_moe \
            --quantized_path $CKPT/${SAVE_NAME} \
            --hf_output_path $HF/${SAVE_NAME} \
            >> "$LOG/$SAVE_NAME.log" 2>&1

        # 3. Evaluation (Benchmark)
        python -m eval.eval_zeroshot_hf \
            --tasks arc_challenge,arc_easy,piqa,winogrande,boolq,hellaswag,mmlu \
            --batch_size 10 \
            --hf_path $HF/$SAVE_NAME \
            --output_path $RES/${SAVE_NAME}_common_mmlu \
            >> "$LOG/$SAVE_NAME.log" 2>&1

        # 4. Cleanup
        if [ -d "$HF/$SAVE_NAME" ]; then
            rm -rf "$HF/$SAVE_NAME"
        fi

        echo "[Done] Lambda: $lmbda on GPU: $gpu_id"
    ) & 
done

wait
echo "All jobs completed."