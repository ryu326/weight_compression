#!/bin/bash
# ##########################################################################
# ##  Reasoning-eval pipeline for Qwen3 4B / 8B (mirrors comp_lm.sh)
# ##  Eval suite (ParoQuant Table 2): MMLU-Pro, GPQA-Diamond, AIME-24/25.
# ##  NOTE: lm_eval 0.4.4 ships mmlu_pro + gpqa_diamond_cot_zeroshot.
# ##        AIME-24 / AIME-25 are NOT in 0.4.4 — add custom task config or
# ##        upgrade lm_eval to enable them. (Marked TODO below.)
# ##########################################################################
PYTHON_BIN="/opt/conda/envs/qwen3/bin/python"   # transformers>=4.51 (Qwen3)
unset PYTHONPATH
export PATH="/opt/conda/envs/qwen3/bin:$PATH"
echo "Running with explicit python: $PYTHON_BIN"

##########################################################################
##                       EXPERIMENT CONFIGURATION                       ##
##########################################################################
comp_model_bases=(
    # '/home/jgryu/workspace/weight_compression/NWC/checkpoint2/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt/rdloss_ql_size16_encdim512_M16_Q4_nRB4R0_m0_batch_size2048_total_iter200000_lr0.0001_seed4.0/seed4'
    '../NWC/checkpoint2/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt/M16'
)
quantize_flags=(
    "--direction col --ql --Q 4 --row_normalize --ldlq --comp_batch_size 64 --ft_epochs 5"
)
experiment_names=(
    'ql_ldlq128_rnorm_ft'
)

##########################################################################
##                           MODEL CONFIGURATION                        ##
##########################################################################
model_names=(
    "Qwen--Qwen3-4B"
    "Qwen--Qwen3-8B"
)
lm_model_paths=(
    "Qwen/Qwen3-4B"
    "Qwen/Qwen3-8B"
)
hess_paths=(
    "../Wparam_dataset/quip_hess/qwen3_4b_1024"
    "../Wparam_dataset/quip_hess/qwen3_8b_1024"
)

##########################################################################
##                              SCRIPT SETUP                            ##
##########################################################################
CKPT="../hf_model_comp/comp_qtip/ckpt"
HF="../hf_model_comp/comp_qtip/hf"
LOG="./log"
RES="../hf_model_comp_results_v2"

mkdir -p "$CKPT" "$HF" "$LOG" "$RES"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HOME=/home/jgryu/.cache/huggingface

# Reasoning-eval is run separately by `eval_reasoning_paroquant.sh` using
# lighteval + vllm (paroquant style). Keep this here for reference only.
# REASONING_TASKS="mmlu_pro,gpqa_diamond_cot_zeroshot"

# Lambda sweep (same as comp_lm.sh)
lmbda_values=(100 300 1000 10000 50 30)

##########################################################################
##                        MAIN EXECUTION LOOP                           ##
##########################################################################
for j in "${!model_names[@]}"; do
    model_name="${model_names[$j]}"
    HESS="${hess_paths[$j]}"
    lm_model_path="${lm_model_paths[$j]}"

    echo "------------------------------------------------------------------------"
    echo "            MODEL: $model_name  (HF id: $lm_model_path)"
    echo "            HESS : $HESS"
    echo "------------------------------------------------------------------------"

    if [ ! -d "$HESS" ]; then
        echo "WARN: Hessian dir missing for $model_name -> $HESS. Run input_hessian_qwen3.sh first." >&2
    fi

    for i in "${!experiment_names[@]}"; do
        exp_name="${experiment_names[$i]}"
        comp_model_base="${comp_model_bases[$i]}"
        current_quantize_flags="${quantize_flags[$i]}"

        echo "========================================================================"
        echo "            STARTING EXPERIMENT SET: $exp_name"
        echo "========================================================================"

        for lmbda in "${lmbda_values[@]}"; do
            SAVE_NAME=${model_name}/${exp_name}/lmbda${lmbda}

            echo "################## Compression | lmbda=${lmbda} | Exp: ${exp_name} | Model: ${model_name} ##################"
            comp_model=$comp_model_base/lmbda${lmbda}_*/best_loss*.pth.tar
            mkdir -p "$(dirname "$LOG/$SAVE_NAME.log")"

            "$PYTHON_BIN" -m quantize_llama.quantize_finetune_llama \
                --save_path "$CKPT/$SAVE_NAME" \
                --base_model "$lm_model_path" \
                --comp_model_path $comp_model \
                --in_hess_path "$HESS" \
                --devset_size 384 --ft_valid_size 128 --batch_size 8 \
                ${current_quantize_flags} \
                2>&1 | tee "$LOG/$SAVE_NAME.log"

            echo "################## hfize | lmbda=${lmbda} | Exp: ${exp_name} | Model: ${model_name} ##################"
            "$PYTHON_BIN" -m quantize_llama.hfize_llama \
                --quantized_path "$CKPT/${SAVE_NAME}" \
                --hf_output_path "$HF/${SAVE_NAME}" \
                --base_model "$lm_model_path" \
                2>&1 | tee -a "$LOG/${SAVE_NAME}.log"

            echo "################## PPL eval | lmbda=${lmbda} | Exp: ${exp_name} | Model: ${model_name} ##################"
            "$PYTHON_BIN" -m eval.eval_ppl_hf \
                --hf_path "$HF/${SAVE_NAME}" \
                --seqlen 2048 \
                --output_path "${RES}/${SAVE_NAME}" \
                --datasets wikitext2,c4 \
                --no_use_cuda_graph 2>&1 | tee -a "$LOG/$SAVE_NAME.log"

            echo "################## Reasoning eval skipped (run eval_reasoning_paroquant.sh separately) ##################"
            # NOTE: HF directory intentionally retained for separate paroquant
            # lighteval+vllm reasoning evaluation. eval_reasoning_paroquant.sh
            # will iterate over the saved HF dirs.

            # ckpt is preserved unconditionally (user requirement). HF dir is
            # cleaned up by eval_reasoning_paroquant.sh after evaluation.
            echo "Keeping checkpoint files for $SAVE_NAME (ckpt preservation enforced)"
        done
    done
done
