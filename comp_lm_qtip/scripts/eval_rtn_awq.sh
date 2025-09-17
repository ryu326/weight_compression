#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/workspace/hf_cache/huggingface_nwc

MODELS=(
    "meta-llama--Meta-Llama-3-8B"
    # "meta-llama--Llama-2-7b-hf"
    # "meta-llama--Llama-2-13b-hf"
)

BITS=(2 3) 

LOG="./logs_eval"
mkdir -p $LOG

for MODEL in "${MODELS[@]}"; do
    for b in "${BITS[@]}"; do

        pretrain_path=/workspace/Weight_compression/hf_model_comp/awq/${MODEL}/w${b}bit-g128-fake-quantized
        output_path=/workspace/Weight_compression/hf_model_comp_results/awq/${MODEL}/w${b}-g128-fake-quantized_common_mmlu

        mkdir -p "$output_path"

        SAVE_NAME=$(echo "${MODEL}_w${b}" | tr '/' '_')
        LOG_FILE=$LOG/${SAVE_NAME}.log

        echo "################## Running benchmark evaluation MODEL=${MODEL}, bit=${b} ##################"
        python -m eval.eval_zeroshot_hf \
            --tasks arc_challenge,arc_easy,piqa,winogrande,boolq,hellaswag,mmlu \
            --batch_size 1 \
            --hf_path $pretrain_path \
            --output_path $output_path \
            2>&1 | tee -a $LOG_FILE

        output_path=/workspace/Weight_compression/hf_model_comp_results/awq/${MODEL}/w${b}-g128-fake-quantized

        python -m eval.eval_ppl_hf \
            --hf_path $pretrain_path \
            --datasets wikitext2,c4 \
            --seqlen 2048 \
            --output_path $output_path \
            --no_use_cuda_graph \
            2>&1 | tee -a $LOG_FILE

    done
done
