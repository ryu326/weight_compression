#!/bin/bash
export CUDA_VISIBLE_DEVICES=3

# 사용할 모델들
MODELS=(
    # "meta-llama--Meta-Llama-3-8B"
    # "meta-llama--Llama-2-7b-hf"
    "meta-llama--Llama-2-13b-hf"
)

# 사용할 비트 수
BITS=(2 3 4 5 6 7 8)   # 필요시 (2 3 4 8) 등으로 확장 가능

# 로그 디렉토리
LOG="./logs_eval"
mkdir -p $LOG

for MODEL in "${MODELS[@]}"; do
    for b in "${BITS[@]}"; do

        pretrain_path=/workspace/Weight_compression/hf_model_comp/awq/${MODEL}/w${b}bit-g128-fake-quantized
        output_path=/workspace/Weight_compression/hf_model_comp_results/awq/${MODEL}/w${b}bit-g128-fake-quantized_mmlu_hs

        mkdir -p "$output_path"

        SAVE_NAME=$(echo "${MODEL}_w${b}" | tr '/' '_')
        LOG_FILE=$LOG/${SAVE_NAME}.log

        echo "################## Running benchmark evaluation MODEL=${MODEL}, bit=${b} ##################"
        python -m eval.eval_zeroshot_hf \
            --tasks mmlu,hellaswag \
            --batch_size 1 \
            --hf_path $pretrain_path \
            --output_path $output_path \
            2>&1 | tee -a $LOG_FILE

        # 필요시 ppl 평가도 추가 가능
        # python -m eval.eval_ppl_hf \
        #     --hf_path $pretrain_path \
        #     --seqlen 2048 \
        #     --output_path $output_path \
        #     --no_use_cuda_graph \
        #     2>&1 | tee -a $LOG_FILE

    done
done
