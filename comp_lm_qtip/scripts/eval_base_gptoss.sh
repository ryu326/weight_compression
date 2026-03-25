#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
export HF_HOME=/home/jgryu/.cache/huggingface

# lm_model_path=mistralai/Mixtral-8x7B-v0.1
# output_base=/home/jgryu/workspace/weight_compression/hf_model_comp_results/mistralai/base_model

lm_model_path=openai/gpt-oss-20b
output_root=/home/jgryu/workspace/weight_compression/hf_model_comp_results_v2/gpt-oss-20b

pids=()

run_eval() {
    local gpu_pair="$1"
    local output_suffix="$2"
    local gptoss_version="${3-}"
    local output_base="${output_root}/${output_suffix}"
    local output_path="${output_base}"
    local log_dir="${output_base}_logs"
    local ppl_log="${log_dir}/ppl.log"
    local zs_log="${log_dir}/zeroshot.log"
    local gptoss_args=()

    mkdir -p "${log_dir}"

    if [[ -n "${gptoss_version}" ]]; then
        gptoss_args=(--gptoss_replace_version "${gptoss_version}")
    fi

    (
        export CUDA_VISIBLE_DEVICES="${gpu_pair}"

        # python -m eval.eval_ppl_hf \
        #     --hf_path "${lm_model_path}" \
        #     --seqlen 2048 \
        #     --output_path "${output_path}" \
        #     --datasets wikitext2,c4 \
        #     --no_use_cuda_graph \
        #     "${gptoss_args[@]}" >"${ppl_log}" 2>&1

        python -m eval.eval_zeroshot_hf \
            --task aime25,aime24 \
            --hf_path "${lm_model_path}" \
            --output_path "${output_path}_aime" \
            --fewshot_as_multiturn \
            --apply_chat_template \
            "${gptoss_args[@]}" >"${zs_log}" 2>&1
            # --batch_size 2 \
            # --num_fewshot 5 \
        

            # --tasks arc_challenge,arc_easy,boolq,piqa,winogrande,hellaswag,mmlu \

        # lm_eval \
        #     --model hf \
        #     --model_args '{"pretrained":"openai/gpt-oss-20b","dtype":"auto","chat_template_args":{"reasoning_effort":"low"},"enable_thinking": true,"think_end_token":200008}' \
        #     --device "auto" \
        #     --tasks mmlu \
        #     --apply_chat_template \
        #     --fewshot_as_multiturn \
        #     --batch_size auto
        #     >"${zs_log}" 2>&1
    ) &

    pids+=($!)
}

run_eval "0" "base_model" "original"
run_eval "1" "base_model_replace_v1.1" "v1.1"
# run_eval "0" "base_model_replace_v2" "v2"
# run_eval "1" "base_model_replace_v3" "v3"


for pid in "${pids[@]}"; do
    wait "${pid}"
done
