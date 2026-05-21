#!/bin/bash
# Compute input Hessians for Qwen3 4B and 8B.
# Effective samples = devset_size x nproc_per_node = 512 x 2 = 1024 per rank-aggregated layer.
# Qwen3 uses Llama-style attention/MLP module names, so input_hessian_llama.py works as-is.

PYTHON_BIN="/opt/conda/envs/qwen3/bin/python"   # transformers>=4.51 (Qwen3 support)
TORCHRUN_BIN="/opt/conda/envs/qwen3/bin/torchrun"

export HF_HOME=/home/jgryu/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0,1

model_ids=(
    "Qwen/Qwen3-4B"
    "Qwen/Qwen3-8B"
)
save_tags=(
    "qwen3_4b_1024"
    "qwen3_8b_1024"
)
batch_sizes=(
    16
    8
)

for i in "${!model_ids[@]}"; do
    model_id="${model_ids[$i]}"
    save_tag="${save_tags[$i]}"
    bs="${batch_sizes[$i]}"
    save_path="../Wparam_dataset/quip_hess/${save_tag}"

    echo "=========================================================================="
    echo " Hessian: ${model_id} -> ${save_path} (batch_size=${bs}, devset_size=512)"
    echo "=========================================================================="

    "${TORCHRUN_BIN}" --nproc_per_node=2 -m quantize_llama.input_hessian_llama \
        --batch_size "${bs}" --devset_size 512 \
        --large_batch_size 512 \
        --ctx_size 4096 \
        --base_model "${model_id}" \
        --save_path "${save_path}"
done
