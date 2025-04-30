# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# nnodes determines the number of GPU nodes to utilize (usually 1 for an 8 GPU node)
# nproc_per_node indicates the number of GPUs per node to employ.
export CUDA_VISIBLE_DEVICES=0
for b in 5 6 7 8 9; do
    torchrun --nnodes=1 --nproc_per_node=1 ptq.py \
    --input_model ../Wparam_dataset/hf_model/meta-llama--Meta-Llama-3-8B \
    --do_train False \
    --do_eval True \
    --per_device_eval_batch_size 1 \
    --model_max_length 2048 \
    --fp16 False \
    --bf16 True \
    --save_safetensors False \
    --w_bits $b \
    --a_bits 16 \
    --w_clip \
    --a_asym \
    --rotate \
    --optimized_rotation_path ./output_rotation/8B_w${b}a16/R.bin \
    --save_qmodel_path ./output_qmodel/8B_w${b}a16.pth \
    --export_to_et \
    2>&1 | tee ./logs/ptq_8B_w${b}a16.log
done

    # --w_groupsize 32 \