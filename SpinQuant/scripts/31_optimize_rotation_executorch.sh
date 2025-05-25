# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# nnodes determines the number of GPU nodes to utilize (usually 1 for an 8 GPU node)
# nproc_per_node indicates the number of GPUs per node to employ.
export CUDA_VISIBLE_DEVICES=0,1,2
for b in 5 6 7; do
    torchrun --nnodes=1 --nproc_per_node=3 optimize_rotation.py \
    --input_model ../Wparam_dataset/hf_model/meta-llama--Llama-2-7b-hf  \
    --output_rotation_path "../hf_model_comp/spinquant/output_rotation/7B_w${b}a16" \
    --output_dir "../hf_model_comp/spinquant/output/7B_w${b}a16" \
    --logging_dir "./logs/7B_w${b}a16" \
    --model_max_length 2048 \
    --fp16 False \
    --bf16 True \
    --log_on_each_node False \
    --per_device_train_batch_size 1 \
    --logging_steps 1 \
    --learning_rate 1.5 \
    --weight_decay 0. \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing True \
    --save_safetensors False \
    --max_steps 100 \
    --w_bits $b \
    --a_bits 16 \
    --k_bits 16 \
    --v_bits 16 \
    --w_clip \
    --a_asym \
    2>&1 | tee ./logs/7B_w${b}a16.log
done

    # --w_groupsize 256 \
