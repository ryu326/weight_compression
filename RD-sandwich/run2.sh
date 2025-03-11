#!/bin/bash
# 1,13,146
# 2,24,270
# 4,43,500
# 7,79,3000
e=150
batch=(1024 1024 1024 1024 1024 1024 1024)
dim=(4096 2048 1024 128)
unit=(2000 2000 2000 2000 2000 2000 2000)

for model_type in attn mlp; do
    for i in "${!dim[@]}"; do
        b="${batch[$i]}"
        d="${dim[$i]}"
        u="${unit[$i]}"
        l=1e-5
        CUDA_VISIBLE_DEVICES=2 python rdub_mlp_wp_v2.py -V \
            --checkpoint_dir checkpoints_v3/llama3_8B_${model_type}_latent_dim_2000/llama3-8B_d${d}_b${b}_e${e}_lr${l}_normalize \
            --dataset /home/jgryu/Weight_compression/Wparam_dataset_v0/TFRecord/meta-llama--Meta-Llama-3-8B/${model_type}/d256/${model_type}_d256_train.tfrecord \
            --data_dim $d --latent_dim 2000 --prior_type maf \
            --maf_stacks 3 \
            --ar_hidden_units 50,50 --ar_activation softplus \
            --posterior_type gaussian --encoder_units ${u},${u},${u} --decoder_units ${u},${u},${u} \
            --encoder_activation softplus --decoder_activation softplus \
            --lambdas 4,43,500 \
            --use_wp_dataset --rpd --normalize \
            train \
            --epochs $e --lr $l --batchsize $b \
            --steps_per_epoch 500 --warmup 30 --startup 40 --patience 10 \
            --max_validation_steps 100 \
            --repeat
    done
done