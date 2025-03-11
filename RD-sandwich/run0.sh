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
        CUDA_VISIBLE_DEVICES=0 python rdub_mlp_wp_v2.py -V \
            --checkpoint_dir checkpoints_v3/llama3_8B_${model_type}_latent_dim_2000/llama3-8B_d${d}_b${b}_e${e}_lr${l}_normalize \
            --dataset /home/jgryu/Weight_compression/Wparam_dataset_v0/TFRecord/meta-llama--Meta-Llama-3-8B/${model_type}/d256/${model_type}_d256_train.tfrecord \
            --data_dim $d --latent_dim 2000 --prior_type maf \
            --maf_stacks 3 \
            --ar_hidden_units 50,50 --ar_activation softplus \
            --posterior_type gaussian --encoder_units ${u},${u},${u} --decoder_units ${u},${u},${u} \
            --encoder_activation softplus --decoder_activation softplus \
            --lambdas 1,13,146 \
            --use_wp_dataset --rpd --normalize \
            train \
            --epochs $e --lr $l --batchsize $b \
            --steps_per_epoch 500 --warmup 30 --startup 40 --patience 10 \
            --max_validation_steps 100 \
            --repeat
    done
done

# e=150
# batch=(1024 1024 1024 1024 1024 1024 1024) 
# dim=(16)
# unit=(2000 2000 2000 2000 2000 2000 2000)
# for i in "${!dim[@]}"; do
#     b="${batch[$i]}"
#     d="${dim[$i]}"
#     u="${unit[$i]}"
#     l=1e-5
#     # for lambda_value in 1 13 146; do
#     # for lambda_value in 2 24 270; do
#     # for lambda_value in 4 43 500; do
#     # for lambda_value in 7 79 3000; do
#     CUDA_VISIBLE_DEVICES=0 python rdub_mlp_wp_v2.py -V \
#         --dataset /home/jgryu/Weight_compression/Wparam_dataset/TFRecord/meta-llama--Meta-Llama-3-8B/per_tensor \
#         --data_dim $d --latent_dim $d --prior_type maf \
#         --maf_stacks 3 \
#         --ar_hidden_units 50,50 --ar_activation softplus \
#         --posterior_type gaussian --encoder_units ${u},${u},${u} --decoder_units ${u},${u},${u} \
#         --encoder_activation softplus --decoder_activation softplus \
#         --lambdas 1,13,146 \
#         --use_wp_dataset --rpd --normalize \
#         train \
#         --epochs $e --lr $l --batchsize $b \
#         --steps_per_epoch 500 --warmup 30 --startup 40 --patience 10 \
#         --max_validation_steps 100 \
#         --repeat
#     done
# done


# e=150
# batch=(1024 1024 1024 1024 1024 1024 1024) 
# dim=(16 32 64 128 256 1024 4096)
# unit=(2000 2000 2000 2000 2000 2000 2000)
# type=(layers-0_q_proj)

# for t in "${type[@]}"; do
#     for i in "${!dim[@]}"; do
#         b="${batch[$i]}"
#         d="${dim[$i]}"
#         u="${unit[$i]}"
#         # lr=(1e-4 5e-5 1e-5 5e-6)
#         lr=(1e-5)
#         for l in "${lr[@]}"; do
#             for lambda_value in 1 13 146; do
#             # for lambda_value in 2 24 270; do
#             # for lambda_value in 4 43 500; do
#             # for lambda_value in 7 79 3000; do
#                 CUDA_VISIBLE_DEVICES=0 python rdub_mlp_wp.py -V \
#                     --checkpoint_dir checkpoints_v3/llama3_8B_${t}_out/llama3-8B_d${d}_b${b}_e${e}_lr${l}_normalize \
#                     --dataset /home/jgryu/Weight_compression/Wparam_dataset/TFRecord/meta-llama--Meta-Llama-3-8B/${t}/d${d}/${t}_d${d}_train_out.tfrecord \
#                     --data_dim $d --latent_dim $d --prior_type maf \
#                     --maf_stacks 3 \
#                     --ar_hidden_units 50,50 --ar_activation softplus \
#                     --posterior_type gaussian --encoder_units ${u},${u},${u} --decoder_units ${u},${u},${u} \
#                     --encoder_activation softplus --decoder_activation softplus \
#                     --lambda ${lambda_value} \
#                     --use_wp_dataset --rpd --normalize \
#                     train \
#                     --epochs $e --lr $l --batchsize $b \
#                     --steps_per_epoch 500 --warmup 30 --startup 40 --patience 10 \
#                     --max_validation_steps 100 \
#                     --repeat
#             done
#         done
#     done
# done

### dim 다 돌리기
# e=150
# batch=(1024 1024 1024 1024)
# dim=(64 128 256 512 4096)
# unit=(2000 2000 2000 2000)
# for i in "${!dim[@]}"; do
#     b="${batch[$i]}"
#     d="${dim[$i]}"
#     u="${unit[$i]}"

#     lr=(1e-4 5e-5 1e-5 5e-6)
#     for i in "${!lr[@]}"; do
#         l="${lr[$i]}"
#         for lambda_value in 1 4 13 43 146 500; do
#         # for lambda_value in 2 7 24 79 270 3000; do
#             CUDA_VISIBLE_DEVICES=0 python rdub_mlp_wp.py -V --checkpoint_dir checkpoints_v2/llama3-8B_d${d}_b${b}_e${e}_lr${l}_normalize \
#                 --dataset /home/jgryu/Weight_compression/Wparam_dataset/TFRecord/meta-llama/Meta-Llama-3-8B/mlp_d${d}_train.tfrecord \
#                 --data_dim $d --latent_dim $d --prior_type maf \
#                 --maf_stacks 3 \
#                 --ar_hidden_units 50,50 --ar_activation softplus \
#                 --posterior_type gaussian --encoder_units ${u},${u},${u} --decoder_units ${u},${u},${u} \
#                 --encoder_activation softplus --decoder_activation softplus \
#                 --lambda ${lambda_value} \
#                 --use_wp_dataset --rpd --normalize \
#                 train \
#                 --epochs $e --lr $l --batchsize $b \
#                 --steps_per_epoch 500 --warmup 10 --startup 20 --patience 10 \
#                 --max_validation_steps 100 \
#                 --repeat
#         done
#     done
# done
##### d 16 잘 된 거 ####
# e=150
# batch=(2048 2048 2048 2048)
# dim=(16 16 16 16)
# lr=(1e-4 5e-5 1e-5 5e-6)
# unit=(500 500 500 500)
# for i in "${!dim[@]}"; do
#     b="${batch[$i]}"
#     d="${dim[$i]}"
#     l="${lr[$i]}"
#     u="${unit[$i]}"
#     for lambda_value in 1 5 10 50 100 500 1000; do
#     # for lambda_value in 3 8 30 80 800 3000; do
#         CUDA_VISIBLE_DEVICES=0 python rdub_mlp_wp.py -V --checkpoint_dir checkpoints_v2/llama3-8B_d${d}_b${b}_e${e}_lr${l}_normalize \
#             --dataset /home/jgryu/Weight_compression/Wparam_dataset/TFRecord/meta-llama/Meta-Llama-3-8B/mlp_d16_train.tfrecord \
#             --data_dim $d --latent_dim $d --prior_type maf \
#             --maf_stacks 3 \
#             --ar_hidden_units 50,50 --ar_activation softplus \
#             --posterior_type gaussian --encoder_units ${u},${u},${u} --decoder_units ${u},${u},${u} \
#             --encoder_activation softplus --decoder_activation softplus \
#             --lambda ${lambda_value} \
#             --use_wp_dataset --rpd --normalize \
#             train \
#             --epochs $e --lr $l --batchsize $b \
#             --steps_per_epoch 500 --warmup 10 --startup 20 --patience 10 \
#             --max_validation_steps 100 \
#             --repeat
#     done
# done
##### d 1024 잘 된 거 ####
# e=150
# batch=(1024 1024 1024 1024)
# dim=(1024 1024 1024 1024)
# lr=(1e-4 5e-5 1e-5 5e-6)
# unit=(2000 2000 2000 2000)
# for i in "${!dim[@]}"; do
#     b="${batch[$i]}"
#     d="${dim[$i]}"
#     l="${lr[$i]}"
#     u="${unit[$i]}"
#     for lambda_value in 1 5 10 50 100 500 1000; do
#     # for lambda_value in 3 8 30 80 800 3000; do
#         CUDA_VISIBLE_DEVICES=0 python rdub_mlp_wp.py -V --checkpoint_dir checkpoints_v2/llama3-8B_d${d}_b${b}_e${e}_lr${l}_normalize_2 \
#             --dataset /home/jgryu/Weight_compression/Wparam_dataset/TFRecord/meta-llama/Meta-Llama-3-8B/mlp_d1024_train.tfrecord \
#             --data_dim $d --latent_dim $d --prior_type maf \
#             --maf_stacks 3 \
#             --ar_hidden_units 50,50 --ar_activation softplus \
#             --posterior_type gaussian --encoder_units ${u},${u},${u} --decoder_units ${u},${u},${u} \
#             --encoder_activation softplus --decoder_activation softplus \
#             --lambda ${lambda_value} \
#             --use_wp_dataset --rpd --normalize \
#             train \
#             --epochs $e --lr $l --batchsize $b \
#             --steps_per_epoch 1000 --warmup 10 --startup 20 --patience 10 \
#             --max_validation_steps 100 \
#             --repeat
#     done
# done

# n=1000
# ## standard normal distribution 에 대해서
# for scale in 0.001 0.1 10; do
#     for latent_dim in 1000; do
#         for lambda in 0.3 1 3 10 30 100 300; do
#             CUDA_VISIBLE_DEVICES=0 python rdub_mlp.py -V \
#                 --dataset gaussian \
#                 --gparams_path data/gaussian_params-dim=$n.npz \
#                 --data_dim $n \
#                 --latent_dim $latent_dim \
#                 --checkpoint_dir checkpoints_v2/gaussian_default_scale${scale} \
#                 --prior_type gmm_1 \
#                 --posterior_type gaussian \
#                 --encoder_activation none \
#                 --decoder_activation leaky_relu \
#                 --decoder_units $n \
#                 --lambda $lambda \
#                 --gaussian_scale $scale \
#                 --rpd train \
#                 --epochs 80 \
#                 --steps_per_epoch 1000 \
#                 --lr 5e-4 \
#                 --batchsize 64 \
#                 --max_validation_steps 1
#         done
#     done
# done

# e=600
# batch=(1024)
# dim=(1024)
# lr=(1e-4)
# unit=(2000)
# for i in "${!dim[@]}"; do
#     b="${batch[$i]}"
#     d="${dim[$i]}"
#     l="${lr[$i]}"
#     u="${unit[$i]}"
#     for lambda_value in 1e-12 1e-10 1e-8 1e-6 0.01 1e-11 1e-9 1e-7 1e-5; do
#     # for lambda_value in 3e-10 3e-8 3e-6 3e-4 0.03 3e-11 3e-9 3e-7 3e-5; do
#         CUDA_VISIBLE_DEVICES=0 python rdub_mlp_wp.py -V --checkpoint_dir checkpoints_v2/llama3-8B_d${d}_b${b}_e${e}  \
#             --dataset /home/jgryu/Weight_compression/Wparam_dataset/TFRecord/meta-llama/Meta-Llama-3-8B/mlp_d1024_train.tfrecord \
#             --data_dim $d --latent_dim $d --prior_type maf \
#             --maf_stacks 3 \
#             --ar_hidden_units 50,50 --ar_activation softplus \
#             --posterior_type gaussian --encoder_units ${u},${u},${u} --decoder_units ${u},${u},${u} \
#             --encoder_activation softplus --decoder_activation softplus \
#             --lambda ${lambda_value} \
#             --use_wp_dataset --rpd \
#             train \
#             --epochs $e --lr $l --batchsize $b \
#             --steps_per_epoch 1000 --warmup 20 --startup 400 --patience 20 \
#             --max_validation_steps 100 \
#             --repeat
#     done
# done


# e=600
# # batch=(1024 512 1024)
# # dim=(2048 4096 1024)
# batch=(1024 1024 1024 1024)
# dim=(128 256 512 1024)
# lr=(1e-4 1e-4 1e-4 1e-4)
# unit=(500 1000 2000 4000)
# for i in "${!dim[@]}"; do
#     b="${batch[$i]}"
#     d="${dim[$i]}"
#     l="${lr[$i]}"
#     u="${unit[$i]}"
#     # for lambda_value in 1e-12 1e-10 1e-8 1e-6 1e-4 0.01; do
#     for lambda_value in 3e-12 3e-10 3e-8 3e-6 3e-4 0.03; do
#     # for lambda_value in 1e-11 1e-9 1e-7 1e-5 1e-3 0.1; do
#     # for lambda_value in 3e-11 3e-9 3e-7 3e-5 3e-3 0.3; do
#         CUDA_VISIBLE_DEVICES=1 python rdub_mlp.py -V --checkpoint_dir checkpoints/v2_llama_attn_d=${d}_b=${b}_e=${e}  \
#             --dataset /workspace/jgryu/Weight_compression/Wparam_dataset/tensor_slices/train/meta-llama/Llama-2-7b-hf \
#             --model_filter self_attn llama \
#             --data_dim $d --latent_dim $d --prior_type maf \
#             --maf_stacks 3 \
#             --ar_hidden_units 50,50 --ar_activation softplus \
#             --posterior_type gaussian --encoder_units ${u},${u} --decoder_units ${u},${u} \
#             --encoder_activation softplus --decoder_activation softplus \
#             --nats --lambda ${lambda_value} \
#             --use_wp_dataset --attn_norm  --rpd \
#             train \
#             --epochs $e --lr $l --batchsize $b \
#             --steps_per_epoch 1000 --warmup 20 --startup 400 --patience 20 \
#             --max_validation_steps 100 --num_data 10000 \
#             --repeat
#     done
# done

            # --dataset /workspace/jgryu/Weight_compression/Wparam_dataset/npy/test.npy \
# '/home/jgryu/Weight_compression/Wparam_dataset/tensor_slices/train/meta-llama/CodeLlama-7b-hf/model-layers-0-self_attn-k_proj-weight'
# /workspace/jgryu/Weight_compression/Wparam_dataset/tensor_slices/train/meta-llama/Llama-2-7b-hf \
# e=100
# # batch=(1024 512 1024)
# # dim=(2048 4096 1024)
# batch=(1024)
# dim=(1024)
# lr=(5e-4)
# for i in "${!dim[@]}"; do
#     b="${batch[$i]}"
#     d="${dim[$i]}"
#     l="${lr[$i]}"
#     for lambda_value in 1e-12 1e-10 1e-8 1e-6 1e-4 0.01; do
#     # for lambda_value in 3e-12 3e-10 3e-8 3e-6 3e-4 0.03; do
#     # for lambda_value in 1e-11 1e-9 1e-7 1e-5 1e-3 0.1; do
#     # for lambda_value in 3e-11 3e-9 3e-7 3e-5 3e-3 0.3; do
#         CUDA_VISIBLE_DEVICES=0 python rdub_mlp.py -V --checkpoint_dir checkpoints/llama-7b_mlp_d=${d}_b=${b}_e=${e}  \
#             --dataset /home/jgryu/Weight_compression/Wparam_dataset/tensor_slices/train/meta-llama/Llama-2-7b-chat-hf \
#             --data_dim $d --latent_dim $d --prior_type maf \
#             --maf_stacks 3 \
#             --ar_hidden_units 50,50 --ar_activation softplus \
#             --posterior_type gaussian --encoder_units 2000,2000 --decoder_units 2000,2000 \
#             --encoder_activation softplus --decoder_activation softplus \
#             --nats --lambda ${lambda_value} \
#             --use_wp_dataset --model_filter mlp \
#             train \
#             --epochs $e --lr $l --batchsize $b \
#             --max_validation_steps 100 \
#             --warmup 20
#     done
# done




        # CUDA_VISIBLE_DEVICES=0 python rdub_mlp.py -V --checkpoint_dir checkpoints/llama-7b_attn_normalized_2_d=${d}_b=${b}_e=${e}  \
        #     --model_filter llama 7b self_attn \
        #     --dataset /home/jgryu/Weight_compression/Wparam_dataset/Wparam_npy/llama_7b_self_attn_d=${d}_val.npy \
        #     --data_dim $d --latent_dim $d --prior_type gmm_1 \
        #     --posterior_type gaussian --encoder_units 2000,2000 --decoder_units 2000,2000 \
        #     --encoder_activation softplus --decoder_activation softplus \
        #     --nats --lambda ${lambda_value} \
        #     --use_wp_dataset \
        #     train \
        #     --epochs $e --lr $l --batchsize $b \
        #     --max_validation_steps 100

# sub_lmbdas = [1e-12, 1e-11, 1e-10, 1e-09, 1e-08, 1e-07, 1e-06, 1e-05, 0.0001, 0.001, 0.01]

# for n in 2048 1024 512 128
    # for lambda_value in 1e-20 1e-18 1e-16 1e-14 1e-12 1e-10 1e-8 1e-6
    # for lambda_value in 3e-20 3e-18 3e-16 3e-14 3e-12 3e-10 3e-8 3e-6
    # for lambda_value in 1e-19 1e-17 1e-15 1e-13 1e-11 1e-9 1e-7 1e-5
    # for lambda_value in 3e-19 3e-17 3e-15 3e-13 3e-11 3e-9 3e-7 3e-5

# CUDA_VISIBLE_DEVICES=0 parallel -j1 python rdub_mlp.py -V --checkpoint_dir checkpoints/llama_7b_attn_d=${n}_b=${b}  \
#     --dataset /home/jgryu/Weight_compression/model_parm_dataset/llama_7b_attn_d=${n}_train.npy \
#     --data_dim $n --latent_dim $n --prior_type gmm_1 \
#     --posterior_type gaussian --encoder_units 500,500 --decoder_units 500,500 \
#     --encoder_activation softplus --decoder_activation softplus \
#     --nats --lambda {} --test \
#     train \
#     --epochs $e --lr 5e-4 --batchsize $b \
#     --max_validation_steps 100 ::: 0.03 0.1 0.3 1

# CUDA_VISIBLE_DEVICES=1 parallel -j1 python rdub_mlp.py -V --checkpoint_dir checkpoints/gemma_2b_attn_d=${n}  \
#     --dataset /home/jgryu/Weight_compression/model_parm_dataset/gemma_2b_attn_d=${n}_train.npy \
#     --data_dim $n --latent_dim $n --prior_type gmm_1 \
#     --posterior_type gaussian --encoder_units 500,500 --decoder_units 500,500 \
#     --encoder_activation softplus --decoder_activation softplus \
#     --nats --lambda {} train \
#     --epochs 100 --lr 5e-4 --batchsize 512 \
#     --max_validation_steps 100 ::: 3 10

# CUDA_VISIBLE_DEVICES=2 parallel -j1 python rdub_mlp.py -V --checkpoint_dir checkpoints/gemma_2b_attn_d=${n}  \
#     --dataset /home/jgryu/Weight_compression/model_parm_dataset/gemma_2b_attn_d=${n}_train.npy \
#     --data_dim $n --latent_dim $n --prior_type gmm_1 \
#     --posterior_type gaussian --encoder_units 500,500 --decoder_units 500,500 \
#     --encoder_activation softplus --decoder_activation softplus \
#     --nats --lambda {} train \
#     --epochs 100 --lr 5e-4 --batchsize 256 \
#     --max_validation_steps 100 ::: 30 100

# CUDA_VISIBLE_DEVICES=3 parallel -j1 python rdub_mlp.py -V --checkpoint_dir checkpoints/gemma_2b_attn_d=${n}  \
#     --dataset /home/jgryu/Weight_compression/model_parm_dataset/gemma_2b_attn_d=${n}_train.npy \
#     --data_dim $n --latent_dim $n --prior_type gmm_1 \
#     --posterior_type gaussian --encoder_units 500,500 --decoder_units 500,500 \
#     --encoder_activation softplus --decoder_activation softplus \
#     --nats --lambda {} train \
#     --epochs 100 --lr 5e-4 --batchsize 512 \
#     --max_validation_steps 100 ::: 300 1000


# CUDA_VISIBLE_DEVICES=0,1 parallel python rdub_mlp.py -V --checkpoint_dir checkpoints/gemma_2b_attn_d=${n}  \
#     --dataset /home/jgryu/Weight_compression/model_parm_dataset/gemma_2b_attn_d=${n}_train.npy \
#     --data_dim $n --latent_dim $n --prior_type gmm_1 \
#     --posterior_type gaussian --encoder_units 500,500 --decoder_units 500,500 \
#     --encoder_activation softplus --decoder_activation softplus \
#     --nats --lambda {} train \
#     --epochs 100 --lr 5e-4 --batchsize 1 \
#     --max_validation_steps 100 ::: 0.3 1 3 10 30 100 300 

# n=128
# CUDA_VISIBLE_DEVICES=0,1,2,3 parallel python rdub_mlp.py -V --checkpoint_dir checkpoints/gemma_2b_attn_d=${n}  \
#     --dataset /home/jgryu/Weight_compression/model_parm_dataset/gemma_2b_attn_d=${n}_train.npy \
#     --data_dim $n --latent_dim $n --prior_type maf --maf_stacks 3 \
#     --posterior_type gaussian --encoder_units 500,500 --decoder_units 500,500 \
#     --encoder_activation softplus --decoder_activation softplus \
#     --ar_hidden_units 50,50 --ar_activation softplus --nats --lambda {} train \
#     --epochs 100 --lr 5e-4 --batchsize 2 \
#     --max_validation_steps 100 ::: 0.3 1 3 10 30 100 300