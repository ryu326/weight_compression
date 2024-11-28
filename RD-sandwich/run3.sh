e=150
batch=(1024 1024 1024 1024 1024 1024 1024) 
dim=(16)
unit=(2000 2000 2000 2000 2000 2000 2000)
for i in "${!dim[@]}"; do
    b="${batch[$i]}"
    d="${dim[$i]}"
    u="${unit[$i]}"
    l=1e-5
    # for lambda_value in 1 13 146; do
    # for lambda_value in 2 24 270; do
    # for lambda_value in 4 43 500; do
    # for lambda_value in 7 79 3000; do
    CUDA_VISIBLE_DEVICES=3 python rdub_mlp_wp_v2.py -V \
        --dataset /home/jgryu/Weight_compression/Wparam_dataset/TFRecord/meta-llama--Meta-Llama-3-8B/per_tensor \
        --data_dim $d --latent_dim $d --prior_type maf \
        --maf_stacks 3 \
        --ar_hidden_units 50,50 --ar_activation softplus \
        --posterior_type gaussian --encoder_units ${u},${u},${u} --decoder_units ${u},${u},${u} \
        --encoder_activation softplus --decoder_activation softplus \
        --lambdas 7,79,3000 \
        --use_wp_dataset --rpd --normalize \
        train \
        --epochs $e --lr $l --batchsize $b \
        --steps_per_epoch 500 --warmup 30 --startup 40 --patience 10 \
        --max_validation_steps 100 \
        --repeat
    done
done