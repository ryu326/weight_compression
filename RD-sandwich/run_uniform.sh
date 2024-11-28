
n=1000
## standard normal distribution 에 대해서
for latent_dim in 1000; do
    for lambda in 0.3 1; do
        CUDA_VISIBLE_DEVICES=0 python rdub_mlp.py -V \
            --dataset uniform \
            --data_dim $n \
            --latent_dim $latent_dim \
            --checkpoint_dir checkpoints_v2/uniform \
            --prior_type gmm_1 \
            --posterior_type gaussian \
            --encoder_activation none \
            --decoder_activation leaky_relu \
            --decoder_units $n \
            --lambda $lambda \
            --rpd train \
            --epochs 80 \
            --steps_per_epoch 1000 \
            --lr 5e-4 \
            --batchsize 64 \
            --max_validation_steps 1
    done
done
