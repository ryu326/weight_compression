# CUDA_VISIBLE_DEVICES=2 taskset -c 0-31 python compress_lm_nwc.py \
#     --model_path ../checkpoint/nwc_hp/block_seq_row_16/lmbda50_*/best_loss*.pth.tar \
#     --direction row \
#     --save_path nwc_hp \
#     --batch_size 8192 \
#     --dataset seq > ../logs/recon0.log 2>&1 &

# CUDA_VISIBLE_DEVICES=3 taskset -c 0-31 python compress_lm_nwc.py \
#     --model_path ../checkpoint/nwc/gaussian_seq_row_16/lmbda30000_*/best_loss*.pth.tar \
#     --direction row \
#     --save_path nwc \
#     --batch_size 8192 \
#     --dataset seq > ../logs/recon3.log 2>&1 &


# CUDA_VISIBLE_DEVICES=2 taskset -c 0-31 python compress_lm_nwc.py \
#     --model_path ../checkpoint/nwc/block_seq_ql_random_col_16/lmbda100_*/best_loss*.pth.tar \
#     --direction col \
#     --save_path nwc

# CUDA_VISIBLE_DEVICES=2 taskset -c 0-31 python compress_lm_nwc.py \
#     --model_path ../checkpoint/nwc/block_seq_ql_random_col_16/lmbda1000_*/best_loss*.pth.tar \
#     --direction col \
#     --save_path nwc

# python recon_lm_nwc_ql_seq.py --cuda 0 \
#     --model_path ../checkpoint/nwc_ql/block_seq_ql_random_col_16/lmbda50_*/best_loss*.pth.tar  \
#     --direction col \
#     > ../logs/recon0.log 2>&1 &

# python recon_lm_nwc_ql_seq.py --cuda 1 \
#     --model_path ../checkpoint/nwc_ql/block_seq_ql_random_col_16/lmbda200_*/best_loss*.pth.tar \
#     --direction col \
#     > ../logs/recon1.log 2>&1 &

# python recon_lm_nwc_ql_seq.py --cuda 2 \
#     --model_path ../checkpoint/nwc_ql/block_seq_ql_random_col_16/lmbda300_*/best_loss*.pth.tar  \
#     --direction col \
#     > ../logs/recon2.log 2>&1 &

# python recon_lm_nwc_ql_seq.py --cuda 3 \
#     --model_path ../checkpoint/nwc_ql/block_seq_ql_random_col_16/lmbda10000_*/best_loss*.pth.tar \
#     --direction col \
#     > ../logs/recon3.log 2>&1 &

# wait

# python recon_lm_nwc_ql_seq.py --cuda 0 \
#     --model_path ../checkpoint/nwc_ql/block_seq_ql_random_col_16/lmbda50_*/best_loss*.pth.tar  \
#     --direction row \
#     > ../logs/recon0.log 2>&1 &

# python recon_lm_nwc_ql_seq.py --cuda 1 \
#     --model_path ../checkpoint/nwc_ql/block_seq_ql_random_col_16/lmbda200_*/best_loss*.pth.tar \
#     --direction row \
#     > ../logs/recon1.log 2>&1 &

# python recon_lm_nwc_ql_seq.py --cuda 2 \
#     --model_path ../checkpoint/nwc_ql/block_seq_ql_random_col_16/lmbda300_*/best_loss*.pth.tar  \
#     --direction row \
#     > ../logs/recon2.log 2>&1 &

python recon_lm_nwc_ql_seq.py --cuda 3 \
    --model_path ../checkpoint/nwc_ql/block_seq_ql_random_col_16/lmbda10000_*/best_loss*.pth.tar \
    --direction col \
    > ../logs/recon3.log 2>&1 &

# CUDA_VISIBLE_DEVICES=2 taskset -c 0-31 python compress_lm_nwc.py \
#     --model_path ../checkpoint/nwc_hp/block_seq_row_16/lmbda100_*/best_loss*.pth.tar \
#     --direction row \
#     --save_path nwc_hp \
#     --batch_size 8192 \
#     --dataset seq > ../logs/recon0.log 2>&1 &

# CUDA_VISIBLE_DEVICES=3 taskset -c 0-31 python compress_lm_nwc.py \
#     --model_path ../checkpoint/nwc_hp/block_seq_row_16/lmbda200_*/best_loss*.pth.tar \
#     --direction row \
#     --save_path nwc_hp \
#     --batch_size 8192 \
#     --dataset seq > ../logs/recon1.log 2>&1 &

# wait

# CUDA_VISIBLE_DEVICES=0 taskset -c 0-31 python compress_lm_nwc.py \
#     --model_path ../checkpoint/nwc_hp/block_seq_row_16/lmbda1000_*/best_loss*.pth.tar \
#     --direction row \
#     --save_path nwc_hp \
#     --batch_size 8192 \
#     --dataset seq > ../logs/recon2.log 2>&1 &

# CUDA_VISIBLE_DEVICES=3 taskset -c 0-31 python compress_lm_nwc.py \
#     --model_path ../checkpoint/nwc_hp/block_seq_row_16/lmbda10000_*/best_loss*.pth.tar \
#     --direction row \
#     --save_path nwc_hp \
#     --batch_size 8192 \
#     --dataset seq > ../logs/recon3.log 2>&1 &

# CUDA_VISIBLE_DEVICES=0 taskset -c 0-31 python compress_lm_nwc.py \
#     --model_path ../checkpoint/nwc/block_col_128/lmbda50_*/best_loss*.pth.tar \
#     --direction col \
#     --save_path nwc \
#     --batch_size 16384 \

# CUDA_VISIBLE_DEVICES=0 taskset -c 0-31 python compress_lm_nwc.py \
#     --model_path ../checkpoint/nwc/block_col_128/lmbda100_*/best_loss*.pth.tar \
#     --direction row \
#     --save_path nwc \
#     --batch_size 16384 \


# CUDA_VISIBLE_DEVICES=0 taskset -c 0-31 python compress_lm_nwc.py \
#     --model_path ../checkpoint/nwc/block_col_128/lmbda100_*/best_loss*.pth.tar \
#     --direction col \
#     --save_path nwc \
#     --batch_size 16384 \

# CUDA_VISIBLE_DEVICES=0 taskset -c 0-31 python compress_lm_nwc.py \
#     --model_path ../checkpoint/nwc/block_seq_row_16/lmbda10000_*/best_loss*.pth.tar \
#     --direction row \
#     --save_path nwc \
#     --batch_size 16384 \
#     --dataset seq \
#     --no_save \
#     > ../logs/recon0.log 2>&1 &


# CUDA_VISIBLE_DEVICES=2 taskset -c 0-31 python compress_lm_nwc.py \
#     --model_path ../checkpoint/nwc/block_seq_row_16/lmbda30000_*/best_loss*.pth.tar \
#     --direction row \
#     --save_path nwc \
#     --batch_size 16384 \
#     --dataset seq \
#     --no_save \
#     > ../logs/recon1.log 2>&1 &

# wait

# CUDA_VISIBLE_DEVICES=0 taskset -c 0-31 python compress_lm_nwc.py \
#     --model_path ../checkpoint/nwc/block_seq_row_16/lmbda10000_*/best_loss*.pth.tar \
#     --direction col \
#     --save_path nwc \
#     --batch_size 16384 \
#     --dataset seq \
#     > ../logs/recon0.log 2>&1 &


# CUDA_VISIBLE_DEVICES=2 taskset -c 0-31 python compress_lm_nwc.py \
#     --model_path ../checkpoint/nwc/block_seq_row_16/lmbda30000_*/best_loss*.pth.tar \
#     --direction col \
#     --save_path nwc \
#     --batch_size 16384 \
#     --dataset seq \
#     > ../logs/recon1.log 2>&1 &

# wait


##########################################

# CUDA_VISIBLE_DEVICES=0 taskset -c 0-31 python compress_lm_nwc_seq.py \
#     --model_path ../checkpoint/nwc/block_seq_row_16/lmbda1000_*/best_loss*.pth.tar \
#     --direction col \
#     --save_path nwc \
#     --batch_size 16384

# wait


# CUDA_VISIBLE_DEVICES=0 taskset -c 0-31 python compress_lm_nwc.py \
#     --model_path ../checkpoint/nwc/gaussian_row_16/lmbda100_*/best_loss*.pth.tar \
#     --direction row \
#     --save_path nwc \
#     --batch_size 16384

# wait

# CUDA_VISIBLE_DEVICES=0 taskset -c 0-31 python compress_lm_nwc.py \
#     --model_path ../checkpoint/nwc/gaussian_row_16/lmbda1000_*/best_loss*.pth.tar \
#     --direction row \
#     --save_path nwc \
#     --batch_size 16384

# wait

# CUDA_VISIBLE_DEVICES=0 taskset -c 0-31 python compress_lm_nwc.py \
#     --model_path ../checkpoint/nwc/gaussian_row_16/lmbda100_*/best_loss*.pth.tar \
#     --direction col \
#     --save_path nwc \
#     --batch_size 16384

# wait

# CUDA_VISIBLE_DEVICES=0 taskset -c 0-31 python compress_lm_nwc.py \
#     --model_path ../checkpoint/nwc/gaussian_row_16/lmbda1000_*/best_loss*.pth.tar \
#     --direction col \
#     --save_path nwc \
#     --batch_size 16384

# wait


# python recon_lm_nwc.py --cuda 2 \
#     --model_path checkpoint/nwc/block_row_16/lmbda100_rdloss_encdim512_batch_size4096_total_iter1500000_lr0.0001_seed100/recent*.pth.tar \
#     --direction row \
#     > logs/recon0.log 2>&1 &

# python recon_lm_nwc.py --cuda 3 \
#     --model_path checkpoint/nwc/block_row_16/lmbda1000_rdloss_encdim512_batch_size4096_total_iter1500000_lr0.0001_seed100/recent*.pth.tar \
#     --direction row \
#     > logs/recon1.log 2>&1 &

# wait

# python recon_lm_nwc.py --cuda 2 \
#     --model_path checkpoint/nwc/block_col_128/lmbda50_rdloss_encdim512_batch_size4096_total_iter1500000_lr0.0001_seed100/best_loss*.pth.tar \
#     --direction row \
#     > logs/recon2.log 2>&1 &

# python recon_lm_nwc.py --cuda 3 \
#     --model_path checkpoint/nwc/block_col_128/lmbda300_rdloss_encdim512_batch_size4096_total_iter1500000_lr0.0001_seed100/best_loss*.pth.tar \
#     --direction row \
#     > logs/recon3.log 2>&1 &

# wait

# python recon_lm_nwc_ql.py --cuda 2 \
#     --model_path /home/jgryu/Weight_compression/VQVAE/checkpoint/nwc_ql/block_ql_col_16/lmbda100_rdloss_ql_encdim512_batch_size4096_total_iter1500000_lr0.0001_seed100/best_loss_model_loss_12.00164_bpp_6.0_MSE_0.02411_total_iter_450000.pth.tar \
#     --direction row

# wait

# python recon_lm_nwc_ql.py --cuda 2 \
#     --model_path /home/jgryu/Weight_compression/VQVAE/checkpoint/nwc_ql/block_ql_col_16/lmbda100_rdloss_ql_encdim512_batch_size4096_total_iter1500000_lr0.0001_seed100/best_loss_model_loss_12.00164_bpp_6.0_MSE_0.02411_total_iter_450000.pth.tar \
#     --direction col

# wait

# python recon_lm_nwc_ql.py --cuda 2 \
#     --model_path checkpoint/nwc_ql_random/block_ql_random_col_128/lmbda10000_rdloss_ql_encdim512_batch_size4096_total_iter1500000_lr0.0001_seed100/best_loss*.pth.tar \
#     --direction col \
#     > logs/recon2.log 2>&1 &

# python recon_lm_nwc_ql.py --cuda 3 \
#     --model_path checkpoint/nwc_ql_random/block_ql_random_col_128/lmbda200_rdloss_ql_encdim512_batch_size4096_total_iter1500000_lr0.0001_seed100/best_loss*.pth.tar \
#     --direction col \
#     > logs/recon3.log 2>&1 &

# wait
# 

# python recon_lm_vqvae.py --cuda 0 \
#     --model_path checkpoint/vqvae_qlike/block_col_16/bpp3.0*/best_loss*.pth.tar \
#     --direction col \
#     > logs/recon0.log 2>&1 &
    
# python recon_lm_vqvae.py --cuda 3 \
#     --model_path checkpoint/vqvae_qlike/block_col_16/bpp4.0*/best_loss*.pth.tar \
#     --direction col \
#     > logs/recon1.log 2>&1 &

# wait

# python recon_lm_vqvae.py --cuda 0 \
#     --model_path checkpoint/vqvae_qlike/block_col_16/bpp3.0*/best_loss*.pth.tar \
#     --direction row \
#     > logs/recon0.log 2>&1 &
    
# python recon_lm_vqvae.py --cuda 3 \
#     --model_path checkpoint/vqvae_qlike/block_col_16/bpp4.0*/best_loss*.pth.tar \
#     --direction row \
#     > logs/recon1.log 2>&1 &

# wait


# python recon_lm_vqvae_mag_vec.py --cuda 0 \
#     --model_path checkpoint/vqvae_mag_vec/row_16_calib/bpp3.0_size16_smse_ne256_de16_K8_P6_encdim512_batch_size256_total_iter1500000_lr0.0001_seed100/best_loss*.pth.tar \
#     > logs/vqvae_mag_bpp3.0.log 2>&1 &


# python recon_lm_vqvae_mag_vec.py --cuda 1 \
#     --model_path checkpoint/vqvae_mag_vec/row_16_calib/bpp4.0_size16_smse_ne256_de16_K8_P8_encdim512_batch_size256_total_iter1500000_lr0.0001_seed100/best_loss*.pth.tar \
#     > logs/vqvae_mag_bpp4.0.log 2>&1 &


# python recon_lm_vqvae_mag_vec.py --cuda 2 \
#     --model_path checkpoint/vqvae_mag_vec/row_16_calib/bpp6.0_size16_smse_ne256_de16_K8_P12_encdim512_batch_size256_total_iter1500000_lr0.0001_seed100/best_loss*.pth.tar \
#     > logs/vqvae_mag_bpp6.0.log 2>&1 &


# python recon_lm_vqvae_mag_vec.py --cuda 3 \
#     --model_path checkpoint/vqvae_mag_vec/row_16_calib/bpp8.0_size16_smse_ne256_de16_K8_P16_encdim512_batch_size256_total_iter1500000_lr0.0001_seed100/best_loss*.pth.tar \
#     > logs/vqvae_mag_bpp8.0.log 2>&1 &

# wait