# train T2L via reconstruction training
export CUDA_VISIBLE_DEVICES=1
env DUMP_LORAS=1 \
WANDB_MODE=disabled uv run python scripts/train_nwc_recon.py configs/hyper_lora_decontam_lol_tasks.yaml \
    --model_dir=mistralai/Mistral-7B-Instruct-v0.2 \
    --emb_model=Alibaba-NLP/gte-large-en-v1.5 \
    --warmup_frac=0.01 --lr=1e-3 --epochs=10000 \
    --n_train_ds=479 --exp_setup=hyper_lora --encoder_type=linear \
    --pred_z_score=True --n_descs_per_ds=128 --n_embs_per_sampled_task=1 \
    --n_tasks_per_batch=4 --factorized=True --delta_w_scaling=10000 --shared_AB_head=True \
    --rdlmbda=10000 --compnet_latent_width=1 --compnet_latent_size=2048 d_enc_in=4096 d_dec_out=4096 \
    --val_freq=10000 --cond_dim=64


nohup env WANDB_MODE=online CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_nwc_recon.py configs/hyper_lora_decontam_lol_tasks.yaml \
    --model_dir=mistralai/Mistral-7B-Instruct-v0.2 \
    --emb_model=Alibaba-NLP/gte-large-en-v1.5 \
    --warmup_frac=0.1 --lr=1e-3 --epochs=10000 \
    --n_train_ds=479 --exp_setup=hyper_lora --encoder_type=linear \
    --pred_z_score=True --n_descs_per_ds=128 --n_embs_per_sampled_task=1 \
    --n_tasks_per_batch=4 --factorized=True --delta_w_scaling=10000 --shared_AB_head=True \
    --rdlmbda=10000 --compnet_latent_width=1 --compnet_latent_size=2048 d_enc_in=4096 d_dec_out=4096 \
    --val_freq=10000 --cond_dim=64 > ./logs/recon_train0.log 2>&1 &

nohup env WANDB_MODE=online CUDA_VISIBLE_DEVICES=1 uv run python scripts/train_nwc_recon.py configs/hyper_lora_decontam_lol_tasks.yaml \
    --model_dir=mistralai/Mistral-7B-Instruct-v0.2 \
    --emb_model=Alibaba-NLP/gte-large-en-v1.5 \
    --warmup_frac=0.1 --lr=1e-3 --epochs=10000 \
    --n_train_ds=479 --exp_setup=hyper_lora --encoder_type=linear \
    --pred_z_score=True --n_descs_per_ds=128 --n_embs_per_sampled_task=1 \
    --n_tasks_per_batch=4 --factorized=True --delta_w_scaling=10000 --shared_AB_head=True \
    --rdlmbda=10000 --compnet_latent_width=1 --compnet_latent_size=2048 d_enc_in=4096 d_dec_out=4096 \
    --val_freq=10000 --cond_dim=64 --use_ortho_whiten=False > ./logs/recon_train1.log 2>&1 &

nohup env WANDB_MODE=online CUDA_VISIBLE_DEVICES=2 uv run python scripts/train_nwc_recon.py configs/hyper_lora_decontam_lol_tasks.yaml \
    --model_dir=mistralai/Mistral-7B-Instruct-v0.2 \
    --emb_model=Alibaba-NLP/gte-large-en-v1.5 \
    --warmup_frac=0.1 --lr=1e-3 --epochs=10000 \
    --n_train_ds=479 --exp_setup=hyper_lora --encoder_type=linear \
    --pred_z_score=True --n_descs_per_ds=128 --n_embs_per_sampled_task=1 \
    --n_tasks_per_batch=4 --factorized=True --delta_w_scaling=10000 --shared_AB_head=True \
    --rdlmbda=10000 --compnet_latent_width=1 --compnet_latent_size=2048 d_enc_in=2048 d_dec_out=2048 \
    --val_freq=10000 --cond_dim=64 --use_ortho_whiten=True > ./logs/recon_train2.log 2>&1 &



nohup env WANDB_MODE=online CUDA_VISIBLE_DEVICES=3 uv run python scripts/train_nwc_recon.py configs/hyper_lora_decontam_lol_tasks.yaml \
    --model_dir=mistralai/Mistral-7B-Instruct-v0.2 \
    --emb_model=Alibaba-NLP/gte-large-en-v1.5 \
    --warmup_frac=0.1 --lr=1e-3 --epochs=1000 \
    --n_train_ds=479 --exp_setup=hyper_lora --encoder_type=linear \
    --pred_z_score=True --n_descs_per_ds=128 --n_embs_per_sampled_task=1 \
    --n_tasks_per_batch=4 --factorized=False --delta_w_scaling=10000 --shared_AB_head=True \
    --rdlmbda=10000 --val_freq=10000 \
    --compnet_v=3 \
    --compnet_latent_width=1 --compnet_latent_size=2048 d_enc_in=2048 d_dec_out=2048 \
    --cond_dim=64 --use_ortho_whiten=True --rank_res=16 --rank_film=16 \
    > ./logs/recon_train3.log 2>&1 &


# WANDB_MODE=disabled uv run python scripts/train_hyper_recon.py configs/hyper_lora_decontam_lol_tasks.yaml \
# --model_dir=mistralai/Mistral-7B-Instruct-v0.2/ \
# --emb_model=Alibaba-NLP/gte-large-en-v1.5 \
# --warmup_frac=0.1 --lr=1e-3 --epochs=10000 \
# --n_train_ds=479 --exp_setup=hyper_lora --encoder_type=linear \
# --pred_z_score=True --n_descs_per_ds=128 --n_embs_per_sampled_task=1 \
# --n_tasks_per_batch=4 --factorized=False --delta_w_scaling=10000 --shared_AB_head=True
