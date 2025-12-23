
# (lmbdas=(10000)
# for lmbda in "${lmbdas[@]}"; do
#     echo "=== Running with λ=${lmbda} ==="
#     CUDA_VISIBLE_DEVICES=2  python -u train_nwc.py \
#         --architecture nwc_ql  --dataset block_seq_ql_random \
#         --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/col_1024_gaussian_padding.pt \
#         --iter 200000 --input_size 16 --M 16 --Q 4 --dim_encoder 512 --batch_size 2048 \
#         --loss rdloss_ql \
#         --run_name n_rb2 --n_resblock 2 \
#         --lmbda $lmbda
# done
# ) > ./logs/run2.log 2>&1 &

# (lmbdas=(10000)
# for lmbda in "${lmbdas[@]}"; do
#     echo "=== Running with λ=${lmbda} ==="
#     CUDA_VISIBLE_DEVICES=3  python -u train_nwc.py \
# done
# ) > ./logs/run3.log 2>&1 &

(lmbdas=(30 100 1000)
for lmbda in "${lmbdas[@]}"; do
    echo "=== Running with λ=${lmbda} ==="
    CUDA_VISIBLE_DEVICES=4  python -u train_nwc.py \
        --architecture nwc_ql  --dataset block_seq_ql_random \
        --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/col_1024_gaussian_padding.pt \
        --iter 200000 --input_size 16 --M 16 --Q 4 --dim_encoder 512 --batch_size 2048 \
        --loss rdloss_ql \
        --no_layernorm \
        --run_name noLN \
        --lmbda $lmbda
done
) > ./logs/run4.log 2>&1 &

(lmbdas=(50 300 10000)
for lmbda in "${lmbdas[@]}"; do
    echo "=== Running with λ=${lmbda} ==="
    CUDA_VISIBLE_DEVICES=5  python -u train_nwc.py \
        --architecture nwc_ql  --dataset block_seq_ql_random \
        --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/col_1024_gaussian_padding.pt \
        --iter 200000 --input_size 16 --M 16 --Q 4 --dim_encoder 512 --batch_size 2048 \
        --loss rdloss_ql \
        --no_layernorm \
        --run_name noLN \
        --lmbda $lmbda
done
) > ./logs/run5.log 2>&1 &

(lmbdas=(30 100 1000)
for lmbda in "${lmbdas[@]}"; do
    echo "=== Running with λ=${lmbda} ==="
    CUDA_VISIBLE_DEVICES=6  python -u train_nwc.py \
        --architecture nwc_ql  --dataset block_seq_ql_random \
        --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/rnormed_col_1024.pt \
        --iter 200000 --input_size 16 --M 16 --Q 4 --dim_encoder 512 --batch_size 2048 \
        --loss rdloss_ql \
        --no_layernorm \
        --run_name noLN_rnorm \
        --lmbda $lmbda
done
) > ./logs/run6.log 2>&1 &

(lmbdas=(50 300 10000)
for lmbda in "${lmbdas[@]}"; do
    echo "=== Running with λ=${lmbda} ==="
    CUDA_VISIBLE_DEVICES=7  python -u train_nwc.py \
        --architecture nwc_ql  --dataset block_seq_ql_random \
        --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/rnormed_col_1024.pt \
        --iter 200000 --input_size 16 --M 16 --Q 4 --dim_encoder 512 --batch_size 2048 \
        --loss rdloss_ql \
        --no_layernorm \
        --run_name noLN_rnorm \
        --lmbda $lmbda
done
) > ./logs/run7.log 2>&1 &

        # --architecture nwc_ql  --dataset block_seq_ql_random \
        # --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/col_1024_gaussian_padding.pt \
        # --iter 200000 --input_size 4 --M 4 --Q 4 --dim_encoder 64 --batch_size 8192 \
        # --loss rdloss_ql \
        # --n_resblock 2 \
        # --lmbda $lmbda

        # --architecture nwc_ql  --dataset block_seq_ql_random \
        # --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/col_1024_gaussian_padding.pt \
        # --iter 200000 --input_size 16 --M 16 --Q 4 --dim_encoder 512 --batch_size 2048 \
        # --loss rdloss_ql \
        # --run_name n_rb2 --n_resblock 2 \
        # --lmbda $lmbda

        #  --architecture nwc_ql  --dataset block_seq_ql_random \
        # --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/col_1024_gaussian_padding.pt \
        # --iter 200000 --input_size 16 --M 16 --Q 4 --dim_encoder 512 --batch_size 2048 \
        # --loss rdloss_ql \
        # --run_name seed3 --seed 3 \
        # --lmbda $lmbda

########## lattice
# (lmbdas=(E8P12)
# for lmbda in "${lmbdas[@]}"; do
#     echo "=== Running with λ=${lmbda} ==="
#     CUDA_VISIBLE_DEVICES=5  python -u train_nwc.py \
#         --architecture nwc_lattice  --dataset block_seq_ql_random \
#         --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/col_1024_gaussian_padding.pt \
#         --iter 200000 --input_size 8 --M 8 --Q 4 --dim_encoder 256 --batch_size 2048 \
#         --loss mseloss \
#         --lattice $lmbda
# done
# ) > ./logs/run5.log 2>&1 &



# (lmbdas=(E8P12)
# for lmbda in "${lmbdas[@]}"; do
#     echo "=== Running with λ=${lmbda} ==="
#     CUDA_VISIBLE_DEVICES=5  python -u train_nwc.py \
#         --architecture nwc_lattice  --dataset block_seq_ql_random \
#         --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/col_1024_gaussian_padding.pt \
#         --iter 200000 --input_size 16 --M 16 --Q 4 --dim_encoder 512 --batch_size 2048 \
#         --loss mseloss \
#         --lattice $lmbda
# done
# ) > ./logs/run5.log 2>&1 &

########## Id
        # --architecture nwc_id  --dataset block_seq_ql_random \
        # --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/col_1024_gaussian_padding.pt \
        # --iter 200000 --input_size 16 --M 16 --Q 4 --dim_encoder 512 --batch_size 2048 \
        # --loss rdloss_ql \
        # --lmbda $lmbda \
        # --learning_rate 1e-4

########## VQ
        # --architecture nwc_vq  --dataset block_seq_ql_random \
        # --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/col_1024_gaussian_padding.pt \
        # --iter 100000 --input_size 16 --M 16 --Q 4 --dim_encoder 512 --batch_size 2048 \
        # --loss vqloss \
        # --K $lmbda --e_dim 1 \
        # --learning_rate 1e-4

        # --architecture nwc_vq  --dataset block_seq_ql_random \
        # --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/col_1024_gaussian_padding.pt \
        # --iter 100000 --input_size 16 --M 16 --Q 4 --dim_encoder 512 --batch_size 2048 \
        # --loss vqloss \
        # --K $lmbda --e_dim 4 \
        # --learning_rate 1e-4

        # --architecture nwc_vq  --dataset block_seq_ql_random \
        # --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/col_1024_gaussian_padding.pt \
        # --iter 200000 --input_size 16 --M 16 --Q 4 --dim_encoder 512 --batch_size 2048 \
        # --loss vqloss \
        # --K $lmbda --e_dim 2 \
        # --learning_rate 1e-4

        # --architecture nwc_vq  --dataset block_seq_ql_random \
        # --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/col_1024_gaussian_padding.pt \
        # --iter 200000 --input_size 16 --M 16 --Q 4 --dim_encoder 512 --batch_size 2048 \
        # --loss vqloss \
        # --K $lmbda \
        # --learning_rate 1e-4



        # --architecture nwc_ql  --dataset block_seq_ql_random \
        # --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/col_1024_gaussian_padding.pt \
        # --iter 200000 --input_size 16 --M 16 --Q 4 --dim_encoder 512 --batch_size 2048 \
        # --loss rdloss_ql_mse \
        # --run_name ablation_rate_bpp_3.4 \
        # --lmbda $lmbda \
        # --learning_rate 1e-4


        # --architecture nwc_ql  --dataset block_seq_ql_random \
        # --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/col_1024_gaussian_padding.pt \
        # --iter 200000 --input_size 16 --M 16 --Q 4 --dim_encoder 512 --batch_size 2048 \
        # --loss rdloss_ql \
        # --run_name double_check \
        # --lmbda $lmbda


    #    --architecture nwc_ql  --dataset block_seq_ql_random \
    #     --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/col_1024_gaussian_padding.pt \
    #     --iter 200000 --input_size 16 --M 16 --Q 4 --dim_encoder 512 --batch_size 2048 \
    #     --loss rdloss_ql_mse \
    #     --run_name ablation_mse \
    #     --lmbda $lmbda \
    #     --learning_rate 1e-4


        # --architecture nwc_ql \
        # --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/col_1024_gaussian_padding.pt \
        # --dataset block_seq_ql_random \
        # --iter 200000 \
        # --input_size 16 \
        # --M 16 \
        # --Q 4 \
        # --run_name ablation_mse \
        # --dim_encoder 512 \
        # --batch_size 2048 \
        # --loss rdloss_ql_mse \
        # --lmbda $lmbda \
        # --learning_rate 1e-4