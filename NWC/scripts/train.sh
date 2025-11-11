(lmbdas=(30)
for lmbda in "${lmbdas[@]}"; do
    echo "=== Running with λ=${lmbda} ==="
    CUDA_VISIBLE_DEVICES=4  python -u train_nwc.py \
        --architecture nwc_ql  --dataset block_seq_ql_random \
        --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/col_1024_gaussian_padding.pt \
        --iter 200000 --input_size 16 --M 16 --Q 4 --dim_encoder 512 --batch_size 2048 \
        --loss rdloss_ql_mse \
        --run_name ablation_rate_bpp_3.4 \
        --lmbda $lmbda \
        --learning_rate 1e-4
done
) > ./logs/run4.log 2>&1 &

(lmbdas=(50)
for lmbda in "${lmbdas[@]}"; do
    echo "=== Running with λ=${lmbda} ==="
    CUDA_VISIBLE_DEVICES=5  python -u train_nwc.py \
        --architecture nwc_ql  --dataset block_seq_ql_random \
        --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/col_1024_gaussian_padding.pt \
        --iter 200000 --input_size 16 --M 16 --Q 4 --dim_encoder 512 --batch_size 2048 \
        --loss rdloss_ql_mse \
        --run_name ablation_rate_bpp_3.4 \
        --lmbda $lmbda \
        --learning_rate 1e-4
done
) > ./logs/run5.log 2>&1 &

(lmbdas=(100)
for lmbda in "${lmbdas[@]}"; do
    echo "=== Running with λ=${lmbda} ==="
    CUDA_VISIBLE_DEVICES=6  python -u train_nwc.py \
        --architecture nwc_ql  --dataset block_seq_ql_random \
        --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/col_1024_gaussian_padding.pt \
        --iter 200000 --input_size 16 --M 16 --Q 4 --dim_encoder 512 --batch_size 2048 \
        --loss rdloss_ql_mse \
        --run_name ablation_rate_bpp_3.4 \
        --lmbda $lmbda \
        --learning_rate 1e-4
done
) > ./logs/run6.log 2>&1 &

(lmbdas=(300)
for lmbda in "${lmbdas[@]}"; do
    echo "=== Running with λ=${lmbda} ==="
    CUDA_VISIBLE_DEVICES=7  python -u train_nwc.py \
        --architecture nwc_ql  --dataset block_seq_ql_random \
        --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/col_1024_gaussian_padding.pt \
        --iter 200000 --input_size 16 --M 16 --Q 4 --dim_encoder 512 --batch_size 2048 \
        --loss rdloss_ql_mse \
        --run_name ablation_rate_bpp_3.4 \
        --lmbda $lmbda \
        --learning_rate 1e-4
done
) > ./logs/run7.log 2>&1 &

(lmbdas=(1000)
for lmbda in "${lmbdas[@]}"; do
    echo "=== Running with λ=${lmbda} ==="
    CUDA_VISIBLE_DEVICES=3  python -u train_nwc.py \
        --architecture nwc_ql  --dataset block_seq_ql_random \
        --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/col_1024_gaussian_padding.pt \
        --iter 200000 --input_size 16 --M 16 --Q 4 --dim_encoder 512 --batch_size 2048 \
        --loss rdloss_ql_mse \
        --run_name ablation_rate_bpp_3.4 \
        --lmbda $lmbda \
        --learning_rate 1e-4
done
) > ./logs/run3.log 2>&1 &


(lmbdas=(10000)
for lmbda in "${lmbdas[@]}"; do
    echo "=== Running with λ=${lmbda} ==="
    CUDA_VISIBLE_DEVICES=2  python -u train_nwc.py \
        --architecture nwc_ql  --dataset block_seq_ql_random \
        --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/col_1024_gaussian_padding.pt \
        --iter 200000 --input_size 16 --M 16 --Q 4 --dim_encoder 512 --batch_size 2048 \
        --loss rdloss_ql_mse \
        --run_name ablation_rate_bpp_3.4 \
        --lmbda $lmbda \
        --learning_rate 1e-4
done
) > ./logs/run2.log 2>&1 &

(lmbdas=(10)
for lmbda in "${lmbdas[@]}"; do
    echo "=== Running with λ=${lmbda} ==="
    CUDA_VISIBLE_DEVICES=1  python -u train_nwc.py \
        --architecture nwc_ql  --dataset block_seq_ql_random \
        --dataset_path ../Wparam_dataset/block_pt/meta-llama--Meta-Llama-3-8B/col_1024_gaussian_padding.pt \
        --iter 200000 --input_size 16 --M 16 --Q 4 --dim_encoder 512 --batch_size 2048 \
        --loss rdloss_ql_mse \
        --run_name ablation_rate_bpp_3.4 \
        --lmbda $lmbda \
        --learning_rate 1e-4
done
) > ./logs/run1.log 2>&1 &

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