log_dir=./training_logs
dim=512
length=8
lambda_value=3000
CUDA_VISIBLE_DEVICES=3 taskset -c 24-31 python -u train.py \
    --lmbda ${lambda_value} --iter 2000000 --u-length 4 --batch-size 8 --seed 100 --dist_port 4568 --slurm \
    2>&1 | tee ${log_dir}/log_lmbda_${lambda_value}.txt