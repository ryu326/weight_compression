
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -u train.py \
#     --image_quality 2 --iter 2000000 --batch-size 8 --seed 100 --dist_port 6044 \
#     --dataset_dir ../Wparam_dataset \
#     2>&1 | tee ./training_logs/test.txt


dataset=models--meta-llama--Meta-Llama-3-8B/mlp_4096_dataset
log_dir=./training_logs/${dataset}
mkdir -p ${log_dir}  # 디렉토리 생성
dim=512
length=8

for lambda_value in 1 2 4 7 13 24 43 79 146 270 500 3000; do
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -u train.py \
        --iter 1000000 --batch-size 64 --seed 100 --dist_port 5786 \
        --dataset_dir ./dataset_wp_one_row/${dataset}.pt \
        --data_dim ${dim} --length ${length} \
        --lmbda ${lambda_value} \
        2>&1 | tee ${log_dir}/lmbda${lambda_value}.txt
done


# for d in 32 64 256 512 1024; do
#     CUDA_VISIBLE_DEVICES=0 python test_image_pretrain.py \
#         --dataset_dir /workspace/jgryu/Weight_compression/Wparam_dataset/image_shape_wp/meta-llama-3-8b_attn_val_json/${d}_${d}
# done
