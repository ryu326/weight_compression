# sbatch tcm_train_quality_1.sh
# sbatch tcm_train_quality_2.sh
# sbatch tcm_train_quality_3.sh
# sbatch tcm_train_quality_4.sh
# # sbatch tcm_train_quality_5.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u train.py --model_name TCM --image_quality 2 --iter 2000000 --batch-size 8 --seed 100 --radius_denominator 8 --dist_port 6044 2>&1 | tee ./training_logs/wp_exp_2.txt

# python -u train.py --model_name TCM --image_quality 1 --iter 2000000 --batch-size 8 --seed 100 --dist_port 6044 > ./training_logs/log_1.txt