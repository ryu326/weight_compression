#!/bin/sh
#SBATCH -J  elic_hf_v2_2
#SBATCH -o  ./training_logs/%j.log_elic_hf_v2_2.txt
#SBATCH -p  3090
#SBATCH -t 72:00:00
#SBATCH   --gres=gpu:4
#SBATCH   --nodes=1
#SBATCH   --ntasks=1
#SBATCH   --tasks-per-node=1
#SBATCH   --cpus-per-task=4

cd  $SLURM_SUBMIT_DIR

echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "CUDA_HOME=$CUDA_HOME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CUDA_VERSION=$CUDA_VERSION"
srun -l /bin/hostname
srun -l /bin/pwd
srun -l /bin/date

echo "Start"
echo "conda PATH "
echo "source  /home/minkyu4506/anaconda3/etc/profile.d/conda.sh"
source  /home/minkyu4506/anaconda3/etc/profile.d/conda.sh
echo "conda activate pytorch_p38"
conda activate pytorch_p38

SAMPLES_DIR=$HOME/nic_with_high_freq_loss_ver2
python -u $SAMPLES_DIR/train.py --model_name ELIC --image_quality 2 --iter 2050000 --batch-size 8 --seed 100 --radius_denominator 8 --slurm --dist_port 6325
date
echo " conda deactivate pytorch_p38  "
conda deactivate
squeue--job $SLURM_JOBID
echo  "##### END #####"


dataset=models--meta-llama--Meta-Llama-3-8B/mlp_4096_dataset
log_dir=./training_logs/${dataset}
mkdir -p ${log_dir}  # 디렉토리 생성
dim=512
length=8

for lambda_value in 1 2 4 7 13 24 43 79 146 270 500 3000; do
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -u train.py \
        --iter 1000000 --batch-size 16 --seed 100 --dist_port 5786 \
        --dataset_dir ./dataset_wp_one_row/${dataset}.pt \
        --data_dim ${dim} --length ${length} \
        --lmbda ${lambda_value} \
        2>&1 | tee ${log_dir}/lmbda${lambda_value}.txt
done


# for d in 32 64 256 512 1024; do
#     CUDA_VISIBLE_DEVICES=0 python test_image_pretrain.py \
#         --dataset_dir /workspace/jgryu/Weight_compression/Wparam_dataset/image_shape_wp/meta-llama-3-8b_attn_val_json/${d}_${d}
# done
