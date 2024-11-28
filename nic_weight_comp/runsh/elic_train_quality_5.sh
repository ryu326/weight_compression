#!/bin/sh
#SBATCH -J  elic_hf_v2_5
#SBATCH -o  ./training_logs/%j.log_elic_hf_v2_5.txt
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
python -u $SAMPLES_DIR/train.py --model_name ELIC --image_quality 5 --iter 2050000 --batch-size 8 --seed 100 --radius_denominator 8 --slurm --dist_port 9742
date
echo " conda deactivate pytorch_p38  "
conda deactivate
squeue--job $SLURM_JOBID
echo  "##### END #####"