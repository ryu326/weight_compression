#!/bin/sh
#SBATCH -J  tr_nwc_lmbda1
#SBATCH -o  ./logs_slurm/tr_nwc_lmbda1.txt
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
# echo "conda PATH "
# echo "source  /home/minkyu4506/anaconda3/etc/profile.d/conda.sh"
# source  /home/minkyu4506/anaconda3/etc/profile.d/conda.sh
# echo "conda activate pytorch_p38"
# conda activate pytorch_p38
echo "Docker run"
docker run -d -it --name nwc_jiyunbae -v /home/jiyunbae/jgryu/:/workspace/jgryu/ -v /data/:/data/ --gpus all --shm-size=400G jegwangryu/nwc:1
echo "Docker exec"
docker exec nwc_jiyunbae bash -c "cd /workspace/jgryu/weight_compression/nic_weight_comp_transformer; python dataset_generation_one_row.py; bash run_lmbda1.sh"
echo "Docker stop"
docker stop nwc_jiyunbae
docker rm nwc_jiyunbae

squeue--job $SLURM_JOBID
echo  "##### END #####"