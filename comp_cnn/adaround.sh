conda run -n aimet python /home/jgryu/workspace/weight_compression/aimet/Examples/torch/quantization/resnet18_weight_only_bw_sweep.py \
  --dataset_dir /data/ILSVRC2012 \
  --use_cuda \
  --arch resnet50 \
  --gpu_ids 3 4 5 6 7 \
  --weight_bits 2 3 4 5 6 \
  --batch_size 512 \
  --num_workers 4 \
  --adaround_num_batches 4 \
  --adaround_iterations 10000
