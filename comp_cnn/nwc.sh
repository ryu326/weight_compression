# export CUDA_VISIBLE_DEVICES=1
# python /home/jgryu/workspace/weight_compression/comp_cnn/compress_resnet_nwc.py \
#   --arch resnet18 \
#   --comp_model_path /home/jgryu/workspace/weight_compression/NWC/checkpoint/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt/M16/lmbda30_rdloss_ql_size16_encdim512_M16_Q4_R0_m0_batch_size2048_total_iter200000_lr0.0001_seed100/best_loss_model_loss_3.46368_bpp_4.27494_MSE_0.02685_total_iter_95000.pth.tar \
#   --direction row \
#   --layer_normalize \
#   --dataset_dir /data/ILSVRC2012 \
#   --save_path ./resnet18_nwc_run \
#   --eval_batch_size 512 \
#   --num_workers 32 \
#   --ql --ql_search_value 2 --Q 4 

python /home/jgryu/workspace/weight_compression/comp_cnn/compress_resnet_nwc.py \
  --arch resnet50 \
  --comp_model_paths \
  /home/jgryu/workspace/weight_compression/NWC/checkpoint/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt/M16/lmbda10_*/best_loss*.pth.tar \
  /home/jgryu/workspace/weight_compression/NWC/checkpoint/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt/M16/lmbda30_*/best_loss*.pth.tar \
  /home/jgryu/workspace/weight_compression/NWC/checkpoint/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt/M16/lmbda50_*/best_loss*.pth.tar \
  /home/jgryu/workspace/weight_compression/NWC/checkpoint/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt/M16/lmbda75_*/best_loss*.pth.tar \
  /home/jgryu/workspace/weight_compression/NWC/checkpoint/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt/M16/lmbda100_*/best_loss*.pth.tar \
  /home/jgryu/workspace/weight_compression/NWC/checkpoint/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt/M16/lmbda300_*/best_loss*.pth.tar \
  /home/jgryu/workspace/weight_compression/NWC/checkpoint/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt/M16/lmbda1000_*/best_loss*.pth.tar \
  /home/jgryu/workspace/weight_compression/NWC/checkpoint/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt/M16/lmbda10000_*/best_loss*.pth.tar \
  --gpu_ids  1 2 3 4 5 6 7 \
  --layer_normalize \
  --dataset_dir /data/ILSVRC2012 \
  --save_path ./resnet50_nwc_lambda_sweep_ql1_ldlq16_row \
  --eval_batch_size 512 \
  --num_workers 8 \
  --hessian_num_batches 16 \
  --hessian_batch_size 32 \
  --sigma_reg 1e-4 \
  --ql --ldlq  --comp_batch_size 16 \
  --direction row --ql_search_value 1 --Q 4 \

