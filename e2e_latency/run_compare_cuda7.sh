cd /home/jgryu/workspace/weight_compression/e2e_latency


# nvidia-smi -i 7 -pl 150

CUDA_VISIBLE_DEVICES=7 python benchmark_llama_offload.py \
  --mode compare \
  --model-id ../Wparam_dataset/hf_model/meta-llama--Meta-Llama-3-8B \
  --dtype float16 \
  --prompt-length 1024 \
  --max-new-tokens 16 \
  --bandwidth-gbps 4 8 12 25 1000 \
  --baseline-vram-gb 12 \
  --baseline-runtime-reserve-gb 1.5 \
  --decode-ms-per-matrix 1.17 \
  --decode-overhead-scale 1.0 \
  --baseline-runtime-reserve-gb 2.0 \
  --json-output ./results/llama8b_compare_12GB_300W.json


# nvidia-smi -i 7 -pl 300
