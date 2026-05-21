#!/bin/bash
# NWC_v2 example launches.  Each command exercises a different
# (encoder × decoder × entropy_model × dataset) combination.
#
# Run from NWC_v2/ root:
#   cd /home/jgryu/workspace/weight_compression/NWC_v2 && bash scripts/train.sh

set -u
set -o pipefail
cd "$(dirname "$0")/.."

GPU="${GPU:-0}"
BATCH="${BATCH:-256}"
ITERS="${ITERS:-200000}"

# 1) baseline: resblock encoder + resblock decoder + compressai EB on Llama-3-8B
CUDA_VISIBLE_DEVICES=$GPU python train.py \
    --dataset llama8b \
    --encoder_transform resblock --decoder_transform resblock \
    --entropy_model compressai \
    --M 16 --n_resblock 4 --dim_encoder 32 \
    --lmbda 100 --iter $ITERS --batch_size $BATCH \
    --run_name llama8b_resblock_compressai

# 2) parametric mixture entropy model
CUDA_VISIBLE_DEVICES=$GPU python train.py \
    --dataset llama8b \
    --encoder_transform resblock --decoder_transform resblock \
    --entropy_model parametric --num_gaussian 3 --num_laplacian 3 \
    --M 16 --lmbda 100 --iter $ITERS --batch_size $BATCH \
    --run_name llama8b_resblock_parametric

# 3) RHT encoder + RHT decoder (input_size == M required)
CUDA_VISIBLE_DEVICES=$GPU python train.py \
    --dataset gaussian \
    --encoder_transform rht --decoder_transform rht \
    --entropy_model compressai \
    --input_size 16 --M 16 --lmbda 100 --iter $ITERS --batch_size $BATCH \
    --run_name gaussian_rht_compressai

# 4) hybrid: linear encoder + resblock decoder + lattice EB
CUDA_VISIBLE_DEVICES=$GPU python train.py \
    --dataset llama8b \
    --encoder_transform linear --decoder_transform resblock \
    --entropy_model lattice \
    --M 16 --lmbda 100 --iter $ITERS --batch_size $BATCH \
    --run_name llama8b_linear-resblock_lattice
