#!/bin/bash
export HF_HOME=/home/jgryu/.cache/huggingface
PYTHON_BIN="/opt/conda/bin/python"
CKPT="../hf_model_comp/comp_qtip/ckpt"
HF="../hf_model_comp/comp_qtip/hf"
RES="../hf_model_comp_results_v2/vision"
LOG="./log"
HF_MODEL_BASE="/home/jgryu/workspace/weight_compression/Wparam_dataset/hf_model"
HESS_BASE="/home/jgryu/workspace/weight_compression/Wparam_dataset/quip_hess"
NWC_BASE="/home/jgryu/workspace/weight_compression/NWC/checkpoint2/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt/MultiSeed_rdloss_ql_size16_encdim512_M16_Q4_nRB4R0_m0_batch_size2048_total_iter200000_lr0.0001_seed4.0"
GPU_IDS=(4 5 6 7)
JOBS_PER_GPU=4
MAX=$((${#GPU_IDS[@]} * JOBS_PER_GPU))  # 16

echo "=== DINOv2 Phase1 re-run (GPU 4~7, 16 simultaneous) ==="
ci=0
for seed in 1 2 3; do
    for lmb in 30 50 100 300 1000 10000; do
        gpu=${GPU_IDS[$((( ci % MAX ) / JOBS_PER_GPU))]}
        save="facebook--dinov2-large-imagenet1k-1-layer/rnorm_ldlq64_seed${seed}/lmbda${lmb}"
        nwc="${NWC_BASE}/seed${seed}/lmbda${lmb}_*/best_loss*.pth.tar"
        log="${LOG}/${save}_comp_rerun.log"
        mkdir -p "$(dirname $log)"
        echo "  >> [GPU $gpu] DINOv2 seed${seed} λ${lmb}"
        (
            export CUDA_VISIBLE_DEVICES=$gpu
            $PYTHON_BIN -m quantize_llama.quantize_finetune_dino \
                --save_path ${CKPT}/${save} \
                --base_model ${HF_MODEL_BASE}/facebook--dinov2-large-imagenet1k-1-layer \
                --comp_model_path $nwc \
                --in_hess_path ${HESS_BASE}/dinov2-large-imagenet1k-1-layer_cc1024 \
                --direction col --ql --Q 4 --row_normalize --comp_batch_size 64 --ldlq --ft_epochs 0 \
                > $log 2>&1
            echo "[COMP DONE] seed${seed}/lmbda${lmb}" >> $log
        ) &
        ((ci++))
        if (( ci % MAX == 0 )); then
            echo "  Waiting for batch (${ci} jobs)..."
            wait
        fi
    done
done
wait
echo "=== DINOv2 Phase1 완료 ==="

# Phase 2 immediately after
echo "=== DINOv2 Phase2 hfize+eval (GPU 4~7) ==="
ei=0
for seed in 1 2 3; do
    for lmb in 30 50 100 300 1000 10000; do
        res="${RES}/facebook--dinov2-large-imagenet1k-1-layer/rnorm_ldlq64_seed${seed}/lmbda${lmb}_imagenet_result.json"
        [ -f "$res" ] && { echo "  SKIP seed${seed}/lmbda${lmb}"; continue; }
        gpu=${GPU_IDS[$((ei % 4))]}
        save="facebook--dinov2-large-imagenet1k-1-layer/rnorm_ldlq64_seed${seed}/lmbda${lmb}"
        log="${LOG}/${save}_p2_rerun.log"
        echo "  >> [GPU $gpu] eval seed${seed} λ${lmb}"
        (
            export CUDA_VISIBLE_DEVICES=$gpu
            echo ">> hfize" > $log
            $PYTHON_BIN -m quantize_llama.hfize_dino \
                --quantized_path ${CKPT}/${save} \
                --base_model ${HF_MODEL_BASE}/facebook--dinov2-large-imagenet1k-1-layer \
                --hf_output_path ${HF}/${save} >> $log 2>&1
            echo ">> eval" >> $log
            $PYTHON_BIN -m eval.eval_dino \
                --hf_path ${HF}/${save} \
                --output_path ${RES}/${save} \
                --imagenet_path /data/ILSVRC2012 >> $log 2>&1
            rm -rf "${HF}/${save}"
            echo "EVAL DONE" >> $log
        ) &
        ((ei++))
        if (( ei % 4 == 0 )); then wait; fi
    done
done
wait
echo "=== DINOv2 완료 ==="
