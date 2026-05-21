#!/bin/bash
# 1. CLIP re-eval (11개, GPU 4~7)
# 2. SigLIP comp + eval (GPU 0~3)  
# 3. DINOv2 comp + eval (GPU 4~7, CLIP 끝난 후)

export HF_HOME=/home/jgryu/.cache/huggingface
PYTHON_BIN="/opt/conda/bin/python"
CKPT="../hf_model_comp/comp_qtip/ckpt"
HF="../hf_model_comp/comp_qtip/hf"
RES="../hf_model_comp_results_v2/vision"
LOG="./log"
NWC_BASE="/home/jgryu/workspace/weight_compression/NWC/checkpoint2/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt/MultiSeed_rdloss_ql_size16_encdim512_M16_Q4_nRB4R0_m0_batch_size2048_total_iter200000_lr0.0001_seed4.0"
HF_MODEL_BASE="/home/jgryu/workspace/weight_compression/Wparam_dataset/hf_model"
HESS_BASE="/home/jgryu/workspace/weight_compression/Wparam_dataset/quip_hess"
SIGLIP_PATH="/home/jgryu/.cache/huggingface/hub/models--google--siglip-base-patch16-224/snapshots/7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"

run_hfize_eval_clip() {
    local seed=$1 lmb=$2 gpu=$3
    local save="openai--clip-vit-large-patch14/rnorm_ldlq64_seed${seed}/lmbda${lmb}"
    local log="${LOG}/${save}_reeval.log"
    mkdir -p "$(dirname $log)"
    (
        export CUDA_VISIBLE_DEVICES=$gpu
        $PYTHON_BIN -m quantize_llama.hfize_clip \
            --quantized_path ${CKPT}/${save} \
            --base_model ${HF_MODEL_BASE}/openai--clip-vit-large-patch14 \
            --hf_output_path ${HF}/${save} > $log 2>&1
        $PYTHON_BIN -m eval.eval_clip_imagenet \
            --hf_path ${HF}/${save} \
            --output_path ${RES}/${save} \
            --model_id openai/clip-vit-large-patch14 >> $log 2>&1
        rm -rf "${HF}/${save}"
        echo "DONE" >> $log
    ) &
}

# ── 1. CLIP re-eval (GPU 4~7) ───────────────────────────────────────
echo "=== CLIP re-eval 시작 ==="
gpus=(4 5 6 7); gi=0
for seed in 1 2 3; do
    for lmb in 30 50 100 300 1000 10000; do
        res="${RES}/openai--clip-vit-large-patch14/rnorm_ldlq64_seed${seed}/lmbda${lmb}_imagenet_result.json"
        [ -f "$res" ] && continue
        run_hfize_eval_clip $seed $lmb ${gpus[$((gi%4))]}
        ((gi++))
        if (( gi % 4 == 0 )); then wait; fi
    done
done
wait
echo "=== CLIP re-eval 완료 ==="

# ── 2. DINOv2 comp + eval (GPU 4~7) ────────────────────────────────
echo "=== DINOv2 Phase1 시작 (GPU 4~7) ==="
ci=0; JOBS_PER_GPU=4; MAX=16
for seed in 1 2 3; do
    for lmb in 30 50 100 300 1000 10000; do
        gpu=${gpus[$((( ci % MAX ) / JOBS_PER_GPU))]}
        save="facebook--dinov2-large-imagenet1k-1-layer/rnorm_ldlq64_seed${seed}/lmbda${lmb}"
        nwc="${NWC_BASE}/seed${seed}/lmbda${lmb}_*/best_loss*.pth.tar"
        log="${LOG}/${save}.log"
        mkdir -p "$(dirname $log)"
        echo "  >> [GPU $gpu] DINOv2 seed$seed λ$lmb"
        (
            export CUDA_VISIBLE_DEVICES=$gpu
            $PYTHON_BIN -m quantize_llama.quantize_finetune_dino \
                --save_path ${CKPT}/${save} \
                --base_model ${HF_MODEL_BASE}/facebook--dinov2-large-imagenet1k-1-layer \
                --comp_model_path $nwc \
                --in_hess_path ${HESS_BASE}/dinov2-large-imagenet1k-1-layer_cc1024 \
                --direction col --ql --Q 4 --row_normalize --comp_batch_size 64 --ldlq --ft_epochs 0 \
                > $log 2>&1
            echo "[COMP DONE]" >> $log
        ) &
        ((ci++))
        if (( ci % MAX == 0 )); then wait; fi
    done
done
wait
echo "=== DINOv2 Phase1 완료 ==="

echo "=== DINOv2 Phase2 hfize+eval (GPU 4~7) ==="
ei=0
for seed in 1 2 3; do
    for lmb in 30 50 100 300 1000 10000; do
        gpu=${gpus[$((ei%4))]}
        save="facebook--dinov2-large-imagenet1k-1-layer/rnorm_ldlq64_seed${seed}/lmbda${lmb}"
        log="${LOG}/${save}.log"
        echo "  >> [GPU $gpu] DINOv2 eval seed$seed λ$lmb"
        (
            export CUDA_VISIBLE_DEVICES=$gpu
            $PYTHON_BIN -m quantize_llama.hfize_dino \
                --quantized_path ${CKPT}/${save} \
                --base_model ${HF_MODEL_BASE}/facebook--dinov2-large-imagenet1k-1-layer \
                --hf_output_path ${HF}/${save} >> $log 2>&1
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
