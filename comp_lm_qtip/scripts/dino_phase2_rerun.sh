#!/bin/bash
export HF_HOME=/home/jgryu/.cache/huggingface
PYTHON_BIN="/opt/conda/bin/python"
CKPT="../hf_model_comp/comp_qtip/ckpt"
HF="../hf_model_comp/comp_qtip/hf"
RES="../hf_model_comp_results_v2/vision"
LOG="./log"
HF_MODEL_BASE="/home/jgryu/workspace/weight_compression/Wparam_dataset/hf_model"
GPU_IDS=(4 5 6 7)

echo "=== DINOv2 Phase2 re-run (GPU 4~7) ==="
ei=0
for seed in 1 2 3; do
    for lmb in 30 50 100 300 1000 10000; do
        res="${RES}/facebook--dinov2-large-imagenet1k-1-layer/rnorm_ldlq64_seed${seed}/lmbda${lmb}_imagenet_result.json"
        [ -f "$res" ] && { echo "  SKIP seed${seed}/lmbda${lmb} (exists)"; continue; }
        gpu=${GPU_IDS[$((ei % 4))]}
        save="facebook--dinov2-large-imagenet1k-1-layer/rnorm_ldlq64_seed${seed}/lmbda${lmb}"
        log="${LOG}/${save}_p2.log"
        mkdir -p "$(dirname $log)"
        echo "  >> [GPU $gpu] DINOv2 eval seed${seed} λ${lmb}"
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
echo "=== DINOv2 Phase2 완료 ==="
