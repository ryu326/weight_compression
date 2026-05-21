#!/bin/bash
export HF_HOME=/home/jgryu/.cache/huggingface
PYTHON_BIN="/opt/conda/bin/python"
CKPT="../hf_model_comp/comp_qtip/ckpt"
HF="../hf_model_comp/comp_qtip/hf"
RES="../hf_model_comp_results_v2/vision"
LOG="./log"
HF_MODEL_BASE="/home/jgryu/workspace/weight_compression/Wparam_dataset/hf_model"
GPU_IDS=(4 5 6 7)

echo "=== CLIP missing eval ==="
gi=0
for seed in 1 2 3; do
    for lmb in 30 50 100 300 1000 10000; do
        res="${RES}/openai--clip-vit-large-patch14/rnorm_ldlq64_seed${seed}/lmbda${lmb}_imagenet_result.json"
        [ -f "$res" ] && continue
        gpu=${GPU_IDS[$((gi % 4))]}
        save="openai--clip-vit-large-patch14/rnorm_ldlq64_seed${seed}/lmbda${lmb}"
        log="${LOG}/${save}_missing.log"
        mkdir -p "$(dirname $log)"
        echo "  >> [GPU $gpu] CLIP seed${seed} λ${lmb}"
        (
            export CUDA_VISIBLE_DEVICES=$gpu
            echo ">> hfize" > $log
            $PYTHON_BIN -m quantize_llama.hfize_clip \
                --quantized_path ${CKPT}/${save} \
                --base_model ${HF_MODEL_BASE}/openai--clip-vit-large-patch14 \
                --hf_output_path ${HF}/${save} >> $log 2>&1
            echo ">> eval" >> $log
            $PYTHON_BIN -m eval.eval_clip_imagenet \
                --hf_path ${HF}/${save} \
                --output_path ${RES}/${save} \
                --model_id openai/clip-vit-large-patch14 >> $log 2>&1
            rm -rf "${HF}/${save}"
            echo "DONE" >> $log
        ) &
        ((gi++))
        if (( gi % 4 == 0 )); then wait; fi
    done
done
wait
echo "=== CLIP eval 완료 ==="
