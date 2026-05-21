#!/bin/bash
export HF_HOME=/home/jgryu/.cache/huggingface
PYTHON_BIN="/opt/conda/bin/python"
CKPT="../hf_model_comp/comp_qtip/ckpt"
HF="../hf_model_comp/comp_qtip/hf"
RES="../hf_model_comp_results_v2/vision"
LOG="./log"
SIGLIP_PATH="/home/jgryu/workspace/weight_compression/Wparam_dataset/hf_model/google--siglip-base-patch16-224"
GPU_IDS=(0 1 2 3)
ei=0
for seed in 1 2 3; do
    for lmb in 30 50 100 300 1000 10000; do
        gpu=${GPU_IDS[$((ei%4))]}
        save="google--siglip-base-patch16-224/rnorm_ldlq64_seed${seed}/lmbda${lmb}"
        log="${LOG}/${save}.log"
        mkdir -p "$(dirname $log)"
        echo "  >> [GPU $gpu] SigLIP seed$seed λ$lmb"
        (
            export CUDA_VISIBLE_DEVICES=$gpu
            $PYTHON_BIN -m quantize_llama.hfize_clip \
                --quantized_path ${CKPT}/${save} \
                --base_model $SIGLIP_PATH \
                --hf_output_path ${HF}/${save} >> $log 2>&1
            $PYTHON_BIN -m eval.eval_siglip_imagenet \
                --hf_path ${HF}/${save} \
                --output_path ${RES}/${save} \
                --model_id google/siglip-base-patch16-224 >> $log 2>&1
            rm -rf "${HF}/${save}"
            echo "EVAL DONE" >> $log
        ) &
        ((ei++))
        if (( ei % 4 == 0 )); then wait; fi
    done
done
wait
echo "SigLIP Phase2 done."
