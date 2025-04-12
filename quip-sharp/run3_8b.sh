# export CUDA_VISIBLE_DEVICES=5
CKPT="./ckpt"
HF="./hf"
LOG="./log"
HESS="/home/minkyu4506/weight_compression_dataset/llama3_8b_6144"
# HESS="../Wparam_dataset/quip_hess/llama3_8b_6144"

mkdir -p $CKPT
mkdir -p $LOG
mkdir -p $HF

for K in 2 3 4
do
    echo "Running quantization for K=$K bits"

    SAVE_PATH="$CKPT/3_8b_${K}bit"
    LOG_FILE="$LOG/3_8b_${K}.txt"
    HF_PATH="$HF/3_8b_${K}bit"

    if [ "$K" -eq 2 ]; then
        CODEBOOK="E8P12"
    elif [ "$K" -eq 3 ]; then
        CODEBOOK="E8P12RVQ3B"
    elif [ "$K" -eq 4 ]; then
        CODEBOOK="E8P12RVQ4B"
    fi

    echo "[Stage: Quantize with Finetuning] K=$K" | tee -a $LOG_FILE
    python -m quantize_llama.quantize_finetune_llama \
        --save_path $SAVE_PATH \
        --codebook $CODEBOOK \
        --scale_override 0.9 \
        --base_model meta-llama/Meta-Llama-3-8B \
        --hessian_path $HESS \
        --devset_size 384 \
        --ft_valid_size 128 2>&1 | tee $LOG_FILE

    echo "[Stage: Convert to HF format] K=$K" | tee -a $LOG_FILE
    python -m quantize_llama.hfize_llama \
        --quantized_path $SAVE_PATH \
        --hf_output_path $HF_PATH 2>&1 | tee -a $LOG_FILE

    echo "[Stage: End-to-End Finetuning] K=$K" | tee -a $LOG_FILE
    python -m quantize_llama.finetune_e2e_llama \
        --base_model meta-llama/Meta-Llama-3-8B \
        --hf_path $HF_PATH \
        --devset_size 384 \
        --ft_valid_size 128 \
        --ft_epochs 8 \
        --ft_bs 1 \
        --ctx_size 4096 \
        --ft_update_freq 2 \
        --ft_train_mode \
        --ckpt_path $SAVE_PATH 2>&1 | tee -a $LOG_FILE

    echo "[Stage: Re-convert to HF (post-e2e)] K=$K" | tee -a $LOG_FILE
    python -m quantize_llama.hfize_llama \
        --quantized_path $SAVE_PATH \
        --hf_output_path $HF_PATH 2>&1 | tee -a $LOG_FILE

    echo "[Stage: Eval PPL] K=$K" | tee -a $LOG_FILE
    python -m eval.eval_ppl \
        --hf_path $HF_PATH 2>&1 | tee -a $LOG_FILE

    echo "[Stage: Eval Zero-shot] K=$K" | tee -a $LOG_FILE
    python -m eval.eval_zeroshot \
        --tasks arc_challenge,arc_easy,boolq,piqa,winogrande \
        --batch_size 4 \
        --hf_path $HF_PATH 2>&1 | tee -a $LOG_FILE

    #### No-Finetune 버전
    SAVE_PATH_no_ft="$CKPT/3_8b_${K}bit_no_ft"
    LOG_FILE_no_ft="$LOG/3_8b_${K}_no_ft.txt"
    HF_PATH_no_ft="$HF/3_8b_${K}bit_no_ft"

    echo "[Stage: Quantize (No Finetuning)] K=$K" | tee -a $LOG_FILE_no_ft
    python -m quantize_llama.quantize_finetune_llama \
        --save_path $SAVE_PATH_no_ft \
        --codebook $CODEBOOK \
        --scale_override 0.9 \
        --base_model meta-llama/Meta-Llama-3-8B \
        --hessian_path $HESS \
        --devset_size 384 \
        --ft_epochs 0 \
        --ft_valid_size 128 2>&1 | tee $LOG_FILE_no_ft

    echo "[Stage: Convert to HF format (No Finetuning)] K=$K" | tee -a $LOG_FILE_no_ft
    python -m quantize_llama.hfize_llama \
        --quantized_path $SAVE_PATH_no_ft \
        --hf_output_path $HF_PATH_no_ft 2>&1 | tee -a $LOG_FILE_no_ft

    echo "[Stage: Eval PPL (No Finetuning)] K=$K" | tee -a $LOG_FILE_no_ft
    python -m eval.eval_ppl \
        --hf_path $HF_PATH_no_ft 2>&1 | tee -a $LOG_FILE_no_ft

    echo "[Stage: Eval Zero-shot (No Finetuning)] K=$K" | tee -a $LOG_FILE_no_ft
    python -m eval.eval_zeroshot \
        --tasks arc_challenge,arc_easy,boolq,piqa,winogrande \
        --batch_size 4 \
        --hf_path $HF_PATH_no_ft 2>&1 | tee -a $LOG_FILE_no_ft

done
