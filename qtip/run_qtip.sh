export CUDA_VISIBLE_DEVICES=0,1,2,3

CKPT="../hf_model_comp/qtip/ckpt"
HF="../hf_model_comp/qtip/hf"
LOG="./log"

HESS="../Wparam_dataset/quip_hess/llama3_8b_6144"
base_model="../Wparam_dataset/hf_model/meta-llama--Meta-Llama-3-8B"

# HESS="../Wparam_dataset/quip_hess/meta-llama--Llama-3.2-3B-256"
# base_model="../Wparam_dataset/hf_model/meta-llama--Llama-3.2-3B"

mkdir -p $CKPT
mkdir -p $LOG
mkdir -p $HF

for K in 5 6 7 8
do
    NAME="3_8b_ft1/3_8b_${K}bit"
    SAVE_PATH="$CKPT/$NAME"
    LOG_FILE="$LOG/$NAME"
    HF_PATH="$HF/$NAME"

    mkdir -p $SAVE_PATH
    mkdir -p $(dirname "$LOG_FILE")

    echo "[Stage: Quantize with Finetuning] K=$K" | tee $LOG_FILE
    python -m quantize_llama.quantize_finetune_llama \
        --save_path $SAVE_PATH \
        --codebook bitshift \
        --base_model $base_model \
        --in_hess_path $HESS \
        --scale_override 0.9 \
        --ft_epochs 5 \
        --td_x 16 \
        --td_y 16 \
        --L 16 \
        --K $K \
        --V 2 \
        --decode_mode quantlut_sym \
        --tlut_bits 9 2>&1 \
        | tee -a $LOG_FILE 

    echo "[Stage: Convert to HF format] K=$K" | tee $LOG_FILE
    python -m quantize_llama.hfize_llama \
        --quantized_path $SAVE_PATH \
        --hf_output_path $HF_PATH \
        --base_model $base_model 2>&1 \
        | tee -a $LOG_FILE 

    echo "[Stage: Eval PPL] K=$K" | tee -a $LOG_FILE
    python -m eval.eval_ppl \
        --hf_path ${HF_PATH} \
        --seqlen 2048  2>&1 | tee -a $LOG_FILE

    echo "[Stage: Eval Zero-shot] K=$K" | tee -a $LOG_FILE
    python -m eval.eval_zeroshot --tasks arc_challenge,arc_easy,boolq,piqa,winogrande \
        --batch_size 4  --hf_path ${HF_PATH} \
        --output_path ${HF_PATH}_zeroshot_result.json 2>&1 \
        | tee -a $LOG_FILE 

    # echo "[Stage: End-to-End Finetuning] K=$K" | tee -a $LOG_FILE
    # python -m quantize_llama.finetune_e2e_llama --base_model $base_model \
    #     --hf_path $HF_PATH --devset_size 640 --ft_valid_size 128 \
    #     --ft_epochs 4 --ft_update_freq 4 --ft_bs 1 --ctx_size 4096 \
    #     --start_dev 2 \
    #     --ft_train_lut --hf_output_path ${HF_PATH}_e2e 2>&1 | tee $LOG_FILE 

    # echo "[Stage: Eval PPL] K=$K" | tee -a $LOG_FILE
    # python -m eval.eval_ppl \
    #     --hf_path ${HF_PATH}_e2e \
    #     --seqlen 2048 | tee -a ${HF_PATH}_e2e_ppl_result.txt 2>&1

    # echo "[Stage: Eval Zero-shot] K=$K" | tee -a $LOG_FILE
    # python -m eval.eval_zeroshot --tasks arc_challenge,arc_easy,boolq,piqa,winogrande \
    #     --batch_size 2  --hf_path ${HF_PATH}_e2e \
    #     --output_path ${HF_PATH}_e2e_zeroshot_result.json 2>&1 | tee $LOG_FILE

    NAME="3_8b_noft/3_8b_${K}bit"
    SAVE_PATH="$CKPT/$NAME"
    LOG_FILE="$LOG/$NAME"
    HF_PATH="$HF/$NAME"

    echo "[Stage: Quantize with Finetuning] K=$K" | tee $LOG_FILE
    python -m quantize_llama.quantize_finetune_llama \
        --save_path $SAVE_PATH \
        --codebook bitshift \
        --base_model $base_model \
        --in_hess_path $HESS \
        --scale_override 0.9 \
        --ft_epochs 0 \
        --td_x 16 \
        --td_y 16 \
        --L 16 \
        --K $K \
        --V 2 \
        --decode_mode quantlut_sym \
        --tlut_bits 9 2>&1 \
        | tee -a $LOG_FILE 

    echo "[Stage: Convert to HF format] K=$K" | tee $LOG_FILE
    python -m quantize_llama.hfize_llama \
        --quantized_path $SAVE_PATH \
        --hf_output_path $HF_PATH \
        --base_model $base_model 2>&1 \
        | tee -a $LOG_FILE 

    echo "[Stage: Eval PPL] K=$K" | tee -a $LOG_FILE
    python -m eval.eval_ppl \
        --hf_path ${HF_PATH} \
        --seqlen 2048  2>&1 | tee -a $LOG_FILE

    echo "[Stage: Eval Zero-shot] K=$K" | tee -a $LOG_FILE
    python -m eval.eval_zeroshot --tasks arc_challenge,arc_easy,boolq,piqa,winogrande \
        --batch_size 4  --hf_path ${HF_PATH} \
        --output_path ${HF_PATH}_zeroshot_result.json 2>&1 \
        | tee -a $LOG_FILE 

done

