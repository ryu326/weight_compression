export CUDA_VISIBLE_DEVICES=0

CKPT="../hf_model_comp/qtip/ckpt"
HF="../hf_model_comp/qtip/hf"
LOG="./log"

mkdir -p $CKPT
mkdir -p $LOG
mkdir -p $HF

for K in 3 4
do
    # SAVE_PATH="$CKPT/3_8b_ft1/3_8b_${K}bit"
    # LOG_FILE="$LOG/eval_3_8b_${K}bit"
    # HF_PATH="$HF/3_8b_ft1/3_8b_${K}bit"
    # HF_PATH="$HF/3_8b_ft1/3_8b_${K}bit"

    # if [ ! -d "$HF_PATH" ]; then
    #     python -m quantize_llama.hfize_llama \
    #         --quantized_path $SAVE_PATH \
    #         --hf_output_path $HF_PATH 2>&1  \
    #         --base_model $base_model \
    #         | tee $LOG_FILE 
    # else
    #     echo "HF_PATH already exists, skipping Python execution."
    # fi

    python -m eval.eval_ppl \
        --hf_path ${HF_PATH} \
        --seqlen 2048  | tee -a $LOG_FILE 

    python -m eval.eval_zeroshot --tasks arc_challenge,arc_easy,boolq,piqa,winogrande \
        --batch_size 2  --hf_path ${HF_PATH} \
        --output_path ${HF_PATH}_zeroshot_result.json | tee -a $LOG_FILE 


done

