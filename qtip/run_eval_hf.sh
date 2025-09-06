export CUDA_VISIBLE_DEVICES=0,1,2,3

LOG="./log"
RES="../hf_model_comp_results/qtip"

for K in 3 4
do
    # NAME="3_8b_ft1/3_8b_${K}bit"
    # SAVE_PATH="$CKPT/$NAME"
    # LOG_FILE="$LOG/$NAME"
    # HF_PATH="$HF/$NAME"

    HF_PATH=relaxml/Llama-2-7b-QTIP-${K}Bit
    LOG_FILE=$LOG/relaxml/Llama-2-7b-QTIP-${K}Bit.log

    mkdir -p $SAVE_PATH
    mkdir -p $(dirname "$LOG_FILE")

    # echo "[Stage: Eval PPL] K=$K" | tee -a $LOG_FILE
    # python -m eval.eval_ppl \
    #     --hf_path ${HF_PATH} \
    #     --output_path ${RES}/${HF_PATH} \
    #     --seqlen 2048  2>&1 | tee -a $LOG_FILE

    # echo "[Stage: Eval Zero-shot] K=$K" | tee -a $LOG_FILE
    # python -m eval.eval_zeroshot --tasks arc_challenge,arc_easy,boolq,piqa,winogrande \
    #     --batch_size 4  --hf_path ${HF_PATH} \
    #     --output_path ${RES}/${HF_PATH} \
    #     2>&1 | tee -a $LOG_FILE

    echo "### [Stage: Eval Zero-shot | K=$K] ###" | tee -a $LOG_FILE
    python -m eval.eval_zeroshot \
        --tasks mmlu \
        --batch_size 8  --hf_path ${HF_PATH} \
        --output_path ${RES}/${HF_PATH}_mmlu 2>&1 | tee -a $LOG_FILE
done

export CUDA_VISIBLE_DEVICES=0,1,2,3

LOG="./log"
RES="../hf_model_comp_results/qtip"

for K in 2 3 4
do
    HF_PATH=relaxml/Llama-2-13b-QTIP-${K}Bit
    LOG_FILE=$LOG/relaxml/Llama-2-13b-QTIP-${K}Bit.log

    mkdir -p $SAVE_PATH
    mkdir -p $(dirname "$LOG_FILE")

    # echo "[Stage: Eval PPL] K=$K" | tee -a $LOG_FILE
    # python -m eval.eval_ppl \
    #     --hf_path ${HF_PATH} \
    #     --output_path ${RES}/${HF_PATH} \
    #     --seqlen 2048  2>&1 | tee -a $LOG_FILE

    # echo "[Stage: Eval Zero-shot] K=$K" | tee -a $LOG_FILE
    # python -m eval.eval_zeroshot --tasks arc_challenge,arc_easy,boolq,piqa,winogrande \
    #     --batch_size 4  --hf_path ${HF_PATH} \
    #     --output_path ${RES}/${HF_PATH} \
    #     2>&1 | tee -a $LOG_FILE

    echo "### [Stage: Eval Zero-shot | K=$K] ###" | tee -a $LOG_FILE
    python -m eval.eval_zeroshot \
        --tasks mmlu \
        --batch_size 4  --hf_path ${HF_PATH} \
        --output_path ${RES}/${NAME}_mmlu 2>&1 | tee -a $LOG_FILE
done

