# export CUDA_VISIBLE_DEVICES=0,1,2,3
# torchrun --nproc_per_node=4 -m quantize_llama.input_hessian_llama \
#     --batch_size 8 --devset_size 256 \
#     --base_model /workspace/Weight_compression/Wparam_dataset/hf_model/lmsys--vicuna-7b-v1.5 \
#     --save_path ../Wparam_dataset/quip_hess/lmsys--vicuna-7b-v1.5_256

export CUDA_VISIBLE_DEVICES=0,1,2,3

CKPT="../hf_model_comp/qtip/ckpt"
HF="../hf_model_comp/qtip/hf"
LOG="./log"
RES="../hf_model_comp_results/qtip"

############################################ 
# ft 7b
############################################

HESS="../Wparam_dataset/quip_hess/Hessians-Llama-2-7b-6144"
base_model="../Wparam_dataset/hf_model/meta-llama--Llama-2-7b-hf"

mkdir -p $CKPT
mkdir -p $LOG
mkdir -p $HF

for K in 2 3 4
do
    NAME="llama2_7b/${K}bit"
    SAVE_PATH="$CKPT/$NAME"
    LOG_FILE="${LOG}/${NAME}.log"
    HF_PATH="$HF/$NAME"

    mkdir -p $SAVE_PATH
    mkdir -p $(dirname "$LOG_FILE")

    echo "[Stage: Quantize with Finetuning] K=$K" | tee -a $LOG_FILE
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

    echo "[Stage: Convert to HF format] K=$K" | tee -a $LOG_FILE
    python -m quantize_llama.hfize_llama \
        --quantized_path $SAVE_PATH \
        --hf_output_path $HF_PATH \
        --base_model $base_model 2>&1 \
        | tee -a $LOG_FILE 

    echo "[Stage: Eval PPL] K=$K" | tee -a $LOG_FILE
    python -m eval.eval_ppl \
        --hf_path ${HF_PATH} \
        --output_path ${RES}/${NAME} \
        --seqlen 2048  2>&1 | tee -a $LOG_FILE

    echo "[Stage: Eval Zero-shot] K=$K" | tee -a $LOG_FILE
    python -m eval.eval_zeroshot --tasks arc_challenge,arc_easy,boolq,piqa,winogrande \
        --batch_size 8  --hf_path ${HF_PATH} \
        --output_path ${RES}/${NAME} \
        2>&1 | tee -a $LOG_FILE
done

############################################ 
# ft 13b
############################################

HESS="../Wparam_dataset/quip_hess/Hessians-Llama-2-13b-6144"
base_model="../Wparam_dataset/hf_model/meta-llama--Llama-2-13b-hf"

for K in 2 3 4
do
    NAME="llama2_13b/${K}bit"
    SAVE_PATH="$CKPT/$NAME"
    LOG_FILE="${LOG}/${NAME}.log"
    HF_PATH="$HF/$NAME"

    mkdir -p $SAVE_PATH
    mkdir -p $(dirname "$LOG_FILE")

    echo "[Stage: Quantize with Finetuning] K=$K" | tee -a $LOG_FILE
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

    echo "[Stage: Convert to HF format] K=$K" | tee -a $LOG_FILE
    python -m quantize_llama.hfize_llama \
        --quantized_path $SAVE_PATH \
        --hf_output_path $HF_PATH \
        --base_model $base_model 2>&1 \
        | tee -a $LOG_FILE 

    echo "[Stage: Eval PPL] K=$K" | tee -a $LOG_FILE
    python -m eval.eval_ppl \
        --hf_path ${HF_PATH} \
        --output_path ${RES}/${NAME} \
        --seqlen 2048  2>&1 | tee -a $LOG_FILE

    echo "[Stage: Eval Zero-shot] K=$K" | tee -a $LOG_FILE
    python -m eval.eval_zeroshot --tasks arc_challenge,arc_easy,boolq,piqa,winogrande \
        --batch_size 8  --hf_path ${HF_PATH} \
        --output_path ${RES}/${NAME} \
        2>&1 | tee -a $LOG_FILE
done

############################################ 
# noft 7b
############################################

HESS="../Wparam_dataset/quip_hess/Hessians-Llama-2-7b-6144"
base_model="../Wparam_dataset/hf_model/meta-llama--Llama-2-7b-hf"

for K in 2 3 4
do
    NAME="llama2_7b/noft_${K}bit"
    SAVE_PATH="$CKPT/$NAME"
    LOG_FILE="${LOG}/${NAME}.log"
    HF_PATH="$HF/$NAME"

    echo "[Stage: Quantize with Finetuning] K=$K" | tee -a $LOG_FILE
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

    echo "[Stage: Convert to HF format] K=$K" | tee -a $LOG_FILE
    python -m quantize_llama.hfize_llama \
        --quantized_path $SAVE_PATH \
        --hf_output_path $HF_PATH \
        --base_model $base_model 2>&1 \
        | tee -a $LOG_FILE 

    echo "[Stage: Eval PPL] K=$K" | tee -a $LOG_FILE
    python -m eval.eval_ppl \
        --hf_path ${HF_PATH} \
        --output_path ${RES}/${NAME} \
        --seqlen 2048  2>&1 | tee -a $LOG_FILE

    echo "[Stage: Eval Zero-shot] K=$K" | tee -a $LOG_FILE
    python -m eval.eval_zeroshot --tasks arc_challenge,arc_easy,boolq,piqa,winogrande \
        --batch_size 8  --hf_path ${HF_PATH} \
        --output_path ${RES}/${NAME} \
        2>&1 | tee -a $LOG_FILE

done


############################################ 
# noft 13b
############################################

HESS="../Wparam_dataset/quip_hess/Hessians-Llama-2-13b-6144"
base_model="../Wparam_dataset/hf_model/meta-llama--Llama-2-13b-hf"

for K in 2 3 4
do
    NAME="llama2_13b/noft_${K}bit"
    SAVE_PATH="$CKPT/$NAME"
    LOG_FILE="${LOG}/${NAME}.log"
    HF_PATH="$HF/$NAME"

    echo "[Stage: Quantize with Finetuning] K=$K" | tee -a $LOG_FILE
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

    echo "[Stage: Convert to HF format] K=$K" | tee -a $LOG_FILE
    python -m quantize_llama.hfize_llama \
        --quantized_path $SAVE_PATH \
        --hf_output_path $HF_PATH \
        --base_model $base_model 2>&1 \
        | tee -a $LOG_FILE 

    echo "[Stage: Eval PPL] K=$K" | tee -a $LOG_FILE
    python -m eval.eval_ppl \
        --hf_path ${HF_PATH} \
        --output_path ${RES}/${NAME} \
        --seqlen 2048  2>&1 | tee -a $LOG_FILE

    echo "[Stage: Eval Zero-shot] K=$K" | tee -a $LOG_FILE
    python -m eval.eval_zeroshot --tasks arc_challenge,arc_easy,boolq,piqa,winogrande \
        --batch_size 8  --hf_path ${HF_PATH} \
        --output_path ${RES}/${NAME} \
        2>&1 | tee -a $LOG_FILE

done

############################################ 
# e2e 7b
############################################

# for K in 2 3 4
# do
#     NAME="vicuna_7b_v1.5/${K}bit"
#     SAVE_PATH="$CKPT/$NAME"
#     LOG_FILE="${LOG}/${NAME}.log"
#     HF_PATH="$HF/$NAME"

#     NAME="vicuna_7b_v1.5/${K}bit_e2e"
#     LOG_FILE="${LOG}/${NAME}.log"

#     mkdir -p $(dirname "$LOG_FILE")

#     echo "[Stage: End-to-End Finetuning] K=$K" | tee -a $LOG_FILE
#     python -m quantize_llama.finetune_e2e_llama --base_model $base_model \
#         --hf_path $HF_PATH --devset_size 640 --ft_valid_size 128 \
#         --ft_epochs 4 --ft_update_freq 4 --ft_bs 1 --ctx_size 4096 \
#         --start_dev 2 \
#         --ft_train_lut --hf_output_path ${HF_PATH}_e2e 2>&1 | tee -a $LOG_FILE 

#     echo "[Stage: Eval PPL] K=$K" | tee -a $LOG_FILE
#     python -m eval.eval_ppl \
#         --hf_path ${HF_PATH}_e2e \
#         --output_path ${RES}/${NAME}_e2e \
#         --seqlen 2048  2>&1 | tee -a $LOG_FILE

#     echo "[Stage: Eval Zero-shot] K=$K" | tee -a $LOG_FILE
#     python -m eval.eval_zeroshot --tasks arc_challenge,arc_easy,boolq,piqa,winogrande \
#         --batch_size 8  --hf_path ${HF_PATH}_e2e \
#         --output_path ${RES}/${NAME}_e2e \
#         2>&1 | tee -a $LOG_FILE
# done

