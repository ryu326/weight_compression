##########################################################################
##                           MODEL CONFIGURATION                        ##
##         (모든 실험에 공통으로 적용되는 설정을 여기에 둡니다)             ##
##########################################################################

# model_name="meta-llama--Llama-2-7b-hf"
# HESS="../Wparam_dataset/quip_hess/Hessians-Llama-2-7b-6144"

model_name="meta-llama--Meta-Llama-3-8B"
HESS="../Wparam_dataset/quip_hess/llama3_8b_6144"

# model_name="meta-llama--Llama-2-13b-hf"
# HESS="../Wparam_dataset/quip_hess/Hessians-Llama-2-13b-6144"

##########################################################################
##                      EXPERIMENT PARAMETERS                           ##
##         (수정된 부분: 실험할 파라미터 조합을 여기에 정의합니다)          ##
##########################################################################

# 여러 양자화 방법을 배열로 정의
quant_methods=("group")
group_sizes=(128)
ql_value=(96)

############################################
##              SCRIPT SETUP              ##
############################################
lm_model_path="../Wparam_dataset/hf_model/$model_name"

CKPT="../hf_model_comp/handcraft/ckpt"
HF="../hf_model_comp/handcraft/hf"
LOG="./log"
RES="../hf_model_comp_results/handcraft"

mkdir -p $CKPT
mkdir -p $HF
mkdir -p $LOG
mkdir -p $RES
export CUDA_VISIBLE_DEVICES=0
export WANDB_SILENT=true

echo "========================================================================"
echo "            STARTING EXPERIMENTS"
echo "========================================================================"

# 루프를 중첩하여 모든 파라미터 조합을 실행
for qm in "${quant_methods[@]}"; do
    if [ "$qm" == "group" ]; then
        group_sz_loop=("${group_sizes[@]}")
    else
        group_sz_loop=("-1") # -1은 "해당 없음"을 의미하는 dummy 값
    fi

    for gs in "${group_sz_loop[@]}"; do
        for qlv in "${ql_value[@]}"; do

            # quant_method와 group_sz에 따라 동적으로 SAVE_NAME과 파라미터 생성
            if [ "$qm" == "group" ]; then
                exp_name="qm_${qm}_gs${gs}_q${qlv}"
                quant_args="--quant_method ${qm} --group_sz ${gs}"
            else
                exp_name="qm_${qm}_q${qlv}"
                quant_args="--quant_method ${qm}"
            fi
            
            SAVE_NAME=${model_name}/jepg_scaleh/${exp_name}
            
            echo "################## Running compression | ${exp_name} ##################"
            mkdir -p $(dirname "$LOG/$SAVE_NAME.log")
            
            taskset -c 0-7 \
            python -m quantize_llama.quantize_finetune_llama --save_path $CKPT/$SAVE_NAME \
                --base_model $lm_model_path \
                --in_hess_path $HESS --devset_size 384 --ft_valid_size 128 --batch_size 8 \
                --ft_epochs 0 \
                --handcraft_mode jp \
                --jp_quality ${qlv} \
                ${quant_args} \
                --scaleH \
                2>&1 | tee $LOG/$SAVE_NAME.log

            echo "################## Running hfize | ${exp_name} ##################"
            python -m quantize_llama.hfize_llama --quantized_path $CKPT/${SAVE_NAME} \
                    --hf_output_path $HF/$SAVE_NAME 2>&1 | tee -a $LOG/$SAVE_NAME.log

            pretrain_path=$HF/$SAVE_NAME
            echo "################## Running PPL evaluation | ${exp_name} ##################"
            # mkdir -p "$log_dir" # log_dir 변수가 정의되지 않아 주석 처리
            echo "Running evaluation for directory: $pretrain_path"
            python -m eval.eval_ppl_hf \
                --hf_path $pretrain_path \
                --seqlen 2048 \
                --output_path $RES/$SAVE_NAME \
                --dataset wikitext2,c4 \
                --no_use_cuda_graph 2>&1 | tee -a $LOG/$SAVE_NAME.log

            # echo "################## Running benchmark evaluation | ${exp_name} ##################"
            # python -m eval.eval_zeroshot_hf \
            #     --tasks arc_challenge,arc_easy,boolq,piqa,winogrande \
            #     --batch_size 1  \
            #     --hf_path $pretrain_path \
            #     --output_path $RES/$SAVE_NAME

            if [ "$pretrain_path" != "$HF" ]; then
                echo "Cleaning up temporary files for $SAVE_NAME"
                rm -rf "$pretrain_path"
                rm -rf "$CKPT/$SAVE_NAME"
            fi
        done
    done
done


echo "========================================================================"
echo "                          All experiments finished."
echo "========================================================================"