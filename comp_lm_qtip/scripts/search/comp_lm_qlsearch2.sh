comp_model_base="../NWC/checkpoint/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt/M16"

model_name="meta-llama--Meta-Llama-3-8B"
HESS="../Wparam_dataset/quip_hess/llama3_8b_6144"
###########################################
lm_model_path="../Wparam_dataset/hf_model/$model_name"

CKPT="../hf_model_comp/comp_qtip/ckpt"
HF="../hf_model_comp/comp_qtip/hf"
LOG="./log"
RES="../hf_model_comp_results"

mkdir -p $CKPT
mkdir -p $HF
mkdir -p $LOG
mkdir -p $RES
export CUDA_VISIBLE_DEVICES=2
export WANDB_SILENT=true

ql_value=1
ql_r=(0.1 0.5 1 3 5 10)
lmbda_values=(30 50 100 300 1000 10000)

for lmbda in "${lmbda_values[@]}"; do
    for r in "${ql_r[@]}"; do
        echo "################## Running compression lmbda=${lmbda} ##################"
        ## ========= Change this =========
        SAVE_NAME=${model_name}/ql_Q2_search/v${ql_value}_r${r}/${lmbda}
        # ## ========= Change this =========

        comp_model=$comp_model_base/lmbda${lmbda}_*/best_loss*.pth.tar
        mkdir -p $(dirname "$LOG/$SAVE_NAME.log")
        
        taskset -c 0-31 \
        python -m quantize_llama.quantize_finetune_llama --save_path $CKPT/$SAVE_NAME \
            --base_model $lm_model_path \
            --comp_model_path $comp_model \
            --in_hess_path $HESS --devset_size 384 --ft_valid_size 128 --batch_size 8 \
            --ft_epochs 0 \
            --ql --Q 4 --ql_search_value $ql_value --ql_search_r $r \
            2>&1 | tee $LOG/$SAVE_NAME.log

        echo "################## Running hfize lmbda=${lmbda} ##################"
        python -m quantize_llama.hfize_llama --quantized_path $CKPT/${SAVE_NAME} \
                --hf_output_path $HF/$SAVE_NAME 2>&1 | tee -a $LOG/$SAVE_NAME.log

        echo "################## Running PPL evaluation lmbda=${lmbda} ##################"
        pretrain_path=$HF/$SAVE_NAME
        mkdir -p "$log_dir"
        echo "Running evaluation for directory: $pretrain_path"
        python -m eval.eval_ppl_hf \
            --hf_path $pretrain_path \
            --seqlen 2048 \
            --output_path $RES/$SAVE_NAME \
            --no_use_cuda_graph 2>&1 | tee -a $LOG/$SAVE_NAME.log


        if [ "$pretrain_path" != "$HF" ]; then
            rm -r "$pretrain_path"
            rm -r "$CKPT/$SAVE_NAME"
        fi
    done
done