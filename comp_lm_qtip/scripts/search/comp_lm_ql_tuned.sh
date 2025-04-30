
comp_model_base="../NWC/checkpoint/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt"

model_name="meta-llama--Meta-Llama-3-8B"
HESS="../Wparam_dataset/quip_hess/llama3_8b_6144"

# ql="../Wparam_dataset/hessian/$model_name/quip_hess_n6144_top3_qlevel3.pt"
# ql="../Wparam_dataset/hessian/$model_name/pileval_n_samples128_seqlen512_top[ 0.1  1.  10. ]_qlevel[3, 2, 1].pt"
# ql='../Wparam_dataset/hessian/meta-llama--Llama-2-7b-hf/quip_hess_n6144_all_layers_top[ 0.1  1.  10. ]_qlevel[3, 2, 1].pt'
############################################

lm_model_path="../Wparam_dataset/hf_model/$model_name"

CKPT="../hf_model_comp/comp_qtip/ckpt"
HF="../hf_model_comp/comp_qtip/hf"
LOG="./log"

mkdir -p $CKPT
mkdir -p $HF
mkdir -p $LOG

lmbda_values=(30 50 100)
ql_search_values=(0)

for lmbda in "${lmbda_values[@]}"; do
    for ql_val in "${ql_search_values[@]}"; do
        echo "#### Running compression lmbda=${lmbda}, ql_search_value=${ql_val}, layer=${layer_name} ####"

        ## ========= Change this =========
        SAVE_NAME=ql_tuned_vo031/layer${layer_name}_val${ql_val}/${model_name}/lmbda${lmbda}
        ## ========= Change this =========

        comp_model=$comp_model_base/lmbda${lmbda}_*/best_loss*.pth.tar
        mkdir -p $(dirname "$LOG/$SAVE_NAME.log")
        
        CUDA_VISIBLE_DEVICES=0 taskset -c 0-31 \
        python -m quantize_llama.quantize_finetune_llama --save_path $CKPT/$SAVE_NAME \
            --base_model $lm_model_path \
            --comp_model_path $comp_model \
            --in_hess_path $HESS --devset_size 384 --ft_valid_size 128 \
            --batch_size 8 \
            --ql \
            --ql_search_value $ql_val \
            --ql_tuned \
            --ft_epochs 0 \
            2>&1 | tee $LOG/$SAVE_NAME.log


        echo "################## Running hfize lmbda=${lmbda} ##################"
        python -m quantize_llama.hfize_llama --quantized_path $CKPT/${SAVE_NAME} \
                --hf_output_path $HF/$SAVE_NAME 2>&1 | tee -a $LOG/$SAVE_NAME.log 

        echo "################## Running evaluation lmbda=${lmbda} ##################"
        pretrain_path=$HF/$SAVE_NAME
        log_path=$(echo "$pretrain_path" | sed 's|_reconstructed|_eval|')_quip_result.txt
        log_dir=$(dirname "$log_path")
        mkdir -p "$log_dir"
        echo "Running evaluation for directory: $pretrain_path"
        export CUDA_VISIBLE_DEVICES=0
        python eval_ppl.py \
            --hf_path $pretrain_path \
            --seqlen 2048 \
            --no_use_cuda_graph 2>&1 | tee -a $LOG/$SAVE_NAME.log

        if [ -d "$pretrain_path" ]; then
            echo "Removing $pretrain_path..."
            rm -r "$pretrain_path"
            
            if [ $? -eq 0 ]; then
                echo "Successfully removed $pretrain_path. Now removing .pt files in $CKPT/$SAVE_NAME..."
                rm "$CKPT/$SAVE_NAME"/*.pt
            else
                echo "Failed to remove $pretrain_path."
            fi
        else
            echo "$pretrain_path does not exist."
        fi
    done
done
