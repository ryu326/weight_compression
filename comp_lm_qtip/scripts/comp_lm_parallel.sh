# comp_model_base="../NWC/checkpoint/nwc/block_seq_row_16"
# comp_model_base="../NWC/checkpoint/nwc_ql/block_seq_ql_random__llama-3-8b-hf/block_seq_ql_random_col_16"
# comp_model_base="../NWC/checkpoint/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__col_4096_RHT.pt"
# comp_model_base="../NWC/checkpoint/nwc_tr_with_hyp/block_seq_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt"
# comp_model_base="../NWC/checkpoint/nwc_tr/block_seq_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt"
comp_model_base="../NWC/checkpoint/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt"
# comp_model_base="../NWC/checkpoint/nwc/block_seq_scaler_meta-llama--Meta-Llama-3-8B__scaled3_RHT_sig1e-06_col_1024.pt"
# comp_model_base="../NWC/checkpoint/nwc/block_seq_scaler_meta-llama--Llama-2-7b-hf__scaled3_RHT_sig0.0001_col_4096.pt"
# comp_model_base="../NWC/checkpoint/nwc/block_seq_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt"
# comp_model_base="../NWC/checkpoint/nwc_ql_cdt/block_seq_ql_random_lstats_scaler_meta-llama--Meta-Llama-3-8B__col_1024_layerwise_stats.pt"

# model_name="meta-llama--Llama-2-7b-hf"
# HESS="../Wparam_dataset/quip_hess/Hessians-Llama-2-7b-6144"

model_name="meta-llama--Meta-Llama-3-8B"
HESS="../Wparam_dataset/quip_hess/llama3_8b_6144"

# model_name="meta-llama--Llama-3.2-3B"
# HESS="../Wparam_dataset/quip_hess/meta-llama--Llama-3.2-3B-256"

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

lmbda_values=(30 50 100 300 1000 10000)
gpu_ids=(0 1 2 3)
i=0
for lmbda in "${lmbda_values[@]}"; do
    gpu_id=${gpu_ids[$((i % 4))]}

    ## ========= Change this =========
    SAVE_NAME=ft_ql_tuned_ldlq/${model_name}/lmbda${lmbda}
    ## ========= Change this =========
    comp_model=$comp_model_base/lmbda${lmbda}_*/best_loss*.pth.tar
    mkdir -p $(dirname "$LOG/$SAVE_NAME.log")


    echo ">> Launching full pipeline on GPU $gpu_id: lmbda=$lmbda"

    (
        export CUDA_VISIBLE_DEVICES=$gpu_id
        # echo "################## Running compression lmbda=${lmbda} ##################"
        # CUDA_VISIBLE_DEVICES=0,1,2,3 taskset -c 0-31 \
        # python -m quantize_llama.quantize_finetune_llama --save_path $CKPT/$SAVE_NAME \
        #     --base_model $lm_model_path \
        #     --comp_model_path $comp_model \
        #     --in_hess_path $HESS --devset_size 384 --ft_valid_size 128 \
        #     --batch_size 8 \
        #     --ql \
        #     --ql_tuned \
        #     --ft_epochs 0 \
        #     2>&1 | tee $LOG/$SAVE_NAME.log

            # --incoh_mode had  --rescale_WH_2  --sigma_reg 1e-4 --use_train_scale \
            # --ldlq --comp_batch_size 1 \
            # --ft_comp_model --ft_comp_lmbda $lmbda --ft_comp_ep 100 --direction row \
            # --ft_comp_model2 --ft_comp_lmbda $lmbda --ft_comp_ep 200 \
            # --ql \

        echo "################## Running hfize lmbda=${lmbda} ##################"
        python -m quantize_llama.hfize_llama --quantized_path $CKPT/${SAVE_NAME} \
                --hf_output_path $HF/$SAVE_NAME 2>&1 | tee -a $LOG/$SAVE_NAME.log 

        echo "################## Running PPL evaluation lmbda=${lmbda} ##################"
        pretrain_path=$HF/$SAVE_NAME
        log_path=$(echo "$pretrain_path" | sed 's|_reconstructed|_eval|')_quip_result.txt
        log_dir=$(dirname "$log_path")
        mkdir -p "$log_dir"
        echo "Running evaluation for directory: $pretrain_path"
        python -m eval.eval_ppl \
            --hf_path $pretrain_path \
            --seqlen 2048 \
            --no_use_cuda_graph 2>&1 | tee -a $LOG/$SAVE_NAME.log


        # echo "################## Running benchmark evaluation lmbda=${lmbda} ##################"
        # pretrain_path=$HF/$SAVE_NAME
        # output_path=$(echo "$pretrain_path" | sed 's|model_reconstructed|model_eval|')_harness_results
        # export CUDA_VISIBLE_DEVICES=0,1,2,3
        # lm_eval --model hf \
        #     --model_args "pretrained=$pretrain_path,parallelize=True" \
        #     --tasks arc_easy,arc_challenge,winogrande,piqa,boolq \
        #     --batch_size 4 \
        #     --output_path $output_path \
        #     --trust_remote_code \
        #     2>&1 | tee -a $LOG/$SAVE_NAME.log

        rm -r $pretrain_path
    ) &

    ((i+=1))    
    if (( i % 4 == 0 )); then
        wait
    fi
done