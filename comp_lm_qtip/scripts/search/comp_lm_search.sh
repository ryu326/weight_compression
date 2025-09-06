# comp_model_base="../NWC/checkpoint/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__scaleH_sig0.0001_rnormed_row_1024.pt/rdloss_ql_size128_encdim1024_M256_Q4_R0_m0_batch_size2048_total_iter200000_lr0.0001_seed100"

comp_model_base="../NWC/checkpoint/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt/M16"

# model_name="meta-llama--Llama-2-7b-hf"
# HESS="../Wparam_dataset/quip_hess/Hessians-Llama-2-7b-6144"

model_name="meta-llama--Meta-Llama-3-8B"
HESS="../Wparam_dataset/quip_hess/llama3_8b_6144"

# model_name="meta-llama--Llama-2-13b-hf"
# HESS="../Wparam_dataset/quip_hess/Hessians-Llama-2-13b-6144"

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
RES="../hf_model_comp_results"

mkdir -p $CKPT
mkdir -p $HF
mkdir -p $LOG
mkdir -p $RES
export CUDA_VISIBLE_DEVICES=3
export WANDB_SILENT=true

ql_value=(1 2 3)
lmbda_values=(30 50 100 300 1000 10000)
for qlr in "${ql_value[@]}"; do
    for lmbda in "${lmbda_values[@]}"; do
        echo "################## Running compression lmbda=${lmbda} ##################"
        ## ========= Change this =========
        # SAVE_NAME=${model_name}/scaleH2/size128_encdim1024_M256__ql${qlr}/${lmbda}
        SAVE_NAME=${model_name}/ql_uniform${qlr}/${lmbda}
        # ## ========= Change this =========

        comp_model=$comp_model_base/lmbda${lmbda}_*/best_loss*.pth.tar
        # comp_model=$comp_model_base/lmbda100_*/best_loss*.pth.tar
        # comp_model=$(ls -t $comp_model_base/lmbda${lmbda}_*/best_loss*.pth.tar | head -n 1)
        mkdir -p $(dirname "$LOG/$SAVE_NAME.log")
        
        taskset -c 0-31 \
        python -m quantize_llama.quantize_finetune_llama --save_path $CKPT/$SAVE_NAME \
            --base_model $lm_model_path \
            --comp_model_path $comp_model \
            --in_hess_path $HESS --devset_size 384 --ft_valid_size 128 --batch_size 8 \
            --ft_epochs 0 \
            --ql_search --ql_search_value $qlr \
            2>&1 | tee $LOG/$SAVE_NAME.log

            # --direction row --scaleH \
            # --row_normalize --rnorm_optim --code_optim_lmbda $lmbda --qmap_optim_iter 5 \
            # --qmap_optim  --code_optim_lmbda $lmbda --qmap_optim_iter 5 \
            # --code_optim --code_optim_it 100 --loss rdloss_ql --code_optim_lmbda $lmbda --code_optim_lr 5e-3 --code_optim_model nwc_ql_sga_vbr --optim_qs \
            # --code_optim --code_optim_it 200 --loss rdloss_ql --code_optim_lmbda $lmbda --code_optim_lr 5e-3 \
            # --code_optim_test \
            # --code_optim --code_optim_it 100 --loss rdloss_ql --code_optim_lmbda $lmbda --code_optim_lr 5e-3 \
            # --incoh_mode had  --rescale_WH_2  --sigma_reg 1e-4 --use_train_scale \
            # --ldlq --comp_batch_size 128 \
            # --ft_comp_model2 --ft_comp_lmbda $lmbda --ft_comp_ep 100 --direction row \
            # --ft_comp_model2 --ft_comp_lmbda $lmbda --ft_comp_ep 200 \
            # --ft_comp_model2 --ft_comp_lmbda $lmbda --ft_comp_steps 400 --direction row --ft_train_dec \
            # --layerwise_scale \
            # --row_normalize \
            # --col_normalize \
            # --ql_tuned \
            # --ql \

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

        # echo "################## Running benchmark evaluation lmbda=${lmbda} ##################"
        # pretrain_path=$HF/$SAVE_NAME
        # python -m eval.eval_zeroshot_hf \
        #     --tasks arc_challenge,arc_easy,boolq,piqa,winogrande \
        #     --batch_size 8  \
        #     --hf_path $pretrain_path \
        #     --output_path $RES/$SAVE_NAME

        if [ "$pretrain_path" != "$HF" ]; then
            rm -r "$pretrain_path"
            rm -r "$CKPT/$SAVE_NAME"
        fi
    done
done

    # output_path=$(echo "$pretrain_path" | sed 's|model_reconstructed|model_eval|')_harness_results
    # lm_eval --model hf \
    #     --model_args "pretrained=$pretrain_path,parallelize=True" \
    #     --tasks arc_easy,arc_challenge,winogrande,piqa,boolq \
    #     --batch_size 4 \
    #     --output_path $output_path \
    #     --trust_remote_code \
    #     2>&1 | tee -a $LOG/$SAVE_NAME.log

