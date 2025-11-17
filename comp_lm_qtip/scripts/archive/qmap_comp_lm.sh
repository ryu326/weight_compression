# comp_model_base="../NWC/checkpoint/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt/use_hyper_rdloss_ql_size16_encdim512_M16_Q4_R0_m0_batch_size2048_total_iter200000_lr0.0001_seed100"
# comp_model_base="/workspace/Weight_compression/NWC/checkpoint/nwc_ql_pe/block_seq_ql_random_pos_scaler_meta-llama--Meta-Llama-3-8B__col_1024_idx_ltype_stats.pt/use_hyper_rdloss_ql_size16_encdim512_M16_Q4_R0_m0_batch_size2048_total_iter200000_lr0.0001_seed100"
# comp_model_base="/workspace/Weight_compression/NWC/checkpoint/nwc_qmap/block_seq_qmap_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt/rdloss_qmap_size16_encdim512_M17_Q0_R0_m0_batch_size2048_total_iter200000_lr0.0001_seed100/ld_min25_max10000_"
# comp_model_base="/workspace/Weight_compression/NWC/checkpoint/nwc_qmap/block_seq_qmap_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt/rdloss_qmap_size16_encdim512_M17_Q0_R0_m0_batch_size2048_total_iter200000_lr0.0001_seed100/ld_min5_max10000_"
# comp_model_base="/workspace/Weight_compression/NWC/checkpoint/nwc_qmap/block_seq_qmap_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt/rdloss_qmap2_size16_encdim512_M17_Q0_R0_m0_batch_size2048_total_iter200000_lr0.0001_seed100/lmbda50_"
# comp_model_base="/workspace/Weight_compression/NWC/checkpoint/nwc_qmap/block_seq_qmap_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt/rdloss_qmap_size16_encdim512_M17_Q0_R0_m0_batch_size2048_total_iter200000_lr0.0001_seed100/ld_min1_max10000_"
# comp_model_base="/workspace/Weight_compression/NWC/checkpoint/nwc_qmap2/block_seq_qmap_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt/rdloss_qmap2_size16_encdim512_M17_Q0_R0_m0_batch_size2048_total_iter200000_lr0.0001_seed100"
comp_model_base="/workspace/Weight_compression/NWC/checkpoint/nwc_qmap3/block_seq_qmap_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt/rdloss_qmap2_size16_encdim512_M16_Q0_R0_m0_batch_size2048_total_iter200000_lr0.0001_seed100"

# model_name="meta-llama--Llama-2-7b-hf"
# HESS="../Wparam_dataset/quip_hess/Hessians-Llama-2-7b-6144"

model_name="meta-llama--Meta-Llama-3-8B"
HESS="../Wparam_dataset/quip_hess/llama3_8b_6144"

# model_name="meta-llama--Llama-2-13b-hf"
# HESS="../Wparam_dataset/quip_hess/Hessians-Llama-2-13b-6144"

# model_name="meta-llama--Llama-3.2-3B"
# HESS="../Wparam_dataset/quip_hess/meta-llama--Llama-3.2-3B-256"

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
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_SILENT=true

lmbda_values=(50)
qmap_value=(0 1 0.5)
for qmap_v in "${qmap_value[@]}"; do
    for lmbda in "${lmbda_values[@]}"; do
        echo "################## Running compression qmap_v=${qmap_v} ##################"
        ## ========= Change this =========
        # SAVE_NAME=${model_name}/ql_qmap_5_10000/qmap_uniform${qmap_v}
        SAVE_NAME=${model_name}/ql_qmap3/uniform${qmap_v}/test_lmbda${lmbda}
        # SAVE_NAME=${model_name}/ql_qmap2/lmbda50_hessian_ql
        # SAVE_NAME=${model_name}/ql_qmap3/lmbda50_hessian_ql
        # ## ========= Change this =========

        comp_model=$comp_model_base/lmbda${lmbda}_*/best_loss*.pth.tar
        mkdir -p $(dirname "$LOG/$SAVE_NAME.log")
        
        taskset -c 0-31 \
        python -m quantize_llama.quantize_finetune_llama --save_path $CKPT/$SAVE_NAME \
            --base_model $lm_model_path \
            --comp_model_path $comp_model \
            --in_hess_path $HESS --devset_size 384 --ft_valid_size 128 --batch_size 8 \
            --ft_epochs 0 \
            --qmap_uniform $qmap_v \
            2>&1 | tee $LOG/$SAVE_NAME.log

            # --qmap_hessian_ql \
            # --qmap_optim  --code_optim_lmbda 50 --qmap_optim_iter 5 \
            # --qmap_uniform $qmap_v \
            # --qmap_hessian --qmap_alpha $qmap_v \
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
            # --datasets c4 \

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