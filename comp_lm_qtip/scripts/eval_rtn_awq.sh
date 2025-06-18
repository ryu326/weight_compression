export CUDA_VISIBLE_DEVICES=0,1,2,3

bit=(2 3 4 5 6 7 8)
bit=(2)
for b in "${bit[@]}"; do

    # pretrain_path=../hf_model_comp/RTN/meta-llama--Meta-Llama-3-8B_W${b}g128
    # pretrain_path=../hf_model_comp/awq/meta-llama--Meta-Llama-3-8B/w${b}-g128-fake-quantized
    # pretrain_path=/workspace/Weight_compression/Wparam_dataset/hf_model/meta-llama--Meta-Llama-3-8B
    pretrain_path=/workspace/Weight_compression/Wparam_dataset/hf_model/meta-llama--Llama-2-13b-hf
    output_path=/workspace/Weight_compression/Wparam_dataset/hf_model/meta-llama--Llama-2-13b-hf
    # pretrain_path=/workspace/Weight_compression/hf_model_comp/awq/meta-llama--Llama-2-13b-hf/w${b}-g128-fake-quantized
    # output_path=/workspace/Weight_compression/hf_model_comp_results/awq/meta-llama--Llama-2-13b-hf/w${b}-g128-fake-quantized
    # mkdir -p "/workspace/Weight_compression/hf_model_comp_results/awq/meta-llama--Llama-2-13b-hf"

    # echo "################## Running PPL evaluation lmbda=${lmbda} ##################"
    # echo "Running evaluation for directory: $pretrain_path"
    # python -m eval.eval_ppl_hf \
    #     --hf_path $pretrain_path \
    #     --seqlen 2048 \
    #     --output_path $output_path \
    #     --no_use_cuda_graph 2>&1 | tee -a $LOG/$SAVE_NAME.log
        # --datasets c4 \

    echo "################## Running benchmark evaluation lmbda=${lmbda} ##################"
    python -m eval.eval_zeroshot_hf \
        --tasks arc_challenge,arc_easy,boolq,piqa,winogrande \
        --batch_size 4  \
        --hf_path $pretrain_path \
        --output_path $output_path


    # echo "################## Running PPL evaluation bit=${b} ##################"
    # echo "Running evaluation for directory: $pretrain_path"
    # python -m eval.eval_ppl_hf \
    #     --hf_path $pretrain_path \
    #     --seqlen 2048 \
    #     --dataset c4 \
    #     --no_use_cuda_graph

    # # echo "################## Running benchmark evaluation bit=${b} ##################"
    # # output_path=$(echo "$pretrain_path" | sed 's|model_reconstructed|model_eval|')_harness_results
    # # lm_eval --model hf \
    # #     --model_args "pretrained=$pretrain_path,parallelize=True" \
    # #     --tasks arc_easy,arc_challenge,winogrande,piqa,boolq \
    # #     --batch_size 8 \
    # #     --output_path $output_path \
    # #     --trust_remote_code \
    # #     --seed $b

    # python -m eval.eval_zeroshot_hf --tasks arc_challenge,arc_easy,boolq,piqa,winogrande \
    #     --batch_size 4  --hf_path $pretrain_path \

done