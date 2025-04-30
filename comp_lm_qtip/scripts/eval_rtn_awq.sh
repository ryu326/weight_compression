export CUDA_VISIBLE_DEVICES=0

bit=(4)
for b in "${bit[@]}"; do

    # pretrain_path=../hf_model_comp/RTN/meta-llama--Meta-Llama-3-8B_W${b}g128
    # pretrain_path=../hf_model_comp/awq/meta-llama--Meta-Llama-3-8B/w${b}-g128-fake-quantized
    pretrain_path=/workspace/Weight_compression/Wparam_dataset/hf_model/meta-llama--Meta-Llama-3-8B

    echo "################## Running PPL evaluation bit=${b} ##################"
    echo "Running evaluation for directory: $pretrain_path"
    python -m eval.eval_ppl_hf \
        --hf_path $pretrain_path \
        --seqlen 2048 \
        --no_use_cuda_graph

    # echo "################## Running benchmark evaluation bit=${b} ##################"
    # output_path=$(echo "$pretrain_path" | sed 's|model_reconstructed|model_eval|')_harness_results
    # lm_eval --model hf \
    #     --model_args "pretrained=$pretrain_path,parallelize=True" \
    #     --tasks arc_easy,arc_challenge,winogrande,piqa,boolq \
    #     --batch_size 8 \
    #     --output_path $output_path \
    #     --trust_remote_code \
    #     --seed $b

    # python -m eval.eval_zeroshot_hf --tasks arc_challenge,arc_easy,boolq,piqa,winogrande \
    #     --batch_size 4  --hf_path $pretrain_path \

done