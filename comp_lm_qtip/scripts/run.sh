# nohup bash scripts/comp_lm.sh > ./log/comp_lm.out 2>&1 &
# nohup bash scripts/comp_lm1.sh > ./log/comp_lm1.out 2>&1 &
# nohup bash scripts/comp_lm2.sh > ./log/comp_lm2.out 2>&1 &
# nohup bash scripts/comp_lm3.sh > ./log/comp_lm3.out 2>&1 &

nohup bash scripts/comp_lm_qlsearch.sh > ./log/comp_lm.out 2>&1 &
nohup bash scripts/comp_lm_qlsearch1.sh > ./log/comp_lm1.out 2>&1 &
nohup bash scripts/comp_lm_qlsearch2.sh > ./log/comp_lm2.out 2>&1 &

# export CUDA_VISIBLE_DEVICES=7
# # python -m eval.eval_ppl \
# #        --hf_path /workspace/Weight_compression/Wparam_dataset/hf_model/meta-llama--Llama-3.2-3B \
# #        --seqlen 2048 \
# #        --no_use_cuda_graph

# # python -m eval.eval_ppl \
# #        --hf_path /workspace/Weight_compression/Wparam_dataset/hf_model/meta-llama--Meta-Llama-3-8B \
# #        --seqlen 2048 \
# #        --no_use_cuda_graph

# # echo "################## Running benchmark evaluation lmbda=${lmbda} ##################"
# pretrain_path=/workspace/Weight_compression/Wparam_dataset/hf_model/meta-llama--Meta-Llama-3-8B
# lm_eval --model hf \
#     --model_args "pretrained=$pretrain_path,parallelize=True" \
#     --tasks arc_easy,arc_challenge,winogrande,piqa,boolq \
#     --batch_size 2 \
#     --output_path /workspace/Weight_compression/Wparam_dataset/hf_model/meta-llama--Meta-Llama-3-8B_harness_result \
#     --trust_remote_code 