

# CUDA_VISIBLE_DEVICES=1,2,3 lm_eval --model hf \
#     --model_args pretrained=/home/jgryu/Weight_compression/model_cache/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920,parallelize=True \
#     --tasks hellaswag \
#     --batch_size 1

CUDA_VISIBLE_DEVICES=0,1,2,3 lm_eval --model hf \
    --model_args pretrained=/home/jgryu/Weight_compression/model_cache_reconstructed/nic/meta-llama--Meta-Llama-3-8B_attn_d256_256,parallelize=True \
    --tasks arc_easy,arc_challenge,hellaswag,winogrande,boolq \
    --batch_size 1


##### original
## wikitext
# | Tasks  |Version|Filter|n-shot|    Metric     |   |Value |   |Stderr|
# |--------|------:|------|-----:|---------------|---|-----:|---|------|
# |wikitext|      2|none  |     0|bits_per_byte  |↓  |0.5348|±  |   N/A|
# |        |       |none  |     0|byte_perplexity|↓  |1.4488|±  |   N/A|
# |        |       |none  |     0|word_perplexity|↓  |7.2602|±  |   N/A|

## hellaswag
# |  Tasks  |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
# |---------|------:|------|-----:|--------|---|-----:|---|-----:|
# |hellaswag|      1|none  |     0|acc     |↑  |0.6017|±  |0.0049|
# |         |       |none  |     0|acc_norm|↑  |0.7907|±  |0.0041|


##### Reconstruncted
## wikitext

# | Tasks  |Version|Filter|n-shot|    Metric     |   |   Value    |   |Stderr|
# |--------|------:|------|-----:|---------------|---|-----------:|---|------|
# |wikitext|      2|none  |     0|bits_per_byte  |↓  |      4.2705|±  |   N/A|
# |        |       |none  |     0|byte_perplexity|↓  |     19.2999|±  |   N/A|
# |        |       |none  |     0|word_perplexity|↓  |7489308.9832|±  |   N/A|

## hellaswag
# |  Tasks  |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
# |---------|------:|------|-----:|--------|---|-----:|---|-----:|
# |hellaswag|      1|none  |     0|acc     |↑  |0.2569|±  |0.0044|
# |         |       |none  |     0|acc_norm|↑  |0.2660|±  |0.0044|