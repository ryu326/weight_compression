# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
from logging import Logger

import torch
import torch.distributed as dist
from transformers import LlamaTokenizerFast, AutoTokenizer
import transformers
from eval_utils.main import ptq_model
from eval_utils.modeling_llama import LlamaForCausalLM
from utils import data_utils, eval_utils, utils
from utils.process_args import process_args_ptq

from lm_eval import evaluator, tasks
from lm_eval.models.huggingface import HFLM

import os
os.sys.path.append(os.path.dirname('os.path.abspath(__file__))'))
import eval.gptq_data_utils as gptq_data_utils
import json
from tqdm import tqdm

log: Logger = utils.get_logger("spinquant")


def train() -> None:
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8))
    model_args, training_args, ptq_args = process_args_ptq()
    local_rank = utils.get_local_rank()

    log.info("the rank is {}".format(local_rank))
    torch.distributed.barrier()

    config = transformers.AutoConfig.from_pretrained(
        model_args.input_model, token=model_args.access_token
    )
    # Llama v3.2 specific: Spinquant is not compatiable with tie_word_embeddings, clone lm_head from embed_tokens
    process_word_embeddings = False
    if config.tie_word_embeddings:
        config.tie_word_embeddings = False
        process_word_embeddings = True
    dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        config=config,
        torch_dtype=dtype,
        token=model_args.access_token,
    )
    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()
    model.cuda()

    model = ptq_model(ptq_args, model, model_args)
    model.seqlen = training_args.model_max_length

    
    model.cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_args.input_model)
    
    lm_eval_model = HFLM(model,
                         tokenizer=tokenizer,
                         batch_size=4)

    limit = None
    num_fewshot = 0
    apply_chat_template = False
    fewshot_as_multiturn = False
    # task_names = "arc_challenge,arc_easy,boolq,piqa,winogrande".split(",")
    task_names = "hellaswag,mmlu".split(",")

    results = evaluator.simple_evaluate(
        model=lm_eval_model,
        tasks=task_names,
        limit=limit,
        num_fewshot=num_fewshot,
        apply_chat_template=apply_chat_template,
        fewshot_as_multiturn=fewshot_as_multiturn)

    for key in results['results']:
        print(key)
        print()
        print(results['results'][key])
        print()
        print()

    def convert(o):
        if isinstance(o, torch.dtype):
            return str(o)
        if isinstance(o, torch.Tensor):
            return o.tolist()
        return None 

    if local_rank == 0:
        result_path = ptq_args.save_qmodel_path if ptq_args.save_qmodel_path not in [None, ''] else ptq_args.load_qmodel_path
        result_path = result_path.split('.pth')[0] + f'_hs_mmlu_zeroshot_results.json'
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        results["config"]["model"] = ptq_args.save_qmodel_path
        if "samples" in results:
            del results["samples"]
        with open(result_path, "w") as f:
            json.dump(results, f, indent=2, default=convert)
            
            
    # if local_rank == 0:
            
    # datasets = ['wikitext2', 'c4']
    # seed = 0
    # seqlen = training_args.model_max_length
    # seqlen = 2048
    # for dataset in datasets:
    #     try:
    #         input_tok = gptq_data_utils.get_test_tokens(dataset,
    #                                                     seed=seed,
    #                                                     seqlen=seqlen,
    #                                                     model=model_args.input_model)
    #     except Exception as e:
    #         print(f"Error loading dataset {dataset}: {e}")
    #         input_tok = gptq_data_utils.get_test_tokens(dataset,
    #                                                     seed=seed,
    #                                                     seqlen=seqlen,
    #                                                     model="../Wparam_dataset/hf_model/meta-llama--Meta-Llama-3-8B")
    #     nsamples = input_tok.numel() // seqlen
    #     input_tok = input_tok[0, :(seqlen * nsamples)].view(
    #         nsamples, seqlen)

    #     loss_fct = torch.nn.CrossEntropyLoss().cuda()
    #     acc_loss = 0.0
    #     progress = tqdm(range(nsamples))
    #     for ii in progress:
    #         input = input_tok[ii, :].cuda().view(1, -1)
    #         output = model(input,
    #                     use_cache=False,
    #                     output_hidden_states=False,
    #                     output_attentions=False)[0]
    #         shift_logits = output[:, :-1, :].contiguous()
    #         shift_labels = input[:, 1:]
    #         loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
    #                         shift_labels.view(-1))
    #         acc_loss += loss.item()
    #         progress.set_description(f"avg_loss = {acc_loss/(ii+1)}")

    #     avg_loss = acc_loss / nsamples

    #     ppl = torch.exp(torch.tensor(avg_loss)).item()
    #     print(f'{dataset} perplexity: {ppl:.3f}')
        
    #     if local_rank == 0:
    #         result_path = ptq_args.save_qmodel_path if ptq_args.save_qmodel_path not in [None, ''] else ptq_args.load_qmodel_path
    #         result_path = result_path.split('.pth')[0] + f'_ppl_results.json'
            
    #         try:
    #             with open(result_path, 'r') as f:
    #                 comp_result= json.load(f)
    #         except:
    #             comp_result = {}
    #             comp_result['ppl'] = {}
    #         if not isinstance(comp_result['ppl'], dict):
    #             comp_result['ppl'] = {}
    #         comp_result['ppl'][dataset] = ppl
            
    #         with open(result_path, 'w') as f:
    #             json.dump(comp_result, f, indent=4)

if __name__ == "__main__":
    train()
