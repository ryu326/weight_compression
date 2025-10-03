import argparse
import json
import math
import os
import random

import sys
import os

import glog
import torch
from tqdm import tqdm
import json

import gptq_data_utils

import transformers

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--hf_path', default='hfized/quantized_hada_70b', type=str)
parser.add_argument('--seqlen', default=4096, type=int)
parser.add_argument('--no_use_cuda_graph', action='store_true')
parser.add_argument('--no_use_flash_attn', action='store_true')


def main(args):
    # datasets = ['wikitext2', 'c4']
    datasets = ['wikitext2']
    model = transformers.AutoModelForCausalLM.from_pretrained(
            args.hf_path, trust_remote_code=True, torch_dtype="auto", cache_dir='./aqlm_cache',
        ).cuda()
    
    if "2-7b" in args.hf_path.lower():
        model_str = "../Wparam_dataset/hf_model/meta-llama--Llama-2-7b-hf"
    elif "2-13b" in args.hf_path.lower():
        model_str = "../Wparam_dataset/hf_model/meta-llama--Llama-2-13b-hf"
    elif "3-8b" in args.hf_path.lower():
        model_str = "../Wparam_dataset/hf_model/meta-llama--Meta-Llama-3-8B"

    for dataset in datasets:
        input_tok = gptq_data_utils.get_test_tokens(dataset,
                                                    seed=args.seed,
                                                    seqlen=args.seqlen,
                                                    model=model_str)
        nsamples = input_tok.numel() // args.seqlen
        input_tok = input_tok[0, :(args.seqlen * nsamples)].view(
            nsamples, args.seqlen)

        if not args.no_use_cuda_graph:
            model.reset()

        loss_fct = torch.nn.CrossEntropyLoss().cuda()
        acc_loss = 0.0
        progress = tqdm(range(nsamples))
        for ii in progress:
            input = input_tok[ii, :].cuda().view(1, -1)
            output = model(input,
                           use_cache=False,
                           output_hidden_states=False,
                           output_attentions=False)[0]
            shift_logits = output[:, :-1, :].contiguous()
            shift_labels = input[:, 1:]
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))
            acc_loss += loss.item()
            progress.set_description(f"avg_loss = {acc_loss/(ii+1)}")

        avg_loss = acc_loss / nsamples

        ppl = torch.exp(torch.tensor(avg_loss)).item()
        glog.info(f'{dataset} perplexity: {ppl}')
        print(f'{dataset} perplexity: {ppl:.3f}')
        
        try:
            with open(f'{args.hf_path}_result.json', 'r') as f:
                comp_result= json.load(f)
        except:
            comp_result = {}
        comp_result['ppl'] = {dataset: ppl}
        with open(f'./{args.hf_path}_result.json', 'w') as f:
            json.dump(comp_result, f, indent=4)

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    main(args)
