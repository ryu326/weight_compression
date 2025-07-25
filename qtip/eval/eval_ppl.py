import argparse
import json
import math
import os
import random

import datasets
import glog
import torch
from tqdm import tqdm

from lib.linear import QuantizedLinear
from lib.utils import gptq_data_utils
from lib.utils.unsafe_import import model_from_hf_path

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--hf_path', default='hfized/quantized_hada_70b', type=str)
parser.add_argument('--seqlen', default=4096, type=int)
parser.add_argument('--manifest', action='store_true')
parser.add_argument('--max_mem_ratio', default=0.7, type=float)
parser.add_argument("--output_path", default=None, type=str)


def main(args):
    # datasets = ['wikitext2', 'c4', 'ptb']
    datasets = ['wikitext2','c4']
    model, model_str = model_from_hf_path(args.hf_path, max_mem_ratio=args.max_mem_ratio)

    if args.manifest:
        # manifest the model in BF/FP16 for faster inference
        # useful for non-kernel supported decode modes
        for module in model.modules():
            if isinstance(module, QuantizedLinear):
                module.mode = 'train-fixW'
    result = {}
    for dataset in datasets:
        input_tok = gptq_data_utils.get_test_tokens(dataset,
                                                    seed=args.seed,
                                                    seqlen=args.seqlen,
                                                    model=model_str)
        nsamples = input_tok.numel() // args.seqlen
        input_tok = input_tok[0, :(args.seqlen * nsamples)].view(
            nsamples, args.seqlen)

        loss_fct = torch.nn.CrossEntropyLoss().cuda()
        acc_loss = 0.0
        progress = tqdm(range(nsamples))
        for ii in progress:
            input = input_tok[ii, :].to(model.device).view(1, -1)
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
        result[dataset] = ppl

        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(f'{args.output_path}_ppl_result.json', 'w') as f:
            json.dump(result, f, indent=2)

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    main(args)
