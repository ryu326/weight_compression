import argparse
import json
import os
import random

import datasets
import glog
import torch
from lm_eval import evaluator, tasks
from transformers import AutoTokenizer

from lib.utils import LMEvalAdaptor
from lm_eval.models.huggingface import HFLM
from lib.utils.unsafe_import import model_from_hf_path
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

parser = argparse.ArgumentParser()
# parser.add_argument('--seed', default=0, type=int)
# parser.add_argument('--hf_path', default='hfized/quantized_hada_70b', type=str)
# parser.add_argument('--batch_size', type=int, default=1, help='batch size')
# parser.add_argument("--tasks", type=str)
# parser.add_argument("--output_path", default=None, type=str)
# parser.add_argument('--num_fewshot', type=int, default=0)
parser.add_argument('--no_use_cuda_graph', action='store_true', default=True)
parser.add_argument('--no_use_flash_attn', action='store_true', default=False)

parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--hf_path', default='hfized/quantized_hada_70b', type=str)
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument("--tasks", type=str)
parser.add_argument("--output_path", default=None, type=str)
parser.add_argument('--num_fewshot', type=int, default=0)
parser.add_argument('--limit', type=int, default=None)
parser.add_argument('--apply_chat_template', action='store_true')
parser.add_argument('--fewshot_as_multiturn', action='store_true')
parser.add_argument('--manifest_model', action='store_true')
parser.add_argument('--max_mem_ratio', type=float, default=0.7)

def main(args):
    model, model_str = model_from_hf_path(
        args.hf_path,
        use_cuda_graph=False,
        use_flash_attn=not args.no_use_flash_attn)
    tokenizer = AutoTokenizer.from_pretrained(model_str)

    glog.info('loaded model!')
    tokenizer.pad_token = tokenizer.eos_token

    task_names = args.tasks.split(",")

    lm_eval_model = HFLM(model,
                    tokenizer=tokenizer,
                    batch_size=args.batch_size)
    results = evaluator.simple_evaluate(
        model=lm_eval_model,
        tasks=task_names,
        limit=args.limit,
        num_fewshot=args.num_fewshot,
        apply_chat_template=args.apply_chat_template,
        fewshot_as_multiturn=args.fewshot_as_multiturn)
    
    # lm_eval_model = LMEvalAdaptor(model_str, model, tokenizer, args.batch_size)
    # results = evaluator.simple_evaluate(
    #     model=lm_eval_model,
    #     tasks=task_names,
    #     batch_size=args.batch_size,
    #     no_cache=True,
    #     num_fewshot=args.num_fewshot,
    # )

    # print(evaluator.make_table(results))
    
    for key in results['results']:
        print(key)
        print()
        print(results['results'][key])
        print()
        print()

    try:
        if args.output_path is not None:
            torch.save(results, args.output_path)

        if args.output_path is not None:
            os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
            # otherwise cannot save
            results["config"]["model"] = args.hf_path
            with open(args.output_path, "w") as f:
                json.dump(results, f, indent=2)
    except:
        pass

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    main(args)
