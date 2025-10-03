import argparse
import json
import os
import random

import datasets
import glog
import torch
from lm_eval import evaluator, tasks
from lm_eval.models.huggingface import HFLM
from transformers import AutoTokenizer
import transformers
import accelerate

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

parser = argparse.ArgumentParser()
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


def model_from_hf_path(path, max_mem_ratio=0.7, device_map=None):
    model_cls = transformers.AutoModelForCausalLM
    model_str = path

    if device_map is None:
        mmap = {
            i: f"{torch.cuda.mem_get_info(i)[1]*max_mem_ratio/(1 << 30)}GiB"
            for i in range(torch.cuda.device_count())
        }
        model = model_cls.from_pretrained(path,
                                          torch_dtype='auto',
                                          low_cpu_mem_usage=True,
                                          attn_implementation='sdpa')
        device_map = accelerate.infer_auto_device_map(
            model,
            no_split_module_classes=['LlamaDecoderLayer'],
            max_memory=mmap)
    model = model_cls.from_pretrained(path,
                                      torch_dtype='auto',
                                      low_cpu_mem_usage=True,
                                      attn_implementation='sdpa',
                                      device_map=device_map)

    return model, model_str


def main(args):
    model, model_str = model_from_hf_path(args.hf_path, max_mem_ratio=args.max_mem_ratio, device_map='balanced')

    # manifest for faster inference
    # use for codebooks without kernel support
    if args.manifest_model:
        for module in model.modules():
            if isinstance(module, QuantizedLinear):
                module.mode = 'train-fixW'

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_str)
    except Exception as e:
        print(f"Error loading Tokenizer: {e}")
        if '8b' in model_str.lower():
            model_str_tmp = "../Wparam_dataset/hf_model/meta-llama--Meta-Llama-3-8B"
        elif '7b' in  model_str.lower():
            model_str_tmp = "../Wparam_dataset/hf_model/meta-llama--Llama-2-7b-hf"
        elif '13b' in model_str.lower():
            model_str_tmp = "../Wparam_dataset/hf_model/meta-llama--Llama-2-13b-hf"
        tokenizer = AutoTokenizer.from_pretrained(model_str_tmp)

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
    
    if args.output_path is None:
        args.output_path = args.hf_path + '_zeroshot_results0.4.4.json'
    else:
        args.output_path = args.output_path + '_zeroshot_results0.4.4.json'
        
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    results["config"]["model"] = args.hf_path
    if "samples" in results:
        del results["samples"]
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2, default=convert)

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    main(args)