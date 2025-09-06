from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
import os
from tqdm import tqdm
import glog
import json
import gc

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import numpy as np
import torch

from huggingface_hub import login
login(token = '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')

def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_wikitext2(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_ptb(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    valdata = load_dataset('ptb_text_only',
                           'penn_treebank',
                           split='validation')

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer("\n\n".join(traindata['sentence']),
                         return_tensors='pt')
    testenc = tokenizer("\n\n".join(valdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset(
        'allenai/c4',
        data_files={'train': 'en/c4-train.00000-of-01024.json.gz'},
        split='train')
    valdata = load_dataset(
        'allenai/c4',
        data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'},
        split='validation')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    import random
    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)

    class TokenizerWrapper:

        def __init__(self, input_ids):
            self.input_ids = input_ids

    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_ptb_new(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4_new(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset(
        'allenai/c4',
        data_files={'train': 'en/c4-train.00000-of-01024.json.gz'},
        split='train')
    valdata = load_dataset(
        'allenai/c4',
        data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'},
        split='validation')

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:

        def __init__(self, input_ids):
            self.input_ids = input_ids

    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_loaders(name, nsamples=128, seed=0, seqlen=2048, model=''):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model)
    if 'ptb' in name:
        if 'new' in name:
            return get_ptb_new(nsamples, seed, seqlen, model)
        return get_ptb(nsamples, seed, seqlen, model)
    if 'c4' in name:
        if 'new' in name:
            return get_c4_new(nsamples, seed, seqlen, model)
        return get_c4(nsamples, seed, seqlen, model)


def get_test_tokens(name, seed=0, seqlen=2048, model=''):
    train_samples = 0
    if name == 'wikitext2':
        return get_wikitext2(train_samples, seed, seqlen,
                             model)[1]['input_ids']
    elif name == 'c4':
        return get_c4(train_samples, seed, seqlen, model)[1].input_ids
    elif name == 'c4_new':
        return get_c4_new(train_samples, seed, seqlen, model)[1].input_ids
    else:
        raise Exception


model_ids = [
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Llama-2-7b-hf",
]
bits_list = [
    [3,4,8],
    [2,3],
    [8],
]
for model_id, bits in zip(model_ids, bits_list):
    datasets = ['wikitext2']
    seqlen = 2048
    seed = 0

    for bit in bits:
        try:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

            tokenizer = AutoTokenizer.from_pretrained(model_id)
            gptq_config = GPTQConfig(bits=bit, dataset="c4", tokenizer=tokenizer)
            quantized_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", quantization_config=gptq_config)

            if "2-7b" in model_id.lower():
                hf_path = f"./hf/meta-llama--Llama-2-7b-hf/{bit}bit"
            elif "2-13b" in model_id.lower():
                hf_path = f"./hf/meta-llama--Llama-2-13b-hf/{bit}bit"
            elif "3-8b" in model_id.lower():
                hf_path = f"./hf/meta-llama--Meta-Llama-3-8B/{bit}bit"

            # quantized_model.to('cpu')
            # quantized_model.save_pretrained(hf_path)
            # quantized_model.to('cuda')

            for dataset in datasets:
                input_tok = get_test_tokens(dataset,seed=seed, seqlen=seqlen, model=model_id)
                nsamples = input_tok.numel() // seqlen
                input_tok = input_tok[0, :(seqlen * nsamples)].view(
                    nsamples, seqlen)

                loss_fct = torch.nn.CrossEntropyLoss().cuda()
                acc_loss = 0.0
                progress = tqdm(range(nsamples))
                for ii in progress:
                    input = input_tok[ii, :].cuda().view(1, -1)
                    output = quantized_model(input,
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
                    with open(f'{hf_path}_result.json', 'r') as f:
                        comp_result= json.load(f)
                except:
                    comp_result = {}
                comp_result['ppl'] = {dataset: ppl}
                with open(f'./{hf_path}_result.json', 'w') as f:
                    json.dump(comp_result, f, indent=4)

        except Exception as e:
            glog.error(f"Error: {e}")
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            continue