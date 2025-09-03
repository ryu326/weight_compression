import logging
from math import sqrt
from typing import Literal

import torch
from tqdm import tqdm
import wandb

logger = logging.getLogger()


def embed_texts(texts, emb_model, emb_tokenizer, task_desc_format_fn, pooling_fn, device, batch_size=None):
    formatted_descs = list(map(task_desc_format_fn, texts))
    tokenized_ds_descs = emb_tokenizer(
        formatted_descs,
        truncation=True,
        padding=True,
        max_length=2**13,
        return_tensors="pt",
    )
    return embed_tokens(tokenized_ds_descs, emb_model, pooling_fn, device, batch_size)


def embed_tokens(tokenized_texts, emb_model, pooling_fn, device, batch_size=None):
    if batch_size is None:
        # Process all at once if no batch size specified
        tokenized_texts = {k: v.to(device) for k, v in tokenized_texts.items()}
        return _embed_tokens_single_batch(tokenized_texts, emb_model, pooling_fn)

    # Process in batches
    n_samples = tokenized_texts["input_ids"].shape[0]
    embeddings = []

    for start_idx in tqdm(range(0, n_samples, batch_size), total=n_samples // batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch = {k: v[start_idx:end_idx].to(device) for k, v in tokenized_texts.items()}
        batch_embeddings = _embed_tokens_single_batch(batch, emb_model, pooling_fn)
        embeddings.append(batch_embeddings)

    return torch.cat(embeddings, dim=0)


def _embed_tokens_single_batch(tokenized_texts, emb_model, pooling_fn):
    outputs = emb_model(**tokenized_texts, output_hidden_states=True)
    task_embs = pooling_fn(outputs, tokenized_texts["attention_mask"]).to(torch.float32)
    return torch.nn.functional.normalize(task_embs) * sqrt(task_embs.shape[-1])


def get_inp_tokenize_fn(
    tokenizer,
    sft_mode: Literal["causal_lm", "completion"],
    is_intx_model: bool,
    inp_max_len: int,
):
    def tokenize_causal_lm(examples):
        # a dict with keys: ["input_ids", "attention_mask"]
        tokenized_seq = tokenizer(
            examples["text"],
            # apply_chat_template should already add all the special tokens
            add_special_tokens=True if not is_intx_model else False,
            truncation=True,
            padding=False,
            max_length=inp_max_len,
        )
        tokenized_seq["labels"] = tokenized_seq["input_ids"]
        return tokenized_seq

    # NOTE: we're not considering multi-turn sft
    # this fn is used to mask out the loss from the prompt
    # and train only on the response
    # see # see https://github.com/huggingface/trl/issues/632#issuecomment-1972630547
    # https://github.com/huggingface/notebooks/blob/main/examples/question_answering.ipynb
    # for more advanced multi-turn training
    def tokenize_prompt_completion(examples):
        # a dict with keys: ["input_ids", "attention_mask"]
        # we can also access seqeunce_ids to differentiate between prompt and response
        tokenized_seq = tokenizer(
            examples["prompt"],
            examples["response"],
            add_special_tokens=False,
            truncation=True,
            padding=False,
            # apply to prompt and response separately
            # i.e., we can get the max sequence length of 2 x inp_max_len
            max_length=inp_max_len,
        )

        tokenized_seq["labels"] = [None] * len(tokenized_seq["input_ids"])
        input_ids = tokenized_seq["input_ids"]
        attention_mask = tokenized_seq["attention_mask"]
        labels = tokenized_seq["labels"]
        for i in range(len(tokenized_seq["input_ids"])):
            if not is_intx_model:
                # manually add bos and eos tokens
                input_ids[i] = [tokenizer.bos_token_id] + input_ids[i] + [tokenizer.eos_token_id]
                attention_mask[i] = [1] + attention_mask[i] + [1]
                sequence_ids = [0] + tokenized_seq.sequence_ids(i) + [1]
            else:
                sequence_ids = tokenized_seq.sequence_ids(i)
            labels[i] = [-100 if sequence_id == 0 else label for sequence_id, label in zip(sequence_ids, input_ids[i])]
        return tokenized_seq

    tokenize_function = tokenize_causal_lm if sft_mode == "causal_lm" else tokenize_prompt_completion
    return tokenize_function


def log_scalar(metric_name, val, curstep):
    if wandb.run is not None:
        wandb.log({metric_name: val}, step=curstep)
    logger.info(f"{metric_name}: {val:.4f}")
