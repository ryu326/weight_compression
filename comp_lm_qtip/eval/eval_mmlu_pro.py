#!/usr/bin/env python3
"""MMLU-Pro evaluation with category-balanced sampling.

Loads TIGER-Lab/MMLU-Pro test split, stratified-samples N_PER_CAT examples
per category (default 143 → ~2002 total, all 14 categories represented), runs
inference via vLLM, and scores with extractive_match.

Output: JSON array at <output_dir>/MMLU-PRO.jsonl
Format matches paroquant lighteval output so notebooks load it transparently.

Usage:
    python eval/eval_mmlu_pro.py \\
        --model /path/to/hf_model \\
        --output-dir /results/model/reasoning \\
        --gpus 0,1,2,3 \\
        [--n-per-cat 143] [--seed 42] [--overwrite]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path

LETTERS = list("ABCDEFGHIJ")

SYSTEM_PROMPT = (
    "Answer the following multiple choice question. "
    "Think step by step before answering. "
    "The last line of your response should be of the following format: "
    "'Answer: $LETTER' (without quotes) where LETTER is one of ABCDEFGHIJ."
)


def _build_prompt(tokenizer, question: str, options: list[str]) -> str:
    opts_str = "\n".join(f"{LETTERS[i]}) {opt}" for i, opt in enumerate(options))
    content = f"{SYSTEM_PROMPT}\n\n{question}\n\n{opts_str}"
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False, add_generation_prompt=True,
            enable_thinking=True,
        )
    except TypeError:
        pass
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True,
    )


def _extractive_match(pred: str, gold_letter: str) -> float:
    """Return 1.0 if 'Answer: {gold}' appears in pred (case-insensitive)."""
    pattern = rf"answer\s*:\s*{re.escape(gold_letter)}\b"
    return 1.0 if re.search(pattern, pred, re.IGNORECASE) else 0.0


def _stratified_sample(dataset, n_per_cat: int, seed: int):
    rng = random.Random(seed)
    by_cat: dict[str, list] = {}
    for ex in dataset:
        by_cat.setdefault(ex["category"], []).append(ex)
    selected = []
    for cat, exs in sorted(by_cat.items()):
        rng.shuffle(exs)
        selected.extend(exs[:n_per_cat])
    rng.shuffle(selected)
    return selected


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model",       required=True)
    p.add_argument("--output-dir",  required=True)
    p.add_argument("--gpus",        default="0,1,2,3,4,5,6,7")
    p.add_argument("--n-per-cat",   type=int, default=143,
                   help="Samples per category (143 × 14 = 2002 total)")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--max-tokens",  type=int, default=4096)
    p.add_argument("--overwrite",   action="store_true")
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "MMLU-PRO.jsonl"

    if out_path.exists() and not args.overwrite:
        print(f"[skip] {out_path} exists")
        return

    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    print("Loading TIGER-Lab/MMLU-Pro ...", flush=True)
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test", trust_remote_code=True)
    samples = _stratified_sample(ds, args.n_per_cat, args.seed)

    cats = {}
    for s in samples:
        cats[s["category"]] = cats.get(s["category"], 0) + 1
    print(f"  {len(samples)} samples across {len(cats)} categories:")
    for c, n in sorted(cats.items()):
        print(f"    {c:25s}: {n}")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tp  = len(args.gpus.split(","))

    llm = LLM(
        model=args.model, dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=32768,
        tensor_parallel_size=tp,
        trust_remote_code=True,
    )

    prompts = [_build_prompt(tok, s["question"], s["options"]) for s in samples]

    print(f"Running inference on {len(prompts)} samples ...", flush=True)
    sp = SamplingParams(temperature=0, max_tokens=args.max_tokens)
    outputs = llm.generate(prompts, sp)
    preds = [o.outputs[0].text.strip() for o in outputs]

    results = []
    for s, prompt, pred in zip(samples, prompts, preds):
        gold_letter = LETTERS[s["answer_index"]]
        score = _extractive_match(pred, gold_letter)
        results.append({
            "full_prompt":    prompt,
            "generated_text": pred,
            "gold":           [gold_letter],
            "metrics":        {"extractive_match": score},
            "category":       s["category"],
        })

    json.dump(results, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    # Summary
    by_cat_scores: dict[str, list] = {}
    for r in results:
        by_cat_scores.setdefault(r["category"], []).append(r["metrics"]["extractive_match"])
    overall = sum(r["metrics"]["extractive_match"] for r in results) / len(results) * 100

    print(f"\n{'='*50}")
    print(f"MMLU-Pro (balanced)  |  overall = {overall:.2f}%  (n={len(results)})")
    print(f"{'='*50}")
    for cat, scores in sorted(by_cat_scores.items()):
        avg = sum(scores) / len(scores) * 100
        print(f"  {cat:25s}: {avg:.1f}%  (n={len(scores)})")
    print(f"{'='*50}")
    print(f"Saved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
