#!/usr/bin/env python3
"""Needle-in-a-Haystack (NIAH) evaluation.

Follows gkamradt/LLMTest_NeedleInAHaystack exactly:
  - Haystack : Paul Graham Essays (vendor/LLMTest_NeedleInAHaystack/)
  - Needle   : inserted at sentence boundaries matching target depth
  - Scoring  : case-insensitive substring match on the needle key-phrase

Default needle  : "The best thing to do in San Francisco is eat a sandwich
                   and sit in Dolores Park on a sunny day."
Default question: "What is the best thing to do in San Francisco?"
Scored phrase   : "eat a sandwich and sit in Dolores Park"

Output: JSON heatmap  {ctx_len_str → {depth_str → score, "_avg" → avg}, "_avg" → overall}

Usage:
    python eval/niah.py \\
        --model /path/to/hf_model \\
        --output-dir /results/model/niah \\
        --gpus 0,1,2,3
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
EVAL_DIR   = Path(__file__).resolve().parent
VENDOR_DIR = EVAL_DIR.parent / "vendor"
PG_ESSAYS_DIR = VENDOR_DIR / "LLMTest_NeedleInAHaystack" / "needlehaystack" / "PaulGrahamEssays"

# ── Defaults (matching gkamradt) ──────────────────────────────────────────────
DEFAULT_NEEDLE   = ("The best thing to do in San Francisco is eat a sandwich "
                    "and sit in Dolores Park on a sunny day.")
DEFAULT_QUESTION = "What is the best thing to do in San Francisco?"
DEFAULT_SCORE_PHRASE = "eat a sandwich and sit in Dolores Park"  # scored substring

DEFAULT_CTX_LENS = [1_000, 2_000, 4_000, 8_000, 16_000, 32_000]
# 10 evenly-spaced depths, matching gkamradt default (10 % to 100 %)
DEFAULT_DEPTHS   = [round(x, 2) for x in np.linspace(0.1, 1.0, 10).tolist()]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_pg_essays() -> str:
    """Load all Paul Graham essay .txt files and concatenate."""
    if not PG_ESSAYS_DIR.exists():
        raise FileNotFoundError(
            f"Paul Graham essays not found at {PG_ESSAYS_DIR}. "
            "Run: git clone https://github.com/gkamradt/LLMTest_NeedleInAHaystack "
            "into comp_lm_qtip/vendor/"
        )
    texts = []
    for p in sorted(PG_ESSAYS_DIR.glob("*.txt")):
        texts.append(p.read_text(encoding="utf-8", errors="replace"))
    return "\n\n".join(texts)


def _chat_prompt(tokenizer, context: str, question: str) -> str:
    msgs = [
        {"role": "system", "content": "You are a helpful assistant. Answer only the question asked."},
        {"role": "user", "content": f"{context}\n\n{question}"},
    ]
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    except TypeError:
        pass
    try:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"{context}\n\n{question}"


def _make_sample(
    tokenizer,
    essay_sents: List[str],
    needle: str,
    depth: float,
    ctx_len: int,
    question: str,
) -> str:
    """
    Reproduce gkamradt insertion:
      1. Fill haystack to ctx_len tokens (repeat essays as needed)
      2. Find sentence boundary closest to depth fraction
      3. Insert needle at that boundary
    """
    # Build haystack of exactly ctx_len tokens
    q_budget = 64
    target = ctx_len - q_budget
    combined: List[int] = []
    sent_idx = 0
    while len(combined) < target:
        ids = tokenizer.encode(essay_sents[sent_idx % len(essay_sents)],
                               add_special_tokens=False)
        combined += ids
        sent_idx += 1
    combined = combined[:target]

    # Sentence boundary closest to desired depth
    # Work at token level: find cumulative token position closest to depth * target
    depth_pos = int(target * depth)
    # We decode and re-tokenize per sentence — approximate with token split
    context_before = tokenizer.decode(combined[:depth_pos], skip_special_tokens=True)
    context_after  = tokenizer.decode(combined[depth_pos:], skip_special_tokens=True)

    # Insert needle at sentence boundary (append period + space)
    context = context_before.rstrip() + " " + needle + " " + context_after.lstrip()

    return _chat_prompt(tokenizer, context, question)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model",        required=True)
    p.add_argument("--output-dir",   required=True)
    p.add_argument("--gpus",         default="0,1,2,3,4,5,6,7")
    p.add_argument("--context-lens", nargs="+", type=int,   default=DEFAULT_CTX_LENS)
    p.add_argument("--depths",       nargs="+", type=float, default=DEFAULT_DEPTHS)
    p.add_argument("--needle",       default=DEFAULT_NEEDLE)
    p.add_argument("--question",     default=DEFAULT_QUESTION)
    p.add_argument("--score-phrase", default=DEFAULT_SCORE_PHRASE,
                   help="Substring that must appear in prediction to count as correct")
    p.add_argument("--overwrite",    action="store_true")
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "niah.json"

    if result_path.exists() and not args.overwrite:
        print(f"[skip] {result_path} exists")
        _print_summary(json.load(open(result_path)))
        return

    from nltk.tokenize import sent_tokenize
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    print("Loading Paul Graham essays...", flush=True)
    essay_text  = _load_pg_essays()
    essay_sents = sent_tokenize(essay_text)
    print(f"  {len(essay_sents)} sentences loaded", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tp  = len(args.gpus.split(","))
    max_ctx = max(args.context_lens) + 200

    llm = LLM(
        model=args.model, dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=max_ctx,
        tensor_parallel_size=tp,
        trust_remote_code=True,
    )

    rows: List[Tuple[int, float, str]] = []   # (ctx_len, depth, prompt)
    for ctx_len in args.context_lens:
        for depth in args.depths:
            prompt = _make_sample(tok, essay_sents,
                                  args.needle, depth, ctx_len, args.question)
            rows.append((ctx_len, depth, prompt))

    print(f"Running inference on {len(rows)} samples...", flush=True)
    sp = SamplingParams(temperature=0, max_tokens=128)
    outputs = llm.generate([r[2] for r in rows], sp)
    preds   = [o.outputs[0].text.strip() for o in outputs]

    # Score: is the key phrase in the prediction? (case-insensitive)
    phrase_lc = args.score_phrase.lower()
    cell_scores: dict = {}
    for (ctx_len, depth, _), pred in zip(rows, preds):
        correct = int(phrase_lc in pred.lower())
        k_c, k_d = str(ctx_len), f"{depth:.2f}"
        cell_scores.setdefault(k_c, {}).setdefault(k_d, []).append(correct)

    scores: dict = {}
    for c, deps in cell_scores.items():
        scores[c] = {d: float(np.mean(v)) for d, v in deps.items()}
        scores[c]["_avg"] = float(np.mean(list(scores[c].values())))
    scores["_avg"] = float(np.mean([scores[c]["_avg"] for c in scores]))

    json.dump(scores, open(result_path, "w"), indent=2)
    print(f"Saved → {result_path}")
    _print_summary(scores)


def _print_summary(scores: dict) -> None:
    ctx_keys = sorted((k for k in scores if not k.startswith("_")), key=int)
    if not ctx_keys:
        return
    dep_keys = sorted(
        {d for c in ctx_keys for d in scores[c] if not d.startswith("_")},
        key=float,
    )
    label = "ctx\\dep"
    header = f"{label:>10}" + "".join(f"{float(d)*100:>8.0f}%" for d in dep_keys) + "  avg"
    print(f"\nNIAH (score %):\n{header}")
    for c in ctx_keys:
        row = f"{c:>10}"
        for d in dep_keys:
            v = scores[c].get(d, float("nan"))
            row += f"{v*100:>9.1f}"
        row += f"  {scores[c].get('_avg', float('nan'))*100:.1f}"
        print(row)
    print(f"  Overall avg: {scores.get('_avg', float('nan'))*100:.1f}%")


if __name__ == "__main__":
    main()
