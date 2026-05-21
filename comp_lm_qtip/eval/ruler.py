#!/usr/bin/env python3
"""RULER benchmark evaluation — follows NVIDIA/RULER exactly.

Data generation uses the same logic as NVIDIA/RULER/scripts/data/synthetic/:
  - Needle template : "One of the special magic {type_needle_v} for {key} is: {value}."
  - Prompt template : official RULER template from constants.py
  - Haystack        : Paul Graham Essays (essay tasks) / noise sentences (noise tasks)
  - Keys            : wonderwords adjective-noun pairs (same as official)
  - Values          : 7-digit random numbers (same as official default)
  - Scoring         : string_match_all from eval/synthetic/constants.py

Tasks implemented (matching synthetic.yaml):
  niah_single_1   : noise haystack, 1 needle, 1 query
  niah_single_2   : essay haystack, 1 needle, 1 query
  niah_single_3   : essay haystack, 1 needle, 1 query, UUID values
  niah_multikey_1 : essay haystack, 2 keys, 1 query
  niah_multikey_2 : essay haystack, 4 keys, 2 queries
  niah_multikey_3 : essay haystack, 8 keys, 4 queries
  niah_multivalue : essay haystack, 1 key, 4 values, 1 query
  niah_multiquery : essay haystack, 1 key, 1 value, 4 queries
  vt              : variable tracking (noise haystack)
  cwe             : common word extraction
  fwe             : frequent word extraction

Context lengths: 4K, 8K, 16K, 32K.
n_samples: 10 per (task, ctx_len).

Usage:
    python eval/ruler.py \\
        --model /path/to/hf_model \\
        --output-dir /results/model/ruler \\
        --gpus 0,1,2,3
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import string
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
EVAL_DIR      = Path(__file__).resolve().parent
VENDOR_RULER  = EVAL_DIR.parent / "vendor" / "RULER"
PG_ESSAYS_JSON = VENDOR_RULER / "scripts" / "data" / "synthetic" / "json" / "PaulGrahamEssays.json"
PG_ESSAYS_TXT  = (EVAL_DIR.parent / "vendor" / "LLMTest_NeedleInAHaystack"
                  / "needlehaystack" / "PaulGrahamEssays")

# ── Official RULER prompt template (from constants.py) ────────────────────────
NIAH_TEMPLATE = (
    "Some special magic {type_needle_v} are hidden within the following text. "
    "Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards.\n"
    "{context}\n"
    "What are all the special magic {type_needle_v} for {query} mentioned in the provided text?"
)
NIAH_NEEDLE = "One of the special magic {type_needle_v} for {key} is: {value}."

NOISE_SENTENCE = ("The grass is green. The sky is blue. The sun is yellow. "
                  "Here we go. There and back again.")

VT_TEMPLATE = (
    "Memorize and track the chain(s) of variable assignment hidden in the following text.\n\n"
    "{context}\n"
    "Question: Find all variables that are assigned the value {query} in the text above."
)

CWE_TEMPLATE = (
    "Below is a numbered list of words. In these words, some appear more often than others. "
    "Memorize the ones that appear most often.\n{context}\n"
    "Question: What are the 10 most common words in the above list?"
)

FWE_TEMPLATE = (
    "Read the following coded text and track the frequency of each coded word. "
    "Find the three most frequently appeared coded words. {context}\n"
    "Question: Do not provide any explanation. Please ignore the dots '....'. "
    "What are the three most frequently appeared words in the above coded text?"
)

# ── Default settings ──────────────────────────────────────────────────────────
DEFAULT_TASKS = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery",
    "vt", "cwe", "fwe",
]
DEFAULT_CTX_LENS = [4096, 8192, 16384, 32768]
N_SAMPLES = 10  # samples per (task, ctx_len) — matches RULER paper

# Task config matching RULER's synthetic.yaml
TASK_CFG = {
    "niah_single_1":   dict(haystack="noise", num_k=1, num_v=1, num_q=1, vtype="numbers"),
    "niah_single_2":   dict(haystack="essay", num_k=1, num_v=1, num_q=1, vtype="numbers"),
    "niah_single_3":   dict(haystack="essay", num_k=1, num_v=1, num_q=1, vtype="uuids"),
    "niah_multikey_1": dict(haystack="essay", num_k=2, num_v=1, num_q=1, vtype="numbers"),
    "niah_multikey_2": dict(haystack="essay", num_k=4, num_v=1, num_q=2, vtype="numbers"),
    "niah_multikey_3": dict(haystack="essay", num_k=8, num_v=1, num_q=4, vtype="numbers"),
    "niah_multivalue": dict(haystack="essay", num_k=1, num_v=4, num_q=1, vtype="numbers"),
    "niah_multiquery": dict(haystack="essay", num_k=1, num_v=1, num_q=4, vtype="numbers"),
    "vt":  dict(haystack="noise", num_chains=1, num_hops=4),
    "cwe": dict(freq_cw=30, freq_ucw=3, num_cw=10),
    "fwe": dict(num_freq_words=3),
}


# ── Load Paul Graham essays ───────────────────────────────────────────────────

_PG_WORDS: List[str] = []   # lazy-loaded word list

def _load_pg_words() -> List[str]:
    global _PG_WORDS
    if _PG_WORDS:
        return _PG_WORDS
    # Try RULER JSON first
    if PG_ESSAYS_JSON.exists():
        import re as _re
        text = json.load(open(PG_ESSAYS_JSON))["text"]
        _PG_WORDS = _re.sub(r'\s+', " ", text).split(" ")
        return _PG_WORDS
    # Fallback: gkamradt txt files
    if PG_ESSAYS_TXT.exists():
        text = "\n\n".join(p.read_text(errors="replace")
                           for p in sorted(PG_ESSAYS_TXT.glob("*.txt")))
        import re as _re
        _PG_WORDS = _re.sub(r'\s+', " ", text).split(" ")
        return _PG_WORDS
    raise FileNotFoundError(
        "Paul Graham essays not found. Run: python "
        "vendor/RULER/scripts/data/synthetic/json/download_paulgraham_essay.py"
    )


# ── Key generation (wonderwords, same as official RULER) ─────────────────────

_WW_WORDS: List[str] = []

def _ww_words() -> List[str]:
    global _WW_WORDS
    if not _WW_WORDS:
        import wonderwords
        nouns = wonderwords.random_word._get_words_from_text_file("nounlist.txt")
        adjs  = wonderwords.random_word._get_words_from_text_file("adjectivelist.txt")
        _WW_WORDS = sorted(set(f"{a}-{n}" for a in adjs for n in nouns))
    return _WW_WORDS


def _gen_value(vtype: str, rng: random.Random) -> str:
    if vtype == "numbers":
        return str(rng.randint(10**6, 10**7 - 1))    # 7-digit number (official RULER)
    if vtype == "uuids":
        return str(uuid.UUID(int=rng.getrandbits(128), version=4))
    # words fallback
    return rng.choice(_ww_words())


# ── Haystack builders ─────────────────────────────────────────────────────────

def _essay_context(num_words: int, needles: List[str], depths_pct: List[int],
                   tokenizer) -> str:
    """Insert needles into Paul Graham essay haystack at given depths (%).
    Matches RULER niah.py essay branch exactly (sentence-level insertion).
    """
    from nltk.tokenize import sent_tokenize
    words = _load_pg_words()
    repeats = (num_words + len(words) - 1) // len(words)
    text = " ".join((words * repeats)[:num_words])
    sents = sent_tokenize(text.strip())

    ins_pos = [0] + sorted(
        int(len(sents) * (d / 100)) for d in depths_pct
    ) + [len(sents)]

    parts = []
    for i in range(1, len(ins_pos)):
        seg = " ".join(sents[ins_pos[i-1]:ins_pos[i]])
        parts.append(seg)
        if i - 1 < len(needles):
            parts.append(needles[i-1])
    return " ".join(parts)


def _noise_context(n_sentences: int, needles: List[str], rng: random.Random,
                   tokenizer) -> str:
    """Noise haystack (repeated noise sentence) with needles inserted."""
    sentences = [NOISE_SENTENCE] * n_sentences
    idxs = sorted(rng.sample(range(n_sentences), len(needles)), reverse=True)
    for idx, nd in zip(idxs, needles):
        sentences.insert(idx, nd)
    return "\n".join(sentences)


def _fit_haystack(tokenizer, ctx_len: int, build_fn, tokens_to_gen: int = 128) -> str:
    """Binary search for haystack size that fits in ctx_len tokens."""
    lo, hi = 50, ctx_len * 4
    best = ""
    for _ in range(20):
        mid = (lo + hi) // 2
        candidate = build_fn(mid)
        n = len(tokenizer.encode(candidate, add_special_tokens=False)) + tokens_to_gen
        if n <= ctx_len:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(tokenizer, full_text: str) -> str:
    """Wrap the RULER-formatted input with chat template (Qwen3 thinking=False)."""
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": full_text}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    except TypeError:
        pass
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": full_text}],
            tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        return full_text


# ── Official string_match_all scoring ────────────────────────────────────────

def _string_match_all(preds: List[str], refs: List[List[str]]) -> float:
    """Exact copy of RULER eval/synthetic/constants.py string_match_all."""
    score = sum(
        sum(1.0 if r.lower() in pred.lower() else 0.0 for r in ref) / len(ref)
        for pred, ref in zip(preds, refs)
    ) / len(preds) * 100
    return round(score, 2)


# ── NIAH data generator ───────────────────────────────────────────────────────

def _gen_niah(tokenizer, ctx_len: int, rng: random.Random, cfg: dict
              ) -> Tuple[str, List[str]]:
    num_k = cfg["num_k"]
    num_v = cfg["num_v"]
    num_q = cfg["num_q"]
    vtype = cfg["vtype"]
    haystack_type = cfg["haystack"]

    keys   = [rng.choice(_ww_words()) for _ in range(num_k)]
    values = [[_gen_value(vtype, rng) for _ in range(num_v)] for _ in range(num_k)]
    needles = [
        NIAH_NEEDLE.format(type_needle_v=vtype, key=k, value=v)
        for k, vs in zip(keys, values)
        for v in vs
    ]
    rng.shuffle(needles)

    depths_pct = sorted(rng.sample(range(0, 101), len(needles)))

    def _build(sz):
        if haystack_type == "essay":
            return _essay_context(sz, needles, depths_pct, tokenizer)
        else:
            return _noise_context(sz, needles, rng, tokenizer)

    context = _fit_haystack(tokenizer, ctx_len, _build)

    # Query  (pick num_q keys to ask about)
    q_idxs  = rng.sample(range(num_k), num_q)
    queries = [keys[i] for i in q_idxs]
    answers = [v for i in q_idxs for v in values[i]]
    query   = (", ".join(queries[:-1]) + ", and " + queries[-1]
               if len(queries) > 1 else queries[0])

    tmpl = NIAH_TEMPLATE
    if num_q * num_v == 1:          # singular grammar fix (official code)
        tmpl = tmpl.replace("Some", "A").replace("are all", "is").replace("are", "is")
        vtype_s = vtype.rstrip("s")
    else:
        vtype_s = vtype

    full_text = tmpl.format(type_needle_v=vtype_s, context=context, query=query)
    return _build_prompt(tokenizer, full_text), answers


# ── Variable Tracking ─────────────────────────────────────────────────────────

def _gen_vt(tokenizer, ctx_len: int, rng: random.Random, cfg: dict
            ) -> Tuple[str, List[str]]:
    num_chains = cfg.get("num_chains", 1)
    num_hops   = cfg.get("num_hops", 4)
    var_names  = list(string.ascii_uppercase)
    rng.shuffle(var_names)

    chain_texts, final_vals = [], []
    for c in range(num_chains):
        chain = var_names[c * (num_hops + 1): (c + 1) * (num_hops + 1)]
        final = str(rng.randint(100, 999))
        final_vals.append(final)
        assignments = [f"{chain[i]} = {chain[i+1]}" for i in range(num_hops - 1)]
        assignments.append(f"{chain[-2]} = {final}")
        rng.shuffle(assignments)
        chain_texts.append(", ".join(assignments) + ".")

    query = final_vals[0]
    answers = [var_names[c * (num_hops + 1)] for c in range(num_chains)
               if final_vals[c] == query]

    def _build(sz):
        sentences = [NOISE_SENTENCE] * sz
        for ct in chain_texts:
            ins = rng.randint(0, len(sentences))
            sentences.insert(ins, ct)
        return "\n".join(sentences)

    context = _fit_haystack(tokenizer, ctx_len, _build)
    full_text = VT_TEMPLATE.format(context=context, query=query)
    return _build_prompt(tokenizer, full_text), answers


# ── CWE / FWE ─────────────────────────────────────────────────────────────────

def _gen_cwe(tokenizer, ctx_len: int, rng: random.Random, cfg: dict
             ) -> Tuple[str, List[str]]:
    freq_cw  = cfg.get("freq_cw",  30)
    freq_ucw = cfg.get("freq_ucw",  3)
    num_cw   = cfg.get("num_cw",   10)
    ww       = _ww_words()
    common   = rng.sample(ww, num_cw)
    uncommon = [w for w in rng.sample(ww, 200) if w not in set(common)]

    word_pool = common * freq_cw + uncommon * freq_ucw
    rng.shuffle(word_pool)
    word_list = "\n".join(f"{i+1}. {w}" for i, w in enumerate(word_pool))

    def _build(_sz):
        return word_list   # CWE context is just the word list

    context = _fit_haystack(tokenizer, ctx_len, _build)
    full_text = CWE_TEMPLATE.format(context=context)
    return _build_prompt(tokenizer, full_text), common


def _gen_fwe(tokenizer, ctx_len: int, rng: random.Random, cfg: dict
             ) -> Tuple[str, List[str]]:
    num_words = cfg.get("num_freq_words", 3)
    ww        = _ww_words()
    top_words = rng.sample(ww, num_words)
    others    = [w for w in rng.sample(ww, 200) if w not in set(top_words)]

    word_pool = top_words * 50 + others * 3
    rng.shuffle(word_pool)
    coded = " .... ".join(word_pool)

    def _build(_sz):
        return coded

    context = _fit_haystack(tokenizer, ctx_len, _build)
    full_text = FWE_TEMPLATE.format(context=context)
    return _build_prompt(tokenizer, full_text), top_words


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    p.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    p.add_argument("--ctx-lens", nargs="+", type=int, default=DEFAULT_CTX_LENS)
    p.add_argument("--n-samples", type=int, default=N_SAMPLES)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "ruler.json"

    if result_path.exists() and not args.overwrite:
        print(f"[skip] {result_path} exists")
        _print_summary(json.load(open(result_path)))
        return

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tp  = len(args.gpus.split(","))
    max_ctx = max(args.ctx_lens) + 300

    llm = LLM(
        model=args.model, dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=max_ctx,
        tensor_parallel_size=tp,
        trust_remote_code=True,
    )

    # Pre-load shared resources
    _load_pg_words()
    _ww_words()

    # Generate all samples
    all_rows: List[Dict[str, Any]] = []
    for task in args.tasks:
        if task not in TASK_CFG:
            print(f"[skip] unknown task: {task}")
            continue
        cfg = TASK_CFG[task]
        for ctx_len in args.ctx_lens:
            rng = random.Random(args.seed + hash(task) + ctx_len)
            for i in range(args.n_samples):
                try:
                    if task.startswith("niah"):
                        prompt, answers = _gen_niah(tok, ctx_len, rng, cfg)
                    elif task == "vt":
                        prompt, answers = _gen_vt(tok, ctx_len, rng, cfg)
                    elif task == "cwe":
                        prompt, answers = _gen_cwe(tok, ctx_len, rng, cfg)
                    elif task == "fwe":
                        prompt, answers = _gen_fwe(tok, ctx_len, rng, cfg)
                    else:
                        continue
                except Exception as e:
                    print(f"  [{task}/{ctx_len}] sample {i} error: {e}", flush=True)
                    continue
                all_rows.append({
                    "task": task, "ctx_len": ctx_len,
                    "answers": answers, "prompt": prompt,
                })
    print(f"Total samples: {len(all_rows)}", flush=True)

    # Run inference (tokens_to_generate=128 per RULER constants)
    sp = SamplingParams(temperature=0, max_tokens=128)
    outputs = llm.generate([r["prompt"] for r in all_rows], sp)
    preds   = [o.outputs[0].text.strip() for o in outputs]

    # Score with official string_match_all
    cell_preds: Dict[str, Dict[str, Tuple[List[str], List[List[str]]]]] = {}
    for row, pred in zip(all_rows, preds):
        t, c = row["task"], str(row["ctx_len"])
        bucket = cell_preds.setdefault(t, {}).setdefault(c, ([], []))
        bucket[0].append(pred)
        bucket[1].append(row["answers"])

    scores: Dict[str, Any] = {}
    for task, ctx_map in cell_preds.items():
        scores[task] = {}
        for c, (ps, rs) in ctx_map.items():
            scores[task][c] = _string_match_all(ps, rs)
        scores[task]["_avg"] = float(np.mean(list(scores[task].values())))

    ctx_avgs: Dict[str, List[float]] = {}
    for task, ctx_map in scores.items():
        for c, v in ctx_map.items():
            if not c.startswith("_"):
                ctx_avgs.setdefault(c, []).append(v)
    scores["_ctx_avg"] = {c: float(np.mean(v)) for c, v in ctx_avgs.items()}
    scores["_avg"] = float(np.mean([scores[t]["_avg"] for t in scores
                                    if not t.startswith("_")]))

    json.dump(scores, open(result_path, "w"), indent=2)
    print(f"Saved → {result_path}")
    _print_summary(scores)


def _print_summary(scores: dict) -> None:
    task_keys = [k for k in scores if not k.startswith("_")]
    ctx_keys = sorted(
        {c for t in task_keys for c in scores[t] if not c.startswith("_")},
        key=int,
    )
    header = f"{'task':<25}" + "".join(f"{c:>10}" for c in ctx_keys) + "   avg"
    print(f"\nRULER (score %):\n{header}")
    for t in task_keys:
        row = f"{t:<25}"
        for c in ctx_keys:
            v = scores[t].get(c, float("nan"))
            row += f"{v*100:>10.1f}"
        row += f"  {scores[t].get('_avg', float('nan'))*100:.1f}"
        print(row)
    if "_ctx_avg" in scores:
        row = f"{'ctx_avg':<25}"
        for c in ctx_keys:
            v = scores["_ctx_avg"].get(c, float("nan"))
            row += f"{v*100:>10.1f}"
        print(row)
    print(f"  Overall avg: {scores.get('_avg', float('nan'))*100:.1f}%")


if __name__ == "__main__":
    main()
