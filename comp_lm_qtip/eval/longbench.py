#!/usr/bin/env python3
"""LongBench V1 evaluation for compressed HF models.

Adapted from turboquant/scripts/eval_longbench.py.
Uses official LongBench prompts/scoring; inference via vLLM with tensor_parallel.

Usage:
    python eval/longbench.py \\
        --model /path/to/hf_model \\
        --output-dir /results/model/longbench \\
        --gpus 0,1,2,3 \\
        [--en-only]  # skip Chinese tasks
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

LONGBENCH_DIR = Path("/home/jgryu/workspace/turboquant/vendor/LongBench/LongBench")

DATASETS_EN = [
    "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa",
    "musique", "gov_report", "qmsum", "multi_news", "trec", "triviaqa",
    "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p",
]
DATASETS_ZH = [
    "multifieldqa_zh", "dureader", "vcsum", "lsht", "passage_retrieval_zh",
]
DATASETS_ALL = DATASETS_EN + DATASETS_ZH

# These tasks expect raw completion, not chat-wrapped prompts
NO_CHAT_WRAP = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}


def _apply_chat(tokenizer, content: str, task: str) -> str:
    if task in NO_CHAT_WRAP:
        return content
    # enable_thinking is passed via **kwargs in apply_chat_template (not an explicit param),
    # so inspect.signature() won't find it — always try passing it directly.
    try:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        # Non-Qwen3 tokenizers don't support enable_thinking
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False, add_generation_prompt=True,
        )
    return prompt


def run_inference(args: argparse.Namespace) -> None:
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    with open(LONGBENCH_DIR / "config" / "dataset2prompt.json") as f:
        dataset2prompt = json.load(f)
    with open(LONGBENCH_DIR / "config" / "dataset2maxlen.json") as f:
        dataset2maxlen = json.load(f)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tp = len(args.gpus.split(","))

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=args.max_model_len,
        tensor_parallel_size=tp,
        trust_remote_code=True,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = args.tasks or (DATASETS_EN if args.en_only else DATASETS_ALL)

    for task in tasks:
        out_path = out_dir / f"{task}.jsonl"
        if out_path.exists() and not args.overwrite:
            print(f"[skip] {task}", flush=True)
            continue
        if task not in dataset2prompt:
            print(f"[skip] {task}: no prompt template", flush=True)
            continue

        fmt = dataset2prompt[task]
        max_gen = dataset2maxlen.get(task, 64)
        max_in = args.max_model_len - max_gen - 32

        print(f"[{task}] loading data...", flush=True)
        data = load_dataset("THUDM/LongBench", task, split="test")

        prompts, meta = [], []
        for ex in data:
            content = fmt.format(**ex)
            ids = tok(content, truncation=False, return_tensors="pt").input_ids[0]
            if len(ids) > max_in:
                h = max_in // 2
                content = (tok.decode(ids[:h], skip_special_tokens=True)
                           + tok.decode(ids[-h:], skip_special_tokens=True))
            prompts.append(_apply_chat(tok, content, task))
            meta.append({
                "answers": ex["answers"],
                "all_classes": ex.get("all_classes", []),
                "length": ex.get("length", 0),
            })

        sp = SamplingParams(temperature=0, max_tokens=max_gen)
        outputs = llm.generate(prompts, sp)
        preds = [o.outputs[0].text.strip() for o in outputs]

        with open(out_path, "w", encoding="utf-8") as f:
            for pred, m in zip(preds, meta):
                json.dump({"pred": pred, **m}, f, ensure_ascii=False)
                f.write("\n")
        print(f"[{task}] {len(preds)} examples → {out_path}", flush=True)


def run_eval(output_dir: str) -> dict:
    out_dir = Path(output_dir)
    model_tag = out_dir.name

    pred_link = LONGBENCH_DIR / "pred" / model_tag
    pred_link.parent.mkdir(parents=True, exist_ok=True)
    if pred_link.exists() or pred_link.is_symlink():
        pred_link.unlink()
    pred_link.symlink_to(out_dir.resolve())

    subprocess.run(
        [sys.executable, str(LONGBENCH_DIR / "eval.py"), "--model", model_tag],
        cwd=str(LONGBENCH_DIR), check=False,
    )

    result_path = pred_link / "result.json"
    if not result_path.exists():
        print("WARNING: LongBench eval.py did not produce result.json", flush=True)
        return {}

    scores = json.load(open(result_path))
    task_scores = {k: v for k, v in scores.items() if not k.startswith("_")}
    avg = float(np.mean(list(task_scores.values())))
    scores["_avg"] = avg

    # Save as longbenchV1.json (consistent with niah.json / ruler.json naming)
    (out_dir / "longbenchV1.json").write_text(json.dumps(scores, indent=2))

    print(f"\n{'='*55}")
    print(f"LongBench V1  |  avg = {avg:.2f}")
    print(f"{'='*55}")
    for t, s in sorted(task_scores.items()):
        print(f"  {t:30s}: {s:.2f}")
    print(f"{'='*55}")
    return scores


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    p.add_argument("--tasks", nargs="+", default=None)
    p.add_argument("--en-only", action="store_true", help="Skip Chinese tasks")
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--skip-pred", action="store_true")
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    if not args.skip_pred:
        run_inference(args)
    run_eval(args.output_dir)


if __name__ == "__main__":
    main()
