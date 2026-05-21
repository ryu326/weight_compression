#!/usr/bin/env python3
"""Compute extractive_match accuracy from paroquant lighteval result JSON files
and POST a summary to Slack.

Usage:
    python _paroquant_notify.py <stage_label> <dir_root> [<dir_root> ...]

Each `dir_root` is scanned for `*_paroquant_reasoning/*.jsonl` (or `.json`
written by lighteval). Accuracies are grouped by (run_dir, dataset).
"""
import json
import os
import re
import sys
import urllib.request

WEBHOOK = os.environ.get(
    "SLACK_WEBHOOK_URL",
    "https://hooks.slack.com/services/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",
)


def parse_metrics(metrics):
    if isinstance(metrics, dict):
        return metrics
    if isinstance(metrics, str):
        return json.loads(metrics.replace("'", '"'))
    return {}


def accuracy(path):
    try:
        data = json.load(open(path))
    except Exception as exc:
        return None, str(exc)
    if not isinstance(data, list) or not data:
        return None, 0
    if not isinstance(data[0], dict) or "metrics" not in data[0]:
        return None, 0
    try:
        score = sum(parse_metrics(it.get("metrics", {})).get("extractive_match", 0) for it in data)
    except Exception:
        return None, 0
    return score / len(data), len(data)


def collect(root):
    """Yield (run_label, dataset_name, accuracy, n) for all jsonl found under root."""
    if os.path.isfile(root):  # already a single jsonl
        ds = os.path.splitext(os.path.basename(root))[0]
        acc, n = accuracy(root)
        yield ("", ds, acc, n)
        return
    for cur, _dirs, files in os.walk(root):
        for f in sorted(files):
            if not f.endswith(".jsonl"):  # restrict to lighteval inference output
                continue
            full = os.path.join(cur, f)
            label = os.path.relpath(cur, start=os.path.dirname(root))
            ds = os.path.splitext(f)[0]
            acc, n = accuracy(full)
            if acc is None:
                continue
            yield (label, ds, acc, n)


def lambda_key(label):
    """Pull integer lambda from a 'lmbda<N>_paroquant_reasoning' label."""
    m = re.search(r"lmbda(-?\d+)", label)
    return int(m.group(1)) if m else float("inf")


def format_message(stage, rows):
    if not rows:
        return f"*[{stage}]*\n(no results found)"
    rows_sorted = sorted(rows, key=lambda r: (lambda_key(r[0]), r[1]))
    lines = [f"*[{stage}]* paroquant reasoning eval results"]
    cur_label = None
    for label, ds, acc, n in rows_sorted:
        if label != cur_label:
            cur_label = label
            short = label.replace("_paroquant_reasoning", "") if label else "(root)"
            lines.append(f"\n• `{short}`")
        if acc is None:
            lines.append(f"    {ds}: error")
        else:
            lines.append(f"    {ds}: {acc:.4f} (n={n})")
    return "\n".join(lines)


def post_slack(text):
    payload = json.dumps({"text": text}).encode("utf-8")
    req = urllib.request.Request(
        WEBHOOK, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.status


def main():
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    stage = sys.argv[1]
    rows = []
    for root in sys.argv[2:]:
        rows.extend(collect(root))
    text = format_message(stage, rows)
    print(text)
    try:
        status = post_slack(text)
        print(f"[slack] HTTP {status}", file=sys.stderr)
    except Exception as exc:
        print(f"[slack] failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
