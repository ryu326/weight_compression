#!/usr/bin/env python3
"""Compare RD curves: EB channels=M (rd_sweep_normnone, gaussian) vs
EB channels=1 (rd_sweep_shared_eb, gaussian).

Reads latest ckpt_iter*.pth.tar per run (NOT best.pth.tar — older best.pth.tar
files were saved on val/mse hardcoded; latest ckpt + log[VAL] gives the
final-state RD point).

Output:
  - checkpoint/rd_curves_eb_compare.png
  - checkpoint/rd_results_eb_compare.csv
"""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def latest_ckpt_iter(d: Path):
    iters = []
    for f in d.glob("ckpt_iter*.pth.tar"):
        m = re.match(r"ckpt_iter(\d+)\.pth\.tar$", f.name)
        if m:
            iters.append(int(m.group(1)))
    return max(iters) if iters else None


def parse_val_at_iter(log_path: Path, target_iter):
    """Return (iter, bpp, mse, loss) from [VAL] line at target_iter, or last
    [VAL] if target_iter is None."""
    if not log_path.exists():
        return None
    last = None
    for line in open(log_path):
        m = re.search(r"\[VAL\] iter (\d+): (\{.*\})", line)
        if not m:
            continue
        it = int(m.group(1))
        if target_iter is not None and it != target_iter:
            continue
        d = eval(m.group(2))  # safe: dict literal in our own logs
        last = (it, float(d["bpp_loss"]), float(d["mse"]), float(d["loss"]))
    return last


def transform_label(a):
    enc = str(a.get("encoder_transform", "?"))
    dec = str(a.get("decoder_transform", "?"))
    en = a.get("encoder_n_resblock") or a.get("n_resblock")
    dn = a.get("decoder_n_resblock") or a.get("n_resblock")
    if enc == dec:
        if enc == "resblock":
            return f"resblock(n={en})"
        return enc
    enc_s = f"resblock(n={en})" if enc == "resblock" else enc
    dec_s = f"resblock(n={dn})" if dec == "resblock" else dec
    return f"{enc_s}/{dec_s}"


def gather(save_root: Path, log_dir: Path, dataset_filter=None):
    rows = []
    if not save_root.exists():
        return rows
    for d in sorted(save_root.iterdir()):
        if not d.is_dir():
            continue
        li = latest_ckpt_iter(d)
        if li is None:
            continue
        ckpt = d / f"ckpt_iter{li}.pth.tar"
        if not ckpt.exists():
            continue
        s = torch.load(ckpt, map_location="cpu", weights_only=False)
        a = s.get("args", {})
        ds = str(a.get("dataset", ""))
        if dataset_filter is not None and ds not in dataset_filter:
            continue
        log = log_dir / f"{d.name}.log"
        res = parse_val_at_iter(log, li) or parse_val_at_iter(log, None)
        if res is None:
            continue
        it, bpp, mse, loss = res
        rows.append({
            "run": d.name,
            "dataset": ds,
            "label": transform_label(a),
            "shared_eb": bool(a.get("shared_eb", False)),
            "lmbda": float(a.get("lmbda", 0)),
            "iter": it,
            "max_iter": int(a.get("iter", 20000)),
            "bpp": bpp,
            "mse": mse,
            "loss": loss,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full_eb_root", default="checkpoint/rd_sweep_normnone")
    ap.add_argument("--shared_eb_root", default="checkpoint/rd_sweep_shared_eb")
    ap.add_argument("--full_log_dir", default="log/rd_sweep_normnone")
    ap.add_argument("--shared_log_dir", default="log/rd_sweep_shared_eb")
    ap.add_argument("--out_png", default="checkpoint/rd_curves_eb_compare.png")
    ap.add_argument("--out_csv", default="checkpoint/rd_results_eb_compare.csv")
    args = ap.parse_args()

    datasets = {"gaussian", "llama8b"}
    rows = []
    rows += gather(Path(args.full_eb_root), Path(args.full_log_dir), dataset_filter=datasets)
    rows += gather(Path(args.shared_eb_root), Path(args.shared_log_dir), dataset_filter=datasets)

    rows.sort(key=lambda r: (r["dataset"], r["label"], r["shared_eb"], r["lmbda"]))

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "run", "dataset", "label", "shared_eb", "lmbda", "iter", "max_iter",
                "bpp", "mse", "loss",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {len(rows)} rows -> {args.out_csv}")

    by = defaultdict(list)  # (dataset, label, shared_eb) -> list[row]
    for r in rows:
        by[(r["dataset"], r["label"], r["shared_eb"])].append(r)

    label_order = ["rht", "linear", "resblock(n=2)", "resblock(n=2)/linear"]
    all_labels = sorted({k[1] for k in by.keys()})
    labels = [l for l in label_order if l in all_labels] + [
        l for l in all_labels if l not in label_order
    ]
    plot_datasets = [d for d in ["gaussian", "llama8b"] if any(k[0] == d for k in by.keys())]

    fig, axes = plt.subplots(1, len(plot_datasets), figsize=(11 * len(plot_datasets), 8), dpi=130)
    if len(plot_datasets) == 1:
        axes = [axes]
    cmap = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for ax, ds in zip(axes, plot_datasets):
        for li, lab in enumerate(labels):
            color = cmap[li % len(cmap)]
            for shared, ls, marker, suffix in [(False, "-", "o", " (M=16)"),
                                                (True, "--", "s", " (shared)")]:
                pts = sorted(by.get((ds, lab, shared), []), key=lambda r: r["lmbda"])
                if not pts:
                    continue
                xs = [r["bpp"] for r in pts]
                ys = [r["mse"] for r in pts]
                ax.plot(
                    xs, ys, marker=marker, linestyle=ls, label=f"{lab}{suffix}",
                    color=color, linewidth=1.5, markersize=7,
                    markerfacecolor=color if not shared else "none",
                )
                for r in pts:
                    ax.annotate(
                        f"λ={int(r['lmbda'])}", (r["bpp"], r["mse"]),
                        fontsize=6, alpha=0.55, xytext=(3, 3), textcoords="offset points",
                    )
        ax.set_yscale("log")
        ax.set_xlabel("bpp (val @ latest ckpt)")
        ax.set_ylabel("MSE / std² (val @ latest ckpt, log)")
        ax.set_title(ds)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle(
        "EB channels=M=16 (solid, filled) vs channels=1 / shared (dashed, hollow)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(args.out_png)
    plt.close(fig)
    print(f"saved plot -> {args.out_png}")


if __name__ == "__main__":
    main()
