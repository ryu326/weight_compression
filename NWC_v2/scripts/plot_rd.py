#!/usr/bin/env python3
"""Aggregate RD-sweep results and plot bpp vs MSE curves.

Reads each run dir's `best.pth.tar` (saved by train.py at the iter where
val MSE was lowest) and assembles a CSV + a 2-panel matplotlib figure
(one panel per dataset; one curve per encoder/decoder transform).
"""
import argparse
import csv
import os
import re
from collections import defaultdict
from pathlib import Path

import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_run_dirs(save_root: str):
    """Yield (dataset, transform, lmbda, ckpt_path) from <save_root>/*/best.pth.tar."""
    save_root = Path(save_root)
    if not save_root.exists():
        raise FileNotFoundError(save_root)
    for run_dir in sorted(save_root.iterdir()):
        if not run_dir.is_dir():
            continue
        ckpt = run_dir / "best.pth.tar"
        if not ckpt.exists():
            continue
        try:
            saved = torch.load(ckpt, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"[warn] failed to load {ckpt}: {e}")
            continue
        a = saved.get("args", {})
        ds = str(a.get("dataset", ""))
        enc = str(a.get("encoder_transform", ""))
        dec = str(a.get("decoder_transform", ""))
        lmbda = float(a.get("lmbda", 0.0))
        if not ds or not enc or enc != dec:
            # Skip runs where encoder/decoder differ (this sweep keeps them paired).
            continue
        metrics = saved.get("best_metrics") or {}
        bpp = float(metrics.get("bpp_loss", float("nan")))
        mse = float(metrics.get("mse", saved.get("best_mse", float("nan"))))
        recon = float(metrics.get("recon_loss", float("nan")))
        loss = float(metrics.get("loss", float("nan")))
        yield {
            "run": run_dir.name,
            "dataset": ds,
            "transform": enc,
            "lmbda": lmbda,
            "iter": int(saved.get("iter", 0)),
            "bpp": bpp,
            "mse_normed": mse,
            "recon_loss": recon,
            "loss": loss,
            "ckpt": str(ckpt),
        }


def write_csv(rows, csv_path: str):
    if not rows:
        print("[warn] no rows to write")
        return
    fields = ["run", "dataset", "transform", "lmbda", "iter",
              "bpp", "mse_normed", "recon_loss", "loss", "ckpt"]
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {len(rows)} rows -> {csv_path}")


def plot_rd(rows, out_path: str, transforms_order=None, datasets_order=None):
    if not rows:
        print("[warn] no rows; skipping plot")
        return

    by_ds_tf = defaultdict(list)
    for r in rows:
        if r["mse_normed"] != r["mse_normed"] or r["bpp"] != r["bpp"]:
            continue  # NaN
        by_ds_tf[(r["dataset"], r["transform"])].append(r)

    datasets = datasets_order or sorted({k[0] for k in by_ds_tf.keys()})
    transforms = transforms_order or sorted({k[1] for k in by_ds_tf.keys()})

    n_ds = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(6.5 * n_ds, 5), dpi=140)
    if n_ds == 1:
        axes = [axes]

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for ax, ds in zip(axes, datasets):
        for ti, tf in enumerate(transforms):
            pts = by_ds_tf.get((ds, tf), [])
            if not pts:
                continue
            pts = sorted(pts, key=lambda r: r["lmbda"])
            xs = [r["bpp"] for r in pts]
            ys = [r["mse_normed"] for r in pts]
            ax.plot(xs, ys, marker="o", label=tf,
                    color=color_cycle[ti % len(color_cycle)], linewidth=1.6)
            for r in pts:
                ax.annotate(
                    f"λ={int(r['lmbda'])}",
                    (r["bpp"], r["mse_normed"]),
                    fontsize=7, alpha=0.6, xytext=(3, 3), textcoords="offset points",
                )
        ax.set_yscale("log")
        ax.set_xlabel("bpp (val)")
        ax.set_ylabel("MSE / std² (val, log)")
        ax.set_title(f"{ds}")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend(title="transform")

    fig.suptitle("RD curves: encoder = decoder, EB = compressai", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"saved plot -> {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--save_root", default="./checkpoint/rd_sweep")
    p.add_argument("--out", default="rd_curves.png")
    p.add_argument("--csv", default="rd_results.csv")
    args = p.parse_args()

    rows = list(parse_run_dirs(args.save_root))
    rows.sort(key=lambda r: (r["dataset"], r["transform"], r["lmbda"]))

    write_csv(rows, args.csv)
    plot_rd(rows, args.out,
            transforms_order=["affine", "rht", "linear", "resblock"],
            datasets_order=["gaussian", "llama8b"])


if __name__ == "__main__":
    main()
