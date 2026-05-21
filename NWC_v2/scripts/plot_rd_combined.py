#!/usr/bin/env python3
"""Combined RD-curve plotter — merges symmetric (rd_sweep) and asymmetric
(rd_sweep_extra) runs into one figure per dataset.

Each run becomes a curve labeled by its (encoder_transform, decoder_transform,
encoder_n_resblock, decoder_n_resblock) tuple. Symmetric runs (enc == dec)
collapse to just the transform name; asymmetric runs use a compact suffix.
"""
import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def _short_n(n):
    """Format n_resblock — '?' if missing."""
    if n is None:
        return "?"
    try:
        return str(int(n))
    except Exception:
        return str(n)


def make_label(args_dict: dict) -> str:
    enc = str(args_dict.get("encoder_transform", "?"))
    dec = str(args_dict.get("decoder_transform", "?"))
    n_default = args_dict.get("n_resblock")
    enc_n = args_dict.get("encoder_n_resblock") or n_default
    dec_n = args_dict.get("decoder_n_resblock") or n_default

    # Truly symmetric requires same transform AND same n (when transform=resblock)
    same_n = (enc != "resblock") or (str(_short_n(enc_n)) == str(_short_n(dec_n)))
    if enc == dec and same_n:
        if enc == "resblock":
            return f"{enc}(n={_short_n(enc_n)})"
        return enc
    # Asymmetric — show both sides
    enc_str = f"{enc}(n={_short_n(enc_n)})" if enc == "resblock" else enc
    dec_str = f"{dec}(n={_short_n(dec_n)})" if dec == "resblock" else dec
    return f"{enc_str}/{dec_str}"


def parse_run_dirs(save_root: str):
    save_root = Path(save_root)
    if not save_root.exists():
        return
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
        if not ds:
            continue
        metrics = saved.get("best_metrics") or {}
        bpp = float(metrics.get("bpp_loss", float("nan")))
        mse = float(metrics.get("mse", saved.get("best_mse", float("nan"))))
        recon = float(metrics.get("recon_loss", float("nan")))
        loss = float(metrics.get("loss", float("nan")))
        label = make_label(a)
        yield {
            "run": run_dir.name,
            "dataset": ds,
            "label": label,
            "encoder_transform": str(a.get("encoder_transform", "")),
            "decoder_transform": str(a.get("decoder_transform", "")),
            "encoder_n_resblock": a.get("encoder_n_resblock") or a.get("n_resblock"),
            "decoder_n_resblock": a.get("decoder_n_resblock") or a.get("n_resblock"),
            "lmbda": float(a.get("lmbda", 0.0)),
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
    fields = [
        "run", "dataset", "label", "encoder_transform", "decoder_transform",
        "encoder_n_resblock", "decoder_n_resblock",
        "lmbda", "iter", "bpp", "mse_normed", "recon_loss", "loss", "ckpt",
    ]
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {len(rows)} rows -> {csv_path}")


def plot_rd(rows, out_path: str, datasets_order=None, label_order=None):
    if not rows:
        print("[warn] no rows; skipping plot")
        return
    by_ds_label = defaultdict(list)
    for r in rows:
        if r["mse_normed"] != r["mse_normed"] or r["bpp"] != r["bpp"]:
            continue
        by_ds_label[(r["dataset"], r["label"])].append(r)

    datasets = datasets_order or sorted({k[0] for k in by_ds_label.keys()})
    all_labels = sorted({k[1] for k in by_ds_label.keys()})
    if label_order:
        # Keep requested order first, append unknowns at the end
        ordered = [l for l in label_order if l in all_labels]
        ordered += [l for l in all_labels if l not in ordered]
        labels = ordered
    else:
        labels = all_labels

    n_ds = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(11 * n_ds, 8), dpi=130)
    if n_ds == 1:
        axes = [axes]

    cmap = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for ax, ds in zip(axes, datasets):
        for li, label in enumerate(labels):
            pts = by_ds_label.get((ds, label), [])
            if not pts:
                continue
            pts = sorted(pts, key=lambda r: r["lmbda"])
            xs = [r["bpp"] for r in pts]
            ys = [r["mse_normed"] for r in pts]
            ax.plot(xs, ys, marker="o", label=label, linewidth=1.5,
                    color=cmap[li % len(cmap)])
            for r in pts:
                ax.annotate(
                    f"λ={int(r['lmbda'])}", (r["bpp"], r["mse_normed"]),
                    fontsize=6, alpha=0.55, xytext=(3, 3), textcoords="offset points",
                )
        ax.set_yscale("log")
        ax.set_xlabel("bpp (val)")
        ax.set_ylabel("MSE / std² (val, log)")
        ax.set_title(f"{ds}")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend(title="encoder/decoder", fontsize=8, loc="best")

    fig.suptitle("Combined RD curves (symmetric + asymmetric configs, EB = compressai)", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"saved plot -> {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--save_roots", nargs="+",
                   default=["./checkpoint/rd_sweep", "./checkpoint/rd_sweep_extra"],
                   help="Directories containing run subdirs with best.pth.tar")
    p.add_argument("--out", default="rd_curves_combined.png")
    p.add_argument("--csv", default="rd_results_combined.csv")
    args = p.parse_args()

    rows = []
    for root in args.save_roots:
        if not Path(root).exists():
            print(f"[warn] save_root missing: {root} (skipping)")
            continue
        rows.extend(parse_run_dirs(root))
    rows.sort(key=lambda r: (r["dataset"], r["label"], r["lmbda"]))

    write_csv(rows, args.csv)
    plot_rd(
        rows, args.out,
        datasets_order=["gaussian", "llama8b"],
        label_order=[
            # Symmetric (from rd_sweep)
            "rht", "affine", "linear",
            "resblock(n=1)",          # encR1/decR1 (asymmetric file but symmetric n=1)
            "resblock(n=2)",          # rd_sweep main symmetric n=2
            # Asymmetric (from rd_sweep_extra)
            "resblock(n=2)/resblock(n=1)",
            "resblock(n=2)/linear",
        ],
    )


if __name__ == "__main__":
    main()
