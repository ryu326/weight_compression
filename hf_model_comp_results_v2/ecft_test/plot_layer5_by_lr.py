#!/usr/bin/env python3
"""Layer5 ECFT 2D sweep plot grouped by lr (each subplot = one lr, lambdas as curves).
Includes QTIP 5_down reference curve."""
import os
import re
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = "/home/jgryu/workspace/weight_compression/hf_model_comp_results_v2/meta-llama--Meta-Llama-3-8B/ecft_layer5_lmbda_lr_sweep"
QTIP_LOG_DIR = "/home/jgryu/workspace/weight_compression/qtip/log/llama3_8b/ft1_layer5_only"
OUT_EPOCH = "/home/jgryu/workspace/weight_compression/hf_model_comp_results_v2/ecft_layer5_by_lr_plot.png"
OUT_PARETO = "/home/jgryu/workspace/weight_compression/hf_model_comp_results_v2/ecft_layer5_by_lr_pareto.png"
QTIP_REF_SUBLAYER = "5_down"
QTIP_K_LIST = [2, 3, 5]
QTIP_RATE_MAP = {2: 2.0, 3: 3.0, 5: 5.0}  # K → approximate bpp


def parse_log(path):
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8 and parts[0] == "STEP" and parts[1] != "stage":
                try:
                    rows.append({
                        "stage": parts[1],
                        "epoch": float(parts[2]),
                        "loss": float(parts[3]),
                        "mse": float(parts[4]),
                        "rate": float(parts[5]),
                        "aux": float(parts[6]),
                        "lambda": float(parts[7]),
                    })
                except ValueError:
                    pass
    return rows


def parse_qtip_sublayer(path, sublayer):
    if not os.path.exists(path):
        return None
    pat_init = re.compile(rf"layer {sublayer} initial loss ([0-9eE.+\-]+)")
    pat_ep = re.compile(rf"layer {sublayer} @ epoch (\d+) new loss ([0-9eE.+\-]+)")
    initial = None
    epochs = {}
    with open(path) as f:
        for line in f:
            m = pat_init.search(line)
            if m:
                initial = float(m.group(1))
            m = pat_ep.search(line)
            if m:
                epochs[int(m.group(1))] = float(m.group(2))
    if initial is None and not epochs:
        return None
    xs = [0] + sorted(epochs.keys())
    ys = [initial] + [epochs[k] for k in sorted(epochs.keys())]
    xs = [x + 1 for x in xs]  # align with ECFT 1-based epoch
    return xs, ys


# --- load data ---
data = {}
for sub in sorted(os.listdir(ROOT)):
    m = re.match(r"lmbda([0-9p]+)_lr([0-9eE+\-.]+)$", sub)
    if not m:
        continue
    ld = float(m.group(1).replace("p", "."))
    lr = float(m.group(2).replace("p", "."))
    logs = glob.glob(os.path.join(ROOT, sub, "ecft_dec_log", "*.log"))
    if not logs:
        continue
    rows = parse_log(logs[0])
    if rows:
        data[(ld, lr)] = rows

if not data:
    raise SystemExit("no data")

# Load multi-K QTIP data
qtip_multi = {}  # K -> (xs, ys)
for K in QTIP_K_LIST:
    log_path = os.path.join(QTIP_LOG_DIR, f"{K}bit.log")
    parsed = parse_qtip_sublayer(log_path, QTIP_REF_SUBLAYER)
    if parsed is not None:
        qtip_multi[K] = parsed
qtip = qtip_multi.get(2)  # backward compat for per-epoch plots
print(f"combos: {len(data)}  qtip K loaded: {sorted(qtip_multi.keys())}")

all_lambdas = sorted({ld for ld, _ in data.keys()})
all_lrs = sorted({lr for _, lr in data.keys()})

# --- per-epoch plot: rows=(train mse, valid mse, valid rate), cols=lrs ---
nrows, ncols = 3, len(all_lrs)
fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), dpi=130,
                          squeeze=False)
cmap = plt.get_cmap("viridis")
lambda_color = {ld: cmap(i / max(len(all_lambdas) - 1, 1))
                for i, ld in enumerate(all_lambdas)}

metric_rows = [
    ("train", "mse", "mse_loss", True),
    ("valid", "mse", "mse_loss", True),
    ("valid", "rate", "rate_loss (bpp)", False),
]
for r, (stage, key, ylabel, log_y) in enumerate(metric_rows):
    for c, lr in enumerate(all_lrs):
        ax = axes[r, c]
        for ld in all_lambdas:
            rows = data.get((ld, lr))
            if not rows:
                continue
            pts = [x for x in rows if x["stage"] == stage]
            if not pts:
                continue
            xs = [p["epoch"] for p in pts]
            ys = [p[key] for p in pts]
            ax.plot(xs, ys, "o-", color=lambda_color[ld], label=f"λ={ld:g}",
                    linewidth=1.5, markersize=4)
        # QTIP reference only on mse panels
        if key == "mse" and qtip is not None:
            ax.plot(qtip[0], qtip[1], "k*--", linewidth=2, markersize=10,
                    label="QTIP 5_down")
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{stage} {key}  |  lr={lr:g}")
        if log_y:
            ax.set_yscale("log")
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.legend(fontsize=7, ncol=2)

fig.suptitle("Layer5 ECFT — grouped by lr (λ curves within)", fontsize=14)
fig.tight_layout()
fig.savefig(OUT_EPOCH)
print(f"saved {OUT_EPOCH}")

# --- Pareto: one panel per lr, final (rate, mse) per lambda ---
fig2, axes2 = plt.subplots(1, len(all_lrs), figsize=(5 * len(all_lrs), 5), dpi=130,
                           squeeze=False)
for c, lr in enumerate(all_lrs):
    ax = axes2[0, c]
    rates, mses, labels = [], [], []
    for ld in all_lambdas:
        rows = data.get((ld, lr))
        if not rows:
            continue
        va = [r for r in rows if r["stage"] == "valid"]
        if not va:
            continue
        rates.append(va[-1]["rate"])
        mses.append(va[-1]["mse"])
        labels.append(f"λ={ld:g}")
        ax.scatter(rates[-1], mses[-1], color=lambda_color[ld], s=100,
                   zorder=3)
        ax.annotate(labels[-1], (rates[-1], mses[-1]),
                    fontsize=8, xytext=(5, 5), textcoords="offset points")
    if rates:
        # connect points sorted by rate
        order = sorted(range(len(rates)), key=lambda i: rates[i])
        ax.plot([rates[i] for i in order], [mses[i] for i in order],
                "-", color="gray", alpha=0.5, zorder=1)

    # QTIP multi-K curve on Pareto
    if qtip_multi:
        qtip_rates_p, qtip_mses_p = [], []
        for K in sorted(qtip_multi.keys()):
            r = QTIP_RATE_MAP.get(K, float(K))
            m = qtip_multi[K][1][-1]  # final epoch MSE
            qtip_rates_p.append(r)
            qtip_mses_p.append(m)
            ax.scatter(r, m, color="black", marker="*", s=300, zorder=5)
            ax.annotate(f"QTIP K={K}", (r, m), fontsize=8, fontweight="bold",
                        xytext=(5, 8), textcoords="offset points")
        ax.plot(qtip_rates_p, qtip_mses_p, "k--", linewidth=2, alpha=0.7,
                label="QTIP (K=2,3,5)")

    ax.set_xlabel("final rate_loss (bpp)")
    ax.set_ylabel("final mse_loss (log)")
    ax.set_yscale("log")
    ax.set_title(f"Pareto (valid)  |  lr={lr:g}")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(fontsize=8)

fig2.suptitle("Layer5 ECFT Pareto — grouped by lr  (+ QTIP 2-bit reference)",
              fontsize=14)
fig2.tight_layout()
fig2.savefig(OUT_PARETO)
print(f"saved {OUT_PARETO}")

# summary
print("\n=== final valid by lr ===")
for lr in all_lrs:
    print(f"\nlr = {lr:g}")
    print(f"{'λ':>6} {'mse':>10} {'rate':>10} {'loss':>10}")
    for ld in all_lambdas:
        rows = data.get((ld, lr))
        if not rows:
            print(f"{ld:>6g} (no data)")
            continue
        va = [r for r in rows if r["stage"] == "valid"]
        last = va[-1] if va else None
        if last is None:
            print(f"{ld:>6g} (no valid)")
        else:
            print(f"{ld:>6g} {last['mse']:10.6f} {last['rate']:10.6f} {last['loss']:10.6f}")

if qtip_multi:
    print("\nQTIP 5_down final mse per K:")
    for K in sorted(qtip_multi.keys()):
        print(f"  K={K}: rate={QTIP_RATE_MAP.get(K, K):.1f} bpp  mse={qtip_multi[K][1][-1]:.6e}")
