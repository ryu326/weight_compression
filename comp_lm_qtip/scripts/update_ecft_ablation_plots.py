#!/usr/bin/env python3
import argparse
import csv
import itertools
import os
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


FIELDS = [
    "target_key",
    "target_rate",
    "decoder_type",
    "ecft_epochs",
    "ecft_lmbda",
    "ecft_mode",
    "ecft_entropy_model",
    "ecft_lambda_ortho",
    "ecft_B_init",
    "bpp",
    "mse_normed",
    "proxy_err",
    "result_pt",
]
NUMERIC_DIMS = {"target_rate", "ecft_epochs", "ecft_lmbda", "ecft_lambda_ortho"}
EXCLUDED_LAMBDAS = {1.0, 10.0}


def _as_float(x: Any) -> float:
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return float(x.detach().item())
        raise ValueError("expected scalar tensor")
    return float(x)


def _safe_slug(s: str) -> str:
    return (
        str(s)
        .replace("/", "_")
        .replace(" ", "_")
        .replace("=", "-")
        .replace(".", "p")
        .replace(",", "_")
    )


def _is_excluded_lambda(value: str) -> bool:
    try:
        return float(str(value)) in EXCLUDED_LAMBDAS
    except (TypeError, ValueError):
        return False


def _read_rows(
    path: str,
    default_mode: str = "noise",
    default_entropy_model: str = "compressai",
    default_lambda_ortho: str = "",
    default_B_init: str = "",
) -> list[dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for r in rows:
        if "ecft_mode" not in r or not r["ecft_mode"]:
            r["ecft_mode"] = default_mode
        if "ecft_entropy_model" not in r or not r["ecft_entropy_model"]:
            r["ecft_entropy_model"] = default_entropy_model
        if "ecft_lambda_ortho" not in r:
            r["ecft_lambda_ortho"] = default_lambda_ortho
        if "ecft_B_init" not in r:
            r["ecft_B_init"] = default_B_init
    return rows


def _write_rows(path: str, rows: list[dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _upsert_row(rows: list[dict[str, str]], row: dict[str, str]) -> list[dict[str, str]]:
    key = (
        row["target_key"],
        row["target_rate"],
        row["decoder_type"],
        row["ecft_epochs"],
        row["ecft_lmbda"],
        row.get("ecft_lambda_ortho", ""),
        row.get("ecft_B_init", ""),
    )
    out: list[dict[str, str]] = []
    replaced = False
    for r in rows:
        rk = (
            r["target_key"],
            r["target_rate"],
            r["decoder_type"],
            r["ecft_epochs"],
            r["ecft_lmbda"],
            r.get("ecft_lambda_ortho", ""),
            r.get("ecft_B_init", ""),
        )
        if rk == key:
            out.append(row)
            replaced = True
        else:
            out.append(r)
    if not replaced:
        out.append(row)
    return out


def _to_dim_sort_key(dim_name: str, value: str):
    if dim_name in NUMERIC_DIMS:
        try:
            return (0, float(value))
        except (TypeError, ValueError):
            # missing/non-numeric values (e.g. ortho="" for compressai rows) sort first
            return (-1, str(value))
    return (0, str(value))


def _format_dim_value(dim_name: str, value: str) -> str:
    if dim_name in NUMERIC_DIMS:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return str(value) if value else "NA"
        if v.is_integer():
            return str(int(v))
        return str(v)
    return str(value) if value else "NA"


def _add_variant_column(rows: list[dict[str, str]]) -> None:
    """Synthesize a per-row 'variant' label that collapses (entropy_model,
    ecft_lambda_ortho, ecft_B_init) into one string. Used as curve_dim for the
    compressai-vs-lattice comparison plot so all lattice_eb (ortho, B_init)
    combos show up as distinct curves alongside compressai."""
    for r in rows:
        em = str(r.get("ecft_entropy_model", "compressai"))
        ortho = str(r.get("ecft_lambda_ortho", "") or "")
        binit = str(r.get("ecft_B_init", "") or "")
        if em == "lattice_eb" and (ortho or binit):
            r["variant"] = f"{em}(ortho={ortho},B={binit})"
        else:
            r["variant"] = em


def _unique_values(rows: list[dict[str, str]], dim_name: str) -> list[str]:
    values = {str(r[dim_name]) for r in rows}
    return sorted(values, key=lambda x: _to_dim_sort_key(dim_name, x))


def _filter_rows(
    rows: list[dict[str, str]],
    *,
    target_key: str,
    fixed: dict[str, str],
) -> list[dict[str, str]]:
    out = []
    for r in rows:
        if r["target_key"] != target_key:
            continue
        ok = True
        for k, v in fixed.items():
            if str(r.get(k, "")) != str(v):
                ok = False
                break
        if ok:
            out.append(r)
    return out


def _plot_series_on_axes(
    ax_mse,
    ax_proxy,
    *,
    xs: list[float],
    ys_mse: list[float],
    ys_proxy: list[float],
    label: str,
    linestyle: str = "-",
    linewidth: float = 1.5,
    marker: str = "o",
    alpha: float = 1.0,
    color=None,
):
    line_mse = ax_mse.plot(
        xs,
        ys_mse,
        marker=marker,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=alpha,
        label=label,
        color=color,
    )[0]
    use_color = line_mse.get_color()
    ax_proxy.plot(
        xs,
        ys_proxy,
        marker=marker,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=alpha,
        label=label,
        color=use_color,
    )


def _plot_family_dual_metrics(
    rows: list[dict[str, str]],
    *,
    plots_dir: str,
    target_key: str,
    scenario_name: str,
    curve_dim: str,
    sweep_dim: str,
    fixed_dims: list[str],
) -> None:
    trows = [r for r in rows if r["target_key"] == target_key]
    if not trows:
        return

    fixed_values = [_unique_values(trows, d) for d in fixed_dims]
    for fixed_combo in itertools.product(*fixed_values):
        fixed = {d: v for d, v in zip(fixed_dims, fixed_combo)}
        subset = _filter_rows(rows, target_key=target_key, fixed=fixed)
        if not subset:
            continue

        curve_values = _unique_values(subset, curve_dim)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=140)
        ax_mse, ax_proxy = axes
        curve_count = 0
        for cv in curve_values:
            pts = [r for r in subset if str(r[curve_dim]) == str(cv)]
            if not pts:
                continue
            pts = sorted(pts, key=lambda r: _to_dim_sort_key(sweep_dim, str(r[sweep_dim])))
            xs = [float(r["bpp"]) for r in pts]
            ys_mse = [float(r["mse_normed"]) for r in pts]
            ys_proxy = [float(r["proxy_err"]) for r in pts]
            curve_count += 1
            _plot_series_on_axes(
                ax_mse,
                ax_proxy,
                xs=xs,
                ys_mse=ys_mse,
                ys_proxy=ys_proxy,
                label=f"{curve_dim}={_format_dim_value(curve_dim, cv)}",
            )

        if scenario_name == "ablate_lmbda_curves_by_target_rate":
            fixed_epoch = fixed.get("ecft_epochs")
            try:
                fixed_epoch_num = int(float(str(fixed_epoch)))
            except (TypeError, ValueError):
                fixed_epoch_num = None

            if fixed_epoch_num is not None and fixed_epoch_num != 0:
                baseline_fixed = dict(fixed)
                baseline_fixed["ecft_epochs"] = "0"
                baseline_rows = _filter_rows(rows, target_key=target_key, fixed=baseline_fixed)
                baseline_curve_values = _unique_values(baseline_rows, curve_dim)
                if baseline_curve_values:
                    baseline_cv = baseline_curve_values[0]
                    baseline_pts = [
                        r for r in baseline_rows if str(r.get(curve_dim)) == str(baseline_cv)
                    ]
                    baseline_pts = sorted(
                        baseline_pts, key=lambda r: _to_dim_sort_key(sweep_dim, str(r[sweep_dim]))
                    )
                    if baseline_pts:
                        xs = [float(r["bpp"]) for r in baseline_pts]
                        ys_mse = [float(r["mse_normed"]) for r in baseline_pts]
                        ys_proxy = [float(r["proxy_err"]) for r in baseline_pts]
                        _plot_series_on_axes(
                            ax_mse,
                            ax_proxy,
                            xs=xs,
                            ys_mse=ys_mse,
                            ys_proxy=ys_proxy,
                            label="initial",
                            linestyle="--",
                            linewidth=2.0,
                            marker="o",
                            color="black",
                        )

        if curve_count == 0:
            plt.close(fig)
            continue

        fixed_desc = ", ".join(f"{k}={_format_dim_value(k, v)}" for k, v in fixed.items())
        fig.suptitle(
            f"{target_key} | {scenario_name}\ncurve={curve_dim}, sweep={sweep_dim} | fixed: {fixed_desc}",
            fontsize=10,
        )
        ax_mse.set_xlabel("bpp")
        ax_mse.set_ylabel("mse_normed (log)")
        ax_mse.set_title("bpp vs mse_normed")
        ax_mse.set_yscale("log")
        ax_mse.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
        ax_mse.legend(fontsize=8)

        ax_proxy.set_xlabel("bpp")
        ax_proxy.set_ylabel("proxy_err (log)")
        ax_proxy.set_title("bpp vs proxy_err")
        ax_proxy.set_yscale("log")
        ax_proxy.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
        ax_proxy.legend(fontsize=8)

        fig.tight_layout()

        out_dir = os.path.join(
            plots_dir,
            scenario_name,
            target_key,
        )
        os.makedirs(out_dir, exist_ok=True)
        fixed_slug = "__".join(f"{k}-{_safe_slug(_format_dim_value(k, v))}" for k, v in fixed.items())
        out_path = os.path.join(out_dir, f"{fixed_slug}.png")
        fig.savefig(out_path)
        plt.close(fig)


def _plot_all(
    rows: list[dict[str, str]],
    *,
    plots_dir: str,
    target_order: list[str],
) -> None:
    rows = [r for r in rows if not _is_excluded_lambda(str(r.get("ecft_lmbda", "")))]
    if not rows:
        return
    os.makedirs(plots_dir, exist_ok=True)
    _add_variant_column(rows)

    has_multiple_modes = len({r.get("ecft_mode", "noise") for r in rows}) > 1

    scenarios = [
        (
            "ablate_target_rate_curves_by_lmbda",
            "target_rate",
            "ecft_lmbda",
            ["decoder_type", "ecft_epochs"],
        ),
        (
            "ablate_lmbda_curves_by_target_rate",
            "ecft_lmbda",
            "target_rate",
            ["decoder_type", "ecft_epochs"],
        ),
        (
            "ablate_decoder_type_curves_by_lmbda",
            "decoder_type",
            "ecft_lmbda",
            ["target_rate", "ecft_epochs"],
        ),
        (
            "ablate_decoder_type_curves_by_target_rate",
            "decoder_type",
            "target_rate",
            ["ecft_lmbda", "ecft_epochs"],
        ),
        (
            "ablate_epochs_curves_by_lmbda",
            "ecft_epochs",
            "ecft_lmbda",
            ["target_rate", "decoder_type"],
        ),
        (
            "ablate_epochs_curves_by_target_rate",
            "ecft_epochs",
            "target_rate",
            ["ecft_lmbda", "decoder_type"],
        ),
    ]

    has_multiple_em = len({r.get("ecft_entropy_model", "compressai") for r in rows}) > 1

    if has_multiple_modes:
        scenarios = [
            (sn, cd, sd, fd if "ecft_mode" in fd else fd + ["ecft_mode"])
            for sn, cd, sd, fd in scenarios
        ]
        scenarios += [
            (
                "ablate_ecft_mode_curves_by_lmbda",
                "ecft_mode",
                "ecft_lmbda",
                ["target_rate", "decoder_type", "ecft_epochs"],
            ),
            (
                "ablate_ecft_mode_curves_by_target_rate",
                "ecft_mode",
                "target_rate",
                ["ecft_lmbda", "decoder_type", "ecft_epochs"],
            ),
        ]

    if has_multiple_em:
        scenarios = [
            (sn, cd, sd, fd if "ecft_entropy_model" in fd else fd + ["ecft_entropy_model"])
            for sn, cd, sd, fd in scenarios
        ]
        scenarios += [
            (
                "ablate_entropy_model_curves_by_lmbda",
                "ecft_entropy_model",
                "ecft_lmbda",
                ["target_rate", "decoder_type", "ecft_epochs", "ecft_mode"],
            ),
            (
                "ablate_entropy_model_curves_by_target_rate",
                "ecft_entropy_model",
                "target_rate",
                ["ecft_lmbda", "decoder_type", "ecft_epochs", "ecft_mode"],
            ),
        ]

    has_multiple_ortho = len({r.get("ecft_lambda_ortho", "") for r in rows}) > 1
    has_multiple_binit = len({r.get("ecft_B_init", "") for r in rows}) > 1

    if has_multiple_ortho:
        scenarios = [
            (sn, cd, sd, fd if "ecft_lambda_ortho" in fd else fd + ["ecft_lambda_ortho"])
            for sn, cd, sd, fd in scenarios
        ]
        scenarios += [
            (
                "ablate_lambda_ortho_curves_by_lmbda",
                "ecft_lambda_ortho",
                "ecft_lmbda",
                ["target_rate", "decoder_type", "ecft_epochs", "ecft_mode",
                 "ecft_entropy_model", "ecft_B_init"],
            ),
            (
                "ablate_lambda_ortho_curves_by_target_rate",
                "ecft_lambda_ortho",
                "target_rate",
                ["ecft_lmbda", "decoder_type", "ecft_epochs", "ecft_mode",
                 "ecft_entropy_model", "ecft_B_init"],
            ),
        ]

    if has_multiple_binit:
        scenarios = [
            (sn, cd, sd, fd if "ecft_B_init" in fd else fd + ["ecft_B_init"])
            for sn, cd, sd, fd in scenarios
        ]
        scenarios += [
            (
                "ablate_B_init_curves_by_lmbda",
                "ecft_B_init",
                "ecft_lmbda",
                ["target_rate", "decoder_type", "ecft_epochs", "ecft_mode",
                 "ecft_entropy_model", "ecft_lambda_ortho"],
            ),
            (
                "ablate_B_init_curves_by_target_rate",
                "ecft_B_init",
                "target_rate",
                ["ecft_lmbda", "decoder_type", "ecft_epochs", "ecft_mode",
                 "ecft_entropy_model", "ecft_lambda_ortho"],
            ),
        ]

    # Composite: one curve per (entropy_model, ortho, B_init) variant, for
    # compressai-vs-lattice_eb comparison on the same axes.
    has_multiple_variants = len({r.get("variant", "") for r in rows}) > 1
    if has_multiple_variants:
        scenarios += [
            (
                "ablate_variant_curves_by_target_rate",
                "variant",
                "target_rate",
                ["ecft_lmbda", "decoder_type", "ecft_epochs", "ecft_mode"],
            ),
            (
                "ablate_variant_curves_by_lmbda",
                "variant",
                "ecft_lmbda",
                ["target_rate", "decoder_type", "ecft_epochs", "ecft_mode"],
            ),
        ]

    for target_key in target_order:
        for scenario_name, curve_dim, sweep_dim, fixed_dims in scenarios:
            _plot_family_dual_metrics(
                rows,
                plots_dir=plots_dir,
                target_key=target_key,
                scenario_name=scenario_name,
                curve_dim=curve_dim,
                sweep_dim=sweep_dim,
                fixed_dims=fixed_dims,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records_csv", type=str, required=True)
    parser.add_argument("--plots_dir", type=str, required=True)
    parser.add_argument("--target_order", type=str, required=True)
    parser.add_argument("--pt_path", type=str, default=None)
    parser.add_argument("--target_key", type=str, default=None)
    parser.add_argument("--target_rate", type=str, default=None)
    parser.add_argument("--decoder_type", type=str, default=None)
    parser.add_argument("--ecft_epochs", type=str, default=None)
    parser.add_argument("--ecft_lmbda", type=str, default=None)
    parser.add_argument("--ste_records_csv", type=str, default=None)
    parser.add_argument("--parametric_records_csv", type=str, default=None,
                        help="Additional records CSV from parametric entropy model (noise mode)")
    parser.add_argument("--parametric_ste_records_csv", type=str, default=None,
                        help="Additional records CSV from parametric entropy model (ste mode)")
    parser.add_argument("--ecft_mode", type=str, default="noise",
                        help="ecft_mode tag for upsert row (pt_path mode) and default for --records_csv reads")
    parser.add_argument("--ecft_entropy_model", type=str, default="compressai",
                        help="ecft_entropy_model tag for upsert row (pt_path mode) and default for --records_csv reads")
    parser.add_argument("--ecft_lambda_ortho", type=str, default="",
                        help="ecft_lambda_ortho tag for upsert row (lattice_eb only; empty otherwise)")
    parser.add_argument("--ecft_B_init", type=str, default="",
                        help="ecft_B_init tag for upsert row (lattice_eb only; empty otherwise)")
    args = parser.parse_args()

    rows = _read_rows(args.records_csv,
                      default_mode=args.ecft_mode,
                      default_entropy_model=args.ecft_entropy_model)

    if args.pt_path:
        saved = torch.load(args.pt_path, map_location="cpu", weights_only=False)
        row = {
            "target_key": str(args.target_key),
            "target_rate": str(args.target_rate),
            "decoder_type": str(args.decoder_type),
            "ecft_epochs": str(args.ecft_epochs),
            "ecft_lmbda": str(args.ecft_lmbda),
            "ecft_mode": str(args.ecft_mode),
            "ecft_entropy_model": str(args.ecft_entropy_model),
            "ecft_lambda_ortho": str(args.ecft_lambda_ortho),
            "ecft_B_init": str(args.ecft_B_init),
            "bpp": str(_as_float(saved["bpp"])),
            "mse_normed": str(_as_float(saved["mse_normed"])),
            "proxy_err": str(_as_float(saved["proxy_err"])),
            "result_pt": str(args.pt_path),
        }
        rows = _upsert_row(rows, row)
        _write_rows(args.records_csv, rows)

    if args.ste_records_csv:
        rows = rows + _read_rows(args.ste_records_csv, default_mode="ste", default_entropy_model="compressai")
    if args.parametric_records_csv:
        rows = rows + _read_rows(args.parametric_records_csv, default_mode="noise", default_entropy_model="parametric")
    if args.parametric_ste_records_csv:
        rows = rows + _read_rows(args.parametric_ste_records_csv, default_mode="ste", default_entropy_model="parametric")

    target_order = [x.strip() for x in args.target_order.split(",") if x.strip()]
    _plot_all(rows, plots_dir=args.plots_dir, target_order=target_order)


if __name__ == "__main__":
    main()
