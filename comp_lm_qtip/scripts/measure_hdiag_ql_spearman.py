#!/usr/bin/env python3

import argparse
import csv
import re
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

try:
    from scipy.stats import pearsonr, spearmanr
except ImportError:
    pearsonr = None
    spearmanr = None

QL_COEFFICIENTS = {
    4: np.array([3.4, 1.2, 0.1, 0.05], dtype=np.float64),
    8: np.array([4.000, 1.707, 0.724, 0.307, 0.130, 0.055, 0.024, 0.010], dtype=np.float64),
    16: np.array(
        [4.000, 2.301, 1.324, 0.761, 0.438, 0.252, 0.145, 0.083, 0.048, 0.028, 0.016, 0.009, 0.005, 0.003, 0.0017, 0.001],
        dtype=np.float64,
    ),
}
QLEVEL_REF_TOP_PERCENTAGES = np.array([10.0, 1.0, 0.1], dtype=np.float64)


def flat_to_sym(v: torch.Tensor, n: int) -> torch.Tensor:
    a = torch.zeros(n, n, dtype=v.dtype, device=v.device)
    idxs = torch.tril_indices(n, n, device=v.device)
    a[idxs.unbind()] = v
    a[idxs[1, :], idxs[0, :]] = v
    return a


def regularize_h2(h: torch.Tensor, n: int, sigma_reg: float) -> torch.Tensor:
    h.div_(torch.diag(h).mean())
    idx = torch.arange(n, device=h.device)
    h[idx, idx] += sigma_reg
    return h


def compute_qlevel_log_cells(coeffs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    log_coeffs = np.log(coeffs)
    upper = 0.5 * (log_coeffs[:-1] + log_coeffs[1:])
    lower = np.empty_like(upper)
    lower[:-1] = 0.5 * (log_coeffs[1:-1] + log_coeffs[2:])
    lower[-1] = 2 * log_coeffs[-1] - upper[-1]
    return lower, upper


def get_qlevel_percentages_for_q(Q: int) -> np.ndarray:
    if Q == 4:
        return QLEVEL_REF_TOP_PERCENTAGES.copy()
    if Q not in QL_COEFFICIENTS:
        raise ValueError(f"Unsupported Q for qlevel percentages: {Q}")

    ref_coeffs = QL_COEFFICIENTS[4]
    coeffs = QL_COEFFICIENTS[Q]
    level_coeffs = coeffs[1:]
    lower, upper = compute_qlevel_log_cells(coeffs)
    level_percentages = np.zeros(Q - 1, dtype=np.float64)

    group_masks = (
        level_coeffs >= ref_coeffs[1],
        (level_coeffs < ref_coeffs[1]) & (level_coeffs >= ref_coeffs[2]),
        level_coeffs < ref_coeffs[2],
    )
    high_group_indices = np.flatnonzero(group_masks[0])
    low_group_indices = np.flatnonzero(group_masks[2])
    group_bounds = (
        (np.log(ref_coeffs[1]), upper[high_group_indices[0]]) if high_group_indices.size else None,
        (np.log(ref_coeffs[2]), np.log(ref_coeffs[1])),
        (lower[low_group_indices[-1]], np.log(ref_coeffs[2])) if low_group_indices.size else None,
    )

    for mask, total_percentage, bounds in zip(group_masks, QLEVEL_REF_TOP_PERCENTAGES, group_bounds):
        group_indices = np.flatnonzero(mask)
        if group_indices.size == 0 or bounds is None:
            continue

        group_lo, group_hi = bounds
        widths = np.minimum(upper[group_indices], group_hi) - np.maximum(lower[group_indices], group_lo)
        widths = np.clip(widths, a_min=0.0, a_max=None)
        if widths.sum() <= 0:
            widths = np.full(group_indices.shape, 1 / group_indices.size, dtype=np.float64)
        else:
            widths = widths / widths.sum()
        level_percentages[group_indices] = total_percentage * widths

    return level_percentages


def get_top_percentages_and_qlevels(Q: int) -> tuple[np.ndarray, np.ndarray]:
    level_percentages = get_qlevel_percentages_for_q(Q)
    qlevels = np.arange(1, Q, dtype=np.int32)
    return level_percentages[::-1], qlevels[::-1]


def get_ql_from_h(h: torch.Tensor, comp_model, args) -> torch.Tensor:
    qlevel = torch.zeros((h.shape[1],), dtype=torch.int32, device=h.device)

    if getattr(args, "ql_random_uniform", False):
        if args.Q in (4, 8, 16):
            top, qlevels = get_top_percentages_and_qlevels(args.Q)
        elif args.Q == 2:
            if args.ql_search_r is None:
                raise ValueError("ql_random_uniform with Q=2 requires ql_search_r")
            qlevel_value = args.ql_search_value if comp_model.Q == 4 else 1
            if qlevel_value is None:
                raise ValueError("ql_random_uniform with Q=2 and comp_model.Q=4 requires ql_search_value")
            top = np.array([args.ql_search_r])
            qlevels = [qlevel_value]
        else:
            raise ValueError(f"Unsupported Q for ql_random_uniform: {args.Q}")

        topk = (top * h.shape[1] / 100).astype(int)
        total = int(topk.sum())
        if total > h.shape[1]:
            raise ValueError(f"ql_random_uniform assigned count {total} exceeds tensor size {h.shape[1]}")

        random_indices = torch.randperm(h.shape[1], device=h.device)[:total]
        start = 0
        for count, value in zip(topk, qlevels):
            count = int(count)
            if count <= 0:
                continue
            indices = random_indices[start:start + count]
            qlevel[indices] = int(value)
            start += count
        return qlevel

    if args.ql is True:
        if args.Q in (4, 8, 16):
            top, qlevels = get_top_percentages_and_qlevels(args.Q)
            in_norm = torch.diag(h)
            topk = (top * len(in_norm) / 100).astype(int)
            qlevel = torch.zeros_like(in_norm, dtype=torch.int32)
            _, topk_indices = torch.topk(in_norm, k=topk.sum())
            start = 0
            for count, value in zip(topk, qlevels):
                indices = topk_indices[start:start + count]
                qlevel[indices] = value
                start += count
        elif args.Q == 2:
            top = np.array([args.ql_search_r])
            qlevels = [args.ql_search_value] if comp_model.Q == 4 else [1]
            in_norm = torch.diag(h)
            topk = (top * len(in_norm) / 100).astype(int)
            qlevel = torch.zeros_like(in_norm, dtype=torch.int32)
            _, topk_indices = torch.topk(in_norm, k=topk.sum())
            start = 0
            for count, value in zip(topk, qlevels):
                indices = topk_indices[start:start + count]
                qlevel[indices] = value
                start += count

    if args.ql_invH is True:
        assert comp_model.Q == 4
        lhr = torch.linalg.cholesky(h)
        h_inv = torch.cholesky_inverse(lhr)
        top = np.array([0.1, 1, 10])
        qlevels = [3, 2, 1]
        diag = torch.diag(h_inv)
        topk = (top * len(diag) / 100).astype(int)
        qlevel = torch.zeros_like(diag, dtype=torch.int32)
        _, topk_indices = torch.topk(diag, k=topk.sum(), largest=False)
        start = 0
        for count, value in zip(topk, qlevels):
            indices = topk_indices[start:start + count]
            qlevel[indices] = value
            start += count

    if args.ql_search:
        if args.ql_search_layer_idx is None:
            ql_search_layer_idx = list(range(40))
        else:
            if isinstance(args.ql_search_layer_idx, (list, tuple)):
                ql_search_layer_idx = [int(x) for x in args.ql_search_layer_idx]
            else:
                ql_search_layer_idx = list(map(int, str(args.ql_search_layer_idx).split(',')))
        ql_search_layer_name = args.ql_search_layer_name.split(',')
        if args.layer_name in ql_search_layer_name and args.layer_idx in ql_search_layer_idx:
            qlevel = torch.full_like(qlevel, args.ql_search_value)
        qlevel = torch.full(
            (h.shape[1],),
            args.ql_search_value,
            dtype=torch.int32,
            device=h.device,
        )

    return qlevel


def parse_mapping(mapping_text: str) -> dict[int, float]:
    mapping = {}
    for entry in mapping_text.split(','):
        entry = entry.strip()
        if not entry:
            continue
        key_text, value_text = entry.split(':', 1)
        mapping[int(key_text.strip())] = float(value_text.strip())
    if not mapping:
        raise ValueError("ql_value_mapping must not be empty")
    return mapping


def parse_layer_metadata(path: Path) -> tuple[int, str]:
    match = re.match(r"^(?P<layer_idx>\d+)_(?P<layer_name>.+)$", path.stem)
    if match is None:
        raise ValueError(
            f"Could not parse layer metadata from {path.name}. "
            "Expected filename like '12_qkv.pt' or '3_up.pt'."
        )
    return int(match.group("layer_idx")), match.group("layer_name")


def list_hessian_files(input_path: Path, pattern: str) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    files = [path for path in input_path.glob(pattern) if path.is_file()]
    if not files:
        raise FileNotFoundError(f"No Hessian files matched '{pattern}' under {input_path}")

    return sorted(files, key=lambda path: parse_layer_metadata(path))


def load_hessian(path: Path, device: torch.device, sigma_reg: float, regularize: bool) -> torch.Tensor:
    try:
        h_data = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        h_data = torch.load(path, map_location="cpu")
    h = flat_to_sym(h_data["flatH"].to(device), int(h_data["n"]))
    if "mu" in h_data:
        mu = h_data["mu"].to(device)
        h += mu[None, :] * mu[:, None]
    if regularize:
        h = regularize_h2(h, int(h_data["n"]), sigma_reg)
    return h


def map_qlevels(qlevel: torch.Tensor, mapping: dict[int, float]) -> torch.Tensor:
    unique_levels = {int(level) for level in torch.unique(qlevel).cpu().tolist()}
    missing_levels = sorted(unique_levels - set(mapping))
    if missing_levels:
        raise KeyError(
            f"ql_value_mapping is missing levels used by Qlevel: {missing_levels}"
        )

    mapped = torch.empty(qlevel.shape, dtype=torch.float32, device=qlevel.device)
    for level, value in mapping.items():
        mapped[qlevel == level] = value
    return mapped


def rankdata_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    sorted_values = values[order]

    start = 0
    while start < sorted_values.shape[0]:
        end = start + 1
        while end < sorted_values.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def pearson_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = np.sqrt((x_centered ** 2).sum() * (y_centered ** 2).sum())
    if denom == 0:
        return float("nan")
    return float((x_centered * y_centered).sum() / denom)


def compute_spearman(diag_values: torch.Tensor, mapped_qvalues: torch.Tensor) -> tuple[float, float]:
    x = diag_values.detach().cpu().numpy().astype(np.float64, copy=False)
    y = mapped_qvalues.detach().cpu().numpy().astype(np.float64, copy=False)

    if spearmanr is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = spearmanr(x, y)
        statistic = getattr(result, "statistic", getattr(result, "correlation", float("nan")))
        return float(statistic), float(result.pvalue)

    x_rank = rankdata_average(x)
    y_rank = rankdata_average(y)
    return pearson_corrcoef(x_rank, y_rank), float("nan")


def compute_pearson(diag_values: torch.Tensor, mapped_qvalues: torch.Tensor) -> tuple[float, float]:
    x = diag_values.detach().cpu().numpy().astype(np.float64, copy=False)
    y = mapped_qvalues.detach().cpu().numpy().astype(np.float64, copy=False)

    if pearsonr is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = pearsonr(x, y)
        statistic = getattr(result, "statistic", getattr(result, "correlation", float("nan")))
        return float(statistic), float(result.pvalue)

    return pearson_corrcoef(x, y), float("nan")


def format_qlevel_counts(qlevel: torch.Tensor) -> str:
    unique_vals, counts = torch.unique(qlevel.cpu(), return_counts=True)
    return ",".join(f"{int(value)}:{int(count)}" for value, count in zip(unique_vals, counts))


def write_csv(rows: list[dict], output_path: Path) -> None:
    fieldnames = [
        "file",
        "layer_idx",
        "layer_name",
        "n",
        "spearman_rho",
        "spearman_p_value",
        "pearson_r",
        "pearson_p_value",
        "diag_min",
        "diag_max",
        "mapped_min",
        "mapped_max",
        "qlevel_counts",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure Spearman and Pearson correlation between diag(H) and mapped ql values "
            "without performing actual compression."
        )
    )
    parser.add_argument("--in_hess_path", type=str, required=True, help="Directory or single Hessian .pt file")
    parser.add_argument("--glob", type=str, default="*.pt", help="Glob pattern used when --in_hess_path is a directory")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device, e.g. cpu or cuda:0")
    parser.add_argument("--sigma_reg", type=float, default=1e-2, help="sigma_reg used by regularize_H2")
    parser.add_argument("--no_regularize_h2", action="store_true", default=False)
    parser.add_argument("--max_files", type=int, default=None, help="Only process the first N files after sorting")
    parser.add_argument("--save_csv", type=str, default=None, help="Optional CSV output path for per-file results")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used by --ql_random_uniform")

    parser.add_argument("--ql", action="store_true", default=False)
    parser.add_argument("--ql_invH", action="store_true", default=False)
    parser.add_argument("--ql_random_uniform", action="store_true", default=False)
    parser.add_argument("--ql_search", action="store_true", default=False)
    parser.add_argument("--ql_search_layer_name", type=str, default="q,k,v,o,up,gate,down")
    parser.add_argument("--ql_search_layer_idx", type=str, default=None)
    parser.add_argument("--ql_search_value", type=int, default=None)
    parser.add_argument("--ql_search_r", type=float, default=None)
    parser.add_argument("--Q", type=int, default=4)
    parser.add_argument(
        "--ql_value_mapping",
        type=str,
        default="0:0.29,1:0.83,2:10,3:20",
        help="Mapping from integer ql to scalar value, e.g. '0:0.29,1:0.83,2:10,3:20'",
    )
    return parser


def validate_args(args) -> None:
    if not any([args.ql, args.ql_invH, args.ql_random_uniform, args.ql_search]):
        args.ql = True

    if args.ql_invH and args.Q != 4:
        raise ValueError("--ql_invH currently expects Q=4")

    if args.ql_search and args.ql_search_value is None:
        raise ValueError("--ql_search requires --ql_search_value")

    if args.ql and args.Q == 2 and args.ql_search_r is None:
        raise ValueError("--ql with Q=2 requires --ql_search_r")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    input_path = Path(args.in_hess_path).expanduser().resolve()
    files = list_hessian_files(input_path, args.glob)
    if args.max_files is not None:
        files = files[:args.max_files]

    device = torch.device(args.device)
    mapping = parse_mapping(args.ql_value_mapping)
    comp_model = SimpleNamespace(Q=args.Q)

    mode = "ql"
    if args.ql_invH:
        mode = "ql_invH"
    elif args.ql_random_uniform:
        mode = "ql_random_uniform"
    elif args.ql_search:
        mode = "ql_search"

    print(f"Input path: {input_path}")
    print(f"Matched files: {len(files)}")
    print(f"Mode: {mode}")
    print(f"Device: {device}")
    print(f"Regularize H: {not args.no_regularize_h2} (sigma_reg={args.sigma_reg})")
    print(f"QL mapping: {mapping}")
    print()

    header = (
        f"{'layer':<12} | {'n':>6} | {'spearman':>10} | {'pearson':>10} | {'sp_p':>12} | {'pr_p':>12} | {'diag[min,max]':>28} | {'qlevel_counts':>20}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    all_diag_values = []
    all_mapped_values = []

    for path in files:
        layer_idx, layer_name = parse_layer_metadata(path)
        args.layer_idx = layer_idx
        args.layer_name = layer_name

        h = load_hessian(
            path=path,
            device=device,
            sigma_reg=args.sigma_reg,
            regularize=not args.no_regularize_h2,
        )
        diag_values = torch.diag(h).to(torch.float32)
        qlevel = get_ql_from_h(h, comp_model, args)
        if qlevel.device != diag_values.device:
            qlevel = qlevel.to(diag_values.device)
        mapped_qvalues = map_qlevels(qlevel, mapping)

        spearman_rho, spearman_p_value = compute_spearman(diag_values, mapped_qvalues)
        pearson_r, pearson_p_value = compute_pearson(diag_values, mapped_qvalues)
        qlevel_counts = format_qlevel_counts(qlevel)

        row = {
            "file": str(path),
            "layer_idx": layer_idx,
            "layer_name": layer_name,
            "n": int(diag_values.numel()),
            "spearman_rho": spearman_rho,
            "spearman_p_value": spearman_p_value,
            "pearson_r": pearson_r,
            "pearson_p_value": pearson_p_value,
            "diag_min": float(diag_values.min().item()),
            "diag_max": float(diag_values.max().item()),
            "mapped_min": float(mapped_qvalues.min().item()),
            "mapped_max": float(mapped_qvalues.max().item()),
            "qlevel_counts": qlevel_counts,
        }
        rows.append(row)
        all_diag_values.append(diag_values.cpu())
        all_mapped_values.append(mapped_qvalues.cpu())

        layer_tag = f"{layer_idx}_{layer_name}"
        print(
            f"{layer_tag:<12} | "
            f"{row['n']:6d} | "
            f"{row['spearman_rho']:10.6f} | "
            f"{row['pearson_r']:10.6f} | "
            f"{row['spearman_p_value']:12.4e} | "
            f"{row['pearson_p_value']:12.4e} | "
            f"[{row['diag_min']:10.4f}, {row['diag_max']:10.4f}] | "
            f"{qlevel_counts:>20}"
        )

        del h, diag_values, qlevel, mapped_qvalues
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print()
    global_diag = torch.cat(all_diag_values, dim=0)
    global_mapped = torch.cat(all_mapped_values, dim=0)
    global_spearman_rho, global_spearman_p_value = compute_spearman(global_diag, global_mapped)
    global_pearson_r, global_pearson_p_value = compute_pearson(global_diag, global_mapped)
    print(
        "Global Spearman correlation: "
        f"rho={global_spearman_rho:.6f}, p_value={global_spearman_p_value:.4e}, num_points={global_diag.numel()}"
    )
    print(
        "Global Pearson correlation: "
        f"r={global_pearson_r:.6f}, p_value={global_pearson_p_value:.4e}, num_points={global_diag.numel()}"
    )
    print(
        "Per-file summary: "
        f"mean_spearman={np.mean([row['spearman_rho'] for row in rows]):.6f}, "
        f"median_spearman={np.median([row['spearman_rho'] for row in rows]):.6f}, "
        f"mean_pearson={np.mean([row['pearson_r'] for row in rows]):.6f}, "
        f"median_pearson={np.median([row['pearson_r'] for row in rows]):.6f}"
    )

    if args.save_csv is not None:
        output_path = Path(args.save_csv).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_csv(rows, output_path)
        print(f"Saved CSV: {output_path}")


if __name__ == "__main__":
    main()
