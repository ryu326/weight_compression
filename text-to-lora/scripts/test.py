#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, argparse, hashlib, sys
from typing import Dict, Tuple, List
import torch
from safetensors.torch import load_file as load_safetensors

def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def compare_json(a_path: str, b_path: str) -> Tuple[bool, Dict]:
    if not (os.path.exists(a_path) and os.path.exists(b_path)):
        return False, {"missing": [p for p in (a_path, b_path) if not os.path.exists(p)]}
    with open(a_path, "r") as fa, open(b_path, "r") as fb:
        ja, jb = json.load(fa), json.load(fb)
    same = ja == jb
    diffs = {}
    if not same:
        keys = set(ja.keys()) | set(jb.keys())
        for k in sorted(keys):
            va, vb = ja.get(k, "<MISSING>"), jb.get(k, "<MISSING>")
            if va != vb:
                diffs[k] = {"A": va, "B": vb}
    return same, diffs

def allclose_stats(tA: torch.Tensor, tB: torch.Tensor, rtol: float, atol: float):
    if tA.shape != tB.shape:
        return {"shape_equal": False, "rtol": rtol, "atol": atol}
    # 같은 dtype으로 비교(정밀도 차이 방지). 필요시 float32로 올려 비교.
    if tA.dtype != tB.dtype:
        tA = tA.to(torch.float32); tB = tB.to(torch.float32)
    diff = (tA - tB).abs()
    return {
        "shape_equal": True,
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "allclose": bool(torch.allclose(tA, tB, rtol=rtol, atol=atol)),
        "rtol": rtol, "atol": atol,
    }

def compare_safetensors(a_path: str, b_path: str, rtol: float, atol: float):
    if not (os.path.exists(a_path) and os.path.exists(b_path)):
        return {"status": "missing", "missing": [p for p in (a_path, b_path) if not os.path.exists(p)]}
    # 1) 파일 해시가 같으면 완전 동일
    ha, hb = sha256(a_path), sha256(b_path)
    if ha == hb:
        return {"status": "identical_file", "sha256": ha}
    # 2) 텐서별 비교
    A = load_safetensors(a_path)
    B = load_safetensors(b_path)
    keys = sorted(set(A.keys()) | set(B.keys()))
    per_key = {}
    n_missing = 0
    n_mismatch = 0
    for k in keys:
        if k not in A or k not in B:
            per_key[k] = {"missing_in": "A" if k not in A else "B"}
            n_missing += 1
            continue
        stats = allclose_stats(A[k], B[k], rtol=rtol, atol=atol)
        per_key[k] = stats
        if not (stats.get("shape_equal") and stats.get("allclose")):
            n_mismatch += 1
    return {
        "status": "compared",
        "sha256": {"A": ha, "B": hb},
        "n_keys_A": len(A.keys()),
        "n_keys_B": len(B.keys()),
        "n_missing": n_missing,
        "n_mismatch": n_mismatch,
        "details": per_key,
    }

def discover_pairs(root: str) -> List[Tuple[str, str, str]]:
    """
    root/generated_loras/<task>/compressed/lora_0  와
    root/generated_loras/<task>_full_eval/compressed/lora_0  를 자동 매칭
    """
    gen = os.path.join(root, "generated_loras")
    tasks = [d for d in os.listdir(gen) if os.path.isdir(os.path.join(gen, d))]
    pairs = []
    for t in tasks:
        if t.endswith("_full_eval"):
            base = t[:-10]  # strip "_full_eval"
            a = os.path.join(gen, base, "compressed", "lora_0")
            b = os.path.join(gen, t,    "compressed", "lora_0")
            if os.path.isdir(a) and os.path.isdir(b):
                pairs.append((base, a, b))
        else:
            # 역방향도 커버(풀 디렉토리만 있는 경우 대비)
            f = t + "_full_eval"
            a = os.path.join(gen, t, "compressed", "lora_0")
            b = os.path.join(gen, f, "compressed", "lora_0")
            if os.path.isdir(a) and os.path.isdir(b):
                pairs.append((t, a, b))
    # 중복 제거
    seen = set(); uniq = []
    for name, a, b in pairs:
        key = (name, os.path.realpath(a), os.path.realpath(b))
        if key not in seen:
            seen.add(key); uniq.append((name, a, b))
    return sorted(uniq, key=lambda x: x[0])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="compnet 작업 루트(예: .../train_outputs/compnet_recon/compnet/v4_xxx)")
    ap.add_argument("--rtol", type=float, default=0.0, help="torch.allclose rtol")
    ap.add_argument("--atol", type=float, default=0.0, help="torch.allclose atol")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    pairs = discover_pairs(args.root)
    if not pairs:
        print("No task pairs found under:", args.root); sys.exit(1)

    summary = []
    for task, dirA, dirB in pairs:
        print(f"\n=== Task: {task} ===")
        a_sft = os.path.join(dirA, "adapter_model.safetensors")
        b_sft = os.path.join(dirB, "adapter_model.safetensors")
        a_cfg = os.path.join(dirA, "adapter_config.json")
        b_cfg = os.path.join(dirB, "adapter_config.json")

        # config 비교
        cfg_same, cfg_diff = compare_json(a_cfg, b_cfg)
        print(f"[config] same={cfg_same}")
        if not cfg_same and args.verbose:
            print(json.dumps(cfg_diff, indent=2, ensure_ascii=False))

        # safetensors 비교
        comp = compare_safetensors(a_sft, b_sft, rtol=args.rtol, atol=args.atol)
        status = comp.get("status")
        print(f"[weights] status={status}")
        if status == "identical_file":
            print(f"  sha256={comp['sha256']}")
            n_mismatch = 0
        elif status == "compared":
            print(f"  sha256(A)={comp['sha256']['A']}\n  sha256(B)={comp['sha256']['B']}")
            print(f"  keys: A={comp['n_keys_A']} B={comp['n_keys_B']} | missing={comp['n_missing']} mismatched={comp['n_mismatch']}")
            n_mismatch = comp["n_mismatch"]
            if args.verbose and (n_mismatch > 0 or comp["n_missing"] > 0):
                # 상위 10개만 요약 출력
                cnt = 0
                for k, v in comp["details"].items():
                    if isinstance(v, dict) and (not v.get("shape_equal", True) or not v.get("allclose", True) or "missing_in" in v):
                        print(f"  - {k}: {v}")
                        cnt += 1
                        if cnt >= 10: break
        else:
            print(f"  missing files: {comp.get('missing')}")
            n_mismatch = -1

        summary.append({
            "task": task,
            "config_same": cfg_same,
            "weights_status": status,
            "mismatched_tensors": n_mismatch
        })

    print("\n=== Summary ===")
    for s in summary:
        print(s)

if __name__ == "__main__":
    main()
