"""Llama-3-8B weight-block dataset.

Three normalization paths:
- `normalize ∈ {'col', 'row'}`: delegated to NWC's
  `get_normed_patch_weight_from_hf` (per-column / per-row std).  NWC's loader
  also divides by global std at the end, so `dataset_stats['std'] == 1`.
- `normalize == 'tensor'`: implemented locally — each `nn.Linear` weight is
  divided by its own scalar std, then global std is divided out.
  `dataset_stats['std'] == 1`.
- `normalize is None`: **truly raw** — no per-row/col/tensor normalization,
  no global std div.  `dataset_stats['std']` is the empirical std of the
  raw patches; `dataset_stats['mean']` is the empirical mean.  The codec
  uses these as `scale` / `shift`.
"""
import sys

import torch
from torch.utils.data import Dataset

# NWC sits next to NWC_v2 in the repo root.  Append (not insert) so we don't
# shadow NWC_v2's own top-level modules.
_NWC_ROOT = "/home/jgryu/workspace/weight_compression"
if _NWC_ROOT not in sys.path:
    sys.path.append(_NWC_ROOT)

from NWC.datasets.get_llm_weight import (  # noqa: E402
    get_normed_patch_weight_from_hf,
    get_blocks,
    get_named_linears,
)

_DEFAULT_HF_PATH = (
    "/home/jgryu/workspace/weight_compression/Wparam_dataset/hf_model/"
    "meta-llama--Meta-Llama-3-8B"
)


class LlamaWeightBlockDataset(Dataset):
    def __init__(self, data_split, dataset_stats, input_size: int):
        self.weights = data_split["weight"]  # (N, L, I)
        self.input_size = int(input_size)
        self.mean = float(dataset_stats["mean"])
        self.std = float(dataset_stats["std"])
        if self.weights.shape[-1] != self.input_size:
            raise ValueError(
                f"Llama dataset's last dim {self.weights.shape[-1]} "
                f"!= input_size {self.input_size}"
            )

    def __len__(self):
        return self.weights.shape[0]

    def __getitem__(self, i):
        return {"weight_block": self.weights[i]}  # (L, I)


def _get_patch_weight_no_normalize(
    hf_path: str, direction: str, L: int = 1024, I: int = 16,
):
    """No normalization at all.  Returns raw patches + their empirical
    mean/std as `dataset_stats`."""
    from einops import rearrange
    from transformers import AutoModelForCausalLM
    from tqdm import tqdm
    import pprint

    assert direction in ("col", "row")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(hf_path, local_files_only=True)
    layers = get_blocks(model)

    raw_data = {"weight": []}
    for i in tqdm(range(len(layers)), desc="layers"):
        named_linears = get_named_linears(layers[i])
        for n, m in named_linears.items():
            W = m.weight.data.detach().to(device)
            w = (W.T if direction == "col" else W).to("cpu")
            patches = rearrange(w, "(h p1) (w p2) -> (h w) p1 p2", p1=L, p2=I)
            raw_data["weight"].append(patches)

    raw_data["weight"] = torch.cat(raw_data["weight"], dim=0)
    print(f"weight total shape (no normalize): {raw_data['weight'].shape}")

    indices = torch.randperm(len(raw_data["weight"]))
    split_index = int(len(raw_data["weight"]) - 500)
    train_idx, val_idx = indices[:split_index], indices[split_index:]
    dataset = {
        "train": {"weight": raw_data["weight"][train_idx]},
        "val": {"weight": raw_data["weight"][val_idx]},
    }
    print("train Weight:", dataset["train"]["weight"].shape,
          "val:", dataset["val"]["weight"].shape)

    dataset_stats = {}
    for split in ("train", "val"):
        d = dataset[split]
        dataset_stats[split] = {
            "mean": float(d["weight"].mean().item()),
            "std": float(d["weight"].std().item()),
            "mean_channel": None,
            "std_channel": None,
        }
    print("---- Dataset_stats (no normalize, raw) ----")
    pprint.pprint(dataset_stats)
    return dataset, dataset_stats


def _get_patch_weight_tensor_normalize(
    hf_path: str, direction: str, L: int = 1024, I: int = 16,
):
    """Per-Linear-weight scalar std normalization (`normalize == 'tensor'`).

    Returns the same `(dataset, dataset_stats)` shape as
    `get_normed_patch_weight_from_hf`.  After this call, train/val patches
    have been divided by the *per-tensor* std, then by the *global* std of
    the resulting patches; `dataset_stats['std'] = 1`.
    """
    from einops import rearrange
    from transformers import AutoModelForCausalLM
    from tqdm import tqdm
    import pprint

    assert direction in ("col", "row")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(hf_path, local_files_only=True)
    layers = get_blocks(model)

    raw_data = {"weight": []}
    for i in tqdm(range(len(layers)), desc="layers"):
        named_linears = get_named_linears(layers[i])
        for n, m in named_linears.items():
            W = m.weight.data.detach().to(device)
            tensor_std = W.std().clamp_min(1e-12)
            Wr = W / tensor_std

            if direction == "col":
                w = Wr.T.to("cpu")
            else:
                w = Wr.to("cpu")
            patches = rearrange(w, "(h p1) (w p2) -> (h w) p1 p2", p1=L, p2=I)
            raw_data["weight"].append(patches)

    raw_data["weight"] = torch.cat(raw_data["weight"], dim=0)
    print(f"weight total shape (tensor-normalized): {raw_data['weight'].shape}")

    indices = torch.randperm(len(raw_data["weight"]))
    split_index = int(len(raw_data["weight"]) - 500)
    train_idx, val_idx = indices[:split_index], indices[split_index:]

    dataset = {
        "train": {"weight": raw_data["weight"][train_idx]},
        "val": {"weight": raw_data["weight"][val_idx]},
    }
    print("train Weight:", dataset["train"]["weight"].shape,
          "val:", dataset["val"]["weight"].shape)

    dataset_stats = {}
    for split in ("train", "val"):
        data = dataset[split]
        dataset_stats[split] = {
            "mean": float(data["weight"].mean().item()),
            "std": float(data["weight"].std().item()),
            "mean_channel": None,
            "std_channel": None,
        }
    print("---- Dataset_stats before std normalization (tensor mode) ----")
    pprint.pprint(dataset_stats)

    # Final global std div, mirroring the upstream loader's convention.
    for split in ("train", "val"):
        s = dataset_stats[split]["std"]
        dataset[split]["weight"] = dataset[split]["weight"] / max(s, 1e-12)
        dataset_stats[split]["std"] = 1.0
    print("---- Dataset_stats after std normalization (tensor mode) -----")
    pprint.pprint(dataset_stats)

    return dataset, dataset_stats


def get_dataset_llama8b(args):
    hf_path = getattr(args, "hf_path", None) or _DEFAULT_HF_PATH
    L = int(getattr(args, "seq_len", 1024))
    I = int(getattr(args, "input_size", 16))
    direction = str(getattr(args, "direction", "row"))
    normalize = getattr(args, "normalize", None)
    if isinstance(normalize, str) and normalize.lower() == "none":
        normalize = None

    if normalize == "tensor":
        data, stats = _get_patch_weight_tensor_normalize(
            hf_path=hf_path, direction=direction, L=L, I=I,
        )
    elif normalize is None:
        # Truly raw — no per-element / global normalization
        data, stats = _get_patch_weight_no_normalize(
            hf_path=hf_path, direction=direction, L=L, I=I,
        )
    else:
        # 'row' or 'col' — delegate to NWC's loader (also applies global std div)
        data, stats = get_normed_patch_weight_from_hf(
            hf_path=hf_path, direction=direction, L=L, I=I, normalize=normalize,
        )

    train = LlamaWeightBlockDataset(data["train"], stats["train"], input_size=I)
    val = LlamaWeightBlockDataset(data["val"], stats["val"], input_size=I)
    return (
        train, val,
        float(stats["train"]["std"]), float(stats["val"]["std"]),
        float(stats["train"]["mean"]), float(stats["val"]["mean"]),
    )
