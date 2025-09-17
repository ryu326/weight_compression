import gc
import os
import json
import logging
import random
from copy import deepcopy
from functools import partial
from glob import glob
import argparse
import yaml
import math
import torch.nn.functional as F
import torch
import pandas as pd
import wandb
from typing import Optional, Tuple, Dict
from hyper_llm_modulator.hyper_modulator import load_hypermod_checkpoint, save_lora
from hyper_llm_modulator.res_aggregator import aggregrate_results_and_save_to_file
from hyper_llm_modulator.utils import generate_simplex_points, get_layers, get_metadata, save_json, log_scalar
from hyper_llm_modulator.data import BENCHMARK_TASK_INFO
from hyper_llm_modulator.utils.lora_formatting import convert_qkv_gate_up_lora_to_splits_vllm
from hyper_llm_modulator.utils.model_loading import get_tokenizer
from hyper_llm_modulator.utils.preprocessing import preprocess_result
from hyper_llm_modulator.utils.utils import embed_texts
from hyper_llm_modulator.vllm_eval import eval

from hyper_llm_modulator.comp_lora import get_compnet_v1
from hyper_llm_modulator.comp_lora_v2 import get_compnet_v2
from hyper_llm_modulator.comp_lora_v3 import get_compnet_v3
from hyper_llm_modulator.comp_lora_v4 import get_compnet_v4
from hyper_llm_modulator.comp_lora_v5 import get_compnet_v5
from hyper_llm_modulator.utils.eval_hypermod import do_eval_task

from hyper_llm_modulator.utils.model_loading import get_emb_model_and_fns
from peft import get_peft_config, load_peft_weights, PeftConfig, PeftModel
from hyper_llm_modulator.utils import (
    get_layers,
    get_lora_module_names,
    lora_state_dict_to_tensor_dict,
    get_model_and_tokenizer,
    get_pooling_fn,
    add_full_stop,
    get_target_lora_dirs,
    lora_tensor_dict_to_state_dict,
    get_mean_lora,
    get_std_lora,
)
logger = logging.getLogger()

def load_compnet_checkpoint(save_dir, device):
    base_dir = save_dir
    if "checkpoint" in base_dir:
        base_dir = base_dir.split("checkpoint")[0]

    args = argparse.Namespace(**yaml.safe_load(open(f"{base_dir}/args.yaml", "r")))

    model, tokenizer = get_model_and_tokenizer(
        args.model_dir, train=False, requires_grad=False,
        peft_config=get_peft_config(PeftConfig.from_json_file(f"{base_dir}/adapter_config.json")),
        model_kwargs={"output_hidden_states": True, "output_attentions": False},
        device=device,
    )

    layer_indices = torch.tensor(range(len(get_layers(model))), dtype=torch.long, device=device)    
    return args, None, model, tokenizer, layer_indices

@torch.no_grad()
def _uniform_symmetric_quantize(
    x: torch.Tensor, bits: int, reduce_dims: Tuple[int, ...]
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert bits >= 2, "bits must be >= 2 for signed symmetric quantization"
    qmax = (1 << (bits - 1)) - 1
    max_abs = x.abs()
    for d in sorted(reduce_dims):
        max_abs = max_abs.max(dim=d, keepdim=True).values
    scale = torch.clamp(max_abs / qmax, min=1e-12)
    q = torch.clamp(torch.round(x / scale), min=-qmax, max=qmax)
    return q * scale, scale

# @torch.no_grad()
# def quantize_along_axis(
#     x: torch.Tensor,
#     bits: int,
#     mode: str = "tensor",          # "tensor" | "channel" | "group"
#     axis: Optional[int] = None,    # channel/group 기준 축 (음수 인덱스 허용)
#     group_size: Optional[int] = None,
# ) -> Dict[str, torch.Tensor]:
#     """
#     mode:
#       - "tensor": per-tensor (axis, group_size 무시)
#       - "channel": per-axis (축별로 1 scale)
#       - "group": 주어진 axis를 group_size씩 나눠 그룹별 scale
#     return:
#       dict(x_hat=..., scale=..., bits=..., axis=..., group_size=..., mode=...)
#     """
#     x = x.contiguous()
#     D = x.dim()
#     if mode == "tensor":
#         # 전체 텐서에 대해 scale 1개
#         reduce_dims = tuple(range(D))
#         x_hat, scale = _uniform_symmetric_quantize(x, bits, reduce_dims)
#         return dict(x_hat=x_hat, scale=scale, bits=torch.tensor(bits), mode="tensor", axis=torch.tensor(-1), group_size=torch.tensor(0))

#     assert axis is not None, "channel/group 모드에서는 axis가 필요합니다."
#     if axis < 0:
#         axis = D + axis
#     assert 0 <= axis < D

#     if mode == "channel":
#         # axis를 제외한 모든 축으로 max-abs, axis마다 하나의 scale
#         reduce_dims = tuple(d for d in range(D) if d != axis)
#         x_hat, scale = _uniform_symmetric_quantize(x, bits, reduce_dims)
#         return dict(x_hat=x_hat, scale=scale, bits=torch.tensor(bits), mode="channel", axis=torch.tensor(axis), group_size=torch.tensor(0))

#     elif mode == "group":
#         assert group_size is not None and group_size > 0
#         # 축(axis)을 그룹 단위로 잘라 그룹별로 scale
#         N = x.shape[axis]
#         n_groups = (N + group_size - 1) // group_size

#         # 그룹 축을 맨 앞으로 이동 -> [G, group_size, ...]로 reshape
#         perm = (axis,) + tuple(i for i in range(D) if i != axis)
#         inv_perm = tuple(sorted(range(D), key=lambda i: perm[i]))
#         x_t = x.permute(perm)  # [N, ...]
#         pad = n_groups * group_size - N
#         if pad > 0:
#             pad_shape = (pad,) + x_t.shape[1:]
#             x_t = torch.cat([x_t, torch.zeros(pad_shape, dtype=x.dtype, device=x.device)], dim=0)

#         x_t = x_t.view(n_groups, group_size, *x_t.shape[1:])  # [G, Gs, ...rest]
#         # 그룹별 scale: 그룹 내부 모든 원소에 대해 max-abs -> reduce dims 전부
#         reduce_dims = tuple(range(1, x_t.dim()))  # (Gs + 나머지 전부)
#         x_hat_t, scale = _uniform_symmetric_quantize(x_t, bits, reduce_dims)
#         x_hat_t = x_hat_t.view(n_groups * group_size, *x_hat_t.shape[2:])[:N]  # pad 제거
#         x_hat = x_hat_t.permute(inv_perm).contiguous()
#         # scale shape: [G, 1, 1, ...] (perm 기준)
#         return dict(x_hat=x_hat, scale=scale.squeeze(), bits=torch.tensor(bits), mode="group", axis=torch.tensor(axis), group_size=torch.tensor(group_size))
#     else:
#         raise ValueError("mode must be one of {'tensor','channel','group'}")

# @torch.no_grad()
# def quantize_lora_tensors(
#     A: torch.Tensor, B: torch.Tensor,
#     cfgA: Dict, cfgB: Dict,
# ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
#     """
#     cfg* 예시:
#       cfgA = {"mode":"channel","bits":8,"axis":-1,"group_size":None}
#       cfgB = {"mode":"group","bits":4,"axis":0,"group_size":64}
#     """
#     # A
#     mA = quantize_along_axis(
#         A, bits=cfgA.get("bits", 8),
#         mode=cfgA.get("mode", "channel"),
#         axis=cfgA.get("axis", -1),
#         group_size=cfgA.get("group_size", None),
#     )
#     # B
#     mB = quantize_along_axis(
#         B, bits=cfgB.get("bits", 8),
#         mode=cfgB.get("mode", "channel"),
#         axis=cfgB.get("axis", 1),
#         group_size=cfgB.get("group_size", None),
#     )
#     meta = {"A": mA, "B": mB}
#     return mA["x_hat"], mB["x_hat"], meta

@torch.no_grad()
def _quantize_group_along_axis(
    x: torch.Tensor,
    bits: int,
    axis_r: int,          # r축 (A: -2, B: -1)
    other_axis: int,      # 행렬의 반대 축 (A: -1, B: -2)
    group_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert group_size and group_size > 0
    D = x.dim()
    axis_r = axis_r % D
    other_axis = other_axis % D

    # 배치축 + r축 + other축 순서로 정렬
    batch_dims = tuple(i for i in range(D) if i not in (axis_r, other_axis))
    perm = batch_dims + (axis_r, other_axis)     # [..., R, C]
    inv_perm = [0]*D
    for i, p in enumerate(perm):
        inv_perm[p] = i

    x_t = x.permute(perm).contiguous()           # [..., R, C]
    R = x_t.shape[-2]
    C = x_t.shape[-1]
    n_groups = (R + group_size - 1) // group_size
    pad = n_groups * group_size - R
    if pad > 0:
        pad_shape = x_t.shape[:-2] + (pad, C)
        x_t = torch.cat([x_t, torch.zeros(pad_shape, dtype=x.dtype, device=x.device)], dim=-2)

    # [..., R, C] -> [..., G, Gs, C]
    new_shape = x_t.shape[:-2] + (n_groups, group_size, C)
    x_t = x_t.view(*new_shape)

    # 그룹 내부(Gs, C) 전체로 max-abs → per-group scale
    x_hat_t, scale = _uniform_symmetric_quantize(x_t, bits, reduce_dims=(-2, -1))  # scale: [..., G, 1, 1]

    # x_hat 복원
    x_hat_t = x_hat_t.view(*x_t.shape[:-3], n_groups * group_size, C)[..., :R, :]
    x_hat = x_hat_t.permute(inv_perm).contiguous()

    # === scale을 r축에 대해 브로드캐스트 가능한 텐서로 복원 ===
    # scale: [..., G, 1, 1] -> [..., G]
    scale_g = scale.squeeze(-1).squeeze(-1)              # [..., G]
    # [..., G] -> [..., R] (그룹 스케일을 각 행에 반복)
    scale_rows = scale_g.repeat_interleave(group_size, dim=-1)[..., :R]   # [..., R]
    # [..., R] -> [..., R, 1]  (x_t의 [..., R, C]에 대비해 C축을 1로 둔다)
    scale_rows = scale_rows.unsqueeze(-1)                 # [..., R, 1]
    # 원래 축 순서로 복원 (other_axis 위치는 길이 1이라 그대로 broadcast)
    scale_b = scale_rows.permute(inv_perm).contiguous()   # 원래 x와 같은 축 순서, r축에 상수, other축=1

    return x_hat, scale_b

@torch.no_grad()
def quantize_A(
    A: torch.Tensor,   # [L, r, in]
    bits: int,
    mode: str = "rank",     # "tensor" | "rank" | "group"
    group_size: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    A는 항상 r(행) 기준:
      - tensor:  r×in 전체 per-tensor
      - rank:    r별(per-row) 스케일 → 열(in) 축만 줄여서 scale shape=[..., r, 1]
      - group:   r축을 group_size로 잘라 그룹별 스케일
    """
    assert A.dim() >= 2
    R_dim, C_dim = -2, -1  # r, in
    if mode == "tensor":
        x_hat, scale = _uniform_symmetric_quantize(A, bits, reduce_dims=(R_dim, C_dim))
        return {"x_hat": x_hat, "scale": scale, "mode": mode, "bits": torch.tensor(bits)}
    if mode == "rank":
        x_hat, scale = _uniform_symmetric_quantize(A, bits, reduce_dims=(C_dim,))  # per-row
        return {"x_hat": x_hat, "scale": scale, "mode": mode, "bits": torch.tensor(bits)}
    if mode == "group":
        x_hat, scale = _quantize_group_along_axis(A, bits, axis_r=R_dim, other_axis=C_dim, group_size=group_size or 32)
        return {"x_hat": x_hat, "scale": scale, "mode": mode, "bits": torch.tensor(bits), "group_size": torch.tensor(group_size or 32)}
    raise ValueError("mode must be one of {'tensor','rank','group'}")

@torch.no_grad()
def quantize_B(
    B: torch.Tensor,   # [L, out, r]
    bits: int,
    mode: str = "rank",     # "tensor" | "rank" | "group"
    group_size: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    B는 항상 r(열) 기준:
      - tensor:  out×r 전체 per-tensor
      - rank:    r별(per-col) 스케일 → 행(out) 축만 줄여서 scale shape=[..., 1, r]
      - group:   r축을 group_size로 잘라 그룹별 스케일
    """
    assert B.dim() >= 2
    R_dim, C_dim = -2, -1  # out, r   (여기서 r은 열)
    if mode == "tensor":
        x_hat, scale = _uniform_symmetric_quantize(B, bits, reduce_dims=(R_dim, C_dim))
        return {"x_hat": x_hat, "scale": scale, "mode": mode, "bits": torch.tensor(bits)}
    if mode == "rank":
        x_hat, scale = _uniform_symmetric_quantize(B, bits, reduce_dims=(R_dim,))  # per-col
        return {"x_hat": x_hat, "scale": scale, "mode": mode, "bits": torch.tensor(bits)}
    if mode == "group":
        x_hat, scale = _quantize_group_along_axis(B, bits, axis_r=C_dim, other_axis=R_dim, group_size=group_size or 32)
        return {"x_hat": x_hat, "scale": scale, "mode": mode, "bits": torch.tensor(bits), "group_size": torch.tensor(group_size or 32)}
    raise ValueError("mode must be one of {'tensor','rank','group'}")

@torch.no_grad()
def quantize_lora_tensors(
    A: torch.Tensor, B: torch.Tensor,
    cfgA: Dict, cfgB: Dict,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    cfgA / cfgB 예시:
      {"mode":"rank","bits":8}            # per-rank
      {"mode":"group","bits":4,"group_size":32}
      {"mode":"tensor","bits":8}
    A는 r(행), B는 r(열)을 기준으로 고정 처리.
    """
    mA = quantize_A(
        A, bits=cfgA.get("bits", 8),
        mode=cfgA.get("mode", "rank"),
        group_size=cfgA.get("group_size", None),
    )
    mB = quantize_B(
        B, bits=cfgB.get("bits", 8),
        mode=cfgB.get("mode", "rank"),
        group_size=cfgB.get("group_size", None),
    )
    return mA["x_hat"], mB["x_hat"], {"A": mA, "B": mB}

from peft import get_peft_config, load_peft_weights
from hyper_llm_modulator.utils.lora_formatting import lora_tensor_dict_to_state_dict
from hyper_llm_modulator.utils import get_lora_module_names, get_target_lora_dirs

def _num_scales_A(shape, mode: str, group_size: int | None) -> int:
    # A: [L, r, in]  (r축 = 1)
    L, r = int(shape[0]), int(shape[1])
    if mode == "tensor":
        return L
    if mode == "rank":
        return L * r
    if mode == "group":
        if not group_size or group_size <= 0:
            raise ValueError("group 모드에서는 group_size > 0 이어야 합니다.")
        return L * int(math.ceil(r / float(group_size)))
    raise ValueError(f"unknown mode: {mode}")

def _num_scales_B(shape, mode: str, group_size: int | None) -> int:
    # B: [L, out, r] (r축 = 2)
    L, r = int(shape[0]), int(shape[2])
    if mode == "tensor":
        return L
    if mode == "rank":
        return L * r
    if mode == "group":
        if not group_size or group_size <= 0:
            raise ValueError("group 모드에서는 group_size > 0 이어야 합니다.")
        return L * int(math.ceil(r / float(group_size)))
    raise ValueError(f"unknown mode: {mode}")


@torch.no_grad()
def generate_loras_with_quantization(
    args,
    model,                  # 베이스 LLM (이름 매핑용)
    layer_indices,
    save_dir,
    device,
    eval_ds_info,
):
    """
    args.quant_cfg 예시:
      args.quant_cfg = {
        "A": {"mode":"channel","bits":8,"axis":-1,"group_size":None},
        "B": {"mode":"channel","bits":8,"axis":0,"group_size":None},
      }
    비트/모드/축을 태스크별로 다르게 하고 싶으면 함수 내부에서 분기하세요.
    """
    # quant_cfg = getattr(args, "quant_cfg", {
    #     "A": {"mode":"channel","bits":8,"axis":-1,"group_size":None},
    #     "B": {"mode":"channel","bits":8,"axis":1,"group_size":None},
    # })
    
    quant_cfg = getattr(args, "quant_cfg", None)
    lora_dirs_map = get_target_lora_dirs(list(eval_ds_info.keys()), args.model_dir)
    module_names = get_lora_module_names(model, args.target_modules, layer_indices)

    all_lora_dirs = {task: [] for task in lora_dirs_map}
    save_dicts    = {task: [] for task in lora_dirs_map}
    metric = {}

    for task, src_lora_dir in lora_dirs_map.items():
        tgt_save_dir = f"{save_dir}/generated_loras/{task}/quantized/lora_0"
        os.makedirs(tgt_save_dir, exist_ok=True)

        # 원본 LoRA 로드 → 텐서 dict
        tgt_sd = load_peft_weights(src_lora_dir)
        td = lora_state_dict_to_tensor_dict(
            tgt_sd, args.target_modules, layer_indices, device=device
        )

        A_hat_all, B_hat_all = {}, {}
        task_metrics = {}

        for layer_type in args.target_modules:
            A = td["A"][layer_type]  # [L, r, in]
            B = td["B"][layer_type]  # [L, out, r]

            # (옵션) z-score 정규화 및 역정규화는 여기선 생략. 원 데이터 직접 양자화.
            A_q, B_q, meta = quantize_lora_tensors(
                A, B,
                cfgA=quant_cfg.get("A", {}),
                cfgB=quant_cfg.get("B", {}),
            )

            # 저장용
            A_hat_all[layer_type] = A_q
            B_hat_all[layer_type] = B_q

            # ---- 메트릭 계산 ----
            mse_A = F.mse_loss(A_q, A, reduction="mean").item()
            mse_B = F.mse_loss(B_q, B, reduction="mean").item()

            # deltaW_hat = torch.bmm(B_q, A_q)          # [L, out, in]
            # deltaW_tgt = torch.bmm(B, A)
            # mse_deltaW = F.mse_loss(deltaW_hat, deltaW_tgt, reduction="mean").item()
            mse_deltaW = None

            cfgA = {**{"mode":"rank","bits":8,"group_size":None,"scale_bits":16},
                    **quant_cfg.get("A", {})}
            cfgB = {**{"mode":"rank","bits":8,"group_size":None,"scale_bits":16},
                    **quant_cfg.get("B", {})}

            bits_w_A = int(cfgA["bits"]) * A.numel()
            bits_w_B = int(cfgB["bits"]) * B.numel()

            n_scales_A = _num_scales_A(A.shape, cfgA["mode"], cfgA.get("group_size", None))
            n_scales_B = _num_scales_B(B.shape, cfgB["mode"], cfgB.get("group_size", None))

            bits_s_A = int(cfgA["scale_bits"]) * n_scales_A
            bits_s_B = int(cfgB["scale_bits"]) * n_scales_B

            bits_A_total = bits_w_A + bits_s_A
            bits_B_total = bits_w_B + bits_s_B
            bits_total   = bits_A_total + bits_B_total

            bppA = bits_A_total / max(1, A.numel())
            bppB = bits_B_total / max(1, B.numel())
            bpp_total = bits_total / max(1, (A.numel() + B.numel()))

            # bpp (= 파라미터 1개당 평균 비트 수)
            bppA = bits_A_total / max(1, A.numel())
            bppB = bits_B_total / max(1, B.numel())
            bpp_total = (bits_A_total + bits_B_total) / max(1, (A.numel() + B.numel()))

            task_metrics[layer_type] = {
                "mse_A": mse_A,
                "mse_B": mse_B,
                "mse_deltaW": mse_deltaW,
                # 원래 저장하던 raw 비트도 유지하고,
                "bits_A_weights_only": bits_w_A,
                "bits_B_weights_only": bits_w_B,
                "bits_A_scales": bits_s_A,
                "bits_B_scales": bits_s_B,
                "bits_A_total": bits_A_total,
                "bits_B_total": bits_B_total,
                "bits_total": bits_total,
                "bppA": bppA,
                "bppB": bppB,
                "bpp_total": bpp_total,
                "num_params_A": int(A.numel()),
                "num_params_B": int(B.numel()),
                # 선택: 최종 사용된 양자화 설정도 저장
                "quant_A": {k: (int(v) if isinstance(v, bool) or isinstance(v, (int,)) else v) for k, v in cfgA.items()},
                "quant_B": {k: (int(v) if isinstance(v, bool) or isinstance(v, (int,)) else v) for k, v in cfgB.items()},
            }
            # ---------------
            # ---------------------

        # 텐서딕트를 state_dict로 변환 후 저장
        recon_td = {"A": A_hat_all, "B": B_hat_all}
        recon_sd = lora_tensor_dict_to_state_dict(
            recon_td, module_names, args.target_modules, layer_indices
        )
        # 원본 adapter_config 재사용
        peft_cfg = get_peft_config(PeftConfig.from_json_file(f"{src_lora_dir}/adapter_config.json"))
        save_lora(recon_sd, peft_cfg, tgt_save_dir)

        # 메트릭 JSON 저장
        metrics_path = os.path.join(tgt_save_dir, "quant_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(task_metrics, f, indent=2)

        metric[task] = task_metrics
        all_lora_dirs[task].append(tgt_save_dir)
        save_dicts[task].append({"src_lora": src_lora_dir, "split": "quantized", "lora_dir": tgt_save_dir})

    return all_lora_dirs, save_dicts, metric

def quant_cfg_to_str(quant_cfg: dict) -> str:
    # {"A": {...}, "B": {...}} → "A_mode-channel_bits-8_axis--1|B_mode-channel_bits-8_axis-1"
    parts = []
    for name in ["A", "B"]:
        cfg = quant_cfg.get(name, {})
        s = "_".join(f"{k}-{v}" for k, v in cfg.items() if v is not None)
        parts.append(f"{name}_{s}")
    return "|".join(parts)


def eval_quantized_lora(save_dir, device, curstep, full_eval, use_icl=False, quant_cfg = None):
    args, _, base_model, tokenizer, layer_indices = load_compnet_checkpoint(save_dir, device = device)
    chat_template = tokenizer.chat_template

    args.quant_cfg = quant_cfg
    quant_str = quant_cfg_to_str(args.quant_cfg)
    if full_eval:
        quant_str = quant_str + '_full_eval'
    else:
        quant_str = quant_str + '_val'
        
    # save_dir = f"{save_dir}/quant_{quant_str}"
    # os.makedirs(save_dir, exist_ok=True)

    eval_ds_info = deepcopy(args.eval_ds_info)
    if not full_eval:
        eval_ds_info = {k: v for k, v in eval_ds_info.items() if k in BENCHMARK_TASK_INFO}
        for k in BENCHMARK_TASK_INFO:
            eval_ds_info[k]["ds_kwargs"] = BENCHMARK_TASK_INFO[k]
    for ds in list(eval_ds_info.keys()):
        if ds.startswith("lol_"):
            eval_ds_info.pop(ds)

    # 2) 양자화된 LoRA 생성
    all_lora_dirs, save_dicts, metric = generate_loras_with_quantization(
        args=args,
        model=base_model,
        layer_indices=layer_indices,
        save_dir=save_dir,
        device=device,
        eval_ds_info=eval_ds_info,
    )

    del base_model, tokenizer, layer_indices
    gc.collect()
    torch.cuda.empty_cache()

    # 3) 태스크 평가
    for eval_ds in eval_ds_info:
        ds_kwargs = eval_ds_info[eval_ds].get("ds_kwargs")
        logger.info(f"++++++++ Start eval {eval_ds} ++++++++++")
        try:
            results = do_eval_task(
                args.model_dir,  # 'mistralai/Mistral-7B-Instruct-v0.2',
                chat_template,
                save_dir,
                all_lora_dirs[eval_ds],
                eval_ds,
                save_dicts[eval_ds],
                ds_kwargs,
                use_icl,
                curstep=quant_str,
            )
            print(results)
        except Exception as e:
            logger.warning(f"Eval failed on {eval_ds}: {e}")
        
        # import gc, torch
        # gc.collect(); torch.cuda.empty_cache()
        # try:
        #     import ray
        #     ray.shutdown()
        # except Exception:
        #     pass
                
    # 4) 집계 및 저장
    df = aggregrate_results_and_save_to_file(
        base_model_dir=args.model_dir,
        mt_lora_dir=args.mt_lora_path,
        hypermod_dir=save_dir,
        hypermod_name="quant_lora",
        curstep=quant_str
    )
    metrics_path = os.path.join(save_dir, f"eval_results/quant_metrics_it{quant_str}.json")
    with open(metrics_path, "w") as f:
        json.dump(metric, f, indent=2)
    return None



# def eval_quantized_lora_checkpoint(checkpoint_path, device, curstep, full_eval, use_icl=False, quant_cfg = None):
#     args, _, base_model, tokenizer, layer_indices = load_compnet_checkpoint(checkpoint_path, device = device)
#     chat_template = tokenizer.chat_template
    
#     save_dir = os.path.dirname(checkpoint_path)
#     args.quant_cfg = quant_cfg
#     quant_str = quant_cfg_to_str(args.quant_cfg)
#     if full_eval:
#         quant_str = quant_str + '_full_eval'
#     else:
#         quant_str = quant_str + '_val'
        
#     # save_dir = f"{save_dir}/quant_{quant_str}"
#     # os.makedirs(save_dir, exist_ok=True)

#     eval_ds_info = deepcopy(args.eval_ds_info)
#     if not full_eval:
#         eval_ds_info = {k: v for k, v in eval_ds_info.items() if k in BENCHMARK_TASK_INFO}
#         for k in BENCHMARK_TASK_INFO:
#             eval_ds_info[k]["ds_kwargs"] = BENCHMARK_TASK_INFO[k]
#     for ds in list(eval_ds_info.keys()):
#         if ds.startswith("lol_"):
#             eval_ds_info.pop(ds)

#     # 2) 양자화된 LoRA 생성
#     all_lora_dirs, save_dicts, metric = generate_loras_with_quantization(
#         args=args,
#         model=base_model,
#         layer_indices=layer_indices,
#         save_dir=save_dir,
#         device=device,
#         eval_ds_info=eval_ds_info,
#     )

#     del base_model, tokenizer, layer_indices
#     gc.collect()
#     torch.cuda.empty_cache()

#     # 3) 태스크 평가
#     for eval_ds in eval_ds_info:
#         ds_kwargs = eval_ds_info[eval_ds].get("ds_kwargs")
#         logger.info(f"++++++++ Start eval {eval_ds} ++++++++++")
#         try:
#             results = do_eval_task(
#                 args.model_dir,  # 'mistralai/Mistral-7B-Instruct-v0.2',
#                 chat_template,
#                 save_dir,
#                 all_lora_dirs[eval_ds],
#                 eval_ds,
#                 save_dicts[eval_ds],
#                 ds_kwargs,
#                 use_icl,
#                 curstep=quant_str,
#             )
#             print(results)
#         except Exception as e:
#             logger.warning(f"Eval failed on {eval_ds}: {e}")
        
#         # import gc, torch
#         # gc.collect(); torch.cuda.empty_cache()
#         # try:
#         #     import ray
#         #     ray.shutdown()
#         # except Exception:
#         #     pass
                
#     # 4) 집계 및 저장
#     df = aggregrate_results_and_save_to_file(
#         base_model_dir=args.model_dir,
#         mt_lora_dir=args.mt_lora_path,
#         hypermod_dir=save_dir,
#         hypermod_name="quant_lora",
#         curstep=quant_str
#     )
#     metrics_path = os.path.join(save_dir, f"eval_results/quant_metrics_it{quant_str}.json")
#     with open(metrics_path, "w") as f:
#         json.dump(metric, f, indent=2)
#     return None
