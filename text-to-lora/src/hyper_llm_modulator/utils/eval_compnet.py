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
from hyper_llm_modulator.utils.model_loading import get_emb_model_and_fns
from peft import get_peft_config, load_peft_weights, PeftConfig, PeftModel


logger = logging.getLogger()

def load_compnet_checkpoint(checkpoint_path, device):
    base_dir = os.path.dirname(checkpoint_path)
    if "checkpoint" in base_dir:
        base_dir = base_dir.split("checkpoint")[0]

    args = argparse.Namespace(**yaml.safe_load(open(f"{base_dir}/args.yaml", "r")))

    model, tokenizer = get_model_and_tokenizer(
        args.model_dir, train=False, requires_grad=False,
        peft_config=get_peft_config(PeftConfig.from_json_file(f"{base_dir}/adapter_config.json")),
        model_kwargs={"output_hidden_states": True, "output_attentions": False},
        device=device,
    )

    # 3) 압축모델 생성 및 가중치 로드
    layer_indices = torch.tensor(range(len(get_layers(model))), dtype=torch.long, device=device)
    args.compnet_v = getattr(args, "compnet_v", 1)
    if args.compnet_v == 1:
        get_compnet = get_compnet_v1
    elif args.compnet_v == 2:    
        get_compnet = get_compnet_v2
    elif args.compnet_v == 3:    
        get_compnet = get_compnet_v3
    elif args.compnet_v == 4:    
        get_compnet = get_compnet_v4
    elif args.compnet_v == 5:    
        get_compnet = get_compnet_v5
    comp_model = get_compnet(args, "lora", device, model, layer_indices, task_emb_size=None, from_scratch=False)
    state_dict = torch.load(checkpoint_path, map_location=device)
    if 'model_state' in state_dict:
        state_dict = state_dict['model_state']
    comp_model.load_state_dict(state_dict, strict=True)
    comp_model.eval()
    comp_model.update()
    
    return args, comp_model, model, tokenizer, layer_indices

# def _sum_log_likelihood(t: torch.Tensor) -> torch.Tensor:
#     # log(0) 방지
#     return torch.log(t.clamp_min(1e-12)).reshape(t.shape[0], -1).sum()

# def _bpp_from_branch_liks(branch_liks: dict, num_params: int) -> float:
#     """
#     branch_liks: {"y": Tensor[...,], "h": Tensor[...,]} 형태 (차원은 [L, *] 또는 [L, R, *] 등 자유)
#     num_params : bpp 정규화 분모 (A면 A_in.numel(), B면 B_in.numel())
#     """
#     s = 0.0
#     for k in ("y", "h"):
#         if k in branch_liks and branch_liks[k] is not None:
#             s = s + _sum_log_likelihood(branch_liks[k])
#     bpp = s / (-math.log(2) * num_params)
#     return float(bpp)

def _sum_log_likelihood(t: torch.Tensor) -> torch.Tensor:
    # t: [L, ...] 또는 [L*R, ...] 등. 안전하게 클램프 후 전체 합산.
    return torch.log(t.clamp_min(1e-12)).reshape(t.shape[0], -1).sum()

def _bpp_from_list(liks: list[torch.Tensor | None], num_params: int) -> float:
    s = torch.tensor(0.0, device=liks[0].device if liks and liks[0] is not None else "cpu")
    for x in liks:
        if x is not None:
            s = s + _sum_log_likelihood(x)
    return float(s / (-math.log(2) * max(1, num_params)))

def split_bpp_from_likelihoods(likelihoods: dict, A_in: torch.Tensor, B_in: torch.Tensor):
    """
    likelihoods가 다음 중 하나의 형태를 지원:
      1) {"A":{"y":..., "h":...}, "B":{"y":..., "h":...}}
      2) {"yA":..., "hA":..., "yB":..., "hB":...}
    A_in: [L, R, in]   B_in: [L, out, R]
    """
    # 분기별 텐서 꺼내기
    if "A" in likelihoods:  # nested 형태
        yA = likelihoods["A"].get("y"); hA = likelihoods["A"].get("h")
        yB = likelihoods["B"].get("y"); hB = likelihoods["B"].get("h")
    else:                   # flat 키 형태
        yA = likelihoods.get("yA"); hA = likelihoods.get("hA")
        yB = likelihoods.get("yB"); hB = likelihoods.get("hB")

    num_params_A = A_in.numel()
    num_params_B = B_in.numel()

    bpp_A = _bpp_from_list([yA, hA], num_params_A)
    bpp_B = _bpp_from_list([yB, hB], num_params_B)

    # 전체 bpp(가중 평균; 참고용)
    total_bpp = (
        (bpp_A * num_params_A) + (bpp_B * num_params_B)
    ) / float(max(1, num_params_A + num_params_B))

    return bpp_A, bpp_B, total_bpp

@torch.no_grad()
def generate_loras_with_compnet(
    args,
    comp_model,
    model,                  # 베이스 LLM (이름 매핑용)
    layer_indices,
    save_dir,
    device,
    eval_ds_info,
):
    # 평가할 타깃 LoRA 경로들 얻기 (train/eval 셋에 맞춰)
    lora_dirs_map = get_target_lora_dirs(list(eval_ds_info.keys()), args.model_dir)

    # 모듈명 매핑(저장용 이름 생성)
    module_names = get_lora_module_names(model, args.target_modules, layer_indices)
    all_lora_dirs = {task: [] for task in lora_dirs_map}
    save_dicts    = {task: [] for task in lora_dirs_map}

    metric = {}
    for task, src_lora_dir in lora_dirs_map.items():
        tgt_save_dir = f"{save_dir}/generated_loras/{task}/compressed/lora_0"
        # if full_eval:
        #     tgt_save_dir = f"{save_dir}/generated_loras/{task}_full_eval/compressed/lora_0"
        # else:
        #     tgt_save_dir = f"{save_dir}/generated_loras/{task}/compressed/lora_0"
        os.makedirs(tgt_save_dir, exist_ok=True)

        # 원본 LoRA 로드 → 텐서딕트
        tgt_sd = load_peft_weights(src_lora_dir)
        td = lora_state_dict_to_tensor_dict(
            tgt_sd, args.target_modules, layer_indices, device=device
        )

        # 모듈별로 압축/복원 실행
        A_hat_all, B_hat_all = {}, {}
        task_metrics = {}
        for layer_type in args.target_modules:
            A = td["A"][layer_type]            # [L, r, in]
            B = td["B"][layer_type]            # [L, out, r]

            # (옵션) z-score 정규화
            if getattr(args, "pred_z_score", False):
                meanA = comp_model.mean_recon_target["A"][layer_type][layer_indices]
                meanB = comp_model.mean_recon_target["B"][layer_type][layer_indices]
                stdA  = comp_model.std_recon_target["A"][layer_type][layer_indices]
                stdB  = comp_model.std_recon_target["B"][layer_type][layer_indices]
                A_in  = (A - meanA) / (stdA + 1e-10)
                B_in  = (B - meanB) / (stdB + 1e-10)
            else:
                A_in, B_in = A, B

            # 압축 → 복원
            # strings, side_info = comp_model.compress(
            #     layer_type=layer_type,
            #     layer_indices=layer_indices,
            #     lora_A=A_in, lora_B=B_in,
            # )
            # A_hat, B_hat = comp_model.decompress(strings, side_info)

            out = comp_model.forward(
                layer_type=layer_type,
                layer_indices=layer_indices,
                lora_A=A_in,
                lora_B=B_in,
            )
            A_hat, B_hat = out["A_hat"], out["B_hat"]

            # (옵션) 역정규화
            if getattr(args, "pred_z_score", False):
                A_hat = A_hat * (stdA + 1e-10) + meanA
                B_hat = B_hat * (stdB + 1e-10) + meanB

            A_hat_all[layer_type] = A_hat
            B_hat_all[layer_type] = B_hat
            
            
            # ---- 메트릭 (A/B 각각의 bpp + MSE들) ----
            mse_A = F.mse_loss(A_hat, A, reduction="mean").item()
            mse_B = F.mse_loss(B_hat, B, reduction="mean").item()
            deltaW_hat = torch.bmm(B_hat, A_hat)          # [L, out, in]
            deltaW_tgt = torch.bmm(B, A)
            mse_deltaW = F.mse_loss(deltaW_hat, deltaW_tgt, reduction="mean").item()
            
            
            bpp_A = bpp_B = None
            bpp_total = None
            likelihoods = out.get("likelihoods", None)
            if isinstance(likelihoods, dict):
                bpp_A, bpp_B, bpp_total = split_bpp_from_likelihoods(out["likelihoods"], A_in, B_in)
                    
            task_metrics[layer_type] = {
                "mse_A": mse_A,
                "mse_B": mse_B,
                "mse_deltaW": mse_deltaW,
                "bpp_A": bpp_A,
                "bpp_B": bpp_B,
                "bpp_total": bpp_total,
                "num_params_A": int(A_in.numel()),
                "num_params_B": int(B_in.numel()),
            }
            # ----------------------------------------
        # 텐서딕트를 state_dict로 변환 후 저장
        recon_td = {"A": A_hat_all, "B": B_hat_all}
        recon_sd = lora_tensor_dict_to_state_dict(
            recon_td, module_names, args.target_modules, layer_indices
        )

        # 모듈별 메트릭 JSON 저장
        metrics_path = os.path.join(tgt_save_dir, "comp_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(task_metrics, f, indent=2)
            
        metric[task] = task_metrics
        
        # 어댑터 설정은 원본 것을 재사용
        peft_cfg = get_peft_config(PeftConfig.from_json_file(f"{src_lora_dir}/adapter_config.json"))
        save_lora(recon_sd, peft_cfg, tgt_save_dir)

        all_lora_dirs[task].append(tgt_save_dir)
        save_dicts[task].append({"src_lora": src_lora_dir, "split": "compressed", "lora_dir": tgt_save_dir})

    return all_lora_dirs, save_dicts, metric


def eval_compnet_checkpoint(checkpoint_path, device, curstep, full_eval, use_icl=False):
    args, comp_model, base_model, tokenizer, layer_indices = load_compnet_checkpoint(checkpoint_path, device)
    chat_template = tokenizer.chat_template
    save_dir = os.path.dirname(checkpoint_path)
    eval_ds_info = deepcopy(args.eval_ds_info)
    if not full_eval:
        eval_ds_info = {k: v for k, v in eval_ds_info.items() if k in BENCHMARK_TASK_INFO}
        for k in BENCHMARK_TASK_INFO:
            eval_ds_info[k]["ds_kwargs"] = BENCHMARK_TASK_INFO[k]
        
    for ds in list(eval_ds_info.keys()):
        if ds.startswith("lol_"):
            eval_ds_info.pop(ds)
            
    subname = ''
    if isinstance(curstep, int) and curstep > 0:
        subname += f'_it{curstep}'
    else:
        subname += f'_latest'
    if full_eval:
        subname += f'_full_eval'

    # 2) 압축/복원 LoRA 생성
    all_lora_dirs, save_dicts, metric = generate_loras_with_compnet(
        args=args,
        comp_model=comp_model,
        model=base_model,
        layer_indices=layer_indices,
        save_dir=save_dir,
        device=device,
        eval_ds_info= eval_ds_info,
    )

    os.makedirs(f"{save_dir}/eval_results{subname}", exist_ok=True)
    metrics_path = os.path.join(f"{save_dir}/eval_results{subname}", "comp_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metric, f, indent=2)

    del comp_model, base_model, tokenizer, layer_indices
    gc.collect()
    torch.cuda.empty_cache()

    all_results = {}
    for eval_ds in eval_ds_info:
        # if not eval_ds in ['mbpp']: continue
        ds_kwargs = eval_ds_info[eval_ds].get("ds_kwargs")
        # ds_kwargs = eval_ds_info[eval_ds]["ds_kwargs"] if eval_ds_info[eval_ds]["ds_kwargs"] else None
        try:
            results = do_eval_task(
                args.model_dir,
                chat_template,
                save_dir,
                all_lora_dirs[eval_ds],
                eval_ds,
                save_dicts[eval_ds],
                ds_kwargs,
                use_icl,
                subname = subname
            )
            print(results[eval_ds][0]['results'])
            all_results[eval_ds] = results[eval_ds][0]['results']
        except Exception as e:
            logger.warning(f"Eval failed on {eval_ds}: {e}")

    df = aggregrate_results_and_save_to_file(
        base_model_dir=args.model_dir,
        mt_lora_dir=args.mt_lora_path,
        hypermod_dir=save_dir,
        hypermod_name="compnet",
        subname=subname
    )
    
    metrics_path = os.path.join(save_dir, f"eval_results/comp_metrics_it{curstep}.json")
    with open(metrics_path, "w") as f:
        json.dump(metric, f, indent=2)
            
    # 로그 키만 compnet으로 맞추거나 기존 키 재사용
    # out = {
    #     ("test" if full_eval else "val") + "/benchmark/acc/random_descs": df["benchmark_avg"].loc[("compnet", "random_descs")],
    # }
    # if wandb.run is not None:
    #     wandb.log(out, step=curstep)
    return None
