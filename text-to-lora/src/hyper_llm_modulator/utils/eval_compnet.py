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
    comp_model = get_compnet(args, "lora", device, model, layer_indices, task_emb_size=None, from_scratch=False)
    state_dict = torch.load(checkpoint_path, map_location=device)
    comp_model.load_state_dict(state_dict, strict=True)
    comp_model.eval()
    comp_model.update()
    
    return args, comp_model, model, tokenizer, layer_indices

@torch.no_grad()
def generate_loras_with_compnet(
    args,
    comp_model,
    model,                  # 베이스 LLM (이름 매핑용)
    layer_indices,
    save_dir,
    device,
):
    # 평가할 타깃 LoRA 경로들 얻기 (train/eval 셋에 맞춰)
    lora_dirs_map = get_target_lora_dirs(list(args.eval_ds_info.keys()), args.model_dir)

    # 모듈명 매핑(저장용 이름 생성)
    module_names = get_lora_module_names(model, args.target_modules, layer_indices)
    all_lora_dirs = {task: [] for task in lora_dirs_map}
    save_dicts    = {task: [] for task in lora_dirs_map}

    for task, src_lora_dir in lora_dirs_map.items():
        tgt_save_dir = f"{save_dir}/generated_loras/{task}/compressed/lora_0"
        os.makedirs(tgt_save_dir, exist_ok=True)

        # 원본 LoRA 로드 → 텐서딕트
        tgt_sd = load_peft_weights(src_lora_dir)
        td = lora_state_dict_to_tensor_dict(
            tgt_sd, args.target_modules, layer_indices, device=device
        )

        # 모듈별로 압축/복원 실행
        A_hat_all, B_hat_all = {}, {}
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

        # 텐서딕트를 state_dict로 변환 후 저장
        recon_td = {"A": A_hat_all, "B": B_hat_all}
        recon_sd = lora_tensor_dict_to_state_dict(
            recon_td, module_names, args.target_modules, layer_indices
        )

        # 어댑터 설정은 원본 것을 재사용
        peft_cfg = get_peft_config(PeftConfig.from_json_file(f"{src_lora_dir}/adapter_config.json"))
        save_lora(recon_sd, peft_cfg, tgt_save_dir)

        all_lora_dirs[task].append(tgt_save_dir)
        save_dicts[task].append({"src_lora": src_lora_dir, "split": "compressed", "lora_dir": tgt_save_dir})

    return all_lora_dirs, save_dicts


def eval_compnet_checkpoint(checkpoint_path, device, curstep, full_eval, use_icl=False):
    # 1) 로드
    args, comp_model, base_model, tokenizer, layer_indices = load_compnet_checkpoint(checkpoint_path, device)
    chat_template = tokenizer.chat_template
    save_dir = os.path.dirname(checkpoint_path)

    # 2) 압축/복원 LoRA 생성
    all_lora_dirs, save_dicts = generate_loras_with_compnet(
        args=args,
        comp_model=comp_model,
        model=base_model,
        layer_indices=layer_indices,
        save_dir=save_dir,
        device=device,
    )

    del comp_model, base_model, tokenizer, layer_indices
    gc.collect()
    torch.cuda.empty_cache()

    for eval_ds in args.eval_ds_info:
        ds_kwargs = args.eval_ds_info[eval_ds].get("ds_kwargs")
        results = do_eval_task(
            args.model_dir,
            chat_template,
            save_dir,
            all_lora_dirs[eval_ds],
            eval_ds,
            save_dicts[eval_ds],
            ds_kwargs,
            use_icl,
            curstep=curstep
        )
        print(results)

    df = aggregrate_results_and_save_to_file(
        base_model_dir=args.model_dir,
        mt_lora_dir=args.mt_lora_path,
        hypermod_dir=save_dir,
        hypermod_name="compnet",
        curstep=curstep
    )

    
    # 로그 키만 compnet으로 맞추거나 기존 키 재사용
    out = {
        ("test" if full_eval else "val") + "/benchmark/acc/random_descs": df["benchmark_avg"].loc[("compnet", "random_descs")],
    }
    if wandb.run is not None:
        wandb.log(out, step=curstep)
    return out
