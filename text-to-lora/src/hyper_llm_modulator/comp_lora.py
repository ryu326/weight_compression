# pip install compressai
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.entropy_models import EntropyBottleneck
from compressai.models.google import CompressionModel

import argparse
import logging
import math
import os
from copy import deepcopy
from types import MethodType
from math import sqrt
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import get_peft_config, load_peft_weights, PeftConfig, PeftModel
from peft.tuners._buffer_dict import BufferDict
from peft.tuners.tuners_utils import BaseTunerLayer, check_target_module_exists
from safetensors.torch import save_file
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
import yaml

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

class MLPResidualBlock(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int,
                 pre_layer_norm: bool = True, post_dropout: bool = True):
        super().__init__()
        layers = []
        if pre_layer_norm:
            layers.append(nn.LayerNorm(d_in))
        layers += [
            nn.Linear(d_in, d_hidden),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(d_hidden, d_out),
            nn.SiLU(),
        ]
        if post_dropout:
            layers.append(nn.Dropout(0.05))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.net(x)

class CondEmbedder(nn.Module):
    """
    layer_indices: LongTensor [L]
    layer_type: str (self.module_to_int로 인덱싱)
    -> [L, cond_dim]
    """
    def __init__(self, max_num_layers: int, target_modules: List[str], latent: int):
        super().__init__()
        self.module_to_int = {m: i for i, m in enumerate(target_modules)}
        d_depth = latent // 2
        d_type  = latent // 2
        self.depth_emb = nn.Sequential(
            nn.Embedding(max_num_layers, d_depth),
            nn.LayerNorm(d_depth),
        )
        self.type_emb = nn.Sequential(
            nn.Embedding(len(target_modules), d_type),
            nn.LayerNorm(d_type),
        )
        self.cond_dim = d_depth + d_type

    def forward(self, layer_indices: torch.Tensor, layer_type: str) -> torch.Tensor:
        if layer_indices.ndim != 1:
            raise ValueError("layer_indices must be 1D LongTensor [L].")
        type_idx = torch.tensor([self.module_to_int[layer_type]],
                                device=layer_indices.device, dtype=torch.long)
        type_e = self.type_emb(type_idx).expand(layer_indices.shape[0], -1)  # [L, d_type]
        depth_e = self.depth_emb(layer_indices)                               # [L, d_depth]
        return torch.cat([depth_e, type_e], dim=-1)                           # [L, cond_dim]


def _flatten_lora(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    A: [L, r, in_feat]
    B: [L, out_feat, r]  (통상 B는 [out, r])
    -> [L, r*in + r*out]
    """
    if A.ndim != 3 or B.ndim != 3:
        raise ValueError("A and B must be [L, r, in] and [L, out, r].")
    L, r, in_feat = A.shape
    L2, out_feat, r2 = B.shape
    if L != L2 or r != r2:
        raise ValueError("A and B must have same L and r.")
    A_flat = A.reshape(L, r * in_feat)
    B_flat = B.transpose(-1, -2).reshape(L, r * out_feat)  # [L, r*out]
    return torch.cat([A_flat, B_flat], dim=-1)             # [L, r*in + r*out]


def _unflatten_lora(x: torch.Tensor, r: int, in_feat: int, out_feat: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    x: [L, r*in + r*out]
    -> A_hat: [L, r, in_feat]
       B_hat: [L, out_feat, r]
    """
    L, total = x.shape
    exp_total = r * in_feat + r * out_feat
    if total != exp_total:
        raise ValueError(f"Unexpected flat size {total}, expected {exp_total}.")
    a_len = r * in_feat
    A_flat = x[..., :a_len]                         # [L, r*in]
    B_flat = x[..., a_len:]                         # [L, r*out]
    A = A_flat.reshape(L, r, in_feat)
    B = B_flat.reshape(L, r, out_feat).transpose(-1, -2)  # [L, out, r]
    return A, B

def zero_lora_param_dict(target_modules, n_layers, r, in_features, out_features):
    return nn.ParameterDict({
        "A": nn.ParameterDict({
            m: nn.Parameter(torch.zeros(n_layers, r, in_features[m]), requires_grad=False)
            for m in target_modules
        }),
        "B": nn.ParameterDict({
            m: nn.Parameter(torch.zeros(n_layers, out_features[m], r), requires_grad=False)
            for m in target_modules
        }),
    })

# ----------------------------
# Shared cores (Encoder/Decoder)
# ----------------------------

class EncoderCore(nn.Module):
    """
    입력:  enc_tail_out: [L, d_enc_in]
          cond:         [L, cond_dim]
    출력:  z:           [L, latent_dim]
    """
    def __init__(self, d_enc_in: int, cond_dim: int, latent_dim: int, width: int = 4):
        super().__init__()
        d_in = d_enc_in + cond_dim
        hidden = d_in * width
        self.mixer = nn.Sequential(
            nn.Dropout(0.05),
            nn.Linear(d_in, hidden),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden, d_in),
            nn.SiLU(),
            nn.Dropout(0.05),
        )
        self.mlp1 = MLPResidualBlock(d_in, hidden, d_in, pre_layer_norm=True, post_dropout=True)
        self.out = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, hidden),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, enc_tail_out: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = torch.cat([enc_tail_out, cond], dim=-1)
        h = self.mixer(h)
        h = self.mlp1(h)
        z = self.out(h)
        return z


class DecoderCore(nn.Module):
    """
    입력:  z_hat: [L, latent_dim]
          cond:  [L, cond_dim]
    출력:  dec_core: [L, d_dec_out]
    """
    def __init__(self, latent_dim: int, cond_dim: int, d_dec_out: int, width: int = 4):
        super().__init__()
        d_in = latent_dim + cond_dim
        hidden = d_in * width
        self.mixer = nn.Sequential(
            nn.Dropout(0.05),
            nn.Linear(d_in, hidden),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden, d_in),
            nn.SiLU(),
            nn.Dropout(0.05),
        )
        self.mlp1 = MLPResidualBlock(d_in, hidden, d_in, pre_layer_norm=True, post_dropout=True)
        self.out = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, hidden),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden, d_dec_out),
        )

    def forward(self, z_hat: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = torch.cat([z_hat, cond], dim=-1)
        h = self.mixer(h)
        h = self.mlp1(h)
        return self.out(h)  # [L, d_dec_out]

class LoRACompressionModel(CompressionModel):
    """
    - 모듈 타입(layer_type) 하나를 선택하고, 그 타입의 여러 레이어(layer_indices)를 배치로 처리.
    - Encoder/Decoder 코어는 공유
    - 모듈별로:
        * enc_tail[m]:  flat_dim_m -> d_enc_in
        * dec_head[m]:  d_dec_out  -> flat_dim_m
    - cond(layer embedding)은 Encoder/Decoder 입력에 concat
    """
    def __init__(
        self,
        *,
        target_modules: List[str],
        in_features: Dict[str, int],
        out_features: Dict[str, int],
        max_num_layers: int,
        r: int,
        latent_dim: int = 128,
        width: int = 4,
        d_enc_in: Optional[int] = None,
        d_dec_out: Optional[int] = None,
        mean_recon_target: Optional[Dict[str, torch.Tensor]] = None,
        std_recon_target: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__()
        self.r = r
        self.in_features = in_features
        self.out_features = out_features
        self.target_modules = target_modules

        # cond embedding
        self.cond = CondEmbedder(max_num_layers=max_num_layers,
                                 target_modules=target_modules,
                                 latent=latent_dim)
        cond_dim = self.cond.cond_dim

        # 모듈별 입력/출력 평탄화 차원
        self.flat_dims = {
            m: r * (in_features[m] + out_features[m]) for m in target_modules
        }

        # 공유 코어 입력/출력 차원
        if d_enc_in is None:
            d_enc_in = latent_dim * 2
        if d_dec_out is None:
            d_dec_out = latent_dim * 2
        self.d_enc_in = d_enc_in
        self.d_dec_out = d_dec_out

        # 모듈별 enc_tail / dec_head
        self.enc_tails = nn.ModuleDict()
        self.dec_heads = nn.ModuleDict()
        for m in target_modules:
            fd = self.flat_dims[m]
            # enc_tail: flat_dim_m -> d_enc_in
            self.enc_tails[m] = nn.Sequential(
                nn.LayerNorm(fd),
                nn.Linear(fd, d_enc_in),
                nn.SiLU(),
            )
            # dec_head: d_dec_out -> flat_dim_m
            self.dec_heads[m] = nn.Sequential(
                nn.LayerNorm(d_dec_out),
                nn.Linear(d_dec_out, fd),
            )

        # 공유 Encoder/Decoder core
        self.encoder_core = EncoderCore(d_enc_in=d_enc_in, cond_dim=cond_dim,
                                        latent_dim=latent_dim, width=width)
        self.decoder_core = DecoderCore(latent_dim=latent_dim, cond_dim=cond_dim,
                                        d_dec_out=d_dec_out, width=width)

        # Entropy bottleneck (shared)
        self.entropy_bottleneck = EntropyBottleneck(channels=latent_dim)
        
        self.mean_recon_target = zero_lora_param_dict(
            self.target_modules, max_num_layers, r, in_features, out_features
        )
        self.std_recon_target = zero_lora_param_dict(
            self.target_modules, max_num_layers, r, in_features, out_features
        )

        if mean_recon_target is not None:
            self.mean_recon_target = lora_tensor_dict_to_param_dict(
                mean_recon_target, requires_grad=False
            )
        if std_recon_target is not None:
            self.std_recon_target = lora_tensor_dict_to_param_dict(
                std_recon_target, requires_grad=False
            )

    # ---------- Utility ----------

    def _make_cond(self, layer_indices: torch.Tensor, layer_type: str) -> torch.Tensor:
        return self.cond(layer_indices, layer_type)  # [L, cond_dim]

    @staticmethod
    def _to_nchw(z: torch.Tensor) -> torch.Tensor:
        return z.unsqueeze(-1).unsqueeze(-1)  # [L, C] -> [L, C, 1, 1]

    @staticmethod
    def _from_nchw(z: torch.Tensor) -> torch.Tensor:
        return z.squeeze(-1).squeeze(-1)      # [L, C, 1, 1] -> [L, C]

    # ---------- Core passes ----------

    def forward(
        self,
        *,
        layer_type: str,
        layer_indices: torch.Tensor,  # [L], long
        lora_A: torch.Tensor,         # [L, r, in_feat]
        lora_B: torch.Tensor,         # [L, out_feat, r]
    ):
        if layer_type not in self.target_modules:
            raise ValueError(f"Unknown layer_type '{layer_type}'.")
        device = lora_A.device
        L, rA, in_feat = lora_A.shape
        Lb, out_feat, rB = lora_B.shape
        assert L == Lb and rA == self.r and rB == self.r, "Shape mismatch for A/B."

        # cond
        cond = self._make_cond(layer_indices.to(device), layer_type)  # [L, cond_dim]

        # flatten -> enc_tail(module-specific) -> encoder_core(shared)
        x_flat = _flatten_lora(lora_A, lora_B)            # [L, flat_dim_m]
        enc_in = self.enc_tails[layer_type](x_flat)       # [L, d_enc_in]
        z = self.encoder_core(enc_in, cond)               # [L, latent_dim]

        # entropy bottleneck
        z_nchw = self._to_nchw(z)                         # [L, C, 1, 1]
        z_hat, likelihoods = self.entropy_bottleneck(z_nchw)      # (train: (z_hat, likelihoods)) / (infer: z_hat)
        z_hat = self._from_nchw(z_hat)                    # [L, C]

        # decoder_core(shared) -> dec_head(module-specific) -> unflatten
        dec_core = self.decoder_core(z_hat, cond)         # [L, d_dec_out]
        x_hat_flat = self.dec_heads[layer_type](dec_core)  # [L, flat_dim_m]
        A_hat, B_hat = _unflatten_lora(x_hat_flat, self.r, in_feat, out_feat)

        # losses
        # mse_A = F.mse_loss(A_hat, lora_A, reduction="mean" if reduce else "none")
        # mse_B = F.mse_loss(B_hat, lora_B, reduction="mean" if reduce else "none")

        # # bpp / param
        # if likelihoods is not None:
        #     num_params = x_flat.numel()
        #     bpparam = (-torch.log2(likelihoods)).sum() / num_params  # [scalar]
        # else:
        #     bpparam = torch.tensor(0.0, device=device)

        # rd_loss = mse_A + mse_B + lambda_bpp * bpparam

        return dict(
            A_hat=A_hat, B_hat=B_hat,
            likelihoods = likelihoods
        )

    @torch.no_grad()
    def compress(
        self,
        *,
        layer_type: str,
        layer_indices: torch.Tensor,
        lora_A: torch.Tensor,
        lora_B: torch.Tensor,
    ):
        if layer_type not in self.target_modules:
            raise ValueError(f"Unknown layer_type '{layer_type}'.")
        device = lora_A.device
        L, rA, in_feat = lora_A.shape
        Lb, out_feat, rB = lora_B.shape
        assert L == Lb and rA == self.r and rB == self.r

        cond = self._make_cond(layer_indices.to(device), layer_type)
        x_flat = _flatten_lora(lora_A, lora_B)
        enc_in = self.enc_tails[layer_type](x_flat)
        z = self.encoder_core(enc_in, cond)          # [L, C]
        z_nchw = self._to_nchw(z)                    # [L, C, 1, 1]

        # compressai API 호환 처리
        strings = self.entropy_bottleneck.compress(z_nchw)
        z_shape = None
        if isinstance(strings, tuple):  # (strings, z_shape)
            strings, z_shape = strings
        if z_shape is None:
            z_shape = z_nchw.size()[2:]

        side_info = dict(
            layer_type=layer_type,
            layer_indices=layer_indices.detach().cpu().tolist(),
            in_feat=in_feat,
            out_feat=out_feat,
            r=self.r,
            latent_dim=z.shape[1],
            z_shape=tuple(z_shape),  # (L, C, 1, 1) (1, 1)
        )
        return strings, side_info

    @torch.no_grad()
    def decompress(self, strings, side_info) -> Tuple[torch.Tensor, torch.Tensor]:
        layer_type = side_info["layer_type"]
        z_shape    = tuple(side_info["z_shape"])  # (L, C, 1, 1) # (1, 1)
        # L, C, _, _ = z_shape
        in_feat  = side_info["in_feat"]
        out_feat = side_info["out_feat"]
        r        = side_info["r"]
        assert r == self.r

        z_hat_nchw = self.entropy_bottleneck.decompress(strings, z_shape)  # [L, C, 1, 1]
        z_hat = self._from_nchw(z_hat_nchw)                                # [L, C]

        layer_indices = torch.tensor(side_info["layer_indices"], dtype=torch.long, device=z_hat.device)
        cond = self._make_cond(layer_indices, layer_type)

        dec_core = self.decoder_core(z_hat, cond)                          # [L, d_dec_out]
        x_hat_flat = self.dec_heads[layer_type](dec_core)                  # [L, flat_dim_m]
        A_hat, B_hat = _unflatten_lora(x_hat_flat, self.r, in_feat, out_feat)
        return A_hat, B_hat

    # def aux_loss(self):
    #     return self.entropy_bottleneck.loss()


def get_std_lora(lora_state_dicts):
    modules_names = lora_state_dicts[0].keys()
    std_lora = dict.fromkeys(modules_names)
    for module_name in modules_names:
        std_lora[module_name] = torch.stack(
            [lora_sd[module_name] for lora_sd in lora_state_dicts], dim=0
        )
        # std_lora[module_name] = torch.std(std_lora[module_name], dim=0)
        std_lora[module_name] = torch.std(std_lora[module_name], dim=(0,1,2), keepdim=True).squeeze(0)
    return std_lora


def get_mean_lora(lora_state_dicts):
    modules_names = lora_state_dicts[0].keys()
    avg_lora = dict.fromkeys(modules_names)
    for module_name in modules_names:
        avg_lora[module_name] = torch.stack(
            [lora_sd[module_name] for lora_sd in lora_state_dicts], dim=0
        )
        # avg_lora[module_name] = torch.mean(avg_lora[module_name], dim=0)
        avg_lora[module_name] = torch.mean(avg_lora[module_name], dim=(0,1,2), keepdim=True).squeeze(0)
    return avg_lora

def lora_tensor_dict_to_param_dict(lora_tensor_dict, requires_grad):
    return nn.ParameterDict(
        {
            "A": nn.ParameterDict(
                {
                    k: nn.Parameter(v, requires_grad)
                    for k, v in lora_tensor_dict["A"].items()
                }
            ),
            "B": nn.ParameterDict(
                {
                    k: nn.Parameter(v, requires_grad)
                    for k, v in lora_tensor_dict["B"].items()
                }
            ),
        }
    )

def get_in_out_features(
    model: PeftModel,
    peft_config: PeftConfig = None,
) -> tuple[dict[str, int], dict[str, int]]:
    if peft_config is None:
        peft_config = model.peft_config["default"]
    in_features = dict()
    out_features = dict()
    for module_name, module in model.named_modules():
        if not check_target_module_exists(peft_config, module_name):
            continue
        if not isinstance(module, BaseTunerLayer):
            continue
        # support just Linear layer for now
        # all modules should be a leave module that is Linear layer
        assert isinstance(module.base_layer, nn.Linear), (
            "all modules should be a leave module that is Linear layer"
        )

        # this should always pass
        name = module_name.split(".")[-1]
        assert name in peft_config.target_modules, (
            f"Module {name} not in target modules"
        )

        if name not in in_features:
            in_features[name] = module.in_features
            out_features[name] = module.out_features
        else:
            # assumes each module has the same input and output features
            assert in_features[name] == module.in_features
            assert out_features[name] == module.out_features

    return in_features, out_features

def get_compnet_v1(
    args, peft_type, device, model, layer_indices, task_emb_size, from_scratch=True
): 
    peft_config = model.peft_config["default"]
    in_features, out_features = get_in_out_features(model, peft_config)
    
    mt_lora_sd = mt_lora_td = mean_recon_target = std_recon_target = None
    if from_scratch:
        if args.mt_lora_path:
            mt_lora_sd = load_peft_weights(args.mt_lora_path)
            mt_lora_td = lora_state_dict_to_tensor_dict(
                mt_lora_sd, args.target_modules, layer_indices, device=device
            )

        if args.training_task == "recon":
            lora_paths = get_target_lora_dirs(args.train_ds_names, args.model_dir)
            target_loras = {
                task: load_peft_weights(path) for task, path in lora_paths.items()
            }
            if args.mt_lora_path:
                target_loras = {
                    task: {k: v - mt_lora_sd[k] for k, v in lora.items()}
                    for task, lora in target_loras.items()
                }
            if args.pred_z_score:
                mean_recon_target = get_mean_lora(list(target_loras.values()))
                mean_recon_target = lora_state_dict_to_tensor_dict(
                    mean_recon_target, args.target_modules, layer_indices, device
                )
                std_recon_target = get_std_lora(list(target_loras.values()))
                std_recon_target = lora_state_dict_to_tensor_dict(
                    std_recon_target, args.target_modules, layer_indices, device
                )
    
    max_num_layers = model.config.num_hidden_layers
    
    comp_model = LoRACompressionModel(
        target_modules = args.target_modules,
        in_features = in_features,
        out_features = out_features,
        max_num_layers = max_num_layers,
        r = peft_config.r,
        latent_dim = args.compnet_latent_size,
        width = args.compnet_latent_width,
        d_enc_in = args.d_enc_in,
        d_dec_out = args.d_dec_out,
        mean_recon_target = mean_recon_target,
        std_recon_target = std_recon_target,
    ).to(device)
    return comp_model