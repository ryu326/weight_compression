# pip install compressai
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
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


class HouseholderOrthogonal(nn.Module):
    def __init__(self, dim: int, n_reflections: int = 4):
        super().__init__()
        self.v = nn.Parameter(torch.randn(n_reflections, dim))

    def forward(self, x):  # [..., dim]
        y = x
        for v in self.v:
            u = F.normalize(v, dim=0)
            proj = (y * u).sum(dim=-1, keepdim=True)
            y = y - 2.0 * proj * u
        return y

class OrthoWhiten(nn.Module):
    def __init__(self, dim: int, n_reflections: int = 4):
        super().__init__()
        self.Q = HouseholderOrthogonal(dim, n_reflections)
        self.log_scale = nn.Parameter(torch.zeros(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):  # [..., dim]
        z = self.Q(x)
        z = z * torch.exp(self.log_scale) + self.bias
        return z

class FiLM(nn.Module):
    """cond -> (gamma, beta) and apply: h = (1+tanh(gamma))*h + beta"""
    def __init__(self, cond_dim: int, target_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, 2 * target_dim),
        )

    def forward(self, h, cond):
        gb = self.net(cond)
        gamma, beta = gb.chunk(2, dim=-1)
        return (1.0 + torch.tanh(gamma)) * h + beta


class EncoderCoreV2(nn.Module):
    def __init__(self, d_enc_in: int, cond_dim: int, latent_dim: int, width: float = 1.0):
        super().__init__()
        d_hidden = int(d_enc_in * width)
        self.pre = nn.Sequential(nn.LayerNorm(d_enc_in), nn.Linear(d_enc_in, d_hidden), nn.SiLU())
        self.film1 = FiLM(cond_dim, d_hidden)
        self.mid = nn.Sequential(nn.Linear(d_hidden, d_hidden), nn.SiLU())
        self.film2 = FiLM(cond_dim, d_hidden)
        self.out = nn.Sequential(nn.LayerNorm(d_hidden), nn.Linear(d_hidden, latent_dim))

    def forward(self, enc_in, cond):
        # d_enc_in --> d_hidden --> d_hidden --> latent_dim
        h = self.pre(enc_in)
        h = self.film1(h, cond)
        h_skip = h
        h = self.mid(h)
        h = self.film2(h, cond)
        h = h + h_skip
        z = self.out(h)
        return z

class DecoderCoreV2(nn.Module):
    def __init__(self, latent_dim: int, cond_dim: int, d_dec_out: int, width: float = 1.0):
        super().__init__()
        # d_hidden = int(max(latent_dim, d_dec_out) * width)
        d_hidden = int(latent_dim * width)
        self.pre = nn.Sequential(nn.LayerNorm(latent_dim), nn.Linear(latent_dim, d_hidden), nn.SiLU())
        self.film1 = FiLM(cond_dim, d_hidden)
        self.mid = nn.Sequential(nn.Linear(d_hidden, d_hidden), nn.SiLU())
        self.film2 = FiLM(cond_dim, d_hidden)
        self.out = nn.Sequential(nn.LayerNorm(d_hidden), nn.Linear(d_hidden, d_dec_out))

    def forward(self, z_hat, cond):
        # latent_dim --> d_hidden --> d_hidden --> d_dec_out
        # h = torch.cat([z_hat, cond], dim=-1)
        h = self.pre(z_hat)
        h = self.film1(h, cond)
        h_skip = h
        h = self.mid(h)
        h = self.film2(h, cond)
        h = h + h_skip
        return self.out(h)

class HyperEnc(nn.Module):
    def __init__(self, C_in: int, C_h: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(C_in),
            nn.Linear(C_in, max(C_h, 16)),
            nn.SiLU(),
            nn.Linear(max(C_h, 16), C_h),
        )
    def forward(self, y_abs):  # [L, C]
        return self.net(y_abs)

class HyperDec(nn.Module):
    def __init__(self, C_h: int, C_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(C_h),
            nn.Linear(C_h, max(C_out, 32)),
            nn.SiLU(),
            nn.Linear(max(C_out, 32), C_out),
        )
        self.softplus = nn.Softplus()  # scale > 0
    def forward(self, h_hat):  # [L, C]
        scales = self.softplus(self.net(h_hat)) + 1e-6
        return scales

class CondEmbedder(nn.Module):
    """
    layer_indices: LongTensor [L]
    layer_type: str (self.module_to_int로 인덱싱)
    -> [L, cond_dim]
    """
    def __init__(self, max_num_layers: int, target_modules: List[str], cond_dim: int):
        super().__init__()
        self.module_to_int = {m: i for i, m in enumerate(target_modules)}
        d_depth = cond_dim // 2
        d_type  = cond_dim // 2
        self.depth_emb = nn.Sequential(
            nn.Embedding(max_num_layers, d_depth),
            nn.LayerNorm(d_depth),
        )
        self.type_emb = nn.Sequential(
            nn.Embedding(len(target_modules), d_type),
            nn.LayerNorm(d_type),
        )
        self.cond_dim = cond_dim
        
    def forward(self, layer_indices: torch.Tensor, layer_type: str) -> torch.Tensor:
        if layer_indices.ndim != 1:
            raise ValueError("layer_indices must be 1D LongTensor [L].")
        type_idx = torch.tensor([self.module_to_int[layer_type]],
                                device=layer_indices.device, dtype=torch.long)
        type_e = self.type_emb(type_idx).expand(layer_indices.shape[0], -1)  # [L, d_type]
        depth_e = self.depth_emb(layer_indices)                               # [L, d_depth]
        return torch.cat([depth_e, type_e], dim=-1)                           # [L, cond_dim]

class LoRACompressionModelV2(CompressionModel):
    def __init__(
        self,
        *,
        target_modules: List[str],
        in_features: Dict[str, int],
        out_features: Dict[str, int],
        max_num_layers: int,
        r: int,
        latent_dim: int = 128,
        cond_dim: int = 64,
        d_enc_in: Optional[int] = None,
        d_dec_out: Optional[int] = None,
        width: float = 1,
        use_ortho_whiten: bool = True,
        ortho_reflections: int = 4,
        mean_recon_target: Optional[Dict[str, torch.Tensor]] = None,
        std_recon_target: Optional[Dict[str, torch.Tensor]] = None,
        autoreg_gen: bool = True,
        learnable_pos_emb: bool = True,
    ):
        super().__init__()
        self.r = r
        self.in_features = in_features
        self.out_features = out_features
        self.target_modules = target_modules
        self.autoreg_gen = autoreg_gen
        self.learnable_pos_emb =learnable_pos_emb
        
        # cond
        self.cond = CondEmbedder(max_num_layers=max_num_layers,
                                 target_modules=target_modules,
                                 cond_dim=cond_dim)
        self.cond_dim = cond_dim

        # flat dims
        self.flat_dims = {m: r * (in_features[m] + out_features[m]) for m in target_modules}
        self.rank_input_dimsA = {m: in_features[m] for m in target_modules}
        self.rank_input_dimsB = {m: out_features[m] for m in target_modules}
        
        if d_enc_in is None:  d_enc_in = latent_dim * 2
        if d_dec_out is None: d_dec_out = latent_dim * 2
        self.d_enc_in, self.d_dec_out = d_enc_in, d_dec_out

        # (옵션) OrthoWhiten & enc_tails / dec_heads (모듈별)
        self.use_ortho_whiten = use_ortho_whiten
        if use_ortho_whiten:
            self.ortho = nn.ModuleDict({m: OrthoWhiten(self.flat_dims[m], ortho_reflections) for m in target_modules})
        else:
            self.ortho = nn.ModuleDict({m: nn.Identity() for m in target_modules})
            
        # self.ortho_rank = nn.ModuleDict({
        #     m: OrthoWhiten(self.rank_input_dims[m], ortho_reflections)
        #     for m in target_modules
        # }) if use_ortho_whiten else nn.ModuleDict({m: nn.Identity() for m in target_modules})

        if autoreg_gen:
            self.enc_tailsA = nn.ModuleDict()
            self.enc_tailsB = nn.ModuleDict()
            self.dec_headsA = nn.ModuleDict()
            self.dec_headsB = nn.ModuleDict()
            for m in target_modules:
                fdA = self.rank_input_dimsA[m]
                fdB = self.rank_input_dimsB[m]
                self.enc_tailsA[m] = nn.Sequential(nn.LayerNorm(fdA), nn.Linear(fdA, d_enc_in), nn.SiLU())
                self.enc_tailsB[m] = nn.Sequential(nn.LayerNorm(fdB), nn.Linear(fdB, d_enc_in), nn.SiLU())
                self.dec_headsA[m] = nn.Sequential(nn.LayerNorm(d_dec_out), nn.Linear(d_dec_out, fdA))
                self.dec_headsB[m] = nn.Sequential(nn.LayerNorm(d_dec_out), nn.Linear(d_dec_out, fdB))
        else:
            self.enc_tails = nn.ModuleDict()
            self.dec_heads = nn.ModuleDict()
            for m in target_modules:
                fd = self.flat_dims[m]
                self.enc_tails[m] = nn.Sequential(nn.LayerNorm(fd), nn.Linear(fd, d_enc_in), nn.SiLU())
                self.dec_heads[m] = nn.Sequential(nn.LayerNorm(d_dec_out), nn.Linear(d_dec_out, fd))
        
        if autoreg_gen:
            if self.learnable_pos_emb:
                self.rank_pos = nn.Embedding(r, d_enc_in)   # 인코더 입력 쪽에 더할 용도
            else:
                self.register_buffer("rank_pos_sin_enc", self._build_sincos(r, d_enc_in), persistent=False)

        # FiLM 코어(funnel)
        self.encoder_core = EncoderCoreV2(d_enc_in=d_enc_in, cond_dim=cond_dim, latent_dim=latent_dim, width=width)
        self.decoder_core = DecoderCoreV2(latent_dim=latent_dim, cond_dim=cond_dim, d_dec_out=d_dec_out, width=width)

        # Heavy-tail hyperprior
        C = latent_dim
        C_h = max(C // 2, 16)
        self.hyper_enc = HyperEnc(C_in=C, C_h=C_h)
        self.hyper_dec = HyperDec(C_h=C_h, C_out=C)
        self.hyper_bottleneck = EntropyBottleneck(channels=C_h)
        self.gaussian_conditional = GaussianConditional(None)  # tables will be built via .update()
        
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

    # ---------- utils ----------
    def _make_cond(self, layer_indices: torch.Tensor, layer_type: str) -> torch.Tensor:
        return self.cond(layer_indices, layer_type)

    @staticmethod
    def _to_nchw(x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1).unsqueeze(-1)  # [L, C] -> [L, C, 1, 1]
    @staticmethod
    def _from_nchw(x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(-1).squeeze(-1)

    # ---------- forward ----------
    def forward(
        self,
        *,
        layer_type: str,
        layer_indices: torch.Tensor,  # [L]
        lora_A: torch.Tensor,         # [L, r, in]
        lora_B: torch.Tensor,         # [L, out, r]
    ):
        if layer_type not in self.target_modules:
            raise ValueError(f"Unknown layer_type '{layer_type}'.")

        device = lora_A.device
        L, rA, in_feat = lora_A.shape
        Lb, out_feat, rB = lora_B.shape
        assert L == Lb and rA == self.r and rB == self.r

        cond = self._make_cond(layer_indices.to(device), layer_type)   # [L, cond_dim]

        # flatten -> (optional OrthoWhiten) -> enc_tail -> encoder_core
        x_flat = _flatten_lora(lora_A, lora_B)                         # [L, fd]
        x_flat = self.ortho[layer_type](x_flat)
        enc_in = self.enc_tails[layer_type](x_flat)                    # [L, d_enc_in]
        y = self.encoder_core(enc_in, cond)                            # [L, C]

        # hyperprior: scales from hyper path
        y_abs = y.abs()
        h = self.hyper_enc(y_abs)                                      # [L, C_h]
        h_nchw = self._to_nchw(h)
        h_hat_nchw, h_likelihoods = self.hyper_bottleneck(h_nchw)      # training: returns tuple
        h_hat = self._from_nchw(h_hat_nchw)                            # [L, C_h]
        scales_hat = self.hyper_dec(h_hat)                              # [L, C]
        scales_hat_nchw = self._to_nchw(scales_hat)

        # quantize y and compute likelihoods
        y_nchw = self._to_nchw(y)
        # y_hat_nchw = self.gaussian_conditional.quantize(
        #     y_nchw, mode="noise" if self.training else "dequantize"
        # )
        # y_likelihoods = self.gaussian_conditional.likelihood(
        #     y_hat_nchw, scales_hat_nchw
        # )  # [L, C, 1, 1]
        y_hat_nchw, y_likelihoods = self.gaussian_conditional(y_nchw, scales_hat_nchw)

        y_hat = self._from_nchw(y_hat_nchw)                            # [L, C]
        dec_core = self.decoder_core(y_hat, cond)                       # [L, d_dec_out]
        x_hat_flat = self.dec_heads[layer_type](dec_core)               # [L, fd]
        A_hat, B_hat = _unflatten_lora(x_hat_flat, self.r, in_feat, out_feat)

        return dict(
            A_hat=A_hat, B_hat=B_hat,
            likelihoods={"y": y_likelihoods, "h": h_likelihoods},
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
        x_flat = self.ortho[layer_type](x_flat)
        enc_in = self.enc_tails[layer_type](x_flat)
        y = self.encoder_core(enc_in, cond)                            # [L, C]

        # hyper path
        h = self.hyper_enc(y.abs())                                    # [L, C_h]
        h_nchw = self._to_nchw(h)
        strings_h, h_shape = self.hyper_bottleneck.compress(h_nchw)
        h_hat_nchw = self.hyper_bottleneck.decompress(strings_h, h_shape)
        h_hat = self._from_nchw(h_hat_nchw)
        scales_hat = self.hyper_dec(h_hat)                              # [L, C]
        scales_hat_nchw = self._to_nchw(scales_hat)

        # build indexes & compress y
        indexes = self.gaussian_conditional.build_indexes(scales_hat_nchw)
        y_nchw = self._to_nchw(y)
        strings_y = self.gaussian_conditional.compress(y_nchw, indexes)

        side_info = dict(
            layer_type=layer_type,
            layer_indices=layer_indices.detach().cpu().tolist(),
            in_feat=in_feat, out_feat=out_feat, r=self.r,
            latent_dim=y.shape[1],
            h_shape=h_shape,  # tuple
        )
        return {"y": strings_y, "h": strings_h}, side_info

    @torch.no_grad()
    def decompress(self, strings: Dict[str, List[bytes]], side_info):
        layer_type = side_info["layer_type"]
        layer_indices = torch.tensor(side_info["layer_indices"], dtype=torch.long, device=self.hyper_bottleneck._buffers["quantized_cdf"].device)
        in_feat  = side_info["in_feat"]; out_feat = side_info["out_feat"]; r = side_info["r"]
        assert r == self.r

        # hyper first
        h_hat_nchw = self.hyper_bottleneck.decompress(strings["h"], tuple(side_info["h_shape"]))
        h_hat = self._from_nchw(h_hat_nchw)
        scales_hat = self.hyper_dec(h_hat)
        scales_hat_nchw = self._to_nchw(scales_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat_nchw)

        # y
        y_hat_nchw = self.gaussian_conditional.decompress(strings["y"], indexes)
        y_hat = self._from_nchw(y_hat_nchw)

        # cond & decode
        cond = self._make_cond(layer_indices, layer_type)
        dec_core = self.decoder_core(y_hat, cond)
        x_hat_flat = self.dec_heads[layer_type](dec_core)
        A_hat, B_hat = _unflatten_lora(x_hat_flat, self.r, in_feat, out_feat)
        return A_hat, B_hat

    def aux_loss(self):
        # hyper bottleneck의 보조 손실(필수)
        return self.hyper_bottleneck.loss()

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

def get_compnet_v2(
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
    
    comp_model = LoRACompressionModelV2(
        target_modules = args.target_modules,
        in_features = in_features,
        out_features = out_features,
        max_num_layers = max_num_layers,
        r = peft_config.r,
        latent_dim = args.compnet_latent_size,
        cond_dim= args.cond_dim,
        d_enc_in = args.d_enc_in,
        d_dec_out = args.d_dec_out,
        width = args.compnet_latent_width,
        use_ortho_whiten = args.use_ortho_whiten,
        ortho_reflections = 4,
        mean_recon_target = mean_recon_target,
        std_recon_target = std_recon_target,
    ).to(device)
    return comp_model