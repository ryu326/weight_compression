import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Literal
import math
from compressai.entropy_models import EntropyBottleneck
from compressai.models.google import CompressionModel
import math
from typing import Dict, List, Optional, Tuple
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
# pip install compressai
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

# ------------------------ FiLM ------------------------
class FiLM(nn.Module):
    """cond -> (gamma, beta); h = (1+tanh(gamma))*h + beta.
       cond=None이면 그대로 통과(비활성화)."""
    def __init__(self, cond_dim: int, target_dim: int):
        super().__init__()
        self.cond_dim = cond_dim
        if cond_dim > 0:
            self.net = nn.Sequential(
                nn.LayerNorm(cond_dim),
                nn.Linear(cond_dim, 2 * target_dim),
            )
        else:
            self.net = None  # disabled

    def forward(self, h, cond: Optional[torch.Tensor]):
        if self.net is None or cond is None:
            return h
        gb = self.net(cond)
        gamma, beta = gb.chunk(2, dim=-1)
        return (1.0 + torch.tanh(gamma)) * h + beta

# ------------------------ Cores ------------------------
class EncoderCoreV2(nn.Module):
    def __init__(self, d_enc_in: int, cond_dim: int, latent_dim: int, width: float = 1.0):
        super().__init__()
        d_hidden = int(d_enc_in * width)
        self.pre = nn.Sequential(nn.LayerNorm(d_enc_in), nn.Linear(d_enc_in, d_hidden), nn.SiLU())
        self.film1 = FiLM(cond_dim, d_hidden)
        self.mid = nn.Sequential(nn.Linear(d_hidden, d_hidden), nn.SiLU())
        self.film2 = FiLM(cond_dim, d_hidden)
        self.out = nn.Sequential(nn.LayerNorm(d_hidden), nn.Linear(d_hidden, latent_dim))

    def forward(self, enc_in, cond: Optional[torch.Tensor]):
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
        d_hidden = int(latent_dim * width)
        self.pre = nn.Sequential(nn.LayerNorm(latent_dim), nn.Linear(latent_dim, d_hidden), nn.SiLU())
        self.film1 = FiLM(cond_dim, d_hidden)
        self.mid = nn.Sequential(nn.Linear(d_hidden, d_hidden), nn.SiLU())
        self.film2 = FiLM(cond_dim, d_hidden)
        self.out = nn.Sequential(nn.LayerNorm(d_hidden), nn.Linear(d_hidden, d_dec_out))

    def forward(self, z_hat, cond: Optional[torch.Tensor]):
        h = self.pre(z_hat)
        h = self.film1(h, cond)
        h_skip = h
        h = self.mid(h)
        h = self.film2(h, cond)
        h = h + h_skip
        return self.out(h)

# ------------------------ CondEmbedder ------------------------
class CondEmbedder(nn.Module):
    """
    layer_indices: LongTensor [L] or None
    layer_type: str or None
    -> 토글(use_layer_indices/use_layer_type)에 따라 필요한 임베딩만 적용.
       꺼진 항목은 입력이 와도 무시.
    """
    def __init__(self, max_num_layers: int, target_modules: List[str], cond_dim: int,
                 use_layer_indices: bool = True, use_layer_type: bool = True):
        super().__init__()
        self.use_layer_indices = use_layer_indices
        self.use_layer_type = use_layer_type
        self.module_to_int = {m: i for i, m in enumerate(target_modules)}

        d_depth = (cond_dim // 2) if use_layer_indices else 0
        d_type  = (cond_dim - d_depth) if use_layer_type else 0
        self.cond_dim = d_depth + d_type

        self.depth_emb = None
        self.type_emb  = None
        if d_depth > 0:
            self.depth_emb = nn.Sequential(
                nn.Embedding(max_num_layers, d_depth),
                nn.LayerNorm(d_depth),
            )
        if d_type > 0:
            self.type_emb = nn.Sequential(
                nn.Embedding(len(target_modules), d_type),
                nn.LayerNorm(d_type),
            )

    def forward(
        self,
        layer_indices: Optional[torch.Tensor],
        layer_type: Optional[str],
        L: int,
        device
    ) -> Optional[torch.Tensor]:
        # 전체 cond 비활성화
        if self.cond_dim == 0 or (self.depth_emb is None and self.type_emb is None):
            return None

        parts = []

        # depth 임베딩: 토글이 켜져 있으면 필요, 꺼져 있으면 입력이 와도 무시
        if self.depth_emb is not None:
            if layer_indices is None:
                raise ValueError("use_layer_indices=True 인데 layer_indices=None 입니다.")
            if layer_indices.ndim != 1:
                raise ValueError("layer_indices must be 1D LongTensor [L].")
            parts.append(self.depth_emb(layer_indices.to(device)))  # [L, d_depth]

        # type 임베딩: 토글이 켜져 있으면 필요, 꺼져 있으면 입력이 와도 무시
        if self.type_emb is not None:
            if layer_type is None:
                raise ValueError("use_layer_type=True 인데 layer_type=None 입니다.")
            if layer_type not in self.module_to_int:
                raise ValueError(f"Unknown layer_type '{layer_type}'.")
            type_idx = torch.tensor([self.module_to_int[layer_type]], device=device, dtype=torch.long)
            parts.append(self.type_emb(type_idx).expand(L, -1))  # [L, d_type]

        return torch.cat(parts, dim=-1) if parts else None

# ------------------------ Main Model ------------------------
class LoRACompressionModelV5(CompressionModel):
    def __init__(
        self,
        *,
        target_modules: List[str],
        in_features: Dict[str, int],
        out_features: Dict[str, int],
        max_num_layers: int,
        r: int,
        block_size:int,
        latent_dim: int = 128,
        cond_dim: int = 64,
        d_enc_in: Optional[int] = None,
        d_dec_out: Optional[int] = None,
        width: float = 1.0,
        learnable_pos_emb: bool = True,  # kept for backward-compat
        rank_pos_mode: Literal["learned", "sincos", "none"] = "learned",
        use_layer_indices: bool = True,
        use_layer_type: bool = True,
        mean_recon_target: Optional[Dict[str, torch.Tensor]] = None,
        std_recon_target: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__()
        self.r = r
        self.in_features = in_features
        self.out_features = out_features
        self.target_modules = target_modules
        self.autoreg_gen = True
        self.blsz = block_size

        # --- rank-pos mode normalization (keep legacy flag for compat) ---
        if learnable_pos_emb is False and rank_pos_mode == "learned":
            rank_pos_mode = "sincos"
        self.rank_pos_mode = rank_pos_mode

        # --- cond embedder ---
        self.cond = CondEmbedder(max_num_layers=max_num_layers,
                                 target_modules=target_modules,
                                 cond_dim=cond_dim,
                                 use_layer_indices=use_layer_indices,
                                 use_layer_type=use_layer_type)
        self.cond_dim = self.cond.cond_dim  # could be 0

        # rank-wise dims
        if d_enc_in is None:  d_enc_in = latent_dim * 2
        if d_dec_out is None: d_dec_out = latent_dim * 2
        self.d_enc_in, self.d_dec_out = d_enc_in, d_dec_out

        # tails/heads
        self.enc_tail = nn.Sequential(nn.LayerNorm(self.blsz), nn.Linear(self.blsz, d_enc_in), nn.SiLU())
        self.dec_head = nn.Sequential(nn.LayerNorm(d_dec_out), nn.Linear(d_dec_out, self.blsz))

        # rank pos emb
        if self.rank_pos_mode == "learned":
            self.rank_pos = nn.Embedding(r, d_enc_in)
        elif self.rank_pos_mode == "sincos":
            self.register_buffer("rank_pos_sin_enc", self._build_sincos(r, d_enc_in), persistent=False)
        else:  # none
            self.rank_pos = None
            self.register_buffer("rank_pos_sin_enc", None, persistent=False)

        # cores
        self.encoder_core = EncoderCoreV2(d_enc_in=d_enc_in, cond_dim=self.cond_dim,
                                          latent_dim=latent_dim, width=width)
        self.decoder_core = DecoderCoreV2(latent_dim=latent_dim, cond_dim=self.cond_dim,
                                          d_dec_out=d_dec_out, width=width)

        # EntropyBottleneck
        self.eb = EntropyBottleneck(channels=latent_dim)

        # recon targets (optional)
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
    def _build_sincos(self, r: int, d: int) -> torch.Tensor:
        pe = torch.zeros(r, d)
        position = torch.arange(0, r, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def _get_rank_pos(self, L: int, R: int, device) -> Optional[torch.Tensor]:
        if self.rank_pos_mode == "learned":
            pos = self.rank_pos.weight  # [R, d_enc_in]
        elif self.rank_pos_mode == "sincos":
            pos = self.rank_pos_sin_enc  # [R, d_enc_in]
        else:
            return None
        pos_lr = pos.unsqueeze(0).expand(L, -1, -1).reshape(L * R, -1)  # [L*R, d_enc_in]
        return pos_lr.to(device)

    def _make_cond(self, layer_indices: Optional[torch.Tensor], layer_type: Optional[str],
                   L: int, device) -> Optional[torch.Tensor]:
        # returns [L, cond_dim] or None
        return self.cond(layer_indices, layer_type, L=L, device=device)

    @staticmethod
    def _to_nchw(x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1).unsqueeze(-1)  # [N,C] -> [N,C,1,1]

    @staticmethod
    def _from_nchw(x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(-1).squeeze(-1)      # [N,C,1,1] -> [N,C]

    @staticmethod
    def _split_blocks(x: torch.Tensor, blsz: int) -> Tuple[torch.Tensor, int, int, int]:
        N, D = x.shape
        n_blk = (D + blsz - 1) // blsz
        pad = n_blk * blsz - D
        if pad > 0:
            x = F.pad(x, (0, pad))
        x = x.view(N, n_blk, blsz)
        return x, D, n_blk, pad

    @staticmethod
    def _merge_blocks(x_blocks: torch.Tensor, orig_len: int) -> torch.Tensor:
        N, n_blk, blsz = x_blocks.shape
        x = x_blocks.reshape(N, n_blk * blsz)
        return x[:, :orig_len]

    # ---------- forward ----------
    def forward(
        self,
        *,
        layer_type: Optional[str],            # may be None if disabled
        layer_indices: Optional[torch.Tensor],# [L] or None if disabled
        lora_A: torch.Tensor,                 # [L, r, in]
        lora_B: torch.Tensor,                 # [L, out, r]
    ):
        device = lora_A.device
        L, rA, in_feat = lora_A.shape
        Lb, out_feat, rB = lora_B.shape
        assert L == Lb and rA == self.r and rB == self.r
        R = rA

        # cond & rank-pos
        cond_L = self._make_cond(layer_indices, layer_type, L=L, device=device)  # [L, cond] or None
        cond_lr = None if cond_L is None else cond_L.unsqueeze(1).expand(L, R, -1).reshape(L * R, -1)
        pos_lr = self._get_rank_pos(L, R, device)  # [L*R, d_enc_in] or None

        # ===== A =====
        A_in = lora_A.reshape(L * R, in_feat)                            # [L*R, in]
        A_blk, A_len, A_nblk, _ = self._split_blocks(A_in, self.blsz)    # [L*R, nA, blsz]
        NA = (L * R) * A_nblk
        A_blk_flat = A_blk.reshape(-1, self.blsz)                        # [NA, blsz]

        cond_A = None if cond_lr is None else cond_lr.repeat_interleave(A_nblk, dim=0)  # [NA, cond] or None
        hA_in = self.enc_tail(A_blk_flat)                                # [NA, d_enc_in]
        if pos_lr is not None:
            pos_A = pos_lr.repeat_interleave(A_nblk, dim=0)              # [NA, d_enc_in]
            hA_in = hA_in + pos_A
        yA = self.encoder_core(hA_in, cond_A)                            # [NA, C]
        yA_hat_nchw, yA_lik = self.eb(self._to_nchw(yA))                 # EB
        yA_hat = self._from_nchw(yA_hat_nchw)                            # [NA, C]
        zA = self.decoder_core(yA_hat, cond_A)                           # [NA, d_dec_out]
        A_rec_blk = self.dec_head(zA).view(L * R, A_nblk, self.blsz)     # [L*R, nA, blsz]
        A_hat_flat = self._merge_blocks(A_rec_blk, A_len)                # [L*R, in]
        A_hat = A_hat_flat.view(L, R, in_feat)                           # [L, R, in]

        # ===== B =====
        B_lr = lora_B.permute(0, 2, 1).contiguous().reshape(L * R, out_feat)   # [L*R, out]
        B_blk, B_len, B_nblk, _ = self._split_blocks(B_lr, self.blsz)          # [L*R, nB, blsz]
        NB = (L * R) * B_nblk
        B_blk_flat = B_blk.reshape(-1, self.blsz)                              # [NB, blsz]

        cond_B = None if cond_lr is None else cond_lr.repeat_interleave(B_nblk, dim=0)
        hB_in  = self.enc_tail(B_blk_flat)                                     # [NB, d_enc_in]
        if pos_lr is not None:
            pos_B = pos_lr.repeat_interleave(B_nblk, dim=0)
            hB_in = hB_in + pos_B
        yB = self.encoder_core(hB_in, cond_B)                                  # [NB, C]
        yB_hat_nchw, yB_lik = self.eb(self._to_nchw(yB))                       # EB
        yB_hat = self._from_nchw(yB_hat_nchw)                                  # [NB, C]
        zB = self.decoder_core(yB_hat, cond_B)                                 # [NB, d_dec_out]
        B_rec_blk = self.dec_head(zB).view(L * R, B_nblk, self.blsz)           # [L*R, nB, blsz]
        B_hat_flat = self._merge_blocks(B_rec_blk, B_len)                      # [L*R, out]
        B_hat = B_hat_flat.view(L, R, out_feat).permute(0, 2, 1).contiguous()  # [L, out, R]

        return dict(A_hat=A_hat, B_hat=B_hat, likelihoods={"yA": yA_lik, "yB": yB_lik})

    # ---------- entropy coding ----------
    @torch.no_grad()
    def compress(
        self,
        *,
        layer_type: Optional[str],
        layer_indices: Optional[torch.Tensor],
        lora_A: torch.Tensor,
        lora_B: torch.Tensor,
    ):
        device = lora_A.device
        L, rA, in_feat = lora_A.shape
        Lb, out_feat, rB = lora_B.shape
        assert L == Lb and rA == self.r and rB == self.r
        R = rA

        cond_L = self._make_cond(layer_indices, layer_type, L=L, device=device)
        cond_lr = None if cond_L is None else cond_L.unsqueeze(1).expand(L, R, -1).reshape(L * R, -1)
        pos_lr = self._get_rank_pos(L, R, device)

        # A
        A_in = lora_A.reshape(L * R, in_feat)
        A_blk, A_len, A_nblk, _ = self._split_blocks(A_in, self.blsz)
        A_blk_flat = A_blk.reshape(-1, self.blsz)
        cond_A = None if cond_lr is None else cond_lr.repeat_interleave(A_nblk, dim=0)
        hA_in  = self.enc_tail(A_blk_flat)
        if pos_lr is not None:
            pos_A  = pos_lr.repeat_interleave(A_nblk, dim=0)
            hA_in  = hA_in + pos_A
        yA = self.encoder_core(hA_in, cond_A)
        strings_yA, yA_shape = self.eb.compress(self._to_nchw(yA))

        # B
        B_lr = lora_B.permute(0, 2, 1).contiguous().reshape(L * R, out_feat)
        B_blk, B_len, B_nblk, _ = self._split_blocks(B_lr, self.blsz)
        B_blk_flat = B_blk.reshape(-1, self.blsz)
        cond_B = None if cond_lr is None else cond_lr.repeat_interleave(B_nblk, dim=0)
        hB_in  = self.enc_tail(B_blk_flat)
        if pos_lr is not None:
            pos_B  = pos_lr.repeat_interleave(B_nblk, dim=0)
            hB_in  = hB_in + pos_B
        yB = self.encoder_core(hB_in, cond_B)
        strings_yB, yB_shape = self.eb.compress(self._to_nchw(yB))

        side_info = dict(
            layer_type=layer_type,  # could be None
            layer_indices=(None if layer_indices is None else layer_indices.detach().cpu().tolist()),
            in_feat=in_feat,
            out_feat=out_feat,
            r=self.r,
            latent_dim=yA.shape[1],
            blsz=self.blsz,
            nblk_A=A_blk.shape[1],
            nblk_B=B_blk.shape[1],
            yA_shape=yA_shape,
            yB_shape=yB_shape,
            rank_pos_mode=self.rank_pos_mode,
            cond_dim=self.cond_dim,
        )
        return {"yA": strings_yA, "yB": strings_yB}, side_info

    @torch.no_grad()
    def decompress(self, strings: Dict[str, List[bytes]], side_info):
        device = self.eb._buffers["quantized_cdf"].device
        layer_type = side_info.get("layer_type", None)
        li = side_info.get("layer_indices", None)
        layer_indices = None if li is None else torch.tensor(li, dtype=torch.long, device=device)
        in_feat  = side_info["in_feat"]
        out_feat = side_info["out_feat"]
        r        = side_info["r"]
        assert r == self.r
        blsz = side_info["blsz"]
        nblk_A = side_info["nblk_A"]
        nblk_B = side_info["nblk_B"]

        # EB 복호화
        yA_hat_nchw = self.eb.decompress(strings["yA"], tuple(side_info["yA_shape"]))
        yB_hat_nchw = self.eb.decompress(strings["yB"], tuple(side_info["yB_shape"]))
        yA_hat = self._from_nchw(yA_hat_nchw)    # [NA, C]
        yB_hat = self._from_nchw(yB_hat_nchw)    # [NB, C]

        # sizes
        if layer_indices is not None:
            L = len(layer_indices)
        else:
            # 복호화에 필요한 L은 nblk와 r, feature 길이로 역산 가능
            # NA = L*R*nblk_A = yA_hat.shape[0]  -> L = NA / (R*nblk_A)
            NA = yA_hat.shape[0]
            L = NA // (self.r * nblk_A)
        R = self.r

        # cond & pos
        cond_L = self._make_cond(layer_indices, layer_type, L=L, device=device)
        cond_lr = None if cond_L is None else cond_L.unsqueeze(1).expand(L, R, -1).reshape(L * R, -1)
        pos_lr = self._get_rank_pos(L, R, device)

        # A 복원
        cond_A = None if cond_lr is None else cond_lr.repeat_interleave(nblk_A, dim=0)
        zA = self.decoder_core(yA_hat, cond_A)
        A_rec_blk = self.dec_head(zA).view(L * R, nblk_A, blsz)
        A_hat_flat = self._merge_blocks(A_rec_blk, in_feat)
        A_hat = A_hat_flat.view(L, R, in_feat)

        # B 복원
        cond_B = None if cond_lr is None else cond_lr.repeat_interleave(nblk_B, dim=0)
        zB = self.decoder_core(yB_hat, cond_B)
        B_rec_blk = self.dec_head(zB).view(L * R, nblk_B, blsz)
        B_hat_flat = self._merge_blocks(B_rec_blk, out_feat)
        B_hat = B_hat_flat.view(L, R, out_feat).permute(0, 2, 1).contiguous()
        return A_hat, B_hat

    def aux_loss(self):
        return self.eb.loss()



def zero_lora_param_dict(target_modules, n_layers, r, in_features, out_features):
    return nn.ParameterDict({
        "A": nn.ParameterDict({
            # m: nn.Parameter(torch.zeros(n_layers, r, in_features[m]), requires_grad=False)
            m: nn.Parameter(torch.zeros(n_layers, 1, 1), requires_grad=False)
            for m in target_modules
        }),
        "B": nn.ParameterDict({
            # m: nn.Parameter(torch.zeros(n_layers, out_features[m], r), requires_grad=False)
            m: nn.Parameter(torch.zeros(n_layers, 1, 1), requires_grad=False)
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


def get_compnet_v5(
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
    
    comp_model = LoRACompressionModelV5(
        target_modules = args.target_modules,
        in_features = in_features,
        out_features = out_features,
        max_num_layers = max_num_layers,
        r = peft_config.r,
        block_size = args.block_size,
        latent_dim = args.compnet_latent_size,
        cond_dim= args.cond_dim,
        d_enc_in = args.d_enc_in,
        d_dec_out = args.d_dec_out,
        width = args.compnet_latent_width,
        mean_recon_target = mean_recon_target,
        std_recon_target = std_recon_target,
        learnable_pos_emb = args.learnable_pos_emb,
        rank_pos_mode = "none",
        use_layer_indices = False,
        use_layer_type = False,
    ).to(device)
    return comp_model