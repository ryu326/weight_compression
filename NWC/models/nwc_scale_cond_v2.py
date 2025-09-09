# file: nwc_models.py
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from compressai.models import CompressionModel
from compressai.entropy_models import EntropyBottleneck, GaussianConditional

# (Optional) Lattice-variant
try:
    from lattice_transform_coding.LTC.entropy_models import EntropyBottleneckLattice
    from lattice_transform_coding.LTC.quantizers import get_lattice
    _HAS_LTC = True
except Exception:
    _HAS_LTC = False


__all__ = [
    "get_scale_table",
    "SCALES_MIN",
    "SCALES_MAX",
    "SCALES_LEVELS",
    "NWCScaleCond",
    "NWCScaleCondLTC",
]


# ====== Utilities ======

SCALES_MIN = 0.11
SCALES_MAX = 256.0
SCALES_LEVELS = 64


def get_scale_table(min_: float = SCALES_MIN, max_: float = SCALES_MAX, levels: int = SCALES_LEVELS) -> Tensor:
    """Log-spaced scale table (Balle et al.)."""
    return torch.exp(torch.linspace(math.log(min_), math.log(max_), levels))


def ste_round(x: Tensor) -> Tensor:
    return torch.round(x) - x.detach() + x


def _permute_BLMD_to_BML(*tensors: Tensor) -> Tuple[Tensor, ...]:
    """[B, L, M] -> [B, M, L] (channel to dim=1 as CompressAI expects)."""
    out = []
    for t in tensors:
        perm = list(range(t.dim()))
        perm[1], perm[-1] = perm[-1], perm[1]
        out.append(t.permute(*perm).contiguous())
    return tuple(out)


def _permute_BML_to_BLMD(*tensors: Tensor) -> Tuple[Tensor, ...]:
    """Inverse of _permute_BLMD_to_BML."""
    out = []
    for t in tensors:
        perm = list(range(t.dim()))
        perm[1], perm[-1] = perm[-1], perm[1]
        out.append(t.permute(*perm).contiguous())
    return tuple(out)


# ====== Building blocks ======

def get_act(name: str):
    name = name.lower()
    return {"relu": nn.ReLU, "silu": nn.SiLU, "gelu": nn.GELU}.get(name, nn.SiLU)()

class LinearResBlock(nn.Module):
    def __init__(self, dim, norm=True, act="silu"):
        super().__init__()
        layers = [nn.Linear(dim, dim)]
        if norm: layers.append(nn.LayerNorm(dim))
        layers.append(get_act(act))
        self.f = nn.Sequential(*layers)
    def forward(self, x): return x + self.f(x)


class MLPEncoder(nn.Module):
    """Simple residual MLP: in -> (resblock x n) -> out."""
    def __init__(self, in_dim: int, n_res_layers: int, hidden: int, out_dim: int, norm: bool):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden)
        self.stack = nn.ModuleList([LinearResBlock(hidden, norm=norm) for _ in range(n_res_layers)])
        self.out_proj = nn.Linear(hidden, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_proj(x)
        for layer in self.stack:
            x = layer(x)
        return self.out_proj(x)


# class FiLM(nn.Module):
#     """cond -> (gamma, beta) and apply: h = (1+tanh(gamma))*h + beta"""
#     def __init__(self, cond_dim: int, target_dim: int):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.LayerNorm(cond_dim),
#             nn.Linear(cond_dim, 2 * target_dim),
#         )

#     def forward(self, h, cond):
#         gb = self.net(cond)
#         gamma, beta = gb.chunk(2, dim=-1)
#         return (1.0 + torch.tanh(gamma)) * h + beta
    
    
class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation with residual & LayerNorm."""
    def __init__(self, cond_dim: int, target_dim: int = None):
        super().__init__()
        target_dim = target_dim if target_dim is not None else cond_dim
        self.affine = nn.Linear(cond_dim, 2 * target_dim)  # -> gamma, beta
        self.norm = nn.LayerNorm(target_dim)

    def forward(self, a_feat: Tensor, b_feat: Tensor) -> Tensor:
        gamma, beta = self.affine(b_feat).chunk(2, dim=-1)
        mod = a_feat * gamma + beta
        return self.norm(a_feat + mod)


class IdxEmbedder(nn.Module):
    """
    layer_indices: LongTensor [L]
    layer_type: str (self.module_to_int로 인덱싱)
    -> [L, cond_dim]
    """
    def __init__(self, num_depth: int, num_type: int, cond_dim: int):
        super().__init__()
        d_depth = cond_dim // 2
        d_type  = cond_dim // 2
        self.depth_emb = nn.Sequential(
            nn.Embedding(num_depth, d_depth),
            nn.LayerNorm(d_depth),
        )
        self.type_emb = nn.Sequential(
            nn.Embedding(num_type, d_type),
            nn.LayerNorm(d_type),
        )
        self.cond_dim = cond_dim
        
    def forward(self, depth: torch.Tensor, ltype: torch.Tensor) -> torch.Tensor:
        depth_e = self.depth_emb(depth)
        type_e = self.depth_emb(ltype)
        return torch.cat([depth_e, type_e], dim=-1)  

# ====== Main model (EntropyBottleneck / optional hyperprior) ======

class NWCScaleCond(CompressionModel):
    """
    Conditional residual-MLP autoencoder with FiLM conditioning and optional hyperprior.

    Inputs (dict):
      - 'weight_block': [B, L, I]
      - 'scale_cond'  : [B, L, I]
      - (optional) 'depth': [B, L, I] indices for depth embedding
      - (optional) 'ltype': [B, L, I] indices for layer-type embedding

    Returns (dict):
      - use_hyper == True:
          {'x_hat': [B, L, I], 'likelihoods': {'y': Tensor, 'z': Tensor}}
        else:
          {'x_hat': [B, L, I], 'likelihoods': Tensor}
    """

    def __init__(
        self,
        input_size: int,
        dim_encoder: int,
        n_resblock: int,
        M: int,
        *,  # keyword-only below
        dim_proj: Optional[int] = None,
        n_clayers: int = 2,
        norm: bool = True,
        mode: str = "aun",           # 'aun' | 'ste' (| 'sga': not implemented)
        pe: bool = False,
        pe_n_depth: int = 42,
        pe_n_ltype: int = 7,
        use_hyper: bool = False,
        pre_normalize: bool = False,  # if False: divide by scale_cond at input and multiply back at output
    ):
        super().__init__()

        assert mode in {"aun", "ste", "sga"}, "mode must be one of {'aun','ste','sga'}"

        self.input_size = input_size
        self.dim_encoder = dim_encoder
        self.M = M
        self.mode = mode
        self.use_hyper = use_hyper
        self.pre_normalize = pre_normalize
        self.pe = pe

        # Projection sizes
        dim_proj = (dim_encoder // input_size) if dim_proj is None else dim_proj
        self.dim_proj = dim_proj

        # Encoder-side FiLM
        self.proj_x = nn.Linear(1, dim_proj)
        self.proj_s = nn.Linear(1, dim_proj)
        self.cond_layers = nn.ModuleList([FiLMLayer(dim_proj) for _ in range(n_clayers)])

        # Decoder-side FiLM
        self.proj_y = nn.Linear(dim_proj, 1)
        self.proj_s_dec = nn.Linear(1, dim_proj)
        self.cond_layers_dec = nn.ModuleList([FiLMLayer(dim_proj) for _ in range(n_clayers)])

        # Core encoder/decoder on flattened [B, L, dim_encoder]
        self.g_a = MLPEncoder(dim_encoder, n_resblock, dim_encoder, M, norm)
        self.g_s = MLPEncoder(M, n_resblock, dim_encoder, dim_encoder, norm)

        # Optional PE (depth & layer-type)
        if self.pe:
            self.pe_embed = IdxEmbedder(pe_n_depth, pe_n_ltype, input_size)
            self.pe_film_en = FiLMLayer(input_size)
            self.pe_film_dec = FiLMLayer(input_size, M)
            # self.depth_embedding = nn.Embedding(pe_n_depth, input_size)
            # self.ltype_embedding = nn.Embedding(pe_n_ltype, input_size)

        # Entropy models
        if self.use_hyper:
            # Hyperprior for y -> (z EB) + GaussianConditional for y
            self.h_a = MLPEncoder(M, max(1, n_resblock // 2), max(1, dim_encoder // 4), M // 2, norm)
            self.h_s_means = MLPEncoder(M // 2, max(1, n_resblock // 2), max(1, dim_encoder // 4), M, norm)
            self.h_s_scales = MLPEncoder(M // 2, max(1, n_resblock // 2), max(1, dim_encoder // 4), M, norm)

            self.gaussian_conditional = GaussianConditional(None)
            self.z_entropy_bottleneck = EntropyBottleneck(M // 2)
            # Backward-compat alias (some trainers inspect .entropy_bottleneck)
            self.entropy_bottleneck = self.z_entropy_bottleneck
        else:
            self.y_entropy_bottleneck = EntropyBottleneck(M)
            self.entropy_bottleneck = self.y_entropy_bottleneck

    # ---- helpers ----

    def _apply_pe(self, x_shift: Tensor, data: Dict[str, Tensor], pe_film: FiLMLayer) -> Tensor:
        if not self.pe:
            return x_shift
        # d_embed = self.depth_embedding(data["depth"])   # [B, L, I]
        # l_embed = self.ltype_embedding(data["ltype"])   # [B, L, I]
        pe = self.pe_embed(data["depth"], data["ltype"])
        return pe_film(x_shift, pe)

    def _encode_conditioned(self, x: Tensor, s: Tensor) -> Tensor:
        """
        x: [B, L, I], s: [B, L, I]  -> returns y: [B, L, M]
        """
        x_p = self.proj_x(x.unsqueeze(-1))  # -> [B, L, I, P]
        s_p = self.proj_s(s.unsqueeze(-1))  # -> [B, L, I, P]
        for layer in self.cond_layers:
            x_p = layer(x_p, s_p)
        x_flat = torch.flatten(x_p, start_dim=-2)       # [B, L, I*P] == [B, L, dim_encoder]
        y = self.g_a(x_flat)                            # [B, L, M]
        return y

    def _decode_conditioned(self, y: Tensor, s: Tensor) -> Tensor:
        """
        y: [B, L, M], s: [B, L, I] -> returns x_hat: [B, L, I]
        """
        h = self.g_s(y)                                 # [B, L, dim_encoder]
        h = h.view(h.size(0), h.size(1), self.input_size, self.dim_proj)  # [B, L, I, P]
        s_p = self.proj_s_dec(s.unsqueeze(-1))          # [B, L, I, P]
        for layer in self.cond_layers_dec:
            h = layer(h, s_p)
        x_hat = self.proj_y(h).squeeze(-1)              # [B, L, I]
        return x_hat

    # ---- main passes ----

    def forward(self, data: Dict[str, Tensor], *, scale: Optional[Tensor] = None, shift: Optional[Tensor] = None):
        x: Tensor = data["weight_block"]     # [B, L, I]
        s: Tensor = data["scale_cond"]       # [B, L, I]

        # (Optional) pre/post normalization via scale_cond
        x_in = x if self.pre_normalize else (x / s)

        # (Optional) PE
        if self.pe:
            x_in = self._apply_pe(x_in, data, self.pe_film_en)

        # Encode with FiLM
        y = self._encode_conditioned(x_in, s)  # [B, L, M]

        if self.use_hyper:
            # Hyperprior path: z EB + GaussianConditional on y
            z = self.h_a(y)                                        # [B, L, M/2]
            z_BML, = _permute_BLMD_to_BML(z)
            z_hat_BML, z_likelihoods_BML = self.z_entropy_bottleneck(z_BML)
            z_hat, = _permute_BML_to_BLMD(z_hat_BML)

            means_hat = self.h_s_means(z_hat)                      # [B, L, M]
            scales_hat = self.h_s_scales(z_hat)                    # [B, L, M]

            y_BML, scales_BML, means_BML = _permute_BLMD_to_BML(y, scales_hat, means_hat)
            indexes = self.gaussian_conditional.build_indexes(scales_BML)
            y_hat_BML, y_likelihoods_BML = self.gaussian_conditional(y_BML, scales_BML, means=means_BML)
            y_hat, = _permute_BML_to_BLMD(y_hat_BML)
            
            if self.pe:
                y_hat = self._apply_pe(y_hat, data, self.pe_film_dec)
                
            x_hat = self._decode_conditioned(y_hat, s)             # [B, L, I]
            if not self.pre_normalize:
                x_hat = x_hat * s

            return {
                "x_hat": x_hat,
                "likelihoods": {
                    "y": _permute_BML_to_BLMD(y_likelihoods_BML)[0],
                    "z": _permute_BML_to_BLMD(z_likelihoods_BML)[0],
                },
            }

        # No hyperprior: use EB on y
        y_BML, = _permute_BLMD_to_BML(y)
        y_hat_BML, y_likelihoods_BML = self.y_entropy_bottleneck(y_BML)

        if self.mode == "ste":
            # STE with median offsets
            med = self.y_entropy_bottleneck._get_medians()
            y_tmp = y_BML - med
            y_hat_BML = ste_round(y_tmp) + med

        y_hat, = _permute_BML_to_BLMD(y_hat_BML)
        if self.pe: 
            y_hat = self._apply_pe(y_hat, data, self.pe_film_dec)
        x_hat = self._decode_conditioned(y_hat, s)                 # [B, L, I]
        if not self.pre_normalize:
            x_hat = x_hat * s

        return {"x_hat": x_hat, "likelihoods": _permute_BML_to_BLMD(y_likelihoods_BML)[0]}

    # ---- bitstream API ----

    def compress(self, data: Dict[str, Tensor]):
        x: Tensor = data["weight_block"]
        s: Tensor = data["scale_cond"]
        x_in = x if self.pre_normalize else (x / s)
        if self.pe:
            x_in = self._apply_pe(x_in, data, self.pe_film_en)
        y = self._encode_conditioned(x_in, s)

        if self.use_hyper:
            # z strings
            z = self.h_a(y)
            z_BML, = _permute_BLMD_to_BML(z)
            z_shape = z_BML.size()[2:]
            z_strings = self.z_entropy_bottleneck.compress(z_BML)
            z_hat_BML = self.z_entropy_bottleneck.decompress(z_strings, z_shape)
            z_hat, = _permute_BML_to_BLMD(z_hat_BML)

            means_hat = self.h_s_means(z_hat)
            scales_hat = self.h_s_scales(z_hat)

            y_BML, scales_BML, means_BML = _permute_BLMD_to_BML(y, scales_hat, means_hat)
            indexes = self.gaussian_conditional.build_indexes(scales_BML)
            y_strings = self.gaussian_conditional.compress(y_BML, indexes, means=means_BML)

            return {"strings": [y_strings, z_strings], "shape": z_shape, "scale_cond": s}

        # no hyper
        y_BML, = _permute_BLMD_to_BML(y)
        y_shape = y_BML.size()[2:]
        y_strings = self.y_entropy_bottleneck.compress(y_BML)
        if self.pe:
            return {"strings": [y_strings], "shape": y_shape, "scale_cond": s, "depth": data["depth"], "ltype": data["ltype"]}
        else:
            return {"strings": [y_strings], "shape": y_shape, "scale_cond": s,}
            
    def decompress(self, enc_data: Dict[str, object]):
        strings = enc_data["strings"]
        shape = enc_data["shape"]
        s: Tensor = enc_data["scale_cond"]  # [B, L, I]

        if self.use_hyper:
            # z -> means/scales -> y
            z_hat_BML = self.z_entropy_bottleneck.decompress(strings[1], shape)
            z_hat, = _permute_BML_to_BLMD(z_hat_BML)
            means_hat = self.h_s_means(z_hat)
            scales_hat = self.h_s_scales(z_hat)
            scales_BML, means_BML = _permute_BLMD_to_BML(scales_hat, means_hat)

            indexes = self.gaussian_conditional.build_indexes(scales_BML)
            y_hat_BML = self.gaussian_conditional.decompress(strings[0], indexes, means=means_BML)
            y_hat, = _permute_BML_to_BLMD(y_hat_BML)
        else:
            y_hat_BML = self.y_entropy_bottleneck.decompress(strings[0], shape)
            y_hat, = _permute_BML_to_BLMD(y_hat_BML)

        if self.pe:
            y_hat = self._apply_pe(y_hat, enc_data, self.pe_film_dec)
        x_hat = self._decode_conditioned(y_hat, s)
        if not self.pre_normalize:
            x_hat = x_hat * s
        return {"x_hat": x_hat}

    # ---- aux loss ----
    # def aux_loss(self) -> Tensor:
    #     # Prefer using available EB loss if present; otherwise zero.
    #     dev = next(self.parameters()).device
    #     eb = getattr(self, "z_entropy_bottleneck", None) if self.use_hyper else getattr(self, "y_entropy_bottleneck", None)
    #     if eb is not None and hasattr(eb, "loss"):
    #         return eb.loss()
    #     return torch.tensor(0.0, device=dev)

