import torch
import torch.nn as nn
import torch.nn.functional as F

from compressai.models.google import CompressionModel
from compressai.layers import EntropyBottleneck


# -------------------------
# 조건부 AdaLN (vector용)
# -------------------------
class AdaLN(nn.Module):
    def __init__(self, feat_dim: int, cond_dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(feat_dim)
        self.to_affine = nn.Linear(cond_dim, 2 * feat_dim, bias=True)  # -> [gamma, beta]

    def forward(self, x, cond):
        h = self.ln(x)
        gamma, beta = self.to_affine(cond).chunk(2, dim=-1)
        return gamma * h + beta


# -------------------------
# 조건 임베딩 (depth/type)
# -------------------------
class LayerCond(nn.Module):
    def __init__(self,
                 depth_vocab_size=32,
                 type_vocab_size=2,
                 ab_vocab_size=2,          # NEW: A/B
                 embed_dim=32,
                 cond_proj_dim=128):
        super().__init__()
        self.depth_emb = nn.Embedding(depth_vocab_size, embed_dim)
        self.depth_ln  = nn.LayerNorm(embed_dim)

        self.type_emb  = nn.Embedding(type_vocab_size, embed_dim)
        self.type_ln   = nn.LayerNorm(embed_dim)

        self.ab_emb    = nn.Embedding(ab_vocab_size, embed_dim)  # NEW
        self.ab_ln     = nn.LayerNorm(embed_dim)                  # NEW

        # concat: depth(32) + type(32) + ab(32) = 96 → proj → cond_proj_dim(=128)
        self.proj = nn.Linear(embed_dim * 3, cond_proj_dim)

    def forward(self, layer_depth: torch.Tensor, layer_type: torch.Tensor, ab_type: torch.Tensor):
        """
        layer_depth: (B,) long
        layer_type:  (B,) long
        ab_type:     (B,) long, 0=A, 1=B
        """
        d  = self.depth_ln(self.depth_emb(layer_depth))
        t  = self.type_ln(self.type_emb(layer_type))
        ab = self.ab_ln(self.ab_emb(ab_type))
        cond = torch.cat([d, t, ab], dim=-1)
        return F.silu(self.proj(cond))  # (B, cond_proj_dim)


# -------------------------
# Residual 블록 (AdaLN 적용)
# x -> Linear -> SiLU -> AdaLN -> Dropout -> Linear -> SiLU -> Dropout -> + skip
# in/out dim이 다르면 proj로 skip 정렬
# -------------------------
class ResidualAdaMLPBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, cond_dim, p=0.05):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.adaln1 = AdaLN(hidden_dim, cond_dim)
        self.drop1 = nn.Dropout(p)

        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.adaln2 = AdaLN(out_dim, cond_dim)
        self.drop2 = nn.Dropout(p)

        self.proj = None
        if in_dim != out_dim:
            self.proj = nn.Linear(in_dim, out_dim, bias=False)  # skip projection

    def forward(self, x, cond):
        h = F.silu(self.fc1(x))
        h = self.adaln1(h, cond)
        h = self.drop1(h)

        h = F.silu(self.fc2(h))
        h = self.adaln2(h, cond)
        h = self.drop2(h)

        skip = x if self.proj is None else self.proj(x)
        return h + skip


# -------------------------
# Encoder/Decoder (Residual 포함)
# -------------------------
class CondMLPEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=1024, latent_dim=256, cond_dim=128, p=0.05, nblocks=2):
        super().__init__()
        blocks = []
        dim_in = in_dim
        dim_hidden = hidden_dim
        for i in range(nblocks):
            blocks.append(ResidualAdaMLPBlock(
                in_dim=dim_in,
                hidden_dim=dim_hidden,
                out_dim=dim_hidden,
                cond_dim=cond_dim,
                p=p
            ))
            dim_in = dim_hidden
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(dim_hidden, latent_dim)  # y

    def forward(self, x, cond):
        h = x
        for blk in self.blocks:
            h = blk(h, cond)
        y = self.head(h)
        return y


class CondMLPDecoder(nn.Module):
    def __init__(self, out_dim, hidden_dim=1024, latent_dim=256, cond_dim=128, p=0.05, nblocks=2,
                 use_global_residual=True):
        super().__init__()
        self.use_global_residual = use_global_residual
        self.in_proj = nn.Linear(latent_dim, hidden_dim)

        blocks = []
        dim_in = hidden_dim
        for i in range(nblocks):
            blocks.append(ResidualAdaMLPBlock(
                in_dim=dim_in,
                hidden_dim=hidden_dim,
                out_dim=hidden_dim,
                cond_dim=cond_dim,
                p=p
            ))
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(hidden_dim, out_dim)

        # 디코더 글로벌 residual: x_hat = head(h) + x_skip
        self.out_dim = out_dim

    def forward(self, y_hat, cond, x_skip_flat=None):
        h = F.silu(self.in_proj(y_hat))
        for blk in self.blocks:
            h = blk(h, cond)
        out = self.head(h)  # (B, out_dim)

        # if self.use_global_residual and (x_skip_flat is not None) and (x_skip_flat.shape[-1] == self.out_dim):
        #     out = out + x_skip_flat
        return out


# -------------------------
# NWC_lora (CompressionModel) with residual-enc/dec
# -------------------------
class NWC_lora(CompressionModel):
    def __init__(self,
                 in_shape,
                 depth_vocab_size=32,
                 type_vocab_size=2,
                 ab_vocab_size=2,                 # NEW
                 embed_dim=32,
                 cond_dim=128,
                 enc_hidden=1024,
                 dec_hidden=1024,
                 latent_dim=256,
                 dropout=0.05,
                 enc_blocks=2,
                 dec_blocks=2,):
        super().__init__()
        # 입력 차원
        if isinstance(in_shape, int):
            self.in_dim = in_shape
            self._orig_shape = None
        else:
            prod = 1
            for s in in_shape:
                prod *= s
            self.in_dim = prod
            self._orig_shape = tuple(in_shape)

        # 조건 공유
        # self.layer_cond = LayerCond(depth_vocab_size, type_vocab_size, embed_dim, cond_dim)
        self.layer_cond = LayerCond(depth_vocab_size, type_vocab_size, ab_vocab_size, embed_dim, cond_dim)

        # 인코더/디코더 (Residual 포함)
        self.g_a = CondMLPEncoder(self.in_dim, enc_hidden, latent_dim, cond_dim, dropout, nblocks=enc_blocks)
        self.g_s = CondMLPDecoder(self.in_dim, dec_hidden, latent_dim, cond_dim, dropout,
                                  nblocks=dec_blocks, use_global_residual=False)

        # 엔트로피 병목
        self.entropy_bottleneck = EntropyBottleneck(latent_dim)

    def _flatten(self, x):
        if x.ndim > 2:
            return x.view(x.size(0), -1)
        return x

    def _reshape(self, x):
        if self._orig_shape is None:
            return x
        return x.view(x.size(0), *self._orig_shape)

    def forward(self, x, layer_depth: torch.Tensor, layer_type: torch.Tensor, ab_type: torch.Tensor):
        x_in = self._flatten(x)
        cond = self.layer_cond(layer_depth, layer_type, ab_type)     # UPDATED
        y = self.g_a(x_in, cond)
        y_hat, y_likelihoods = self.entropy_bottleneck(y, training=self.training)
        x_hat_flat = self.g_s(y_hat, cond, x_skip_flat=x_in)
        x_hat = self._reshape(x_hat_flat)
        return {"x_hat": x_hat, "likelihoods": {"y": y_likelihoods}}

    @torch.no_grad()
    def compress(self, x, layer_depth: torch.Tensor, layer_type: torch.Tensor, ab_type: torch.Tensor):
        x_in = self._flatten(x)
        cond = self.layer_cond(layer_depth, layer_type, ab_type)     # UPDATED
        y = self.g_a(x_in, cond)
        y_strings = self.entropy_bottleneck.compress(y)
        return {"strings": y_strings, "shape": y.size()}

    @torch.no_grad()
    def decompress(self, strings, shape, layer_depth: torch.Tensor, layer_type: torch.Tensor, ab_type: torch.Tensor):
        y_hat = self.entropy_bottleneck.decompress(strings, shape)
        cond = self.layer_cond(layer_depth, layer_type, ab_type)     # UPDATED
        x_hat_flat = self.g_s(y_hat, cond, x_skip_flat=None)  # 추론에서는 global residual 미사용 권장
        x_hat = self._reshape(x_hat_flat)
        return {"x_hat": x_hat}

    # def aux_loss(self):
    #     return self.entropy_bottleneck.loss()
