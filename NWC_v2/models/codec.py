"""`NWCv2Codec` — encoder → entropy bottleneck → decoder.

Forward pass:
    x = data["weight_block"]            # (B, T, I)
    x_norm = (x - shift) / scale
    y = g_a(x_norm)                     # (B, T, M)  via encoder transform
    y_perm = y.permute(0, 2, 1)         # (B, M, T)  to match EB's (N, C, *) layout
    y_hat_perm, likelihoods = entropy_bottleneck(y_perm)
    y_hat = y_hat_perm.permute(0, 2, 1) # (B, T, M)
    x_hat = g_s(y_hat) * scale + shift  # (B, T, I)

Output: {"x", "x_hat", "likelihoods"}.
"""
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from transforms import get_transform
from entropy import get_entropy_model


class NWCv2Codec(nn.Module):
    def __init__(self, args, scale: Tensor, shift: Tensor):
        super().__init__()
        self.register_buffer("scale", scale.detach().clone())
        self.register_buffer("shift", shift.detach().clone())

        # Per-side n_resblock — falls back to shared --n_resblock when the
        # per-side flag is not provided (preserves existing behavior).
        n_resblock_default = int(getattr(args, "n_resblock", 4))
        enc_n_resblock = int(getattr(args, "encoder_n_resblock", None) or n_resblock_default)
        dec_n_resblock = int(getattr(args, "decoder_n_resblock", None) or n_resblock_default)

        # Encoder: input_size → M
        self.g_a = get_transform(
            args.encoder_transform,
            in_dim=int(args.input_size),
            out_dim=int(args.M),
            n_resblock=enc_n_resblock,
            dim_encoder=int(getattr(args, "dim_encoder", 32)),
            norm=not bool(getattr(args, "no_layernorm", False)),
            rht_seed=int(getattr(args, "rht_seed", 0)),
        )
        # Decoder: M → input_size.  When both ends are 'rht', tie the decoder
        # to the encoder's inverse — separate-rht-pair is broken because the
        # two RHTs are not initialized as inverses of each other.
        if args.encoder_transform == "rht" and args.decoder_transform == "rht":
            from transforms.rht import RHTInverse
            self.g_s = RHTInverse(self.g_a)
        else:
            self.g_s = get_transform(
                args.decoder_transform,
                in_dim=int(args.M),
                out_dim=int(args.input_size),
                n_resblock=dec_n_resblock,
                dim_encoder=int(getattr(args, "dim_encoder", 32)),
                norm=not bool(getattr(args, "no_layernorm", False)),
                rht_seed=int(getattr(args, "rht_seed", 0)) + 1,
            )

        self.shared_eb = bool(getattr(args, "shared_eb", False))
        eb_channels = 1 if self.shared_eb else int(args.M)
        self.entropy_bottleneck = get_entropy_model(
            args.entropy_model,
            channels=eb_channels,
            num_gaussian=int(getattr(args, "num_gaussian", 3)),
            num_laplacian=int(getattr(args, "num_laplacian", 3)),
        )

        self.input_size = int(args.input_size)
        self.M = int(args.M)
        self.entropy_model_name = str(args.entropy_model)

    def aux_loss(self) -> Tensor:
        eb = self.entropy_bottleneck
        if hasattr(eb, "loss"):
            return eb.loss()
        return torch.zeros((), device=self.scale.device)

    def update(self, force: bool = True, **kwargs) -> bool:
        eb = self.entropy_bottleneck
        if hasattr(eb, "update"):
            try:
                return eb.update(force=force, **kwargs)
            except TypeError:
                return eb.update(force=force)
        return True

    def forward(self, data, scale: Optional[Tensor] = None, shift: Optional[Tensor] = None):
        x = data["weight_block"]                # (B, T, I)
        scale = scale if scale is not None else self.scale
        shift = shift if shift is not None else self.shift
        x_norm = (x - shift) / scale

        y = self.g_a(x_norm)                    # (B, T, M)
        # Permute to (B, M, T) — compressai EB expects channels at dim=1
        y_perm = y.permute(0, 2, 1).contiguous()
        if self.shared_eb:
            B, M, T = y_perm.shape
            y_in = y_perm.reshape(B, 1, M * T)
            y_hat_in, likelihoods = self.entropy_bottleneck(y_in)
            y_hat_perm = y_hat_in.reshape(B, M, T)
            likelihoods = likelihoods.reshape(B, M, T)
        else:
            y_hat_perm, likelihoods = self.entropy_bottleneck(y_perm)
        y_hat = y_hat_perm.permute(0, 2, 1).contiguous()  # (B, T, M)

        x_hat = self.g_s(y_hat) * scale + shift  # (B, T, I)

        return {
            "x": x,
            "x_hat": x_hat,
            "likelihoods": likelihoods,
            "y": y,
            "y_hat": y_hat,
        }
