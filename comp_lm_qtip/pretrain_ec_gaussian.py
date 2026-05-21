#!/usr/bin/env python3
"""Pretrain ec_linear entropy model (+ qs, + B for lattice_eb) on N(0, 1) data.

Produces a checkpoint that `ec_linear_ft.load_pretrained_ec_state` can use to
initialize the `qs`, `entropy_bottleneck`, and (for lattice_eb) `B` fields of a
freshly-constructed EntropyConstrainedLinear when quantizing an LLM.

The `latent`, `left_diag`, `right_diag` are frozen during pretraining — those
are per-layer / per-weight and must be re-derived at LLM time.

Usage examples:
  # compressai variant (EntropyBottleneck channels=1)
  python -m pretrain_ec_gaussian \
      --variant compressai \
      --in_features 4096 --out_features 4096 \
      --save_path ec_pretrained/compressai.pt

  # lattice_eb variant (channels = lattice_dim)
  python -m pretrain_ec_gaussian \
      --variant lattice_eb --lattice_dim 16 --B_init orthogonal \
      --lambda_ortho 0.1 \
      --in_features 4096 --out_features 4096 \
      --save_path ec_pretrained/lattice_eb_n16.pt
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim


def build_layer(args, device):
    # Full encode → quantize → decode pipeline is used in the training loop.
    # left_diag / right_diag are frozen below (per-layer at LLM time anyway).
    common = dict(
        in_features=args.in_features,
        out_features=args.out_features,
        bias=False,
        decoder_type=args.decoder_type,
        rht_seed=args.seed,
        device=device,
        dtype=torch.float32,
    )
    if args.variant == "compressai":
        from lib.linear.ec_linear import EntropyConstrainedLinear
        return EntropyConstrainedLinear(**common)
    if args.variant == "lattice_eb":
        from lib.linear.ec_linear_lattice_eb import EntropyConstrainedLinear
        common.update(
            lattice_dim=args.lattice_dim,
            lambda_ortho=args.lambda_ortho,
            B_init=args.B_init,
        )
        return EntropyConstrainedLinear(**common)
    raise ValueError(f"Unknown variant: {args.variant}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["compressai", "lattice_eb"], required=True)
    parser.add_argument("--in_features", type=int, default=4096)
    parser.add_argument("--out_features", type=int, default=4096)
    parser.add_argument("--decoder_type", default="rht",
                        choices=["identity", "rht", "dft"],
                        help="Encoder/decoder type for the full pipeline used "
                             "during pretraining. 'rht' matches typical LLM use.")
    parser.add_argument("--lattice_dim", type=int, default=16)
    parser.add_argument("--B_init", default="orthogonal",
                        choices=["identity", "orthogonal", "uniform"])
    parser.add_argument("--lambda_ortho", type=float, default=0.0)
    parser.add_argument("--num_steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--aux_lr", type=float, default=1e-3)
    parser.add_argument("--qs_lr", type=float, default=1e-2)
    parser.add_argument("--lmbda", type=float, default=0.1,
                        help="Rate weight in total_loss = mse + lmbda * rate (+ ortho)")
    parser.add_argument("--qs_init", type=float, default=1.0)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    layer = build_layer(args, device)
    layer.quantize_mode = "noise"
    layer.train()
    # All parameters (latent, left_diag, right_diag, B, entropy_bottleneck)
    # are trainable. `latent` is also overwritten each step with encode(w)
    # for a fresh Gaussian weight w — the optimizer's update on latent gets
    # squashed by that copy, which is fine.

    # Learnable scalar qs, parameterized in log-space for positivity.
    log_qs = nn.Parameter(
        torch.log(torch.tensor([args.qs_init], device=device, dtype=torch.float32))
    )

    main_params: list[nn.Parameter] = []
    aux_params: list[nn.Parameter] = []
    for n, p in layer.named_parameters():
        if not p.requires_grad:
            continue
        (aux_params if ".quantiles" in n else main_params).append(p)

    optimizer = optim.Adam(
        [
            {"params": main_params, "lr": args.lr},
            {"params": [log_qs], "lr": args.qs_lr},
        ]
    )
    aux_optimizer = optim.Adam(aux_params, lr=args.aux_lr) if aux_params else None

    for step in range(args.num_steps):
        # Fresh Gaussian weight — encode it into latent space (frozen
        # left_diag/right_diag + decoder-type orthogonal transform). No grad
        # through encode: latent is a (frozen) parameter set via data copy.
        w = torch.randn_like(layer.latent.data)
        with torch.no_grad():
            layer.latent.data.copy_(layer.encode_weight(w))

        optimizer.zero_grad(set_to_none=True)
        if aux_optimizer is not None:
            aux_optimizer.zero_grad(set_to_none=True)

        qs_val = log_qs.exp().reshape(())
        # quantize in latent space, then decode back to weight space.
        quantized_latent, _ = layer.quantize_latent(training=True, qs=qs_val)
        rate = layer._last_rate_loss
        recon = layer.decode_latent(quantized_latent)
        mse = torch.mean((recon - w) ** 2)
        total = mse + args.lmbda * rate

        if args.variant == "lattice_eb" and args.lambda_ortho > 0:
            total = total + args.lambda_ortho * layer.orthogonality_loss()

        total.backward()
        optimizer.step()

        if aux_optimizer is not None:
            aux_loss = layer.aux_loss()
            aux_loss.backward()
            aux_optimizer.step()

        if (step + 1) % args.log_every == 0 or step == 0:
            print(
                f"step {step + 1:5d}/{args.num_steps}: "
                f"mse={mse.item():.4e}  rate={rate.item():.4f}  "
                f"qs={log_qs.exp().item():.4f}",
                flush=True,
            )

    # Bake CDF tables into EB internals (so load-time is fast).
    layer.update_entropy_model(force=True, update_quantiles=True)

    ckpt = {
        "variant": args.variant,
        "config": {
            "in_features": args.in_features,
            "out_features": args.out_features,
            "decoder_type": args.decoder_type,
            "lattice_dim": args.lattice_dim if args.variant == "lattice_eb" else None,
            "lambda_ortho": args.lambda_ortho,
            "B_init": args.B_init,
            "lmbda": args.lmbda,
            "num_steps": args.num_steps,
            "seed": args.seed,
        },
        "qs": log_qs.exp().detach().cpu().reshape(()),
        "entropy_bottleneck": {
            k: v.detach().cpu() for k, v in layer.entropy_bottleneck.state_dict().items()
        },
    }
    if args.variant == "lattice_eb":
        ckpt["B"] = layer.B.detach().cpu()

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, save_path)
    print(f"saved: variant={args.variant}  qs={ckpt['qs'].item():.4f}  -> {save_path}")


if __name__ == "__main__":
    main()
