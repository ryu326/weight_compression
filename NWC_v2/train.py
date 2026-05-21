"""Single-GPU training entry point for NWC_v2.

    encoder transform × decoder transform × entropy model
    × dataset (gaussian | llama8b)

Logs to wandb every `--log_every` iters; runs val every `--eval_every`.
"""
import argparse
import math
import os
import random
import sys
import time

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

from datasets import get_dataset
from entropy import get_entropy_model  # noqa: F401  (import for early failure)
from loss import RDLoss
from models import get_model
from transforms import get_transform  # noqa: F401
from utils import AverageMeter, configure_optimizers, make_logger, save_checkpoint


def parse_args(argv):
    p = argparse.ArgumentParser(description="NWC_v2 trainer")
    # dataset
    p.add_argument("--dataset", default="llama8b", choices=["gaussian", "llama8b"])
    p.add_argument("--num_blocks", type=int, default=100_000, help="gaussian only")
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--input_size", type=int, default=16)
    p.add_argument("--gaussian_std", type=float, default=1.0, help="gaussian only")
    p.add_argument("--hf_path", type=str, default=None, help="llama8b only (override)")
    p.add_argument("--direction", type=str, default="row", choices=["row", "col"],
                   help="llama8b only: patch flatten direction")
    p.add_argument("--normalize", type=str, default="none",
                   choices=["none", "row", "col", "tensor"],
                   help="llama8b only: per-Linear-weight std normalization "
                        "('tensor' = scalar std per Linear)")
    # model
    p.add_argument("--M", type=int, default=16)
    p.add_argument(
        "--encoder_transform", default="resblock",
        choices=["affine", "rht", "linear", "resblock"],
    )
    p.add_argument(
        "--decoder_transform", default="resblock",
        choices=["affine", "rht", "linear", "resblock"],
    )
    p.add_argument("--n_resblock", type=int, default=4)
    p.add_argument("--encoder_n_resblock", type=int, default=None,
                   help="Override n_resblock for encoder only (resblock transform). "
                        "If None, falls back to --n_resblock.")
    p.add_argument("--decoder_n_resblock", type=int, default=None,
                   help="Override n_resblock for decoder only (resblock transform). "
                        "If None, falls back to --n_resblock.")
    p.add_argument("--dim_encoder", type=int, default=32)
    p.add_argument("--no_layernorm", action="store_true")
    p.add_argument("--rht_seed", type=int, default=0)
    # entropy model
    p.add_argument(
        "--entropy_model", default="compressai",
        choices=["compressai", "parametric", "lattice"],
    )
    p.add_argument("--num_gaussian", type=int, default=3, help="parametric only")
    p.add_argument("--num_laplacian", type=int, default=3, help="parametric only")
    p.add_argument("--shared_eb", action="store_true",
                   help="Use EntropyBottleneck(channels=1) instead of channels=M; "
                        "treats all latent dims as samples from one shared distribution.")
    # optimization
    p.add_argument("--lmbda", type=float, default=100.0)
    p.add_argument("--iter", type=int, default=200_000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--aux_learning_rate", type=float, default=1e-3)
    p.add_argument("--clip_max_norm", type=float, default=1.0)
    # early stopping (based on val loss)
    p.add_argument("--early_stop_patience", type=int, default=0,
                   help="Stop training if val loss does not improve for this many "
                        "consecutive evals.  0 = disabled.")
    p.add_argument("--early_stop_min_iter", type=int, default=0,
                   help="Don't trigger early stop before this iter (warmup grace).")
    p.add_argument("--early_stop_metric", type=str, default="loss",
                   choices=["loss", "mse"],
                   help="Which val metric to track for early stopping.")
    # logging / saving
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--eval_every", type=int, default=5000)
    p.add_argument("--save_dir", type=str, default="./checkpoint")
    p.add_argument("--run_name", type=str, default="")
    p.add_argument("--wandb_project", type=str, default="nwc_v2")
    p.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


@torch.no_grad()
def evaluate(model, val_loader, criterion, device, val_std):
    model.eval()
    meters = {k: AverageMeter() for k in ("loss", "recon_loss", "bpp_loss", "mse")}
    for data in val_loader:
        data = {k: v.to(device) for k, v in data.items()}
        out = model(data)
        losses = criterion(data, out)
        bs = data["weight_block"].shape[0]
        meters["loss"].update(losses["loss"].item(), bs)
        meters["recon_loss"].update(losses["recon_loss"].item(), bs)
        meters["bpp_loss"].update(losses["bpp_loss"].item(), bs)
        # raw mse / val_std² for direct comparability across datasets
        mse = ((out["x_hat"] - data["weight_block"].reshape(out["x_hat"].shape)) ** 2).mean().item()
        meters["mse"].update(mse / (val_std ** 2 + 1e-12), bs)
    model.train()
    return {k: m.avg for k, m in meters.items()}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Save dir + logger
    run_name = args.run_name or (
        f"{args.dataset}_{args.encoder_transform}-{args.decoder_transform}"
        f"_{args.entropy_model}_M{args.M}_lmbda{args.lmbda}"
    )
    save_path = os.path.join(args.save_dir, run_name)
    os.makedirs(save_path, exist_ok=True)
    logger = make_logger("nwc_v2", os.path.join(save_path, "train.log"))
    logger.info(f"args: {vars(args)}")

    # wandb
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=vars(args),
        mode=args.wandb_mode,
    )

    # dataset
    train_dataset, val_dataset, train_std, val_std, train_mean, val_mean = get_dataset(args)
    logger.info(
        f"dataset={args.dataset} train_size={len(train_dataset)} "
        f"val_size={len(val_dataset)} train_std={train_std:.5f} "
        f"train_mean={train_mean:.5f}"
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # model — shift = dataset mean (NWC parity), scale = dataset std
    scale = torch.tensor(float(train_std), dtype=torch.float32)
    shift = torch.tensor(float(train_mean), dtype=torch.float32)
    model = get_model(args, scale=scale, shift=shift).to(device)
    logger.info(f"model:\n{model}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"trainable params: {n_params:,}")

    # criterion + optimizers
    criterion = RDLoss(std=train_std, lmbda=args.lmbda).to(device)
    optimizer, aux_optimizer = configure_optimizers(model, args)
    aux_str = f"aux: {sum(1 for n, p in model.named_parameters() if '.quantiles' in n)} params"
    logger.info(f"main optimizer + {aux_str}")

    # checkpoint resume.  best_metric tracks val[args.early_stop_metric] (the
    # quantity used both for best.pth.tar selection AND early-stop firing).
    best_metric_key = "loss" if args.early_stop_metric == "loss" else "mse"
    start_iter = 0
    best_metric = math.inf
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        # Materialize compressai EB's CDF buffers before load (they're sized
        # via `update()`; freshly-built models start with shape (0,) so the
        # checkpoint's (M,…) buffers won't load directly).
        model.update(force=True)
        sd = ckpt["state_dict"]
        # Drop any EB CDF buffers that still mismatch — `update()` rebuilds
        # them from the loaded `quantiles` after the load.
        for k in ("entropy_bottleneck._offset",
                  "entropy_bottleneck._quantized_cdf",
                  "entropy_bottleneck._cdf_length"):
            sd.pop(k, None)
        model.load_state_dict(sd, strict=False)
        model.update(force=True)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if aux_optimizer and "aux_optimizer" in ckpt and ckpt["aux_optimizer"]:
            aux_optimizer.load_state_dict(ckpt["aux_optimizer"])
        start_iter = int(ckpt.get("iter", 0))
        # Don't load prior best_metric — when resuming with a (possibly) new
        # `early_stop_metric`, the prior best is over the wrong quantity.
        # Caller's run will refresh best at the next eval.
        logger.info(
            f"resumed from {args.checkpoint} @ iter {start_iter} "
            f"(best.pth.tar will track val/{best_metric_key} from this point)"
        )

    # training loop
    total_iter = start_iter
    # Early-stop tracking — based on val[args.early_stop_metric].
    best_es_metric = math.inf
    no_improve_count = 0
    early_stopped = False
    t0 = time.time()
    while total_iter < args.iter:
        for data in train_loader:
            if total_iter >= args.iter:
                break
            data = {k: v.to(device, non_blocking=True) for k, v in data.items()}

            optimizer.zero_grad(set_to_none=True)
            if aux_optimizer is not None:
                aux_optimizer.zero_grad(set_to_none=True)

            out = model(data)
            losses = criterion(data, out)
            losses["loss"].backward()

            if args.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
            optimizer.step()

            aux_loss_val = 0.0
            if aux_optimizer is not None:
                try:
                    aux_loss = model.aux_loss()
                except AttributeError:
                    aux_loss = torch.zeros((), device=device)
                if aux_loss.requires_grad:
                    aux_loss.backward()
                    aux_optimizer.step()
                aux_loss_val = float(aux_loss.detach().item())

            total_iter += 1

            if total_iter % args.log_every == 0 or total_iter == 1:
                throughput = total_iter / max(time.time() - t0, 1e-9)
                msg = (
                    f"iter {total_iter}/{args.iter} "
                    f"loss={losses['loss'].item():.5f} "
                    f"recon={losses['recon_loss'].item():.5f} "
                    f"bpp={losses['bpp_loss'].item():.5f} "
                    f"aux={aux_loss_val:.5f} "
                    f"({throughput:.1f} it/s)"
                )
                logger.info(msg)
                wandb.log({
                    "train/loss": losses["loss"].item(),
                    "train/recon_loss": losses["recon_loss"].item(),
                    "train/bpp_loss": losses["bpp_loss"].item(),
                    "train/aux_loss": aux_loss_val,
                    "iter": total_iter,
                })

            if total_iter % args.eval_every == 0 or total_iter == args.iter:
                model.update(force=True)
                val_metrics = evaluate(model, val_loader, criterion, device, val_std)
                logger.info(f"[VAL] iter {total_iter}: {val_metrics}")
                wandb.log({**{f"val/{k}": v for k, v in val_metrics.items()}, "iter": total_iter})

                ckpt_path = os.path.join(save_path, f"ckpt_iter{total_iter}.pth.tar")
                save_checkpoint({
                    "iter": total_iter,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict() if aux_optimizer else None,
                    "best_metric": best_metric,
                    "best_metric_key": best_metric_key,
                    "args": vars(args),
                }, ckpt_path)

                if val_metrics[best_metric_key] < best_metric:
                    best_metric = val_metrics[best_metric_key]
                    save_checkpoint({
                        "iter": total_iter,
                        "state_dict": model.state_dict(),
                        "best_metric": best_metric,
                        "best_metric_key": best_metric_key,
                        "best_metrics": val_metrics,
                        "args": vars(args),
                    }, os.path.join(save_path, "best.pth.tar"))
                    logger.info(
                        f"  best/{best_metric_key} updated -> {best_metric:.6f}  "
                        f"(bpp={val_metrics['bpp_loss']:.4f}, mse={val_metrics['mse']:.6f})"
                    )

                # Early stopping — tracks the same metric as best.pth.tar.
                if args.early_stop_patience > 0:
                    es_key = best_metric_key
                    cur = float(val_metrics[es_key])
                    if cur < best_es_metric - 1e-9:
                        best_es_metric = cur
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                        logger.info(
                            f"  early-stop: no-improve count = {no_improve_count}/"
                            f"{args.early_stop_patience} (best {es_key}={best_es_metric:.6f}, "
                            f"current={cur:.6f})"
                        )
                        if (no_improve_count >= args.early_stop_patience
                                and total_iter >= args.early_stop_min_iter):
                            logger.info(
                                f"  early-stop: triggered at iter {total_iter} "
                                f"(no improvement on val/{es_key} for "
                                f"{no_improve_count} evals)"
                            )
                            early_stopped = True
                            break  # break inner data-loader loop
        if early_stopped:
            break  # break outer while-loop

    if early_stopped:
        logger.info(f"training stopped early at iter {total_iter}")
    else:
        logger.info("training complete")
    wandb.finish()


if __name__ == "__main__":
    main()
