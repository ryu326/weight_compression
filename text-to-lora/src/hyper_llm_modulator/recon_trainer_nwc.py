import logging
import random

import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm
import math
import os
from hyper_llm_modulator.utils.eval_hypermod import eval_hypermod_checkpoint
from hyper_llm_modulator.utils.eval_compnet import eval_compnet_checkpoint

logger = logging.getLogger()


def train(
    args,
    comp_model,
    train_data,
    layer_indices,
    n_batches,
    n_minibatches,
    tasks_per_batch,
    optimizer,
    aux_optimizer,
    lr_scheduler,
    device,
    save_dir,
):
    curstep = 1

    for ep in (pbar := tqdm(range(1, args.epochs + 1), total=args.epochs)):
        tasks = random.sample(args.train_ds_names, len(args.train_ds_names))
        # avg train unnorm error over all tasks
        
        err = 0
        losses = []
        grad_norms = []
        bpp_list, mse_list, aux_list = [], [], []
        
        for start_idx in range(0, len(tasks), tasks_per_batch):
            # sample tasks (so that we train all the layer/depth embedding for sampled tasks)
            batch_tasks = tasks[start_idx : start_idx + tasks_per_batch]
            batch_data = {task: train_data[task] for task in batch_tasks}

            loss, aux_loss, log_items = compute_loss(args, comp_model, batch_data, layer_indices, device)
            
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(comp_model.parameters(), 1.0)
            optimizer.step()

            aux_optimizer.zero_grad()
            aux_loss.backward()
            aux_optimizer.step()
            
            lr_scheduler.step()
            
            losses.append(loss.item())
            aux_list.append(aux_loss.item())
            
            grad_norms.append(float(grad_norm))
            bpp_list.append(float(log_items["bpp"]))
            mse_list.append(float(log_items["mse"]))
            # mseA_list.append(float(log_items["mse_A"]))
            # mseB_list.append(float(log_items["mse_B"]))
            

            # pbar.update(1)
            pbar.set_description(f"loss: {loss.item():.4E}")
            if (curstep % args.logging_freq == 0) or (curstep == n_batches):
                keys = [
                    "train/loss",
                    "train/aux_loss",
                    "train/mse",
                    "train/bpp_loss",
                    "train/grad_norm",
                    "train/lr",
                ]
                vals = [
                    torch.tensor(losses).mean(),
                    torch.tensor(aux_list).mean(),
                    torch.tensor(mse_list).mean(),
                    # torch.tensor(mseA_list).mean(),
                    # torch.tensor(mseB_list).mean(),
                    torch.tensor(bpp_list).mean(),
                    torch.tensor(grad_norms).mean(),
                    lr_scheduler.get_last_lr()[0],
                ]
                for k, v in zip(keys, vals):
                    log_scalar(k, v, curstep)

            # if (curstep % args.logging_freq == 0):
            if (curstep % args.val_freq == 0):
                os.makedirs(save_dir, exist_ok=True)
                ckpt_path = os.path.join(save_dir, "checkpoints", f"latest_it{curstep}.pt")
                torch.save(
                    {
                        "epoch": ep,
                        "step": curstep,
                        "model_state": comp_model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "aux_optimizer_state": aux_optimizer.state_dict(),
                        "lr_scheduler_state": lr_scheduler.state_dict(),
                        "args": vars(args),  # 재시작 시 편의
                    },
                    ckpt_path,
                )
                # 직전 에폭 파일 삭제
                if curstep > args.val_freq:
                    prev_path = os.path.join(save_dir, "checkpoints", f"latest_it{curstep-args.val_freq}.pt")
                    if os.path.exists(prev_path):
                        os.remove(prev_path)

                comp_model.eval()
                sd = comp_model.state_dict()
                save_path = f"{save_dir}/comp_model.pt"
                torch.save(sd, save_path)
                try:
                    eval_compnet_checkpoint(save_path, device, curstep, full_eval=False)
                except Exception as e:
                    logger.warning(f"Eval failed: {e}")
                comp_model.train()

            curstep += 1

        # err = err / n_minibatches
        # if ep % args.logging_freq == 0:
        #     logger.info(f"Epoch {ep}: avg unnorm error: {err:.4E}")
        #     wandb.log({"train/unnorm_err": err}, step=curstep)

        # ================================================
        
        # 재시작 시
        # ckpt = torch.load(os.path.join(save_dir, "latest.pt"), map_location=device)
        # comp_model.load_state_dict(ckpt["model_state"])
        # optimizer.load_state_dict(ckpt["optimizer_state"])
        # aux_optimizer.load_state_dict(ckpt["aux_optimizer_state"])
        # lr_scheduler.load_state_dict(ckpt["lr_scheduler_state"])
        # start_epoch = ckpt["epoch"] + 1
        # curstep = ckpt["step"]

    # save final model
    comp_model.eval()
    sd = comp_model.state_dict()
    save_path = f"{save_dir}/comp_model.pt"
    torch.save(sd, save_path)
    # eval_hypermod_checkpoint(save_path, device, curstep, full_eval=True)


def log_scalar(metric_name, val, curstep):
    if wandb.run is not None:
        wandb.log({metric_name: val}, step=curstep)
    logger.info(f"{metric_name}: {val:.4f}")

def compute_loss(args, model, batch_data, layer_indices, device):
    target_modules = args.target_modules
    loss = 0.0
    aux_loss = 0.0
    mse_loss_sum = 0.0
    bpp_loss_sum = 0.0
    # mseA_sum = 0.0
    # mseB_sum = 0.0

    batch_tasks = list(batch_data.keys())
    layer_indices = torch.tensor(layer_indices)

    for layer_type in target_modules:
        A_list, B_list, idx_list = [], [], []
        for t in batch_tasks:
            A_t = batch_data[t]["lora_A"][layer_type].to(device)  # [L, r, in_feat_m]
            B_t = batch_data[t]["lora_B"][layer_type].to(device)  # [L, out_feat_m, r]
                        
            if args.pred_z_score:
                avg_A = model.mean_recon_target["A"][layer_type]
                avg_B = model.mean_recon_target["B"][layer_type]
                std_A = model.std_recon_target["A"][layer_type]
                std_B = model.std_recon_target["B"][layer_type]
                A_t = (A_t - avg_A) / (std_A + 1e-10)
                B_t = (B_t - avg_B) / (std_B + 1e-10)       
            
            A_list.append(A_t)
            B_list.append(B_t)
            idx_list.append(layer_indices.to(device))             # [L]

        A_cat = torch.cat(A_list, dim=0)                   # [T*L, r, in]
        B_cat = torch.cat(B_list, dim=0)                   # [T*L, out, r]
        rep_indices = torch.cat(idx_list, dim=0)           # [T*L]
            
        out = model.forward(
            layer_type=layer_type,
            layer_indices=rep_indices,
            lora_A=A_cat,
            lora_B=B_cat,
        ) ## out: dict(A_hat, B_hat, likelihoods)
        
        if args.factorized:
            mse_A = F.mse_loss(out['A_hat'], A_cat, reduction="mean")
            mse_B = F.mse_loss(out['B_hat'], B_cat, reduction="mean")
            mse_loss = (mse_A + mse_B) / 2
        else:
            A_hat = out["A_hat"]        # [N, r, in]
            B_hat = out["B_hat"]        # [N, out, r]
            
            deltaW_hat = torch.bmm(B_hat, A_hat)     # [N, out, in]
            deltaW_tgt = torch.bmm(B_cat, A_cat)     # [N, out, in]
            # scale = getattr(args, "delta_w_scaling", 1.0)
            # mse_loss = F.mse_loss(deltaW_hat, deltaW_tgt * scale)
            mse_loss = F.mse_loss(deltaW_hat, deltaW_tgt)
            
        num_param = A_cat.numel() + B_cat.numel()
        if isinstance(out["likelihoods"], dict):
            bpp_loss = sum([torch.log(l).sum() for l in out["likelihoods"].values()]) / (-math.log(2) * num_param)
        else:
            bpp_loss = torch.log(out["likelihoods"]).sum() / (-math.log(2) * num_param)
        
        loss += args.rdlmbda * mse_loss + bpp_loss
        # loss += args.rdlmbda * mse_loss ## test
        aux_loss += model.aux_loss()
        
        mse_loss_sum += float(mse_loss)
        bpp_loss_sum += float(bpp_loss)
        # mseA_sum    += float(mse_A)
        # mseB_sum    += float(mse_B)

        # unnormalized error: A/B L1 평균(로그용)
        # with torch.no_grad():
        #     l1A = F.l1_loss(out["A_hat"], A_cat)
        #     l1B = F.l1_loss(out["B_hat"], B_cat)
        #     unnorm_err_sum += 0.5 * float(l1A + l1B)

    # 모듈 평균
    loss = loss / len(target_modules)
    aux_loss = aux_loss / len(target_modules)
    mse_avg = mse_loss_sum / len(target_modules)
    bpp_avg      = bpp_loss_sum      / len(target_modules)
    # mseA_avg     = mseA_sum     / len(target_modules)
    # mseB_avg     = mseB_sum     / len(target_modules)
    # unnorm_err   = unnorm_err_sum / len(target_modules)

    log_items = {
        "bpp": bpp_avg,
        "mse": mse_avg,
    }
    return loss, aux_loss, log_items

