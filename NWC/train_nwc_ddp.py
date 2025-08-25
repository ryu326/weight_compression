import math
import operator
import os
import random
import shutil
import socket
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
# import torchvision
import wandb
from loss import *
from models import get_model
from datasets import get_datasets
from torch.utils.data import DataLoader
from utils.optimizers import *
from utils.util import *

import argparse
import math
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler



def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument("--iter", default=2000000, type=int)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("-lr", "--learning_rate", default=1e-4, type=float)
    parser.add_argument("--aux_learning_rate",
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument("-n", "--num_workers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--global_batch_size", type=int, default=None,
                        help="(DDP) 전체 GPU 합산 기준의 글로벌 배치사이즈. 설정 시 gradient accumulation로 맞춥니다.")
    parser.add_argument("--seed", type=float, default=100)
    parser.add_argument("--input_size", type=int, default=16)
    parser.add_argument("--dim_encoder", type=int, default=32)
    parser.add_argument("--n_resblock", type=int, default=4)
    parser.add_argument("--M", type=int, default=0)
    parser.add_argument("--N", type=int, default=0)
    parser.add_argument("--ltc_N", type=int, default=0)
    parser.add_argument("--R", type=int, default=0)
    parser.add_argument("--m", type=int, default=0)
    parser.add_argument("--Q", type=int, default=0)
    parser.add_argument("--C", type=int, default=0)
    parser.add_argument("--clip_max_norm", default=1.0, type=float)
    parser.add_argument("--save_dir", default="", type=str)
    parser.add_argument("--architecture", default=None, type=str)
    parser.add_argument("--loss", default="rdloss", type=str)
    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--lmbda", type=int, default=None)
    parser.add_argument("--lmbda_min", type=int, default=None)
    parser.add_argument("--lmbda_max", type=int, default=None)
    parser.add_argument("--dataset_stat_type", type=str, choices=['scaler', 'channel'], default='scaler')
    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--no_layernorm", action='store_true', default=False)
    parser.add_argument("--use_pe", action='store_true', default=False)
    parser.add_argument("--use_hyper", action='store_true', default=False)
    parser.add_argument("--uniform_scale_max", type=float, default=None)
    parser.add_argument("--dist_backend", type=str, default="nccl")
    parser.add_argument("--pre_normalize", action='store_true', default=False)
    args = parser.parse_args(argv)
    return args

def setup_distributed(backend="nccl"):
    """환경변수 (torchrun) 기준으로 DDP 초기화. 반환: (is_dist, rank, local_rank, world_size)"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return True, rank, local_rank, world_size
    else:
        return False, 0, 0, 1

def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def test(test_dataset, model, criterion, args):
    mean_MSE = 0
    avg_bpp = 0
    mean_loss = 0
    mean_recon_loss = 0
    mean_bpp_loss = 0
    device = next(model.parameters()).device
    mse_func = nn.MSELoss()
    
    with torch.no_grad():
        for idx, data in enumerate(test_dataset):
            data = {key: tensor.unsqueeze(0).to(device) for key, tensor in data.items()}
            
            out_net = model(data)
            out_loss = criterion(data= data, output = out_net)
            
            mean_loss += out_loss['loss'].item()
            mean_recon_loss += out_loss['recon_loss'].item()
            mean_bpp_loss += out_loss['bpp_loss'].item()
            
            num_pixels = data['weight_block'].numel()
            
            if 'ltc' in args.architecture.lower():
                out_dec  = model.forward_eval(data)
                bpp = torch.log(out_dec["likelihoods"]).sum().item() / (-math.log(2) * num_pixels)
            else :
                out_enc = model.compress(data)
                out_dec = model.decompress(out_enc)
                bpp = 0
                for s in out_enc["strings"]:
                    bpp += len(s[0]) * 8.0 / num_pixels

            x_hat = out_dec["x_hat"].clone().detach()
            if 'scale_cond' in out_dec.keys():
                x_hat = x_hat * out_dec["scale_cond"]

            mean_MSE += mse_func(data['weight_block'], x_hat).item()
            avg_bpp += bpp

    avg_bpp /= len(test_dataset)
    mean_MSE /= len(test_dataset)
    mean_loss /= len(test_dataset)
    mean_recon_loss /= len(test_dataset)
    mean_bpp_loss /= len(test_dataset)
    return {'TEST MSE': mean_MSE, 'TEST BPP': avg_bpp, 'TEST loss': mean_loss, 'TEST recon_loss': mean_recon_loss, 'TEST bpp_loss': mean_bpp_loss}


# === main 수정 ===
def main(args):
    # ---- DDP 세팅 ----
    is_dist, rank, local_rank, world_size = setup_distributed(args.dist_backend)
    is_master = (rank == 0)

    # wandb는 rank0만 활성화
    run_name = '_'.join(filter(None, [args.run_name, args.architecture, f"{args.lmbda}"]))
    if is_master:
        wandb.init(project="NWC_train", name=run_name)
        wandb.config.update(args)
    else:
        wandb.init(project="NWC_train", name=run_name, mode="disabled")

    # 로그 핸들러
    logger = args.logger
    gpu_num = world_size  # 전체 GPU 수
    device_id = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    # 시드/디터미니즘(프로세스별 동일하게 유지, 셔플은 Sampler가 관리)
    seed = args.seed
    random.seed(seed)
    np.random.seed(int(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    if is_master:
        logger.info("Create experiment save folder")

    csv_file = pd.DataFrame(columns=["iter", "TEST BPP", "TEST MSE"])

    # ---- 데이터 ----
    train_dataset, test_dataset, train_std, test_std = get_datasets(args)

    # DDP에서 sampler 사용
    if is_dist:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        shuffle_flag = False
    else:
        train_sampler = None
        shuffle_flag = True

    # 글로벌 배치 2048을 맞추기 위한 accumulation 설정
    # - args.batch_size는 "GPU당" (local) 배치로 해석
    local_bs = args.batch_size
    accum_steps = 1
    if args.global_batch_size:
        effective_per_step = local_bs * world_size
        accum_steps = max(1, math.ceil(args.global_batch_size / effective_per_step))
        if is_master:
            logger.info(f"[DDP] world_size={world_size}, local_bs={local_bs}, global_target={args.global_batch_size} -> accum_steps={accum_steps}")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_bs,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=shuffle_flag,
        sampler=train_sampler,
        drop_last=True,
    )

    # ---- 모델/옵티마 ----
    device = device_id
    scale = train_dataset.std.to(device)
    shift = train_dataset.mean.to(device)

    net = get_model(args.architecture, args, scale=scale, shift=shift)
    net = net.to(device)

    # DataParallel 제거하고 DDP로 래핑
    if is_dist and gpu_num > 1:
        net = DDP(net, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    criterion = get_loss_fn(args, std=train_std, device=device)

    if is_master:
        logger.info("Training mode : scratch!")
        logger.info(f"local_batch_size : {local_bs}  (accum_steps={accum_steps}, world_size={world_size})")
        if args.global_batch_size:
            logger.info(f"==> Effective global batch per optimizer step ≈ {local_bs * world_size * accum_steps}")
        logger.info(f"num of gpus: {gpu_num}")
        logger.info(args)

    best_mse = float("inf")
    best_bpp = float("inf")
    best_loss = float("inf")
    total_iter = 0  # "optimizer step" 기준으로 world_size를 곱해 증가

    recent_saved_model_path = ""
    best_mse_model_path = ""
    best_bpp_model_path = ""
    best_loss_model_path = ""

    # ---- 프리트레인 로드 (rank0만 로그 출력) ----
    if args.pretrained_path is not None:
        if is_master:
            logger.info(f"========= From pretrain path =========")
            logger.info(f"PRETRAINED PATH : {args.pretrained_path}")
        pt_checkpoint = torch.load(args.pretrained_path, map_location="cpu")

        if 'scale' in pt_checkpoint["state_dict"]:
            del pt_checkpoint["state_dict"]['scale']
        if 'shift' in pt_checkpoint["state_dict"]:
            del pt_checkpoint["state_dict"]['shift']

        # DDP 래핑 유무 대응
        try:
            net.load_state_dict(pt_checkpoint["state_dict"], strict=False)
        except:
            new_state_dict = {k.replace("module.", ""): v for k, v in pt_checkpoint["state_dict"].items()}
            net.load_state_dict(new_state_dict, strict=False)

        # 런타임 scale/shift 갱신
        try:
            net.module.scale = scale
            net.module.shift = shift
        except AttributeError:
            net.scale = scale
            net.shift = shift

        if is_master:
            logger.info(f"===== Loaded state dict =====")

    # ---- 체크포인트 로드 ----
    checkpoint = args.checkpoint
    if checkpoint != "None":
        if is_master:
            logger.info("===== RESUME TRAINING =====")
            logger.info(f"Loading {checkpoint}")
        checkpoint = torch.load(checkpoint, map_location="cpu")
        total_iter = checkpoint.get("total_iter", 0)

        try:
            try:
                net.load_state_dict(checkpoint["state_dict"])
            except:
                new_state_dict = {}
                for k, v in checkpoint["state_dict"].items():
                    new_state_dict[k.replace("module.", "")] = v
                net.load_state_dict(new_state_dict)
        except:
            try:
                net.module.load_state_dict(checkpoint["state_dict"])
            except:
                new_state_dict = {}
                for k, v in checkpoint["state_dict"].items():
                    new_state_dict[k.replace("module.", "")] = v
                net.module.load_state_dict(new_state_dict)

        # optimizers
        optimizer.load_state_dict(checkpoint["optimizer"])
        if aux_optimizer is not None and checkpoint.get("aux_optimizer") is not None:
            aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])

        criterion.load_state_dict(checkpoint["criterion"])

        best_mse = checkpoint.get("best_mse", best_mse)
        best_bpp = checkpoint.get("best_bpp", best_bpp)
        best_loss = checkpoint.get("best_loss", best_loss)

        recent_saved_model_path = checkpoint.get("recent_saved_model_path", recent_saved_model_path)
        best_mse_model_path = checkpoint.get("best_mse_model_path", best_mse_model_path)
        best_bpp_model_path = checkpoint.get("best_bpp_model_path", best_bpp_model_path)
        best_loss_model_path = checkpoint.get("best_loss_model_path", best_loss_model_path)

        del checkpoint

    save_path = args.save_path

    optimizer.zero_grad(set_to_none=True)
    if aux_optimizer is not None:
        aux_optimizer.zero_grad(set_to_none=True)

    # 에폭 개념 없이 반복 → Sampler 셔플 위해 epoch 카운터 유지
    epoch = 0

    while True:
        if total_iter >= args.iter:
            break

        net.train()
        if is_dist and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        epoch += 1

        if hasattr(train_dataset, "shuffle_hesseigen"):
            train_dataset.shuffle_hesseigen()

        microstep = 0  # accumulation 마이크로스텝 카운트

        for i, data in enumerate(train_dataloader):
            if total_iter >= args.iter:
                break

            data = {key: tensor.to(device, non_blocking=True) for key, tensor in data.items()}

            # forward & loss
            out_net = net(data)
            out_loss_dict = criterion(data=data, output=out_net)
            loss = out_loss_dict['loss'] / accum_steps
            loss.backward()
            microstep += 1

            if microstep % accum_steps == 0:
                # gradient clip
                if args.clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip_max_norm)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                # aux loss는 step단위로 1회만
                try:
                    aux_loss_t = net.module.aux_loss()
                except AttributeError:
                    aux_loss_t = net.aux_loss()

                if aux_optimizer is not None:
                    aux_loss_t.backward()
                    aux_optimizer.step()
                    aux_optimizer.zero_grad(set_to_none=True)

                # global iteration 증가 (기존 코드와 동일하게 * gpu_num)
                # total_iter += 1 * gpu_num
                # 글로벌 배치(optimizer step) 1회당 total_iter를 1 증가
                total_iter += 1

                # if (total_iter % 100 == 0 or total_iter == (1 * gpu_num)) and is_master:
                if (total_iter % 100 == 0 or total_iter == (1)) and is_master:
                    logger.info(
                        f"Train iter. {total_iter}/{args.iter} ({100. * total_iter / args.iter:.2f}%): "
                        f"\tLoss: {out_loss_dict['loss'].item():.6f}"
                        f"\trecon_loss: {out_loss_dict['recon_loss'].item():.6f}"
                        f"\tbpp_loss: {out_loss_dict['bpp_loss'].item():.6f}"
                        f"\taux_loss: {aux_loss_t.item() if torch.is_tensor(aux_loss_t) else float(aux_loss_t):.6f}"
                    )
                    wandb.log(out_loss_dict)

                # ---- 주기적 평가/저장 (rank0만) ----
                trigger = (total_iter % 2000) == 0
                # if (trigger or total_iter == (1 * gpu_num)) and is_master:
                if (trigger or total_iter == 1) and is_master:
                    torch.cuda.empty_cache()
                    with torch.no_grad():
                        net_eval = get_model(args.architecture, args, scale=scale, shift=shift)
                        try:
                            net_eval.load_state_dict(net.module.state_dict())
                        except AttributeError:
                            net_eval.load_state_dict(net.state_dict())
                        net_eval = net_eval.eval().to(device)
                        net_eval.requires_grad_(False)
                        # compressai module들 업데이트 필요 시
                        try:
                            net_eval.update()
                        except Exception:
                            pass

                        test_result = test(test_dataset, net_eval, criterion, args)
                        test_result['TEST MSE'] = test_result['TEST MSE'] / (train_std**2)
                        logger.info(test_result)
                        wandb.log(test_result)

                    # 체크포인트 저장
                    try:
                        state_dict = net.module.state_dict()
                    except AttributeError:
                        state_dict = net.state_dict()

                    def save_ckpt(path):
                        torch.save(
                            {
                                "state_dict": state_dict,
                                "optimizer": optimizer.state_dict(),
                                "aux_optimizer": aux_optimizer.state_dict() if aux_optimizer is not None else None,
                                "criterion": criterion.state_dict(),
                                "best_mse": best_mse,
                                "best_bpp": best_bpp,
                                "best_loss": best_loss,
                                "recent_saved_model_path": recent_saved_model_path,
                                "best_mse_model_path": best_mse_model_path,
                                "best_bpp_model_path": best_bpp_model_path,
                                "best_loss_model_path": best_loss_model_path,
                                "total_iter": total_iter,
                            },
                            path,
                        )

                    # best 갱신 로직 동일
                    if test_result['TEST MSE'] < best_mse:
                        best_mse = test_result['TEST MSE']
                        try: os.remove(best_mse_model_path)
                        except: logger.info("can not find prev_mse_best_model!")
                        best_mse_model_path = os.path.join(save_path, f"best_mse_model_loss_{round(test_result['TEST loss'], 5)}_bpp_{round(test_result['TEST BPP'], 5)}_MSE_{round(test_result['TEST MSE'], 5)}_total_iter_{total_iter}.pth.tar")
                        save_ckpt(best_mse_model_path)

                    if test_result['TEST BPP'] < best_bpp:
                        best_bpp = test_result['TEST BPP']
                        try: os.remove(best_bpp_model_path)
                        except: logger.info("can not find prev_bpp_best_model!")
                        best_bpp_model_path = os.path.join(save_path, f"best_bpp_model_loss_{round(test_result['TEST loss'], 5)}_bpp_{round(test_result['TEST BPP'], 5)}_MSE_{round(test_result['TEST MSE'], 5)}_total_iter_{total_iter}.pth.tar")
                        save_ckpt(best_bpp_model_path)

                    if test_result['TEST loss'] < best_loss:
                        best_loss = test_result['TEST loss']
                        try: os.remove(best_loss_model_path)
                        except: logger.info("can not find prev_bpp_best_model!")
                        best_loss_model_path = os.path.join(save_path, f"best_loss_model_loss_{round(test_result['TEST loss'], 5)}_bpp_{round(test_result['TEST BPP'], 5)}_MSE_{round(test_result['TEST MSE'], 5)}_total_iter_{total_iter}.pth.tar")
                        save_ckpt(best_loss_model_path)

                    try: os.remove(recent_saved_model_path)
                    except: logger.info("can not find recent_saved_model!")
                    recent_saved_model_path = os.path.join(save_path, f"recent_model_loss_{round(test_result['TEST loss'], 5)}_bpp_{round(test_result['TEST BPP'], 5)}_MSE_{round(test_result['TEST MSE'], 5)}_total_iter_{total_iter}.pth.tar")
                    save_ckpt(recent_saved_model_path)

                    new_row = {"iter": total_iter, "TEST loss": test_result['TEST loss'], "Best Loss": best_loss, "TEST BPP": test_result['TEST BPP'], "Best BPP": best_bpp, "TEST MSE": test_result['TEST MSE'], "Best MSE": best_mse}
                    csv_file = pd.concat([csv_file, pd.DataFrame([new_row])], ignore_index=True)
                    csv_file.to_csv(os.path.join(save_path, "res.csv"), index=False)

                    del net_eval
                    torch.cuda.empty_cache()

    if is_master:
        wandb.finish()
    cleanup_distributed()

def before_main(argvs):
    args = parse_args(argvs)

    # DDP rank 확인 (torchrun 실행 시 환경변수 존재)
    rank = int(os.environ.get("RANK", "0"))
    is_master = (rank == 0)

    checkpoint = "None"
    save_path = "./"

    if args.uniform_scale_max is not None:
        args.dataset += f'{args.uniform_scale_max}'

    folder_name = '_'.join([
        args.dataset,
        args.dataset_stat_type,
        '__'.join(args.dataset_path.split('/')[-2:])
    ])

    subfolder = '_'.join(filter(None, [
        args.run_name,
        args.loss,
        f"size{args.input_size}",
        f"encdim{args.dim_encoder}",
        f"M{args.M}",
        f"Q{args.Q}",
        f"R{args.R}",
        f"m{args.m}",
        f"batch_size{args.batch_size}",
        f"total_iter{args.iter}",
        f"lr{args.learning_rate}",
        f"seed{args.seed}"
    ]))

    if args.lmbda is not None:
        folder_name = os.path.join(folder_name, subfolder, f"lmbda{args.lmbda}_")
    else:
        folder_name = os.path.join(folder_name, subfolder, f"ld_min{args.lmbda_min}_max{args.lmbda_max}_")

    if args.seed is not None:
        save_path = os.path.join("./checkpoint", args.save_dir, args.architecture, folder_name)

        if is_master:
            if os.path.exists(save_path):
                logger = logger_setup(log_file_name="logs", log_file_folder_name=save_path)
                logger.info("find checkpoint...")

                file_list = os.listdir(save_path)
                for file_name in file_list:
                    if ("recent_model" in file_name) and (".pth.tar" in file_name):
                        logger.info(f"checkpoint exist, name: {file_name}")
                        checkpoint = f"{save_path}/{file_name}"
                if checkpoint == "None":
                    logger.info("no checkpoint is here")
            else:
                create_exp_folder(save_path)
                logger = logger_setup(log_file_name="logs", log_file_folder_name=save_path)
                logger.info("Create new exp folder!")
            logger.info(f"seed : {args.seed}")
            logger.info(f"exp name : {folder_name}")
        else:
            # rank0에서 만든 로그 핸들에 접근 위해 더미 로거 생성
            logger = logger_setup(log_file_name=None, log_file_folder_name=None)

    setattr(args, "checkpoint", checkpoint)
    setattr(args, "save_path", save_path)
    setattr(args, "dev.num_gpus", torch.cuda.device_count())

    # rank0만 config.json 저장
    if is_master:
        args_dict = vars(args)
        with open(os.path.join(save_path, "config.json"), "w") as f:
            json.dump(args_dict, f, indent=4)

    # if is_master:
    #     args_dict = {k: v for k, v in vars(args).items() if k != "logger"}
    #     with open(os.path.join(save_path, "config.json"), "w") as f:
    #         json.dump(args_dict, f, indent=4)
            
    setattr(args, "logger", logger)

    main(args)
        
if __name__ == "__main__":

    before_main(sys.argv[1:])
