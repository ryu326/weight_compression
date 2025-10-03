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
    parser.add_argument("--seed", type=float, default=100)
    parser.add_argument("--input_size", type=int, default=16)
    parser.add_argument("--dim_encoder", type=int, default=32)
    parser.add_argument("--n_resblock", type=int, default=4)
    parser.add_argument("--M", type=int, default=0)
    parser.add_argument("--N", type=int, default=0)
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
    parser.add_argument("--normalize", action='store_true', default=False)
    parser.add_argument("--learnable_s", action='store_true', default=False)
    parser.add_argument("--aug_scale", action='store_true', default=False)
    parser.add_argument("--aug_log_uniform", action='store_true', default=False)
    parser.add_argument("--aug_update_cond", action='store_true', default=False)
    parser.add_argument("--aug_scale_p", type=float, default=0.1)
    parser.add_argument("--aug_scale_max", type=float, default=2.0)
    parser.add_argument("--aug_scale_min", type=float, default=0.5)
    parser.add_argument("--aug_scale_mode", type=str, default='block')
    parser.add_argument("--ql_scale_cond", action='store_true', default=False)
    
    
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
            # if 'scale_cond' in out_dec.keys() and args.pre_normalize:
            #     x_hat = x_hat * out_dec["scale_cond"]

            mean_MSE += mse_func(data['weight_block'], x_hat).item()
            avg_bpp += bpp

    avg_bpp /= len(test_dataset)
    mean_MSE /= len(test_dataset)
    mean_loss /= len(test_dataset)
    mean_recon_loss /= len(test_dataset)
    mean_bpp_loss /= len(test_dataset)
    return {'TEST MSE': mean_MSE, 'TEST BPP': avg_bpp, 'TEST loss': mean_loss, 'TEST recon_loss': mean_recon_loss, 'TEST bpp_loss': mean_bpp_loss}

def main(args):
    run_name = '_'.join(filter(None, [args.run_name, args.architecture, f"{args.lmbda}"]))
    # wandb.init(project="NWC_train", name=run_name, mode = 'disabled')
    wandb.init(project="NWC_train", name=run_name)
    wandb.config.update(args)

    gpu_num = getattr(args, "dev.num_gpus")

    node_rank = 0
    device_id = "cuda"
    logger = args.logger
        

    seed = args.seed

    random.seed(seed)
    np.random.seed(int(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    if node_rank == 0:
        logger.info("Create experiment save folder")

    csv_file = pd.DataFrame(columns=["iter", "TEST BPP", "TEST MSE"])
    
    train_dataset, test_dataset, train_std, test_std  = get_datasets(args)
    
    device = device_id

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )
        
    scale = train_dataset.std.to(device)
    shift = train_dataset.mean.to(device)
    
    net = get_model(args.architecture, args, scale=scale, shift=shift)        
    # wandb.config.update(args, allow_val_change=True)

    net = net.to(device)

    if gpu_num > 1:
        net = nn.DataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    
    criterion = get_loss_fn(args, std=train_std, device = device)

    if node_rank == 0:
        logger.info("Training mode : scratch!")
        logger.info(f"batch_size : {args.batch_size}")
        logger.info(f"num of gpus: {gpu_num}")
        logger.info(args)

    ########################################

    best_mse = float("inf")
    best_bpp = float("inf")
    best_loss = float("inf")
    total_iter = 0

    recent_saved_model_path = ""
    best_mse_model_path = ""
    best_bpp_model_path = ""
    best_loss_model_path = ""
    
    if args.pretrained_path != None:
        if node_rank == 0:
            logger.info(f"========= From pretrain path =========")
            logger.info(f"PRETRAINED PATH : {args.pretrained_path}")
        pt_checkpoint = torch.load(args.pretrained_path)
        
        if 'scale' in pt_checkpoint["state_dict"]:
            del pt_checkpoint["state_dict"]['scale']
        if 'shift' in pt_checkpoint["state_dict"]:
            del pt_checkpoint["state_dict"]['shift']
        print('shift: ', shift, ' scale:', scale)

        net.load_state_dict(pt_checkpoint["state_dict"], strict = False)
        net.scale = scale
        net.shift = shift
        
        # try:
        #     try:
        #         net.load_state_dict(pt_checkpoint["state_dict"], strict=False)
        #     except:
        #         new_state_dict = {}
        #         for k, v in pt_checkpoint["state_dict"].items():
        #             new_state_dict[k.replace("module.", "")] = v
        #         net.load_state_dict(new_state_dict, strict=False)
        # except:
        #     try:
        #         net.module.load_state_dict(pt_checkpoint["state_dict"],strict=False)
        #     except:
        #         new_state_dict = {}
        #         for k, v in pt_checkpoint["state_dict"].items():
        #             new_state_dict[k.replace("module.", "")] = v
        #         net.module.load_state_dict(new_state_dict, strict=False)
                
        # net.scale = scale
        # net.shift = shift
        
        if node_rank == 0:
            logger.info(f"===== Loaded state dict =====")
        

    checkpoint = args.checkpoint
    if checkpoint != "None":  # load from previous checkpoint
        print()
        if node_rank == 0:
            logger.info("===== RESUME TRAINING =====")
            logger.info(f"Loading {checkpoint}")

        checkpoint = torch.load(checkpoint)
        total_iter = checkpoint["total_iter"]

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

        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])

        criterion.load_state_dict(checkpoint["criterion"])

        best_mse = checkpoint["best_mse"]
        best_bpp = checkpoint["best_bpp"]
        best_loss = checkpoint["best_loss"]

        recent_saved_model_path = checkpoint["recent_saved_model_path"]
        best_mse_model_path = checkpoint["best_mse_model_path"]
        best_bpp_model_path = checkpoint["best_bpp_model_path"]
        best_loss_model_path = checkpoint["best_loss_model_path"]

        del checkpoint

    save_path = args.save_path

    optimizer.zero_grad()
    if aux_optimizer is not None:
        aux_optimizer.zero_grad()

    while 1:
        net.train()
        device = next(net.parameters()).device

        if total_iter >= args.iter:
            break
        
        if hasattr(train_dataset, "shuffle_hesseigen"):
            train_dataset.shuffle_hesseigen()

        for i, (data) in enumerate(train_dataloader):
            if total_iter >= args.iter: break
            
            with torch.autograd.detect_anomaly():                
                data = {key: tensor.to(device) for key, tensor in data.items()}
                optimizer.zero_grad()
                if aux_optimizer is not None:
                    aux_optimizer.zero_grad()
                out_net = net(data)

                out_loss = criterion(data= data, output = out_net)
                out_loss['loss'].backward()

                if args.clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip_max_norm)

                optimizer.step()
                
                try:
                    aux_loss = net.aux_loss()
                except:
                    aux_loss = net.module.aux_loss()
                
                if aux_optimizer is not None:
                    aux_loss.backward()
                    aux_optimizer.step()
                
            optimizer.zero_grad()
            if aux_optimizer is not None:
                aux_optimizer.zero_grad()
            
            total_iter += 1 * gpu_num
            if ((total_iter % 100 == 0) or (total_iter == (1 * gpu_num))) and node_rank == 0:
                logger.info(
                    f"Train iter. {total_iter}/{args.iter} ({100. * total_iter / args.iter}%): "
                    f"\tLoss: {out_loss['loss'].item()}"
                    f"\trecon_loss: {out_loss['recon_loss'].item()}"
                    f"\tbpp_loss: {out_loss['bpp_loss'].item()}"
                    f"\taux_loss: {aux_loss.item()}"
                )
                wandb.log(out_loss)

            # 몇 번마다 성능 측정할까? 만 번마다 해보자 + 맨 첨에도 해보자. 궁금하니까.
            trigger = (total_iter % max(1, int(5000 / max(1, (args.batch_size // 1000))))) == 0
            if trigger or (total_iter == (1 * gpu_num)):
            # if (total_iter % ((5000 / (args.batch_size // 1000))) == 0) or (total_iter == (1 * gpu_num)):
                torch.cuda.empty_cache()

                with torch.no_grad():
                    net_eval = get_model(args.architecture, args, scale=scale, shift=shift)        

                    try:
                        net_eval.load_state_dict(net.module.state_dict())
                    except:
                        net_eval.load_state_dict(net.state_dict())

                    net_eval = net_eval.eval().to(device)
                    net_eval.requires_grad_(False)
                    net_eval.update()
                    
                    if node_rank == 0:
                        test_result = test(test_dataset, net_eval, criterion, args)
                        test_result['TEST MSE'] = test_result['TEST MSE'] / train_std**2
                        logger.info(test_result)
                        wandb.log(test_result)
                    else:
                        _ = test(test_dataset, net_eval, criterion, args)

                torch.cuda.empty_cache()
                # 모델 저장
                if node_rank == 0:
                    try:
                        state_dict = net.module.state_dict()
                    except:
                        state_dict = net.state_dict()

                    if test_result['TEST MSE'] < best_mse:
                        best_mse = test_result['TEST MSE']
                        try:
                            os.remove(best_mse_model_path)  # 이전에 최고였던 모델 삭제
                        except:
                            logger.info("can not find prev_mse_best_model!")
                        best_mse_model_path = (
                            save_path + "/" + f"best_mse_model_loss_{round(test_result['TEST loss'], 5)}_bpp_{round(test_result['TEST BPP'], 5)}_MSE_{round(test_result['TEST MSE'], 5)}_total_iter_{total_iter}.pth.tar"
                        )
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
                            best_mse_model_path,
                        )

                    if test_result['TEST BPP'] < best_bpp:
                        best_bpp = test_result['TEST BPP']
                        try:
                            os.remove(best_bpp_model_path)  # 이전에 최고였던 모델 삭제
                        except:
                            logger.info("can not find prev_bpp_best_model!")
                        best_bpp_model_path = (
                            save_path + "/" + f"best_bpp_model_loss_{round(test_result['TEST loss'], 5)}_bpp_{round(test_result['TEST BPP'], 5)}_MSE_{round(test_result['TEST MSE'], 5)}_total_iter_{total_iter}.pth.tar"
                        )
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
                                "best_loss_model_path": best_loss_model_path,
                                "best_bpp_model_path": best_bpp_model_path,
                                "total_iter": total_iter,
                            },
                            best_bpp_model_path,
                        )
                    
                    if test_result['TEST loss'] < best_loss:
                        best_loss = test_result['TEST loss']
                        try:
                            os.remove(best_loss_model_path)  # 이전에 최고였던 모델 삭제
                        except:
                            logger.info("can not find prev_bpp_best_model!")
                        best_loss_model_path = (
                            save_path + "/" + f"best_loss_model_loss_{round(test_result['TEST loss'], 5)}_bpp_{round(test_result['TEST BPP'], 5)}_MSE_{round(test_result['TEST MSE'], 5)}_total_iter_{total_iter}.pth.tar"
                        )
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
                                "best_loss_model_path": best_loss_model_path,
                                "best_bpp_model_path": best_bpp_model_path,
                                "total_iter": total_iter,
                            },
                            best_loss_model_path,
                        )
                        
                    try:
                        os.remove(recent_saved_model_path)  # 이전 에폭에서 저장했던 모델 삭제
                    except:
                        logger.info("can not find recent_saved_model!")

                    recent_saved_model_path = (
                        save_path + "/" + f"recent_model_loss_{round(test_result['TEST loss'], 5)}_bpp_{round(test_result['TEST BPP'], 5)}_MSE_{round(test_result['TEST MSE'], 5)}_total_iter_{total_iter}.pth.tar"
                    )
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
                        recent_saved_model_path,
                    )                    
                    
                    new_row = {"iter": total_iter, "TEST loss": test_result['TEST loss'], "Best Loss": best_loss, "TEST BPP": test_result['TEST BPP'], "Best BPP": best_bpp, "TEST MSE": test_result['TEST MSE'], "Best MSE": best_mse}
                    csv_file = pd.concat([csv_file, pd.DataFrame([new_row])], ignore_index=True)
                    csv_file.to_csv(save_path + "/" + "res.csv", index=False)
                    
                del net_eval

                torch.cuda.empty_cache()

    wandb.finish()

def before_main(argvs):

    args = parse_args(argvs)
    checkpoint = "None"
    save_path = "./"
    
    # folder_name = f"{args.dataset}_{args.dataset_stat_type}_{'__'.join(args.dataset_path.split('/')[-2:])}"
    # folder_name += f"/{args.run_name}{args.loss}_size{args.input_size}_encdim{args.dim_encoder}_M{args.M}_Q{args.Q}_R{args.R}_m{args.m}"
    # folder_name += f"_batch_size{args.batch_size}_total_iter{args.iter}_lr{args.learning_rate}_seed{args.seed}/lmbda{args.lmbda}"
    
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

    setattr(args, "checkpoint", checkpoint)

    setattr(args, "save_path", save_path)
    setattr(args, "dev.num_gpus", torch.cuda.device_count())


    # args_dict = {arg.dest: getattr(args, arg.dest, None) for arg in args._parser._actions}
    args_dict = vars(args)
    # 사전을 JSON으로 변환하여 저장
    with open(save_path + "/config.json", "w") as f:
        json.dump(args_dict, f, indent=4)
    
    setattr(args, "logger", logger)
    main(args)
        
if __name__ == "__main__":

    before_main(sys.argv[1:])
