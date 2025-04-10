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
    parser.add_argument("--dataset_stat_type", type=str, choices=['scaler', 'channel'], default='scaler')
    # parser.add_argument("--vector", action='store_true')
    
    args = parser.parse_args(argv)
    return args

def test(test_dataset, model, criterion):
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
            
            out_enc = model.compress(data)
            out_dec = model.decompress(out_enc)
            
            # try:
            #     out_dec = model.decompress(out_enc["strings"], out_enc["shape"], data["q_level"])
            # except:
            #     out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
            
            # out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
                

            num_pixels = data['weight_block'].numel()
            
            bpp = 0
            for s in out_enc["strings"]:
                bpp += len(s[0]) * 8.0 / num_pixels

            x_hat = out_dec["x_hat"].clone().detach()
            mean_MSE += mse_func(data['weight_block'], x_hat).item()
            avg_bpp += bpp

    avg_bpp /= len(test_dataset)
    mean_MSE /= len(test_dataset)
    mean_loss /= len(test_dataset)
    mean_recon_loss /= len(test_dataset)
    mean_bpp_loss /= len(test_dataset)
    return {'TEST MSE': mean_MSE, 'TEST BPP': avg_bpp, 'TEST loss': mean_loss, 'TEST recon_loss': mean_recon_loss, 'TEST bpp_loss': mean_bpp_loss}

def main(opts):
    wandb.init(project="NWC_VQVAE", name=opts.architecture)
    wandb.config.update(opts)

    gpu_num = getattr(opts, "dev.num_gpus")

    node_rank = 0
    device_id = "cuda"
    logger = opts.logger
        

    seed = opts.seed

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
    
    train_dataset, test_dataset, train_std, test_std  = get_datasets(opts)
    
    device = device_id

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=opts.batch_size,
        num_workers=opts.num_workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )
        
    ## 함수로 따로 만들기
    scale = train_dataset.std.to(device)
    shift = train_dataset.mean.to(device)
    
    net = get_model(opts.architecture, opts, scale=scale, shift=shift)        
    # wandb.config.update(opts, allow_val_change=True)

    net = net.to(device)

    if gpu_num > 1:
        net = nn.DataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, opts)
    
    criterion = get_loss_fn(opts, std=train_std, device = device)

    if node_rank == 0:
        logger.info("Training mode : scratch!")
        logger.info(f"batch_size : {opts.batch_size}")
        logger.info(f"num of gpus: {gpu_num}")
        logger.info(opts)

    ########################################

    best_mse = float("inf")
    best_bpp = float("inf")
    best_loss = float("inf")
    total_iter = 0

    recent_saved_model_path = ""
    best_mse_model_path = ""
    best_bpp_model_path = ""
    best_loss_model_path = ""

    checkpoint = opts.checkpoint
    if checkpoint != "None":  # load from previous checkpoint
        print("##### RESUME TRAINING #####")
        if node_rank == 0:
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

    save_path = opts.save_path

    optimizer.zero_grad()
    aux_optimizer.zero_grad()

    while 1:
        net.train()
        device = next(net.parameters()).device

        if total_iter >= opts.iter:
            break
        
        if hasattr(train_dataset, "shuffle_hesseigen"):
            train_dataset.shuffle_hesseigen()

        for i, (data) in enumerate(train_dataloader):
            if total_iter >= opts.iter: break
            
            with torch.autograd.detect_anomaly():                
                data = {key: tensor.to(device) for key, tensor in data.items()}
                optimizer.zero_grad()
                aux_optimizer.zero_grad()
                out_net = net(data)

                out_loss = criterion(data= data, output = out_net)
                out_loss['loss'].backward()

                if opts.clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), opts.clip_max_norm)

                optimizer.step()
                
                try:
                    aux_loss = net.aux_loss()
                except:
                    aux_loss = net.module.aux_loss()
                    
                aux_loss.backward()
                aux_optimizer.step()
                
            optimizer.zero_grad()
            aux_optimizer.zero_grad()
            
            total_iter += 1 * gpu_num
            if ((total_iter % 100 == 0) or (total_iter == (1 * gpu_num))) and node_rank == 0:
                logger.info(
                    f"Train iter. {total_iter}/{opts.iter} ({100. * total_iter / opts.iter}%): "
                    f"\tLoss: {out_loss['loss'].item()}"
                    f"\trecon_loss: {out_loss['recon_loss'].item()}"
                    f"\tbpp_loss: {out_loss['bpp_loss'].item()}"
                    f"\taux_loss: {aux_loss.item()}"
                )
                wandb.log(out_loss)

            # 몇 번마다 성능 측정할까? 만 번마다 해보자 + 맨 첨에도 해보자. 궁금하니까.
            if (total_iter % 5000 == 0) or (total_iter == (1 * gpu_num)):
                torch.cuda.empty_cache()

                with torch.no_grad():
                    net_eval = get_model(opts.architecture, opts, scale=scale, shift=shift)        

                    try:
                        net_eval.load_state_dict(net.module.state_dict())
                    except:
                        net_eval.load_state_dict(net.state_dict())

                    net_eval = net_eval.eval().to(device)
                    net_eval.requires_grad_(False)
                    net_eval.update()
                    
                    if node_rank == 0:
                        test_result = test(test_dataset, net_eval, criterion)
                        test_result['TEST MSE'] = test_result['TEST MSE'] / train_std**2
                        logger.info(test_result)
                        wandb.log(test_result)
                    else:
                        _ = test(test_dataset, net_eval, criterion)

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
                                "aux_optimizer": aux_optimizer.state_dict(),
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
                                "aux_optimizer": aux_optimizer.state_dict(),
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
                                "aux_optimizer": aux_optimizer.state_dict(),
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
                            "aux_optimizer": aux_optimizer.state_dict(),
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

    opts = parse_args(argvs)
    checkpoint = "None"
    save_path = "./"
    
    folder_name = f"{opts.dataset}_{opts.dataset_stat_type}_{'__'.join(opts.dataset_path.split('/')[-2:])}"
    folder_name += f"/lmbda{opts.lmbda}_{opts.loss}_size{opts.input_size}_encdim{opts.dim_encoder}_M{opts.M}_Q{opts.Q}_R{opts.R}_m{opts.m}"
    folder_name += f"_batch_size{opts.batch_size}_total_iter{opts.iter}_lr{opts.learning_rate}_seed{opts.seed}"

    if opts.seed is not None:
        save_path = os.path.join("./checkpoint", opts.save_dir, opts.architecture, folder_name)

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

        logger.info(f"seed : {opts.seed}")
        logger.info(f"exp name : {folder_name}")

    setattr(opts, "checkpoint", checkpoint)

    setattr(opts, "save_path", save_path)
    setattr(opts, "dev.num_gpus", torch.cuda.device_count())


    # args_dict = {arg.dest: getattr(opts, arg.dest, None) for arg in opts._parser._actions}
    args_dict = vars(opts)
    # 사전을 JSON으로 변환하여 저장
    with open(save_path + "/config.json", "w") as f:
        json.dump(args_dict, f, indent=4)
    
    setattr(opts, "logger", logger)
    main(opts)
        
if __name__ == "__main__":

    before_main(sys.argv[1:])
