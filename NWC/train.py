# CUDA_VISIBLE_DEVICES=0,1,2,3 taskset -c 0-31 nohup python -u train.py --lmbda 1 --iter 2000000 --u-length 4 --batch-size 8 --seed 100 --dist_port 6044 > ./training_logs/log_lmbda_1.txt 2>&1 &

# 노헙 안돌리는거
# CUDA_VISIBLE_DEVICES=2 taskset -c 14-23 python -u train.py --lmbda 1 --iter 2000000 --u-length 4 --batch-size 8 --seed 100 --dist_port 6044 --slurm
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
import torch.distributed as dist
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
    parser.add_argument("--dist_port", type=int, default=6006, required=True)
    parser.add_argument("--iter", default=2000000, type=int)
    # parser.add_argument("--dataset_path", type=str, default='../Wparam_dataset/dataset_block/meta-llama/Meta-Llama-3-8B/mlp_attn_16_row_dataset.pt')
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("-lr", "--learning_rate", default=1e-4, type=float)
    parser.add_argument("-n", "--num_workers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=float, default=100)
    parser.add_argument("--input_size", type=int, default=16)
    parser.add_argument("--dim_encoder", type=int, default=32)
    parser.add_argument("--P", type=int, default=4)
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--n_embeddings", type=int, default=None)
    parser.add_argument("--dim_embeddings", type=int, default=16)
    parser.add_argument("--n_resblock", type=int, default=4)
    parser.add_argument("--vq_beta", type=float, default=0.25)
    parser.add_argument("--clip_max_norm", default=1.0, type=float)
    parser.add_argument("--save_dir", default="vqvae_calib", type=str)
    parser.add_argument("--architecture", default=None, type=str)
    parser.add_argument("--loss", default="smse", type=str)
    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--block_direction", default='row', type=str)
    # parser.add_argument("--vector", action='store_true')
    
    args = parser.parse_args(argv)
    return args

def test(test_dataset, model, criterion):
    mean_MSE = 0
    mean_loss = 0
    mean_recon_loss = 0
    mean_embedding_loss = 0
    
    device = next(model.parameters()).device
    mse_func = nn.MSELoss()
    
    for idx, data in enumerate(test_dataset):
        data = {key: tensor.unsqueeze(0).to(device) for key, tensor in data.items()}
        out_net = model(data)
        out_loss = criterion(data= data, output = out_net)
        
        mean_loss += out_loss['loss'].item()
        mean_recon_loss += out_loss['recon_loss'].item()
        mean_embedding_loss += out_loss['embedding_loss'].item()
        x_hat = out_net["x_hat"].clone().detach()
        mean_MSE += mse_func(data['weight_block'], x_hat).item()

    mean_MSE /= len(test_dataset)
    mean_loss /= len(test_dataset)
    mean_recon_loss /= len(test_dataset)
    mean_embedding_loss /= len(test_dataset)
    
    return {'TEST MSE': mean_MSE, 'TEST loss': mean_loss, 'TEST recon_loss': mean_recon_loss, 'TEST embedding_loss': mean_embedding_loss}

def main(opts):
    wandb.init(project="NWC_VQVAE", name=opts.save_dir)
    wandb.config.update(opts)

    gpu_num = getattr(opts, "dev.num_gpus")

    if gpu_num > 1:
        node_rank = getattr(opts, "ddp.rank", 0)
        device_id = getattr(opts, "dev.device_id", torch.device("cpu"))
        logger = logger_setup(log_file_name="logs", log_file_folder_name=opts.save_path)

    else:
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

    csv_file = pd.DataFrame(columns=["iter", "TEST Loss", "TEST MSE"])
    
    train_dataset, test_dataset, train_std, test_std  = get_datasets(opts)
    
    device = device_id

    if gpu_num > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        batch_sampler_train = torch.utils.data.BatchSampler(train_sampler, opts.batch_size, drop_last=True)

        train_dataloader = DataLoader(
            train_dataset, batch_sampler=batch_sampler_train, num_workers=opts.num_workers, pin_memory=True
        )

    else:
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
        net = torch.nn.parallel.DistributedDataParallel(
            net,
            device_ids=[device_id],
            output_device=device_id,
            find_unused_parameters=False,
        )

    optimizer = optim.Adam(net.parameters(), lr=opts.learning_rate)
    
    criterion = get_loss_fn(opts, std=train_std)

    if node_rank == 0:
        logger.info("Training mode : scratch!")
        logger.info(f"batch_size : {opts.batch_size}")
        logger.info(f"num of gpus: {gpu_num}")
        logger.info(opts)

    ########################################

    best_mse = float("inf")
    best_loss = float("inf")
    total_iter = 0

    recent_saved_model_path = ""
    best_mse_model_path = ""
    best_loss_model_path = ""

    checkpoint = opts.checkpoint
    if checkpoint != "None":  # load from previous checkpoint

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

        criterion.load_state_dict(checkpoint["criterion"])

        best_mse = checkpoint["best_mse"]
        best_loss = checkpoint["best_loss"]

        recent_saved_model_path = checkpoint["recent_saved_model_path"]
        best_mse_model_path = checkpoint["best_mse_model_path"]
        best_loss_model_path = checkpoint["best_loss_model_path"]

        del checkpoint

    save_path = opts.save_path

    if gpu_num > 1:
        dist.barrier()

    optimizer.zero_grad()

    while 1:
        net.train()
        device = next(net.parameters()).device

        if total_iter >= opts.iter:
            break

        for i, (data) in enumerate(train_dataloader):
            if total_iter >= opts.iter: break
            
            with torch.autograd.detect_anomaly():
                # weight = data['weight_block']
                # weight = weight.to(device)
                # tensor_block_idx = data['tensor_block_idx']
                # tensor_block_idx = tensor_block_idx.to(device)
                
                ## 더 최적화하는 방법?
                data = {key: tensor.to(device) for key, tensor in data.items()}
                
                optimizer.zero_grad()
                out_net = net(data)

                if gpu_num > 1:
                    dist.barrier()

                out_loss = criterion(data= data, output = out_net)
                out_loss['loss'].backward()

                if opts.clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), opts.clip_max_norm)

                optimizer.step()
            optimizer.zero_grad()
            total_iter += 1 * gpu_num
            if ((total_iter % 1000 == 0) or (total_iter == (1 * gpu_num))) and node_rank == 0:
                logger.info(
                    f"Train iter. {total_iter}/{opts.iter} ({100. * total_iter / opts.iter}%): "
                    f"\tLoss: {out_loss['loss'].item()}"
                    f"\trecon_loss: {out_loss['recon_loss'].item()}"
                    f"\tembedding_loss: {out_loss['embedding_loss'].item()}"
                    f'\tbpp: {out_net["min_encoding_indices"].numel()/opts.batch_size/opts.input_size * opts.K}'
                )
                wandb.log(out_loss)
            if gpu_num > 1:
                dist.barrier()

            # 몇 번마다 성능 측정할까? 만 번마다 해보자 + 맨 첨에도 해보자. 궁금하니까.
            if (total_iter % 10000 == 0) or (total_iter == (1 * gpu_num)):
                torch.cuda.empty_cache()

                if gpu_num > 1:
                    dist.barrier()

                with torch.no_grad():
                    net_eval = get_model(opts.architecture, opts, scale=scale, shift=shift)        

                    try:
                        net_eval.load_state_dict(net.module.state_dict())
                    except:
                        net_eval.load_state_dict(net.state_dict())

                    net_eval = net_eval.eval().to(device)
                    net_eval.requires_grad_(False)

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
                            save_path + "/" + f"best_mse_model_Loss_{round(test_result['TEST loss'], 5)}_MSE_{round(test_result['TEST MSE'], 5)}_total_iter_{total_iter}.pth.tar"
                        )
                        torch.save(
                            {
                                "state_dict": state_dict,
                                "optimizer": optimizer.state_dict(),
                                "criterion": criterion.state_dict(),
                                "best_mse": best_mse,
                                "best_loss": best_loss,
                                "recent_saved_model_path": recent_saved_model_path,
                                "best_mse_model_path": best_mse_model_path,
                                "best_loss_model_path": best_loss_model_path,
                                "total_iter": total_iter,
                            },
                            best_mse_model_path,
                        )

                    if test_result['TEST loss'] < best_loss:
                        best_loss = test_result['TEST loss']
                        try:
                            os.remove(best_loss_model_path)  # 이전에 최고였던 모델 삭제
                        except:
                            logger.info("can not find prev_loss_best_model!")
                        best_loss_model_path = (
                            save_path + "/" + f"best_loss_model_Loss_{round(best_loss, 5)}_MSE_{round(test_result['TEST MSE'], 5)}_total_iter_{total_iter}.pth.tar"
                        )
                        torch.save(
                            {
                                "state_dict": state_dict,
                                "optimizer": optimizer.state_dict(),
                                "criterion": criterion.state_dict(),
                                "best_mse": best_mse,
                                "best_loss": best_loss,
                                "recent_saved_model_path": recent_saved_model_path,
                                "best_mse_model_path": best_mse_model_path,
                                "best_loss_model_path": best_loss_model_path,
                                "total_iter": total_iter,
                            },
                            best_loss_model_path,
                        )
                    try:
                        os.remove(recent_saved_model_path)  # 이전 에폭에서 저장했던 모델 삭제
                    except:
                        logger.info("can not find recent_saved_model!")

                    recent_saved_model_path = (
                        save_path + "/" + f"recent_model_Loss_{round(test_result['TEST loss'], 5)}_MSE_{round(test_result['TEST MSE'], 5)}_total_iter_{total_iter}.pth.tar"
                    )
                    torch.save(
                        {
                            "state_dict": state_dict,
                            "optimizer": optimizer.state_dict(),
                            "criterion": criterion.state_dict(),
                            "best_mse": best_mse,
                            "best_loss": best_loss,
                            "recent_saved_model_path": recent_saved_model_path,
                            "best_mse_model_path": best_mse_model_path,
                            "best_loss_model_path": best_loss_model_path,
                            "total_iter": total_iter,
                        },
                        recent_saved_model_path,
                    )                    
                    
                    new_row = {"iter": total_iter, "TEST loss": test_result['TEST loss'], "Best Loss": best_loss, "TEST MSE": test_result['TEST MSE'], "Best MSE": best_mse}
                    csv_file = pd.concat([csv_file, pd.DataFrame([new_row])], ignore_index=True)
                    csv_file.to_csv(save_path + "/" + "res.csv", index=False)
                    
                del net_eval

                torch.cuda.empty_cache()
            if gpu_num > 1:
                dist.barrier()

    wandb.finish()


def distributed_init(opts) -> int:
    ddp_url = getattr(opts, "ddp.dist_url", None)

    node_rank = getattr(opts, "ddp.rank", 0)
    is_master_node = node_rank == 0

    if ddp_url is None:
        # 따로 지정한게 없어서 무조건 이쪽으로 들어옴
        ddp_port = opts.dist_port
        hostname = socket.gethostname()
        ddp_url = "tcp://{}:{}".format(hostname, ddp_port)
        setattr(opts, "ddp.dist_url", ddp_url)

    node_rank = getattr(opts, "ddp.rank", 0)
    world_size = getattr(opts, "ddp.world_size", 0)

    if torch.distributed.is_initialized():
        print("DDP is already initialized and cannot be initialize twice!")
    else:
        print("distributed init (rank {}): {}".format(node_rank, ddp_url))

        dist_backend = getattr(opts, "ddp.backend", "nccl")  # "gloo"

        if dist_backend is None and dist.is_nccl_available():
            dist_backend = "nccl"
            if is_master_node:
                print("Using NCCL as distributed backend with version={}".format(torch.cuda.nccl.version()))
        elif dist_backend is None:
            dist_backend = "gloo"

        dist.init_process_group(
            backend=dist_backend,
            init_method=ddp_url,
            world_size=world_size,
            rank=node_rank,
        )

        # perform a dummy all-reduce to initialize the NCCL communicator
        if torch.cuda.is_available():
            dist.all_reduce(torch.zeros(1).cuda())

    node_rank = torch.distributed.get_rank()
    setattr(opts, "ddp.rank", node_rank)
    return node_rank


def distributed_worker(i, main, opts):
    setattr(opts, "dev.device_id", i)
    torch.cuda.set_device(i)
    setattr(opts, "dev.device", torch.device(f"cuda:{i}"))

    ddp_rank = i
    setattr(opts, "ddp.rank", ddp_rank)

    node_rank = distributed_init(opts)
    setattr(opts, "ddp.rank", node_rank)
    main(opts)


def ddp_or_single_process(argvs):

    opts = parse_args(argvs)
    checkpoint = "None"
    save_path = "./"
    
    if opts.K is not None:
        opts.n_embeddings = 2**opts.K
    
    opts.bpp = opts.P * opts.K / opts.input_size
    # opts.bpp = opts.P * math.log2(opts.n_embeddings) / opts.input_size
    
    folder_name = f"{opts.dataset}_{opts.block_direction}_{opts.input_size}"
    folder_name += f"/bpp{opts.bpp}_{opts.loss}_ne{opts.n_embeddings}_de{opts.dim_embeddings}_K{opts.K}_P{opts.P}_encdim{opts.dim_encoder}"
    folder_name += f"_batch_size{opts.batch_size}_total_iter{opts.iter}_lr{opts.learning_rate}_seed{opts.seed}"

    if opts.seed is not None:
        save_path = f"./checkpoint/{opts.save_dir}/{folder_name}"

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

    if torch.cuda.device_count() > 1:
        setattr(opts, "ddp.world_size", torch.cuda.device_count())

        logger.info(f"opts: {opts}")

        torch.multiprocessing.spawn(
            fn=distributed_worker,
            args=(main, opts),
            nprocs=getattr(opts, "dev.num_gpus"),
        )
    else:
        setattr(opts, "logger", logger)
        main(opts)


if __name__ == "__main__":

    ddp_or_single_process(sys.argv[1:])
