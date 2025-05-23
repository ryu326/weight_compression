# CUDA_VISIBLE_DEVICES=0,1,2,3 taskset -c 0-31 nohup python -u train.py --lmbda 1 --iter 2000000 --u-length 4 --batch-size 8 --seed 100 --dist_port 6044 > ./training_logs/log_lmbda_1.txt 2>&1 &

# 노헙 안돌리는거
# CUDA_VISIBLE_DEVICES=2 taskset -c 14-23 python -u train.py --lmbda 1 --iter 2000000 --u-length 4 --batch-size 8 --seed 100 --dist_port 6044 --slurm
import operator
import os
import random
import shutil
import socket
import sys

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import wandb
from datasets_weight_matrix import get_datasets
from models import Learnable_SEEDLM
from torch.utils.data import DataLoader
from utils.optimizers import *
from utils.util import *

# 시간 측정해보기








def test(total_iter, test_dataset, model, save_path, logger, mse_func, node_rank=0):

    mean_MSE = 0

    device = next(model.parameters()).device

    for idx, weight in enumerate(test_dataset):

        weight = weight.to(device)
        x = weight.unsqueeze(0).to(device)  # [1, 512, 512] 같음

        out = model(x.to(device))

        x_hat = out["x_hat"].clone().detach()

        mse = mse_func(x, x_hat).item()

        mean_MSE += mse

        if node_rank == 0:

            logger.info(f"Test total_iter: {total_iter}, File name: {idx}, MSE: {mse}")

    mean_MSE /= len(test_dataset)

    if node_rank == 0:
        logger.info(f"Average_MSE: {mean_MSE}")

    return mean_MSE


def main(opts):
    wandb.init(project="NN-based SEEDLM", name="nn_seedlm")
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

    csv_file = pd.DataFrame(None, index=["Best_MSE"], columns=["MSE", "iter"])

    train_dataset, test_dataset = get_datasets(opts.dataset_path)

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

    net = Learnable_SEEDLM(
        input_dim=opts.data_dim, u_length=opts.u_length
    )  # 들어갈 변수 많으니 argument 받아서 잘 집어넣어주는 코드 짜보자 (hint: opt 이용)
    net = net.to(device)

    if gpu_num > 1:
        net = torch.nn.parallel.DistributedDataParallel(
            net,
            device_ids=[device_id],
            output_device=device_id,
            find_unused_parameters=False,
        )

    optimizer = optim.Adam(net.parameters(), lr=opts.learning_rate)
    criterion = nn.MSELoss()
    # criterion = RateDistortionLoss(lmbda=opts.lmbda)

    ########################################
    # test 준비
    mse_func = nn.MSELoss()

    if node_rank == 0:
        logger.info("Training mode : scratch!")
        logger.info(f"lmbda : {opts.lmbda}")

        logger.info(f"batch_size : {opts.batch_size}")

        logger.info(f"num of gpus: {gpu_num}")
    ########################################

    best_mse = float("inf")
    total_iter = 0

    recent_saved_model_path = ""
    best_mse_model_path = ""

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

        recent_saved_model_path = checkpoint["recent_saved_model_path"]
        best_mse_model_path = checkpoint["best_mse_model_path"]

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

        for i, (weight) in enumerate(train_dataloader):

            if total_iter >= opts.iter:
                break

            with torch.autograd.detect_anomaly():
                weight = weight.to(device)

                optimizer.zero_grad()

                out_net = net(weight)

                if gpu_num > 1:
                    dist.barrier()

                updating_time = {}

                out_criterion = criterion(out_net["x"], out_net["x_hat"])
                out_criterion.backward()

                if opts.clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), opts.clip_max_norm)

                optimizer.step()

            wandb.log({"train_loss": out_criterion.item()})
            optimizer.zero_grad()

            total_iter += 1 * gpu_num

            if ((total_iter % 5000 == 0) or (total_iter == (1 * gpu_num))) and node_rank == 0:

                logger.info(
                    f"Train iter. {total_iter}/{opts.iter} ({100. * total_iter / opts.iter}%): "
                    f"\tLoss: {out_criterion.item()}"
                )

            if gpu_num > 1:
                dist.barrier()

            # 몇 번마다 성능 측정할까? 만 번마다 해보자 + 맨 첨에도 해보자. 궁금하니까.
            if (total_iter % 50000 == 0) or (total_iter == (1 * gpu_num)):
                # if (total_iter % 50000 == 0):
                torch.cuda.empty_cache()

                if gpu_num > 1:
                    dist.barrier()

                with torch.no_grad():
                    if node_rank == 0:
                        # 결과 이미지 저장하는 경로
                        if os.path.exists(f"{save_path}/figures/total_iter_{total_iter}"):
                            shutil.rmtree(f"{save_path}/figures/total_iter_{total_iter}")

                        try:
                            os.mkdir(f"{save_path}/figures/total_iter_{total_iter}/best_group")
                            os.mkdir(f"{save_path}/figures/total_iter_{total_iter}/worst_group")
                        except:
                            os.makedirs(f"{save_path}/figures/total_iter_{total_iter}/best_group")
                            os.makedirs(f"{save_path}/figures/total_iter_{total_iter}/worst_group")

                    net_eval = Learnable_SEEDLM(
                        input_dim=opts.data_dim
                    )  # 들어갈 변수 많으니 argument 받아서 잘 집어넣어주는 코드 짜보자 (hint: opt 이용)

                    try:
                        net_eval.load_state_dict(net.module.state_dict())
                    except:
                        net_eval.load_state_dict(net.state_dict())

                    net_eval = net_eval.eval().to(device)
                    net_eval.requires_grad_(False)

                    if node_rank == 0:
                        mean_MSE = test(
                            total_iter,
                            test_dataset,
                            net_eval,
                            f"{save_path}/figures/total_iter_{total_iter}",
                            logger,
                            mse_func,
                            node_rank,
                        )
                        wandb.log({"Test_loss": mean_MSE})
                    else:
                        _ = test(
                            total_iter,
                            test_dataset,
                            net_eval,
                            f"{save_path}/figures/total_iter_{total_iter}",
                            logger,
                            mse_func,
                            node_rank,
                        )

                torch.cuda.empty_cache()
                # 모델 저장
                if node_rank == 0:
                    try:
                        state_dict = net.module.state_dict()
                    except:
                        state_dict = net.state_dict()

                    if mean_MSE < best_mse:
                        best_mse = mean_MSE

                        try:
                            os.remove(best_mse_model_path)  # 이전에 최고였던 모델 삭제
                        except:
                            logger.info("can not find prev_mse_best_model!")

                        csv_file.loc["Best_MSE", :] = [round(mean_MSE, 8), total_iter]

                        best_mse_model_path = (
                            save_path + "/" + f"best_mse_model_MSE_{round(mean_MSE, 5)}_total_iter_{total_iter}.pth.tar"
                        )

                        torch.save(
                            {
                                "state_dict": state_dict,
                                "optimizer": optimizer.state_dict(),
                                "criterion": criterion.state_dict(),
                                "best_mse": best_mse,
                                "recent_saved_model_path": recent_saved_model_path,
                                "best_mse_model_path": best_mse_model_path,
                                "total_iter": total_iter,
                            },
                            best_mse_model_path,
                        )

                    csv_file.to_csv(save_path + "/" + "best_res.csv")

                    # 이번 epoch에 학습한 친구도 저장하기
                    try:
                        os.remove(recent_saved_model_path)  # 이전 에폭에서 저장했던 모델 삭제
                    except:
                        logger.info("can not find recent_saved_model!")

                    recent_saved_model_path = (
                        save_path + "/" + f"recent_model_MSE_{round(mean_MSE, 5)}_total_iter_{total_iter}.pth.tar"
                    )

                    torch.save(
                        {
                            "state_dict": state_dict,
                            "optimizer": optimizer.state_dict(),
                            "criterion": criterion.state_dict(),
                            "best_mse": best_mse,
                            "recent_saved_model_path": recent_saved_model_path,
                            "best_mse_model_path": best_mse_model_path,
                            "total_iter": total_iter,
                        },
                        recent_saved_model_path,
                    )

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

    folder_name = (
        "/".join(opts.dataset_path.split("/")[-2:])
        + f"/lmbda_{opts.lmbda}_u_length_{opts.u_length}_batch_size{opts.batch_size}_total_iter{opts.iter}_seed{opts.seed}"
    )

    if opts.seed is not None:
        # save_path = f'./checkpoint/exp2_NIC_Fair_model_{opts.model_name}_lmbda_{opts.lmbda}_seed_{opts.seed}_batch_size_{opts.batch_size}_radius_denominator_{opts.radius_denominator}_total_iter_{opts.iter}'

        save_path = f"./checkpoint/{folder_name}"

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
