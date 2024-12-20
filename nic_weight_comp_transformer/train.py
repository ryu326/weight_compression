# nohup python -u train.py --model_name TCM --lmbda 1 --iter 2000000 --batch-size 8 --seed 100 --dist_port 6044 > ./training_logs/log_lmbda_1.txt 2>&1 &

# 노헙 안돌리는거
# CUDA_VISIBLE_DEVICES=0 taskset -c 0-7 python -u train.py --model_name ELIC --lmbda 1 --iter 2000000 --batch-size 8 --seed 100 --dist_port 6044 --slurm
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
from datasets_WeightParam import WParam_dataset
from models import TR_NIC
from torch.utils.data import DataLoader
from utils.optimizers import *
from utils.util import *

# 시간 측정해보기








def test(total_iter, test_dataset, model, save_path, logger, mse_func, node_rank=0):

    avg_bpp = 0.0
    mean_MSE = 0

    device = next(model.parameters()).device

    # for idx, (image_name, image) in enumerate(test_dataset) :
    for idx, image in enumerate(test_dataset):

        img = image.to(device)
        x = img.unsqueeze(0).to(device)  # [1, 4, 512] 같음

        out_enc = model.compress(x.to(device))
        out_dec = model.decompress(out_enc["strings"], out_enc["shape"])

        num_pixels = x.size(1) * x.size(2)
        bpp = 0

        for s in out_enc["strings"]:
            if s != [0]:  #
                bpp += len(s[0]) * 8.0 / num_pixels

        x_hat = out_dec["x_hat"].clone().detach()

        mse = mse_func(x, x_hat).item()

        avg_bpp += bpp
        mean_MSE += mse

        if node_rank == 0:

            logger.info(f"Test total_iter: {total_iter}, File name: {idx}, MSE: {mse}, BPP: {bpp}")

    avg_bpp /= len(test_dataset)
    mean_MSE /= len(test_dataset)

    if node_rank == 0:

        logger.info(f"Average_MSE: {mean_MSE}, Average_Bit-rate: {avg_bpp} bpp")

    return avg_bpp, mean_MSE


def main(opts):
    wandb.init(project="Neural Weight Compression_v2", name="tr_nwc")
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

    # def get_rank():
    #     if not is_dist_avail_and_initialized():
    #         return 0
    #     return dist.get_rank()
    # # fix the seed for reproducibility
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

    csv_file = pd.DataFrame(None, index=["Best_MSE", "Best_bpp"], columns=["bpp", "MSE", "iter"])

    # if opts.slurm == True:
    #     # [batch, seq_len, dim] 형식으로 나오게 (embedding sequence 처럼 나오게)
    #     # dataset_dir = /home/jgryu/Weight_compression/Wparam_dataset
    #     train_dataset = WParam_dataset(dataset_folder= opts.dataset_dir, split='train', param_type = 'mlp', data_dim=opts.data_dim, length=opts.length, seed = 100)
    # else:

    train_dataset = WParam_dataset(
        dataset_folder=opts.dataset_dir, split="train", data_dim=opts.data_dim, length=opts.length, seed=100
    )

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

    net = TR_NIC(
        input_dim=opts.data_dim
    )  # 들어갈 변수 많으니 argument 받아서 잘 집어넣어주는 코드 짜보자 (hint: opt 이용)
    net = net.to(device)

    if gpu_num > 1:
        net = torch.nn.parallel.DistributedDataParallel(
            net,
            device_ids=[device_id],
            output_device=device_id,
            find_unused_parameters=False,
        )

    optimizer, aux_optimizer = configure_optimizers(net, opts)

    if opts.model_name == "TCM":
        milestones = [int((opts.iter / gpu_num) * 0.9), int((opts.iter / gpu_num) * 0.96)]
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    elif opts.model_name == "FTIC":
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.3, patience=16)
    elif opts.model_name == "ELIC":
        milestones = [2000000]
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    criterion = RateDistortionLoss(lmbda=opts.lmbda, radius_denominator=opts.radius_denominator)

    ########################################
    # test 준비
    test_dataset = WParam_dataset(
        dataset_folder=opts.dataset_dir, split="val", data_dim=opts.data_dim, length=opts.length, seed=100
    )

    mse_func = nn.MSELoss()

    if node_rank == 0:
        logger.info("Training mode : scratch!")
        logger.info(f"lmbda : {opts.lmbda}")

        logger.info(f"batch_size : {opts.batch_size}")

        logger.info(f"num of gpus: {gpu_num}")
    ########################################

    best_mse = float("inf")
    best_bpp = float("inf")

    recent_saved_model_path = ""
    best_mse_model_path = ""

    total_iter = 0

    recent_saved_model_path = ""
    best_psnr_model_path = ""
    best_ms_ssim_model_path = ""
    best_bpp_model_path = ""
    best_lpips_model_path = ""
    log_loss = {
        "mse_loss_high_freq": [],
        "mse_loss_low_freq": [],
        "mse_loss": [],
        "bpp_loss": [],
        "bpp_loss_low_freq": [],
    }

    checkpoint = opts.checkpoint
    if checkpoint != "None":  # load from previous checkpoint

        if node_rank == 0:
            logger.info(f"Loading {checkpoint}")

        checkpoint = torch.load(checkpoint)
        total_iter = checkpoint["total_iter"]
        log_loss = checkpoint["log_loss"]

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

        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        criterion.load_state_dict(checkpoint["criterion"])

        best_mse = checkpoint["best_mse"]
        best_bpp = checkpoint["best_bpp"]

        recent_saved_model_path = checkpoint["recent_saved_model_path"]

        best_mse_model_path = checkpoint["best_mse_model_path"]
        best_bpp_model_path = checkpoint["best_bpp_model_path"]

        del checkpoint

    save_path = opts.save_path

    if gpu_num > 1:
        dist.barrier()

    # dummy run
    dummy_image = torch.randn(1, opts.length, opts.data_dim).to("cuda")

    for _ in range(10):
        _ = net(dummy_image)

    optimizer.zero_grad()
    aux_optimizer.zero_grad()

    training_starter = torch.cuda.Event(enable_timing=True)
    training_ender = torch.cuda.Event(enable_timing=True)

    while 1:
        net.train()
        device = next(net.parameters()).device

        if total_iter >= opts.iter:
            break

        for i, (img) in enumerate(train_dataloader):

            if total_iter >= opts.iter:
                break

            with torch.autograd.detect_anomaly():
                img = img.to(device)

                optimizer.zero_grad()
                aux_optimizer.zero_grad()

                out_net = net(img)

                torch.cuda.synchronize()

                if gpu_num > 1:
                    dist.barrier()

                updating_time = {}

                training_starter.record()

                # leraning_percentage = float(total_iter)/float(opts.iter)

                out_criterion = criterion(out_net)
                training_ender.record()
                torch.cuda.synchronize()
                updating_time["loss"] = training_starter.elapsed_time(training_ender)

                training_starter.record()
                out_criterion["loss"].backward()

                if opts.clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), opts.clip_max_norm)
                optimizer.step()
                training_ender.record()
                torch.cuda.synchronize()
                updating_time["update"] = training_starter.elapsed_time(training_ender)

                training_starter.record()
                # aux loss
                try:
                    aux_loss = net.aux_loss()
                except:
                    aux_loss = net.module.aux_loss()
                training_ender.record()
                torch.cuda.synchronize()
                updating_time["aux_loss"] = training_starter.elapsed_time(training_ender)

                training_starter.record()
                aux_loss.backward()
                aux_optimizer.step()
                training_ender.record()
                torch.cuda.synchronize()
                updating_time["aux_update"] = training_starter.elapsed_time(training_ender)

                if (opts.model_name == "TCM") or (opts.model_name == "ELIC"):
                    lr_scheduler.step()
                elif opts.model_name == "FTIC":
                    if total_iter + (1 * gpu_num) % 5000 == 0 and total_iter > 0:
                        lr_scheduler.step(loss)

            wandb.log(
                {
                    "train_loss": out_criterion["loss"],
                    "train_mse_loss": out_criterion,
                    "train_bpp_loss": out_criterion["bpp_loss"],
                }
            )
            optimizer.zero_grad()
            aux_optimizer.zero_grad()

            total_iter += 1 * gpu_num

            if ((total_iter % 5000 == 0) or (total_iter == (1 * gpu_num))) and node_rank == 0:

                # log_loss["mse_loss_high_freq"].append(out_criterion["mse_loss_high_freq"].mean().item())
                # log_loss["mse_loss_low_freq"].append(out_criterion["mse_loss_low_freq"].mean().item())
                log_loss["mse_loss"].append(out_criterion["mse_loss"].mean().item())
                log_loss["bpp_loss"].append(out_criterion["bpp_loss"].item())

                logger.info(
                    f"Train iter. {total_iter}/{opts.iter} ({100. * total_iter / opts.iter}%): "
                    f'\tLoss: {out_criterion["loss"].item()} |'
                    f'\tMSE loss: {out_criterion["mse_loss"].mean().item()} |'
                    # f'\tSWT-domain fidelity loss: {out_criterion["swt_loss"].item():.6f} |'
                    # f'\tMSE loss ratio (low freq/high freq): {out_criterion["ratio_between_low_and_high"].item():.6f} |'
                    # f'\tMSE loss (low freq.): {out_criterion["mse_loss_low_freq"].mean().item():.6f} |'
                    # f'\tMSE loss (high freq.): {out_criterion["mse_loss_high_freq"].mean().item():.6f} |'
                    f'\tBpp loss: {out_criterion["bpp_loss"].item()} |'
                    f"\tAux loss: {aux_loss.item()}"
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

                    # if opts.model_name == 'TCM':
                    #     net_eval = TCM(config=[2,2,2,2,2,2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0.0, N=64, M=320, radius_denominator = opts.radius_denominator)
                    # elif opts.model_name == 'FTIC':
                    #     net_eval = FrequencyAwareTransFormer(radius_denominator = opts.radius_denominator)
                    # elif opts.model_name == 'ELIC':
                    #     config = model_config()
                    #     net_eval = ELIC(config, radius_denominator = opts.radius_denominator)
                    net_eval = TR_NIC(
                        input_dim=opts.data_dim
                    )  # 들어갈 변수 많으니 argument 받아서 잘 집어넣어주는 코드 짜보자 (hint: opt 이용)

                    try:
                        net_eval.load_state_dict(net.module.state_dict())
                    except:
                        net_eval.load_state_dict(net.state_dict())

                    net_eval = net_eval.eval().to(device)
                    net_eval.requires_grad_(False)
                    net_eval.update()

                    if node_rank == 0:
                        avg_bpp, mean_MSE = test(
                            total_iter,
                            test_dataset,
                            net_eval,
                            f"{save_path}/figures/total_iter_{total_iter}",
                            logger,
                            mse_func,
                            node_rank,
                        )
                        wandb.log({"Test_avg_bpp": avg_bpp, "Test_loss": mean_MSE})
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
                            logger.info("can not find prev_bpp_best_model!")

                        csv_file.loc["Best_MSE", :] = [round(avg_bpp, 8), round(mean_MSE, 8), total_iter]

                        best_mse_model_path = (
                            save_path
                            + "/"
                            + f"best_mse_model_MSE_{round(mean_MSE, 5)}_BPP_{round(avg_bpp, 5)}_total_iter_{total_iter}.pth.tar"
                        )

                        # recon_damaged_optimizer, comp_optimizer, comp_aux_optimizer
                        torch.save(
                            {
                                "state_dict": state_dict,
                                "optimizer": optimizer.state_dict(),
                                "aux_optimizer": aux_optimizer.state_dict(),
                                "lr_scheduler": lr_scheduler.state_dict(),
                                "criterion": criterion.state_dict(),
                                "best_mse": best_mse,
                                "best_bpp": best_bpp,
                                "recent_saved_model_path": recent_saved_model_path,
                                "best_mse_model_path": best_mse_model_path,
                                "best_bpp_model_path": best_bpp_model_path,
                                "log_loss": log_loss,
                                "total_iter": total_iter,
                            },
                            best_mse_model_path,
                        )

                    if avg_bpp < best_bpp:
                        best_bpp = avg_bpp

                        try:
                            os.remove(best_bpp_model_path)  # 이전에 최고였던 모델 삭제
                        except:
                            logger.info("can not find prev_bpp_best_model!")

                        csv_file.loc["Best_bpp", :] = [round(avg_bpp, 8), round(mean_MSE, 8), total_iter]

                        best_bpp_model_path = (
                            save_path
                            + "/"
                            + f"best_bpp_model_MSE_{round(mean_MSE, 5)}_BPP_{round(avg_bpp, 5)}_total_iter_{total_iter}.pth.tar"
                        )
                        torch.save(
                            {
                                "state_dict": state_dict,
                                "optimizer": optimizer.state_dict(),
                                "aux_optimizer": aux_optimizer.state_dict(),
                                "lr_scheduler": lr_scheduler.state_dict(),
                                "criterion": criterion.state_dict(),
                                "best_mse": best_mse,
                                "best_bpp": best_bpp,
                                "recent_saved_model_path": recent_saved_model_path,
                                "best_mse_model_path": best_mse_model_path,
                                "best_bpp_model_path": best_bpp_model_path,
                                "log_loss": log_loss,
                                "total_iter": total_iter,
                            },
                            best_bpp_model_path,
                        )

                    csv_file.to_csv(save_path + "/" + "best_res.csv")

                    # 이번 epoch에 학습한 친구도 저장하기
                    try:
                        os.remove(recent_saved_model_path)  # 이전 에폭에서 저장했던 모델 삭제
                    except:
                        logger.info("can not find recent_saved_model!")

                    recent_saved_model_path = (
                        save_path
                        + "/"
                        + f"recent_model_MSE_{round(mean_MSE, 5)}_BPP_{round(avg_bpp, 5)}_total_iter_{total_iter}.pth.tar"
                    )

                    torch.save(
                        {
                            "state_dict": state_dict,
                            "optimizer": optimizer.state_dict(),
                            "aux_optimizer": aux_optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                            "criterion": criterion.state_dict(),
                            "best_mse": best_mse,
                            "best_bpp": best_bpp,
                            "recent_saved_model_path": recent_saved_model_path,
                            "best_mse_model_path": best_mse_model_path,
                            "best_bpp_model_path": best_bpp_model_path,
                            "log_loss": log_loss,
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
    if opts.seed is not None:
        # save_path = f'./checkpoint/exp2_NIC_Fair_model_{opts.model_name}_lmbda_{opts.lmbda}_seed_{opts.seed}_batch_size_{opts.batch_size}_radius_denominator_{opts.radius_denominator}_total_iter_{opts.iter}'
        save_path = f"./checkpoint/{'/'.join(opts.dataset_dir.split('/')[-2:])}/lmbda{opts.lmbda}_batch_size{opts.batch_size}_total_iter{opts.iter}_seed{opts.seed}"

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
        logger.info(
            f"exp name : exp_NIC_Fair_model_{opts.model_name}_lmbda_{opts.lmbda}_seed_{opts.seed}_batch_size_{opts.batch_size}_radius_denominator_{opts.radius_denominator}_total_iter_{opts.iter}"
        )

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
