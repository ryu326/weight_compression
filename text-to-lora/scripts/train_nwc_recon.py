from copy import deepcopy
import gc
import os
import random
import string
import time
import shutil
from math import ceil

import wandb
import torch
from transformers import get_scheduler, set_seed
from peft import get_peft_config, PeftConfig, load_peft_weights

from hyper_llm_modulator.comp_lora import get_compnet_v1
from hyper_llm_modulator.comp_lora_v2 import get_compnet_v2
from hyper_llm_modulator.comp_lora_v3 import get_compnet_v3

from hyper_llm_modulator.configs import ArgumentParser, TrainingArguments
from hyper_llm_modulator.data import get_embs_dict, get_recon_train_data
from hyper_llm_modulator.hyper_modulator import create_hypermod
# from hyper_llm_modulator.recon_trainer import train
from hyper_llm_modulator.recon_trainer_nwc import train
from hyper_llm_modulator.utils import (
    create_logger,
    get_layers,
    get_num_params,
    save_yaml,
    get_model_and_tokenizer,
    get_target_lora_dirs,
    get_metadata,
    get_pooling_fn,
    add_full_stop,
)
from hyper_llm_modulator.utils.model_loading import get_emb_model_and_fns
import torch.optim as optim

def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {n for n, p in net.named_parameters() if ".quantiles" not in n and p.requires_grad}
    aux_parameters = {n for n, p in net.named_parameters() if ".quantiles" in n and p.requires_grad}

    print(aux_parameters)  # {'module.entropy_bottleneck_z.quantiles'}

    params_dict = dict(net.named_parameters())

    optimizer = optim.AdamW(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.lr,
    )
    if aux_parameters:
        aux_optimizer = optim.Adam(
            (params_dict[n] for n in sorted(aux_parameters)),
            lr=args.aux_lr,
        )
    else :
        aux_optimizer = None
    return optimizer, aux_optimizer

def main(args):
    args.train_ds_names = args.train_ds_names[: args.n_train_ds]
    args.training_task = "recon"
    # get task metadata and save to the corresponding run folder
    metadata = get_metadata(args.train_ds_names, args.use_per_task_emb)
    metadata["run_name"] = args.run_name
    metadata["save_dir"] = save_dir = args.save_dir
    os.makedirs(f"{save_dir}/checkpoints", exist_ok=True)
    save_yaml(vars(args), f"{save_dir}/args.yaml")
    set_seed(args.seed)

    wandb_dir = f"{os.environ['HOME']}/.wandb/logs/{os.environ['WANDB_PROJECT']}/"
    os.makedirs(wandb_dir, exist_ok=True)
    wandb.init(
        project=os.environ["WANDB_PROJECT"],
        config=vars(args),
        group=args.run_name,
        name=args.run_name,
        dir=wandb_dir,
        notes=args.notes,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    lora_dirs = get_target_lora_dirs(args.train_ds_names, args.model_dir)
    args.train_ds_names = list(lora_dirs.keys())
    args.n_train_ds = len(lora_dirs)

    # assumes all target LORAs have the same config
    lora_dir = list(lora_dirs.values())[0]
    adapter_config_path = f"{lora_dir}/adapter_config.json"
    peft_config = get_peft_config(PeftConfig.from_json_file(adapter_config_path))
    shutil.copy(adapter_config_path, f"{save_dir}/adapter_config.json")

    model, tokenizer = get_model_and_tokenizer(
        args.model_dir,
        train=False,
        requires_grad=False,
        peft_config=peft_config,
    )

    logger.debug(f"Model config: {model.config}")
    logger.debug(f"Model: {model}")

    use_explicit_emb_model = False
    task_emb_size = None
    emb_model = None
    emb_tokenizer = None
    task_desc_format_fn = None
    pooling_fn = None
    use_explicit_emb_model = False

    if not args.use_one_hot_task_emb:
        emb_model = model
        emb_tokenizer = deepcopy(tokenizer)
        task_desc_format_fn = add_full_stop
        pooling_fn = get_pooling_fn("last_token")

        if args.emb_model:
            use_explicit_emb_model = True
            emb_model, emb_tokenizer, task_desc_format_fn, pooling_fn = get_emb_model_and_fns(args.emb_model, device)
            logger.debug(f"emb_model: {emb_model}")
        emb_model.eval()
        task_emb_size = emb_model.config.hidden_size

    n_layers = len(get_layers(model))
    layer_indices = list(range(n_layers))  # target all layers

    # hypermod = create_hypermod(args, peft_config.peft_type.lower(), device, model, layer_indices, task_emb_size)
    # hypermod.train()
    # logger.debug(f"Hypermod: {hypermod}")

    args.compnet_v = getattr(args, "compnet_v", 1)
    if args.compnet_v == 1:
        get_compnet = get_compnet_v1
    elif args.compnet_v == 2:    
        get_compnet = get_compnet_v2
    elif args.compnet_v == 3:    
        get_compnet = get_compnet_v3

    comp_model = get_compnet(args, peft_config.peft_type.lower(), device, model, layer_indices, task_emb_size)
    comp_model.train()
    # hypermod.train()
    logger.debug(f"comp_model: {comp_model}")

    task_embs_dict = get_embs_dict(
        args,
        emb_model,
        emb_tokenizer,
        task_desc_format_fn,
        pooling_fn,
        args.train_ds_names,
        metadata,
        device,
    )

    del model, tokenizer
    if use_explicit_emb_model:
        del emb_model, emb_tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    # data prep
    target_loras = {task: load_peft_weights(d) for task, d in lora_dirs.items()}
    if args.factorized and args.mt_lora_path:
        mt_lora = load_peft_weights(args.mt_lora_path)
        target_loras = {
            task: {k: v - mt_lora[k] for k, v in lora_sd.items()} for task, lora_sd in target_loras.items()
        }
        del mt_lora

    logger.info(f"# of target LORAs: {len(target_loras)}")

    train_data = dict()

    for task, state_dict in target_loras.items():
        train_data[task] = get_recon_train_data(state_dict, args.target_modules, layer_indices, device)

    wandb.watch(comp_model, log="all")

    logger.debug("Trainable hypernet parameters:")
    for name, p in comp_model.named_parameters():
        if p.requires_grad:
            logger.debug(f"{name}, dtype:{p.dtype}")
    _, num_trainable_params = get_num_params(comp_model)
    logger.info(f"trainable params: {num_trainable_params:,d}")

    # training
    # optimizer = torch.optim.AdamW(comp_model.parameters(), lr=args.lr, weight_decay=1e-3)
    optimizer, aux_optimizer = configure_optimizers(comp_model, args)
    tasks_per_batch = min(args.n_tasks_per_batch, len(args.train_ds_names))
    n_minibatches = ceil(len(args.train_ds_names) / tasks_per_batch)
    n_batches = n_minibatches * args.epochs

    num_warmup_steps = args.warmup_frac * n_batches
    lr_scheduler = get_scheduler(
        "linear",
        optimizer,
        num_warmup_steps=int(num_warmup_steps),
        num_training_steps=int(n_batches),
    )   

    train(
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
    )

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["WANDB_PROJECT"] = "hypermod_recon"
    os.environ["WANDB_WATCH"] = "all"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    parser = ArgumentParser((TrainingArguments,))
    args = parser.parse()
    assert args.exp_setup == "hyper_lora"

    uuid = "".join([random.choice(string.ascii_letters + string.digits) for _ in range(8)])
    args.run_name = time.strftime("%Y%m%d-%H%M%S") + f"_{uuid}"
    args.save_dir = f"train_outputs/compnet_recon/{args.exp_setup}/{args.run_name}"
    global logger
    logger = create_logger(args.save_dir, debug=args.debug)
    logger.debug(f"CMD: {' '.join(os.sys.argv)}")
    logger.debug(f"args: {args}")
    logger.debug(f"Is CUDA available: {torch.cuda.is_available()}")
    logger.debug(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    main(args)
