"""
Utilities for fine tuning
"""
import copy
import os
from operator import attrgetter

import glog
import torch
from torch import nn

from lib import codebook, utils
from lib.linear import *

from . import quip


def finetune_decoder_layer(layer, name, device, train_dl, valid_dl, args):
    layer = layer.to(device)

    susv_params, params = utils.extract_susv_params(layer)
    optim = utils.get_susv_adam(susv_params, params, args)

    best_loss = utils.calculate_mse_loss(layer, valid_dl, device)
    best_sd = copy.deepcopy(layer.state_dict())
    glog.info(f'layer {name} initial loss {best_loss}')
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    worse_ct = 0
    position_ids = None

    for epoch in range(args.ft_epochs):
        for bidx, (source, targets) in enumerate(train_dl):
            if position_ids is None:
                position_ids = torch.arange(source.shape[1], device=device).unsqueeze(0)
            with torch.autocast(device_type='cuda',
                                dtype=torch.float16,
                                enabled=True):
                output = layer(source.to(device), position_ids=position_ids)[0]
                loss = nn.MSELoss()(output, targets.to(device))
            scaler.scale(loss).backward()
            if bidx % args.ft_update_freq == args.ft_update_freq - 1 or bidx == len(
                    train_dl) - 1:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()

        if epoch % args.ft_valid_freq == (args.ft_valid_freq - 1):
            test_loss = utils.calculate_mse_loss(layer, valid_dl, device)
            if test_loss < best_loss:
                glog.info(
                    f'layer {name} @ epoch {epoch} new loss {test_loss} old loss {best_loss} BETTER'
                )
                best_loss = test_loss
                best_sd = copy.deepcopy(layer.state_dict())
                worse_ct = 0
            else:
                glog.info(
                    f'layer {name} @ epoch {epoch} new loss {test_loss} old loss {best_loss} WORSE'
                )
                worse_ct += 1
                if worse_ct >= args.ft_early_stop:
                    break

    del optim, train_dl, valid_dl

    layer.load_state_dict(best_sd)
    utils.clean()
    layer = layer.cpu()


def quantize_finetune_decoder_layer(mixed_layer, quant_order, idx, cb, args,
                                    device, pre_orig_emb, orig_emb):
    torch.manual_seed(idx)
    torch.set_num_threads(args.num_cpu_threads)

    codebook_id = codebook.get_id(args.codebook)

    mixed_layer = mixed_layer.float()

    train_dl, valid_dl = utils.split_data(pre_orig_emb, orig_emb, args)

    shared_args = (cb.codesz, cb.packsz, cb.pack_out, str(cb.idx_dtype),
                   cb.version)
    shared_kwargs = {
        'rank': args.lora_rank,
        'rescale_WH': args.rescale_WH,
        'resid_scale_override': args.resid_scale_override,
        'bias': False,
        'train_mode': args.ft_train_mode,
        'grad_ckpt': args.ft_grad_ckpt,
    }

    for quant_i, (linear_attr, name) in enumerate(quant_order):
        orig_linear = attrgetter(linear_attr)(mixed_layer)
        if orig_linear.bias is not None:
            # not implemented yet
            raise Exception
        save_path = f'{args.save_path}/{idx}_{name}.pt'
        hessian_path = f'{args.hessian_path}/{idx}_{name}.pt'
        with torch.no_grad():
            if isinstance(orig_linear, FusedLinear):
                weights = torch.split(orig_linear.weight,
                                      orig_linear.fuse_sizes, 0)
            else:
                weights = [orig_linear.weight]
            quip.quantize_linear(weights, save_path, hessian_path, cb, args,
                                 device)
            saved_linear = torch.load(save_path,
                                      map_location=torch.device('cpu'))
            if saved_linear['fused']:
                quant_linear = FusedQuantizedLinear(
                    -1, [_[0] for _ in saved_linear['shapes']],
                    saved_linear['shapes'][0][1],
                    sum([_[0] for _ in saved_linear['shapes']]), *shared_args,
                    **shared_kwargs)
                for i in range(len(saved_linear['scales'])):
                    quant_linear.fuse_scales[i].copy_(
                        saved_linear['scales'][i])
            else:
                quant_linear = QuantizedLinear(saved_linear['shapes'][0][1],
                                               saved_linear['shapes'][0][0],
                                               *shared_args, **shared_kwargs)
            utils.unpack_quip(quant_linear, saved_linear, codebook_id,
                              cb.codesz)
        quant_linear.SU = nn.Parameter(quant_linear.SU.float(),
                                       requires_grad=True)
        quant_linear.SV = nn.Parameter(quant_linear.SV.float(),
                                       requires_grad=True)
        split_attr = linear_attr.split('.')
        setattr(
            attrgetter('.'.join(split_attr[:-1]))(mixed_layer), split_attr[-1],
            quant_linear)
        if quant_i < len(quant_order) - 1:
            finetune_decoder_layer(mixed_layer, f'{idx}_{name}', device,
                                   train_dl, valid_dl, args)

    with torch.no_grad():
        utils.clean()
        for i, (linear_attr, name) in enumerate(quant_order):
            utils.save_susv(
                attrgetter(linear_attr)(mixed_layer),
                f'{args.save_path}/{idx}_{name}.pt')

    mixed_layer = mixed_layer.to(torch.float16).cpu()
    utils.clean()
    torch.set_grad_enabled(False)


def finetune_susv_e2e(model, orig_logits, emb, position_ids, attention_mask,
                      save_fn, args):

    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinear) or isinstance(
                module, FusedQuantizedLinear):
            module.SU = nn.Parameter(module.SU.float(), requires_grad=True)
            module.SV = nn.Parameter(module.SV.float(), requires_grad=True)
    model.float()

    train_dl, valid_dl = utils.split_data(emb, orig_logits, args)

    susv_params, params = utils.extract_susv_params(model)
    optim = utils.get_susv_adam(susv_params, params, args)

    glog.info(f'+++++++ Calculating initial loss')
    best_loss = utils.calculate_ce_loss(model, position_ids, attention_mask,
                                        valid_dl)
    # best_loss = 0
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    best_sd = copy.deepcopy(model.state_dict())
    glog.info(f'initial loss {best_loss}')
    worse_ct = 0
    for epoch in range(args.ft_epochs):
        for bidx, (source, targets) in enumerate(train_dl):
            with torch.autocast(device_type='cuda',
                                dtype=torch.float16,
                                enabled=True):
                output = model(
                    source,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                )[:, :-1].contiguous()
                loss = nn.CrossEntropyLoss()(output.view(-1, output.shape[-1]),
                                             targets.to(0).view(
                                                 -1, targets.shape[-1]))
            scaler.scale(loss).backward()
            if bidx % args.ft_update_freq == args.ft_update_freq - 1 or bidx == len(
                    train_dl) - 1:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()

        if epoch % args.ft_valid_freq == (args.ft_valid_freq - 1):
            test_loss = utils.calculate_ce_loss(model, position_ids,
                                                attention_mask, valid_dl)
            if test_loss < best_loss:
                glog.info(
                    f'epoch {epoch} new loss {test_loss} old loss {best_loss} BETTER'
                )
                best_loss = test_loss
                best_sd = copy.deepcopy(model.state_dict())
                worse_ct = 0
            else:
                glog.info(
                    f'epoch {epoch} new loss {test_loss} old loss {best_loss} WORSE'
                )
                worse_ct += 1
                if worse_ct >= args.ft_early_stop:
                    break

    with torch.no_grad():
        model.load_state_dict(best_sd)
        # save_fn(model) 
        # model.save_pretrained(args.hf_output_path, safe_serialization=True) ## not working


## from qtip
import copy
import math
from operator import attrgetter

import glog
import torch
from torch import multiprocessing as mp
from torch import nn
from transformers import AutoModelForCausalLM

from lib import codebook, utils
# from lib.linear import QuantizedLinear


def infer(args, end_dev, n_layers, in_q, out_q):
    with torch.no_grad():
        fake_dev_map = {
            'model.embed_tokens': 0,
            'model.rotary_emb': 0,
            'model.norm': end_dev - 1,
            'lm_head': end_dev - 1
        }
        per_dev = math.ceil(n_layers / end_dev)
        for i in range(n_layers):
            fake_dev_map[f'model.layers.{i}'] = (i + 1) // per_dev
        
        # num_gpus = 4
        # # num_gpus = end_dev  # 사용 가능한 GPU 총 개수 (예: 4)
        # # embedding과 rotary, norm, lm_head도 순환 방식으로 할당합니다.
        # fake_dev_map = {
        #     'model.embed_tokens': 2,
        #     'model.rotary_emb': 1,
        #     'model.norm': 2,
        #     'lm_head': 3
        # }
        # # 레이어들을 GPU 0,1,2,3에 순환식으로 할당
        # for i in range(n_layers):
        #     fake_dev_map[f'model.layers.{i}'] = i % num_gpus


        model = AutoModelForCausalLM.from_pretrained(args.base_model,
                                                     torch_dtype='auto',
                                                     device_map=fake_dev_map,
                                                     low_cpu_mem_usage=True)
        while True:
            data = in_q.get()
            if data is None:
                return
            out_q.put(
                model(data.to(0))['logits'][:, :-1].contiguous().softmax(
                    dim=-1).cpu())

def calculate_ce_loss_model(model, dataloader, start_dev, in_q, out_q):
    model.eval()
    total_loss = 0
    ct = 0
    with torch.no_grad():
        for source, target in dataloader:
            in_q.put(source)
            output = model(source.to(start_dev))['logits'][:, :-1].contiguous()
            output = output.view(-1, output.shape[-1])
            target = out_q.get().to(output.device)
            target = target.view(-1, target.shape[-1])
            total_loss += nn.CrossEntropyLoss()(output, target)
            ct += 1
    model.train()
    return (total_loss / ct).cpu().item()

def calculate_mse_loss_quip(layer, dataloader, device):
    layer.eval()
    total_loss = 0
    ct = 0
    position_ids = None
    with torch.no_grad():
        for source, target in dataloader:
            if position_ids is None:
                position_ids = torch.arange(source.shape[1], device=device).unsqueeze(0)
            total_loss += nn.MSELoss()(layer(source.to(device), position_ids=position_ids)[0],
                                       target.to(device))
            ct += 1
    layer.train()
    return (total_loss / ct).cpu().item()

# def finetune_susv_e2e_qtip(quant_model, start_dev, devset, orig_dtype, args):
#     # start_dev = 2
#     in_q = mp.Queue()
#     out_q = mp.Queue()
#     p = mp.Process(target=infer,
#                    args=(args, start_dev, len(quant_model.model.layers), in_q,
#                          out_q))
#     p.start()

#     train_dl, valid_dl = utils.split_data(devset, devset, args)

#     optim = torch.optim.Adam(quant_model.parameters(), lr=args.ft_lr)

#     best_loss = calculate_ce_loss_model(quant_model, valid_dl, start_dev,
#                                               in_q, out_q)
#     # best_loss = 0 # for test
    
#     scaler = torch.cuda.amp.GradScaler(enabled=True)

#     best_sd = copy.deepcopy(quant_model.state_dict())
#     glog.info(f'initial loss {best_loss}')
#     worse_ct = 0
    
#     # 체크포인트 저장 경로 설정
#     if hasattr(args, 'checkpoint_path') and args.checkpoint_path:
#         checkpoint_dir = args.checkpoint_path
#     elif hasattr(args, 'hf_output_path') and args.hf_output_path:
#         checkpoint_dir = os.path.join(os.path.dirname(args.hf_output_path), 'checkpoints')
#     else:
#         checkpoint_dir = './checkpoints'
#     os.makedirs(checkpoint_dir, exist_ok=True)
#     glog.info(f'Checkpoint directory: {checkpoint_dir}')
    
#     for epoch in range(args.ft_epochs):
#         for bidx, (source, _) in enumerate(train_dl):
#             in_q.put(source)
#             with torch.autocast(device_type='cuda',
#                                 dtype=orig_dtype,
#                                 enabled=True):
#                 output = quant_model(
#                     source.to(start_dev))['logits'][:, :-1].contiguous()
#                 target = out_q.get().to(output.device)
#                 target = target.view(-1, target.shape[-1])
#                 loss = nn.CrossEntropyLoss()(output.view(-1, output.shape[-1]),
#                                              target)
#             scaler.scale(loss).backward()
#             if bidx % args.ft_update_freq == args.ft_update_freq - 1 or bidx == len(
#                     train_dl) - 1:
#                 scaler.step(optim)
#                 scaler.update()
#                 optim.zero_grad()

#         # 매 epoch마다 체크포인트 저장
#         checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
#         checkpoint = {
#             'epoch': epoch,
#             'model_state_dict': quant_model.state_dict(),
#             'optimizer_state_dict': optim.state_dict(),
#             'scaler_state_dict': scaler.state_dict(),
#             'best_loss': best_loss,
#             'worse_ct': worse_ct,
#         }
#         torch.save(checkpoint, checkpoint_path)
#         glog.info(f'Saved checkpoint at epoch {epoch} to {checkpoint_path}')

#         if epoch % args.ft_valid_freq == (args.ft_valid_freq - 1):
#             test_loss = calculate_ce_loss_model(quant_model, valid_dl,
#                                                       start_dev, in_q, out_q)
#             if test_loss < best_loss:
#                 glog.info(
#                     f'epoch {epoch} new loss {test_loss} old loss {best_loss} BETTER'
#                 )
#                 best_loss = test_loss
#                 best_sd = copy.deepcopy(quant_model.state_dict())
#                 worse_ct = 0
#             else:
#                 glog.info(
#                     f'epoch {epoch} new loss {test_loss} old loss {best_loss} WORSE'
#                 )
#                 worse_ct += 1
#                 if worse_ct >= args.ft_early_stop:
#                     break

#     in_q.put(None)
#     p.join()
#     with torch.no_grad():
#         quant_model.load_state_dict(best_sd)



from tqdm import tqdm  # Make sure to import tqdm

def finetune_susv_e2e_qtip(quant_model, start_dev, devset, orig_dtype, args):
    in_q = mp.Queue()
    out_q = mp.Queue()
    p = mp.Process(target=infer,
                   args=(args, start_dev, len(quant_model.model.layers), in_q,
                         out_q))
    p.start()

    train_dl, valid_dl = utils.split_data(devset, devset, args)

    optim = torch.optim.Adam(quant_model.parameters(), lr=args.ft_lr)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    if hasattr(args, 'checkpoint_path') and args.checkpoint_path:
        checkpoint_dir = args.checkpoint_path
    elif hasattr(args, 'hf_output_path') and args.hf_output_path:
        checkpoint_dir = os.path.join(os.path.dirname(args.hf_output_path), 'checkpoints')
    else:
        checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    glog.info(f'Checkpoint directory: {checkpoint_dir}')

    # --- Resume Logic ---
    start_epoch = 0
    best_loss = float('inf')
    worse_ct = 0
    best_sd = None

    if hasattr(args, 'resume_ckpt') and args.resume_ckpt and os.path.exists(args.resume_ckpt):
        glog.info(f"Resuming training from checkpoint: {args.resume_ckpt}")
        # Load checkpoint onto the correct device
        # checkpoint = torch.load(args.resume_ckpt, map_location=f'cuda:{start_dev}')
        checkpoint = torch.load(args.resume_ckpt)
        
        new_sd = {}
        sd_ = checkpoint['model_state_dict']
        for key, value in sd_.items():
            if key.endswith(".Qidxs_0"):
                new_key = key.replace(".Qidxs_0", ".Qidxs")
                new_sd[new_key] = value
            else:
                new_sd[key] = value
        quant_model.load_state_dict(new_sd, strict=False)        
        # quant_model.load_state_dict(checkpoint['model_state_dict'])
        
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        worse_ct = checkpoint['worse_ct']
        
        state_dict_to_load = checkpoint.get('best_model_state_dict', checkpoint['model_state_dict'])
        best_sd = {k: v.to(start_dev) for k, v in state_dict_to_load.items()}
        
        glog.info(f"Resumed from epoch {checkpoint['epoch']}. Starting next epoch: {start_epoch}")
        glog.info(f"Resumed best_loss: {best_loss}, worse_ct: {worse_ct}")

    else:
        glog.info("Starting training from scratch.")
        # Calculate initial loss only if not resuming
        best_loss = calculate_ce_loss_model(quant_model, valid_dl, start_dev,
                                              in_q, out_q)
        best_sd = copy.deepcopy(quant_model.state_dict())
        glog.info(f'Initial loss {best_loss}')
        

    previous_checkpoint_path = None # To track and delete old checkpoints

    for epoch in range(start_epoch, args.ft_epochs):
        quant_model.train() # Set model to training mode
        
        # Wrap train_dl with tqdm for a progress bar
        train_iter = tqdm(train_dl, desc=f"Epoch {epoch}/{args.ft_epochs} Training", unit="batch")

        for bidx, (source, _) in enumerate(train_iter):
            in_q.put(source)
            with torch.autocast(device_type='cuda',
                                dtype=orig_dtype,
                                enabled=True):
                output = quant_model(
                    source.to(start_dev))['logits'][:, :-1].contiguous()
                target = out_q.get().to(output.device)
                target = target.view(-1, target.shape[-1])
                loss = nn.CrossEntropyLoss()(output.view(-1, output.shape[-1]),
                                             target)
            
            # Update tqdm postfix with current loss
            train_iter.set_postfix(loss=loss.item())

            scaler.scale(loss).backward()
            if bidx % args.ft_update_freq == args.ft_update_freq - 1 or bidx == len(
                    train_dl) - 1:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()

        # --- Checkpoint Saving ---
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': quant_model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_loss': best_loss,
            'worse_ct': worse_ct,
            'best_model_state_dict': best_sd,  # Save the best state dict
        }
        torch.save(checkpoint, checkpoint_path)
        glog.info(f'Saved checkpoint at epoch {epoch} to {checkpoint_path}')

        # Delete the previous epoch's checkpoint
        if previous_checkpoint_path and os.path.exists(previous_checkpoint_path):
            os.remove(previous_checkpoint_path)
            glog.info(f'Removed previous checkpoint: {previous_checkpoint_path}')
        previous_checkpoint_path = checkpoint_path
        # --- End Checkpoint Saving ---

        if epoch % args.ft_valid_freq == (args.ft_valid_freq - 1):
            quant_model.eval() # Set model to evaluation mode for validation
            test_loss = calculate_ce_loss_model(quant_model, valid_dl,
                                                      start_dev, in_q, out_q)
            quant_model.train() # Set back to train mode

            if test_loss < best_loss:
                glog.info(
                    f'epoch {epoch} new loss {test_loss} old loss {best_loss} BETTER'
                )
                best_loss = test_loss
                best_sd = copy.deepcopy(quant_model.state_dict())
                worse_ct = 0
                
                # Optionally save a separate 'best_model.pt' as well
                best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
                torch.save(best_sd, best_model_path)
                glog.info(f'Saved new best model state to {best_model_path}')

            else:
                glog.info(
                    f'epoch {epoch} new loss {test_loss} old loss {best_loss} WORSE'
                )
                worse_ct += 1
                if worse_ct >= args.ft_early_stop:
                    glog.info(f"Early stopping at epoch {epoch} due to {worse_ct} worse epochs.")
                    break # Exit the training loop

    in_q.put(None)
    p.join()
    
    # After training, load the best model state
    with torch.no_grad():
        if best_sd:
            quant_model.load_state_dict(best_sd)
        else:
            glog.warning("best_sd was never set (or loaded); model state is from the last training step.")
            
    # Optional: Clean up the final 'checkpoint_epoch_X.pt' file
    if previous_checkpoint_path and os.path.exists(previous_checkpoint_path):
        os.remove(previous_checkpoint_path)
        glog.info(f'Removed final epoch checkpoint: {previous_checkpoint_path}')

    glog.info(f"Finetuning finished. Loaded best model with validation loss: {best_loss}")