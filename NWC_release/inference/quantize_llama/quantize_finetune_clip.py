import argparse
import os
import time

import glog, json

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

import torch
import torch.multiprocessing as mp
from transformers import CLIPModel, AutoModel
from transformers.modeling_attn_mask_utils import \
    _prepare_4d_causal_attention_mask

from lib import utils
from lib.algo import finetune_clip
from lib.algo import finetune
from operator import attrgetter

import sys
notebook_dir = os.path.dirname(os.path.abspath("__file__"))
project_root = os.path.abspath(os.path.join(notebook_dir, ".."))

std = 0.012528747320175171
if project_root not in sys.path:
    sys.path.append(project_root)
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from NWC.models import get_model

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_cpu_threads', default=8, type=int)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--devset_size', default=384, type=int)
parser.add_argument('--ctx_size', default=4096, type=int)
parser.add_argument('--save_path', type=str)
parser.add_argument('--in_hess_path', type=str)
parser.add_argument('--in_hess_eig_path', type=str)
parser.add_argument('--whiten', action='store_true', default=False)
parser.add_argument('--base_model', type=str)
parser.add_argument('--sigma_reg', default=1e-2, type=float)
parser.add_argument('--sigma_reg2', default=1e-2, type=float)
parser.add_argument('--scale_override', default=-1, type=float)
parser.add_argument('--use_fp64', action='store_true')
parser.add_argument('--no_use_buffered', action='store_true')
parser.add_argument('--sample_proc', default=1, type=int)
parser.add_argument('--lowmem_ldlq', action='store_true')
parser.add_argument('--ft_lr', default=3e-6, type=float)
parser.add_argument('--ft_bs', default=4, type=int)
parser.add_argument('--ft_update_freq', default=1, type=int)
parser.add_argument('--ft_epochs', default=0, type=int)
parser.add_argument('--ft_valid_freq', default=1, type=int)
parser.add_argument('--ft_valid_size', default=128, type=float)
parser.add_argument('--ft_early_stop', default=5, type=int)
parser.add_argument('--ft_grad_ckpt', action='store_true')
parser.add_argument('--skip_list', default=None, type=str)

parser.add_argument("--bundle", action='store_true', default = True)
parser.add_argument("--ql", action='store_true')
parser.add_argument("--ql_path", type=str, default = None)
parser.add_argument("--hesseigen", type=str, default = None)
parser.add_argument("--gptq", action='store_true', default = False)
parser.add_argument("--ldlq", action='store_true', default = False)
parser.add_argument("--comp_model_path", type=str, default=None)
parser.add_argument("--direction", type=str, default='col')
parser.add_argument("--comp_batch_size", type=int, default=2048)
parser.add_argument('--quip_tune_iters', default=0, type=int)
parser.add_argument('--rescale_WH', action='store_true')
parser.add_argument('--rescale_WH_2', action='store_true')
parser.add_argument('--incoh_mode',
                    default='none',
                    type=str,
                    choices=['had', 'kron', 'none'])
parser.add_argument('--lora_rank',
                    default=0,
                    type=int,
                    help='if <=0 then turned off')
parser.add_argument('--ql_invH', action='store_true', default=False)
parser.add_argument('--use_train_scale', action='store_true', default=False)
parser.add_argument('--layerwise_cdt', action='store_true', default=False)
parser.add_argument('--ft_comp_model', action='store_true', default=False)
parser.add_argument('--ft_comp_model2', action='store_true', default=False)
parser.add_argument('--ft_comp_lmbda', type=int, default=None)
parser.add_argument('--ft_comp_ep', type=int, default=None)
parser.add_argument('--ft_comp_steps', type=int, default=None)
parser.add_argument("--ft_comp_learning_rate", default=1e-4, type=float)
parser.add_argument("--ft_comp_aux_learning_rate", default=1e-3,)
parser.add_argument("--ft_train_dec", action='store_true', default=False)
parser.add_argument("--ql_search", action='store_true', default=False)
parser.add_argument("--ql_search_layer_name", type=str, default=None)
# parser.add_argument("--ql_search_layer_idx", type=str, default='0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31')
parser.add_argument("--ql_search_layer_idx", type=str, default='')
parser.add_argument("--ql_search_value", type=int, default=None)
parser.add_argument("--ql_tuned", action='store_true', default=False)
parser.add_argument('--layerwise_scale', action='store_true', default=False)
parser.add_argument('--layer_normalize', action='store_true', default=False)
parser.add_argument('--channelwise_scale', action='store_true', default=False)
parser.add_argument('--optim_code', action='store_true', default=False)
parser.add_argument('--row_normalize', action='store_true', default=False)
parser.add_argument('--row_normalize2', action='store_true', default=False)
parser.add_argument('--col_normalize', action='store_true', default=False)
parser.add_argument('--code_optim', action='store_true', default=False)
parser.add_argument('--code_optim_it', type=int, default=False)
parser.add_argument('--code_optim_lr', type=float, default=5e-3)
parser.add_argument('--code_optim_lmbda', type=int, default=None)
parser.add_argument('--code_optim_test', action='store_true', default=False)
parser.add_argument('--code_optim_model', type=str, default='nwc_ql_sga')
parser.add_argument('--optim_qs', action='store_true', default=False)
parser.add_argument('--loss', type=str, default='rdloss_ql')
parser.add_argument('--Q', type=int, default=4)
parser.add_argument('--use_codes', action='store_true', default=False)
parser.add_argument('--qmap_uniform', type=float, default=None)
parser.add_argument('--qmap_hessian', action='store_true', default=False)
parser.add_argument('--qmap_hessian_ql', action='store_true', default=False)
parser.add_argument('--qmap_alpha', type=float, default=False)
parser.add_argument('--qmap_optim_iter', type=int, default=5)
parser.add_argument('--qmap_optim', action='store_true', default=False)
# parser.add_argument('--optim_norm', action='store_true', default=False)
parser.add_argument('--cnorm_optim', action='store_true', default=False)
parser.add_argument('--rnorm_optim', action='store_true', default=False)
parser.add_argument('--ft_rnorm', action='store_true', default=False)
parser.add_argument('--ft_scale_cond0', action='store_true', default=False)
parser.add_argument('--ft_metadata', action='store_true', default=False)
parser.add_argument('--ft_Wr', action='store_true', default=False)
parser.add_argument('--ft_y', action='store_true', default=False)
parser.add_argument('--ft_bpp_loss', action='store_true', default=False)
parser.add_argument('--scaleH', action='store_true', default=False)
parser.add_argument('--smooth_scaleH_alpha', type=float, default=None)
parser.add_argument('--lb_scaleH', type=float, default=None)
parser.add_argument('--scaleHinv', action='store_true', default=False)
parser.add_argument('--scale_std', type=float, default=None)
parser.add_argument('--ste', action='store_true', default=False)
parser.add_argument("--handcraft_mode", type=str, choices=["all", "bpg", "jp2k", "jp", "webp"], default=None)
parser.add_argument("--quant_method", type=str, choices=['per_tensor', 'per_channel', 'group'], default=None)
parser.add_argument("--group_sz", type=int, default=-1)
parser.add_argument("--jp_quality", type=int, default=-1)
parser.add_argument("--bpg_quality", type=int, default=-1)
parser.add_argument("--webp_quality", type=int, default=-1)
parser.add_argument("--nic_model", type=str, choices=["tcm", "ftic", "illm"], default=None)
parser.add_argument("--nic_checkpoint", type=str, default=None)
parser.add_argument("--nic_patch_size", type=int, default=-1)
parser.add_argument("--nic_norm_patch_size", type=int, default=-1)
parser.add_argument("--illm_quality", type=int, default=-1)
parser.add_argument('--scale_cond0', action='store_true', default=False)
parser.add_argument('--scale_cond_ub', type=float, default=None)
parser.add_argument('--scale_cond', action='store_true', default=False)
parser.add_argument('--fp_iter', action='store_true', default=False)
parser.add_argument('--fp_iter_max', type=int, default=None)
parser.add_argument('--fp_tol', type=float, default=1e-5)

def check_exist(idx, args):
    suffix = ['q', 'k', 'v', 'o', 'fc1', 'fc2', 'layernorm']
    for _ in suffix:
        test = f'{args.save_path}/{idx}_{_}.pt'
        if not os.path.exists(test):
            return False
    return True


class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def compress_clip_decoder(layer, idx, comp_model, q_level,  args, device, pre_orig_emb,
                          orig_emb, model_config, skip_list):
    if check_exist(idx, args):
        return

    if skip_list is None:
        skip_list = []
        
    ql_i =  q_level[idx] if q_level is not None else None

    # CLIP 모델에 맞춘 양자화 대상
    quant_order = []
    for thing in [('self_attn.v_proj', 'v', 'qkv', 'v', 'col'),
                  ('self_attn.q_proj', 'q', 'qkv', 'q', 'col'),
                  ('self_attn.k_proj', 'k', 'qkv', 'k', 'col'),
                  ('self_attn.out_proj', 'o', 'o', 'o', 'row'),
                  ('mlp.fc1', 'fc1', 'fc1', 'fc1', 'col'),
                  ('mlp.fc2', 'fc2', 'fc2', 'fc2', 'row')]:
        name = f'{idx}_{thing[1]}'
        if name not in skip_list:
            quant_order.append(thing)
        else:
            # 해당 weight 업데이트 제외
            attrgetter(thing[0])(layer).weight.requires_grad = False
            print(f'skipping {name}')

    # 양자화 및 파인튜닝
    # finetune_clip.compress_finetune_decoder_layer_clip(layer, quant_order, idx, comp_model, ql_i, args,
    #                                          device, pre_orig_emb, orig_emb)
    finetune.compress_finetune_decoder_layer(layer, quant_order, idx, comp_model, ql_i, args,
                                             device, pre_orig_emb, orig_emb)

    # 레이어 노름 저장 (layer_norm1, layer_norm2)
    norm_dict = {}
    if hasattr(layer, 'layer_norm1'):
        norm_dict['layer_norm1'] = layer.layer_norm1.weight
    if hasattr(layer, 'layer_norm2'):
        norm_dict['layer_norm2'] = layer.layer_norm2.weight

    torch.save(norm_dict, f'{args.save_path}/{idx}_layernorm.pt')


def main(args):
    if args.skip_list is not None:
        args.skip_list = args.skip_list.split(',')
        
    dtype_ = torch.float64 if args.use_fp64 else torch.float32

    # model = CLIPModel.from_pretrained(args.base_model,
    model = AutoModel.from_pretrained(args.base_model,
                                                 torch_dtype='auto',
                                                 low_cpu_mem_usage=True,
                                                 local_files_only=True,)

    # save configs
    all_config = {'quant_args': args, 'model_config': model.config}
    torch.save(all_config, os.path.join(args.save_path, 'config.pt'))

    all_config = {'quant_args': vars(args), 'model_config': model.config.to_dict()}
    with open(os.path.join(args.save_path, 'config.json'), 'w') as f:
        json.dump(all_config, f, indent=4)

    glog.info('loaded model')

    ## load comp model
    config = os.path.join(os.path.dirname(args.comp_model_path), 'config.json')
    with open(config, 'r', encoding='utf-8') as file:
        config = json.load(file)
    config = Config(**config)
    
    shift, scale = None, None
    if config.architecture == 'nwc_ql' and not hasattr(config, "Q"):
        config.Q = 4
    comp_model = get_model(config.architecture, config, scale=scale, shift=shift)      
    ckpt = torch.load(args.comp_model_path, weights_only=False)
    
    if (args.use_train_scale or args.layerwise_cdt or args.layer_normalize \
            or args.row_normalize or args.col_normalize or args.scaleH):
        try:
            scale = ckpt["state_dict"]["scale"]
            shift = ckpt["state_dict"]["shift"]
            print('Use train scale and shift')
        except:
            scale, shift  = torch.zeros(1), torch.zeros(1)
    else:
        if 'scale' in ckpt["state_dict"]:
            del ckpt["state_dict"]['scale']
        if 'shift' in ckpt["state_dict"]:
            del ckpt["state_dict"]['shift']
        shift, scale = utils.get_model_weight_stats(model, args, config.input_size)
    print(shift, scale)

    comp_model.load_state_dict(ckpt["state_dict"], strict = False)
    comp_model.scale = scale
    comp_model.shift = shift
    print(comp_model.scale, comp_model.shift)
    
    q_level = None
    if args.ql_path is not None:
        assert args.direction == 'col'
        q_level = torch.load(args.ql_path, weights_only=False)
    glog.info('loaded compression model')
    nproc = torch.cuda.device_count()

    def quantize_layer_block(layer_list, prefix):
        proc_list = [None for _ in range(nproc)]
        cur_device = 0
        for i in range(len(layer_list)):
            glog.info(f'{prefix} layer {i} gpu {cur_device}')
            if proc_list[cur_device] is not None:
                proc_list[cur_device][0].join()
                layer_list[proc_list[cur_device][1]] = None
                utils.clean()

            if cur_device + 1 < nproc and proc_list[cur_device + 1] is not None:
                proc_list[cur_device + 1][0].join()

            utils.clean()
            st = time.time()

            layer_list[i].to(cur_device)
            layer_list[i].eval()  # ensure eval mode

            layer_list[i].cpu()
            utils.clean()
            glog.info(f'computed original embedding for {prefix} layer {i} in {time.time() - st:.2f}s')

            proc_list[cur_device] = (mp.Process(target=compress_clip_decoder,  # reuse same fn
                                                args=(
                                                    layer_list[i],
                                                    f'{prefix}_{i}',  # for logging
                                                    comp_model,
                                                    q_level,
                                                    args,
                                                    cur_device,
                                                    None,  # placeholder for emb input
                                                    None,  # placeholder for emb output
                                                    all_config['model_config'],
                                                    args.skip_list
                                                )), i)
            proc_list[cur_device][0].start()
            cur_device = (cur_device + 1) % nproc

        for p in proc_list:
            if p is not None:
                p[0].join()
    
    quantize_layer_block(model.vision_model.encoder.layers, 'vision')
    quantize_layer_block(model.text_model.encoder.layers, 'text')

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    mp.set_start_method('spawn')
    mp.set_sharing_strategy('file_system')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.save_path, exist_ok=True)
    main(args)
