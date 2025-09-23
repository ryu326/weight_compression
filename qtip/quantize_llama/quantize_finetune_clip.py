import argparse
import os
import time

import glog

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

import torch
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPModel, AutoModel
from transformers.modeling_attn_mask_utils import \
    _prepare_4d_causal_attention_mask

from lib import utils
from lib.algo import finetune_clip
from lib.codebook import bitshift
from operator import attrgetter

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_cpu_threads', default=8, type=int)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--devset_size', default=384, type=int)
parser.add_argument('--ctx_size', default=4096, type=int)
parser.add_argument('--save_path', type=str)
parser.add_argument('--in_hess_path', type=str)
parser.add_argument('--base_model', type=str)
parser.add_argument('--sigma_reg', default=1e-2, type=float)
parser.add_argument('--sigma_reg2', default=1e-2, type=float)
parser.add_argument('--scale_override', default=-1, type=float)
parser.add_argument('--codebook', type=str)
parser.add_argument('--use_fp64', action='store_true')
parser.add_argument('--no_use_buffered', action='store_true')
parser.add_argument('--sample_proc', default=1, type=int)
parser.add_argument('--lowmem_ldlq', action='store_true')
parser.add_argument('--ft_lr', default=3e-6, type=float)
parser.add_argument('--ft_bs', default=4, type=int)
parser.add_argument('--ft_update_freq', default=1, type=int)
parser.add_argument('--ft_epochs', default=5, type=int)
parser.add_argument('--ft_valid_freq', default=1, type=int)
parser.add_argument('--ft_valid_size', default=128, type=float)
parser.add_argument('--ft_early_stop', default=5, type=int)
parser.add_argument('--ft_grad_ckpt', action='store_true')
parser.add_argument('--td_x', default=16, type=int)
parser.add_argument('--td_y', default=16, type=int)
parser.add_argument('--L', default=16, type=int)
parser.add_argument('--K', default=2, type=int)
parser.add_argument('--V', default=2, type=int)
parser.add_argument('--tlut_bits', default=0, type=int)
parser.add_argument('--decode_mode', default='lut', type=str)
parser.add_argument('--ft_train_lut', action='store_true')
parser.add_argument('--split_for_tp', action='store_true')
parser.add_argument('--tp_rank', default=8, type=int)
parser.add_argument('--skip_list', default=None, type=str)


def check_exist(idx, args):
    suffix = ['q', 'k', 'v', 'o', 'fc1', 'fc2', 'layernorm']
    for _ in suffix:
        test = f'{args.save_path}/{idx}_{_}.pt'
        if not os.path.exists(test):
            return False
    return True


def quantize_clip_decoder(layer, idx, cb, args, device, pre_orig_emb,
                          orig_emb, model_config, skip_list):
    if check_exist(idx, args):
        return

    if skip_list is None:
        skip_list = []

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
    finetune_clip.quantize_finetune_decoder_layer_clip(layer, quant_order, idx, cb, args,
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

    cb = bitshift.bitshift_codebook(L=args.L,
                                    K=args.K,
                                    V=args.V,
                                    tlut_bits=args.tlut_bits,
                                    decode_mode=args.decode_mode)
    model = AutoModel.from_pretrained(args.base_model,
                                      torch_dtype='auto',
                                      low_cpu_mem_usage=True,)


    # save configs
    all_config = {'quant_args': args, 'model_config': model.config}
    quip_params = {
        'codebook': args.codebook,
        'codebook_version': cb.version,
        'L': args.L,
        'K': args.K,
        'V': args.V,
        'tlut_bits': args.tlut_bits,
        'decode_mode': args.decode_mode,
        'td_x': args.td_x,
        'td_y': args.td_y,
        'split_for_tp': args.split_for_tp,
        'skip_list': args.skip_list,
    }
    all_config['model_config'].update({'quip_params': quip_params})
    torch.save(all_config, os.path.join(args.save_path, 'config.pt'))

    glog.info('loaded model')

    glog.info('loaded dataset and devset')
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

            proc_list[cur_device] = (mp.Process(target=quantize_clip_decoder,  # reuse same fn
                                                args=(
                                                    layer_list[i],
                                                    f'{prefix}_{i}',  # for logging
                                                    cb,
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
