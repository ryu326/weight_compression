import argparse
import datetime
import os
import random
from copy import deepcopy

from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import numpy
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import CLIPModel, CLIPProcessor
# CLIP은 causal mask가 필요 없으므로 text의 경우도 일반적인 attention_mask를 사용합니다.
from lib import utils

# from huggingface_hub import login
# login(token = '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')

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=2, type=int,
                    help="Transformer layer마다 처리할 미니배치 사이즈 (vision/text 각각)")
parser.add_argument('--large_batch_size', default=512, type=int,
                    help="전체 샘플 배치 (sample_concat 함수로부터 리턴되는 샘플 수)")
parser.add_argument('--devset_size', default=8192, type=int,
                    help="전체 데이터셋 샘플 수")
# text의 경우 최대 토큰 수, vision은 processor에서 결정됨 (CLIP은 정해진 해상도)
parser.add_argument('--ctx_size', default=77, type=int,
                    help="텍스트 모달리티의 최대 토큰 길이 (CLIP 기본값: 77)")
parser.add_argument('--base_model',
                    default='openai/clip-vit-base-patch32',
                    type=str)
parser.add_argument('--save_path', default='hessians/clip', type=str)
parser.add_argument('--sample_proc', default=32, type=int,
                    help="데이터 샘플 추출에 사용할 프로세스 수")
parser.add_argument('--seed', default=0, type=int)


def main(args):
    gpu_id = int(os.environ["LOCAL_RANK"])
    
    print("loading CLIP model and processor...")
    model = CLIPModel.from_pretrained(args.base_model, torch_dtype="auto", low_cpu_mem_usage=True)
    processor = CLIPProcessor.from_pretrained(args.base_model)
    model.eval()  # 평가 모드로 전환

    # ----------------------------------------------------------------------------
    # 데이터셋 로드
    # utils.sample_clip_concat 함수는 LAION 등에서 이미지-텍스트 샘플을 추출하여
    # [{"image": PIL.Image, "text": str}, ...] 형태로 반환한다고 가정합니다.
    print("loading dataset...")
    # devset = utils.sample_clip_concat(args.devset_size, nproc=args.sample_proc)
    devset = utils.sample_cc12m_concat(args.devset_size)
    
    # 전체 샘플을 args.large_batch_size 단위로 분할
    devset = torch.split(torch.tensor(range(len(devset))), args.large_batch_size)
    # 전체 샘플 리스트는 따로 보관 (여기서는 인덱스로 사용)
    all_samples = devset  # 실제 이미지/텍스트 정보는 utils.sample_clip_concat() 내부에 저장되어 있다고 가정

    # ----------------------------------------------------------------------------
    # Vision 모달리티 처리: 이미지 전처리 및 임베딩 추출
    # utils.sample_clip_concat에서 추출한 이미지와 텍스트 리스트를 분리합니다.
    clip_samples = utils.sample_clip_concat(args.devset_size, nproc=args.sample_proc)
    images = [s["image"] for s in clip_samples]
    texts  = [s["text"]  for s in clip_samples]

    # 전처리: 이미지는 processor를 통해 pixel_values로 변환
    vision_inputs = processor(images=images, return_tensors="pt")
    # vision embedding: CLIP vision 모듈의 embeddings를 통과
    image_emb_full = model.vision_model.embeddings(vision_inputs["pixel_values"])
    # image_emb_full: [N, seq_len, hidden_dim]
    # 배치 단위로 나누기 (args.batch_size)
    dev_image_emb = list(torch.split(image_emb_full, args.batch_size))

    # vision의 경우, 별도의 position_ids는 필요하지 않으므로 attention mask는 전체 1로 설정
    # (만약 transformer layer 내부에서 mask 필요시 생성)
    dummy_attn_mask = torch.ones((args.batch_size, image_emb_full.size(1)), dtype=torch.long).cuda()

    # ----------------------------------------------------------------------------
    # Text 모달리티 처리: 텍스트 전처리 및 임베딩 추출
    text_inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=args.ctx_size)
    # text embedding: CLIP text 모듈의 embeddings (내부에서 position embedding 포함)
    text_emb_full = model.text_model.embeddings(input_ids=text_inputs["input_ids"])
    dev_text_emb = list(torch.split(text_emb_full, args.batch_size))
    text_attn_mask = text_inputs["attention_mask"].cuda()

    # ----------------------------------------------------------------------------
    # Vision Transformer Layers에 대해 후킹 및 Hessian 데이터 추출
    transformer_layer_index = 0
    # model.vision_model.encoder.layers 는 CLIP ViT의 transformer layer 리스트입니다.
    while len(model.vision_model.encoder.layers) > 0:
        print(gpu_id, "processing vision layer", transformer_layer_index)
        layer = model.vision_model.encoder.layers[0]
        layer = layer.cuda()
        save_pfx = f'/dev/shm/vision_{transformer_layer_index}'
        # 각 layer 내부 리니어 계층에 대해 후킹 (예: self_attn의 q_proj, o_proj와 MLP의 두 linear)
        done_qkv = utils.register_input_H_hook(layer.self_attn.q_proj,
                                               f'{save_pfx}_qkv', gpu_id)
        done_o = utils.register_input_H_hook(layer.self_attn.out_proj,
                                             f'{save_pfx}_o', gpu_id)
        # CLIP ViT의 MLP: Hugging Face 구현에서는 보통 mlp.fc1, mlp.fc2로 명명되어 있음
        done_fc1 = utils.register_input_H_hook(layer.mlp.fc1,
                                               f'{save_pfx}_fc1', gpu_id)
        done_fc2 = utils.register_input_H_hook(layer.mlp.fc2,
                                               f'{save_pfx}_fc2', gpu_id)
        # forward 실행: 각 배치 단위로 처리
        for di in range(len(dev_image_emb)):
            # import ipdb; ipdb.set_trace()
            tmp_input = dev_image_emb[di].cuda()
            # vision transformer layer의 forward는 (hidden_states, attention_mask)를 받습니다.
            out = layer(tmp_input, causal_attention_mask=None, attention_mask =None)[0].cpu()
            dev_image_emb[di] = out
            tmp_input.cpu()
            del tmp_input
            utils.clean()
        layer = layer.cpu()
        del layer, model.vision_model.encoder.layers[0]
        utils.clean()
        fn_dict = {
            'qkv': done_qkv,
            'o': done_o,
            'fc1': done_fc1,
            'fc2': done_fc2
        }
        for key in fn_dict:
            fn_dict[key]()
            utils.clean()
        # dist.barrier()
        if gpu_id == 0:
            for key in fn_dict:
                save_file = f"{args.save_path}/vision_{transformer_layer_index}_{key}.pt"
                if os.path.exists(save_file):
                    data = torch.load(save_file, map_location=torch.device('cpu'))
                    data['flatH'] = data['flatH'].to(torch.float64) * data['ct']
                else:
                    data = None
                gi = 0
                gi_path = f"/dev/shm/vision_{transformer_layer_index}_{key}_{gi}.pt"
                while os.path.exists(gi_path):
                    print("Combining", gi_path)
                    d2 = torch.load(gi_path, map_location=torch.device('cpu'))
                    if data is not None:
                        data['flatH'] += utils.sym_to_flat(d2['H'])
                        data['ct'] += d2['ct']
                        del d2
                        utils.clean()
                    else:
                        data = d2
                        data['flatH'] = utils.sym_to_flat(data['H'])
                        del data['H']
                    os.remove(gi_path)
                    gi += 1
                    gi_path = f"/dev/shm/vision_{transformer_layer_index}_{key}_{gi}.pt"
                data['flatH'] /= data['ct']
                data['flatH'] = data['flatH'].float()
                torch.save(data, save_file)
                del data
                utils.clean()
        # dist.barrier()
        print(f"done processing vision layer {transformer_layer_index}")
        transformer_layer_index += 1

    # ----------------------------------------------------------------------------
    # Text Transformer Layers에 대해 후킹 및 Hessian 데이터 추출
    transformer_layer_index = 0
    while len(model.text_model.encoder.layers) > 0:
        print(gpu_id, "processing text layer", transformer_layer_index)
        layer = model.text_model.encoder.layers[0]
        layer = layer.cuda()
        save_pfx = f'/dev/shm/text_{transformer_layer_index}'
        done_qkv = utils.register_input_H_hook(layer.self_attn.q_proj,
                                               f'{save_pfx}_qkv', gpu_id)
        done_o = utils.register_input_H_hook(layer.self_attn.out_proj,
                                             f'{save_pfx}_o', gpu_id)
        done_fc1 = utils.register_input_H_hook(layer.mlp.fc1,
                                               f'{save_pfx}_fc1', gpu_id)
        done_fc2 = utils.register_input_H_hook(layer.mlp.fc2,
                                               f'{save_pfx}_fc2', gpu_id)
        for di in range(len(dev_text_emb)):
            tmp_input = dev_text_emb[di].cuda()
            # text transformer layer forward: (hidden_states, attention_mask)
            print(torch.all(text_attn_mask == 1).item())
            out = layer(tmp_input, causal_attention_mask=None, attention_mask=None)[0].cpu()
            dev_text_emb[di] = out
            tmp_input.cpu()
            del tmp_input
            utils.clean()
        layer = layer.cpu()
        del layer, model.text_model.encoder.layers[0]
        utils.clean()
        fn_dict = {
            'qkv': done_qkv,
            'o': done_o,
            'fc1': done_fc1,
            'fc2': done_fc2
        }
        for key in fn_dict:
            fn_dict[key]()
            utils.clean()
        # dist.barrier()
        if gpu_id == 0:
            for key in fn_dict:
                save_file = f"{args.save_path}/text_{transformer_layer_index}_{key}.pt"
                if os.path.exists(save_file):
                    data = torch.load(save_file, map_location=torch.device('cpu'))
                    data['flatH'] = data['flatH'].to(torch.float64) * data['ct']
                else:
                    data = None
                gi = 0
                gi_path = f"/dev/shm/text_{transformer_layer_index}_{key}_{gi}.pt"
                while os.path.exists(gi_path):
                    print("Combining", gi_path)
                    d2 = torch.load(gi_path, map_location=torch.device('cpu'))
                    if data is not None:
                        data['flatH'] += utils.sym_to_flat(d2['H'])
                        data['ct'] += d2['ct']
                        del d2
                        utils.clean()
                    else:
                        data = d2
                        data['flatH'] = utils.sym_to_flat(data['H'])
                        del data['H']
                    os.remove(gi_path)
                    gi += 1
                    gi_path = f"/dev/shm/text_{transformer_layer_index}_{key}_{gi}.pt"
                data['flatH'] /= data['ct']
                data['flatH'] = data['flatH'].float()
                torch.save(data, save_file)
                del data
                utils.clean()
        # dist.barrier()
        print(f"done processing text layer {transformer_layer_index}")
        transformer_layer_index += 1

    # ----------------------------------------------------------------------------
    # 정리
    del dev_image_emb, dev_text_emb
    utils.clean()
    del model
    utils.clean()


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    # dist.init_process_group(backend="nccl")
    # gpu_id = int(os.environ["LOCAL_RANK"])
    # device = f"cuda:{gpu_id}"
    # torch.cuda.set_device(device)
    # torch.manual_seed(gpu_id)
    # random.seed(gpu_id)
    # numpy.random.seed(gpu_id)
    # main(args)
    # dist.destroy_process_group()


    os.environ["LOCAL_RANK"] = '0'
    main(args)
