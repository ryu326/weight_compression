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
# AutoModelForImageClassification를 사용하도록 변경
from transformers import AutoImageProcessor, AutoModelForImageClassification

# lib.utils는 제공된 코드에 따라 Hessian 계산을 위한 훅(hook) 등록 및 파일 처리 함수를 포함한다고 가정합니다.
from lib import utils

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=2, type=int,
                    help="Transformer layer마다 처리할 미니배치 사이즈")
parser.add_argument('--large_batch_size', default=512, type=int,
                    help="전체 샘플 배치 (데이터 로더로부터 리턴되는 샘플 수)")
parser.add_argument('--devset_size', default=8192, type=int,
                    help="Hessian 계산에 사용할 전체 데이터셋 샘플 수")
# ImageNet으로 fine-tuning된 모델을 기본값으로 설정
parser.add_argument('--base_model',
                    default='facebook/dinov2-base-imagenet1k-1-layer',
                    type=str)
parser.add_argument('--save_path', default='hessians/dinov2_cls', type=str)
parser.add_argument('--sample_proc', default=32, type=int,
                    help="데이터 샘플 추출에 사용할 프로세스 수")
parser.add_argument('--seed', default=0, type=int)


def main(args):
    gpu_id = int(os.environ["LOCAL_RANK"])
    
    print("loading DINOv2 For Image Classification model and processor...")
    # Dinov2ForImageClassification 모델을 로드
    model = AutoModelForImageClassification.from_pretrained(args.base_model, torch_dtype="auto", low_cpu_mem_usage=True)
    image_processor = AutoImageProcessor.from_pretrained(args.base_model)
    
    model.eval()  # 평가 모드로 전환
    
    print("loading dataset...")
    cc12m_samples = utils.sample_cc12m_concat(args.devset_size)
    images = [s["image"] for s in cc12m_samples]

    
    vision_inputs = image_processor(images=images, return_tensors="pt")
    
    image_emb_full = model.dinov2.embeddings(vision_inputs["pixel_values"])

    dev_image_emb = list(torch.split(image_emb_full, args.batch_size))

    transformer_layer_index = 0
    while len(model.dinov2.encoder.layer) > 0:
        print(gpu_id, "processing vision layer", transformer_layer_index)
        layer = model.dinov2.encoder.layer[0]
        layer = layer.cuda()
        save_pfx = f'/dev/shm/vision_{transformer_layer_index}'

        done_q = utils.register_input_H_hook(layer.attention.attention.query,
                                              f'{save_pfx}_q', gpu_id)
        done_k = utils.register_input_H_hook(layer.attention.attention.key,
                                              f'{save_pfx}_k', gpu_id)
        done_v = utils.register_input_H_hook(layer.attention.attention.value,
                                              f'{save_pfx}_v', gpu_id)
        done_o = utils.register_input_H_hook(layer.attention.output.dense,
                                             f'{save_pfx}_o', gpu_id)
        done_fc1 = utils.register_input_H_hook(layer.mlp.fc1,
                                               f'{save_pfx}_fc1', gpu_id)
        done_fc2 = utils.register_input_H_hook(layer.mlp.fc2,
                                               f'{save_pfx}_fc2', gpu_id)
                                               
        for di in range(len(dev_image_emb)):
            tmp_input = dev_image_emb[di].cuda()
            out = layer(tmp_input)[0].cpu()
            dev_image_emb[di] = out
            tmp_input.cpu()
            del tmp_input
            utils.clean()
            
        layer = layer.cpu()
        # 모델 구조 변경에 따라 .dinov2 추가
        del layer, model.dinov2.encoder.layer[0]
        utils.clean()

        fn_dict = {
            'q': done_q,
            'k': done_k,
            'v': done_v,
            'o': done_o,
            'fc1': done_fc1,
            'fc2': done_fc2
        }

        for key in fn_dict:
            fn_dict[key]()
            utils.clean()

        dist.barrier()
        
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
        
        dist.barrier()
        print(f"done processing vision layer {transformer_layer_index}")
        transformer_layer_index += 1

    # (선택 사항) 최종 classifier 레이어의 Hessian을 계산하려면 여기에 로직 추가 가능
    
    del dev_image_emb
    utils.clean()
    del model
    utils.clean()


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    dist.init_process_group(backend="nccl")
    gpu_id = int(os.environ["LOCAL_RANK"])
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)
    torch.manual_seed(gpu_id)
    random.seed(gpu_id)
    numpy.random.seed(gpu_id)

    main(args)

    dist.destroy_process_group()