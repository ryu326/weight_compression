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
# LLaVA 모델 및 프로세서 import
from transformers import (AutoProcessor, LlavaForConditionalGeneration,
                          PreTrainedTokenizerFast)
from transformers.modeling_attn_mask_utils import \
    _prepare_4d_causal_attention_mask

from lib import utils
import pandas as pd
from datasets.arrow_dataset import Dataset as HFDataset
from PIL import Image, UnidentifiedImageError

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=1, type=int, help="Batch size per GPU. LLaVA is memory intensive, so a small value is recommended.")
parser.add_argument('--large_batch_size', default=256, type=int)
parser.add_argument('--devset_size', default=2048, type=int)
# LLaVA에 맞는 context size로 수정
parser.add_argument('--ctx_size', default=2048, type=int, help="Context size for the language model part.")
# LLaVA 모델 경로로 수정
parser.add_argument('--base_model',
                    default='llava-hf/llava-1.5-7b-hf',
                    type=str)
parser.add_argument('--save_path', default='hessians/llava-1.5-7b', type=str)
parser.add_argument('--sample_proc', default=32, type=int)

from datasets import load_dataset
from torch.utils.data import DataLoader
from datasets import Features, Value, Sequence, Image


def main(args):
    gpu_id = int(os.environ["LOCAL_RANK"])
    device = f"cuda:{gpu_id}"

    print("loading model and processor...")
    processor = AutoProcessor.from_pretrained(args.base_model)
    print("loaded model and processor!")
    COCO_BASE_PATH = "/data/COCO/coco_data/train2017" 

    df = pd.read_json("/workspace/hf_cache/huggingface_nwc/hub/datasets--liuhaotian--LLaVA-Instruct-150K/snapshots/9d451dc7629cfe0469f6ae4432b765cd603d5fcb/llava_v1_5_mix665k.json").drop(columns=["id", "model"])
    
    print(f"Original dataset size: {len(df)}")
    df = df[df['image'].str.startswith('coco', na=False)]
    print(f"Filtered dataset size (COCO only): {len(df)}")

    df.dropna(subset=['image'], inplace=True)
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    
    dataset = HFDataset.from_pandas(df, preserve_index=False)
    dataset = dataset.shuffle(seed=42).select(range(args.devset_size*4))
    
    def is_sample_valid(example):
        from PIL import Image, UnidentifiedImageError

        if 'conversations' not in example or not isinstance(example['conversations'], list):
            return False
        prompts = [conv['value'] for conv in example['conversations'] if conv.get('from') == 'human']
        if not prompts or not prompts[0]:  # 프롬프트가 없거나, 있어도 비어있는 경우
            return False

        try:
            full_image_path = os.path.join(COCO_BASE_PATH, example['image'].split('/')[-1])
            with Image.open(full_image_path) as img:
                img.load()
            return True
        except (FileNotFoundError, UnidentifiedImageError, OSError, TypeError, AttributeError):
            return False

    print("Verifying image files... (This may take a while)")
    verified_dataset = dataset.filter(is_sample_valid, num_proc=4) 
    print(f"Found {len(verified_dataset)} samples with valid and existing images.")

    if args.devset_size > len(verified_dataset):
        print(f"Warning: devset_size({args.devset_size}) is larger than the number of valid images({len(verified_dataset)}). Adjusting.")
        args.devset_size = len(verified_dataset)

    dataset = verified_dataset.shuffle(seed=42).select(range(args.devset_size))
    
    # ------------------------------------------------------------------
    def preprocess_function(examples):
            from PIL import Image
            full_image_path = os.path.join(COCO_BASE_PATH, str(examples['image'].split('/')[-1]))
            image = Image.open(full_image_path).convert("RGB")
            
            # [핵심 수정] 여러 개의 human 프롬프트 중 첫 번째 프롬프트만 선택
            human_prompts = [conv['value'] for conv in examples['conversations'] if conv['from'] == 'human']
            prompt = human_prompts[0] # is_sample_valid 필터에서 최소 1개는 보장됨
            
            # processor에 항상 단일 프롬프트를 전달하여 출력 구조의 일관성 보장
            inputs = processor(text=prompt, images=image, return_tensors="pt", padding="max_length", truncation=True, max_length=args.ctx_size)
            return {'pixel_values': inputs.pixel_values.squeeze(0), 'input_ids': inputs.input_ids.squeeze(0)}
    
    print(f"Applying final preprocessing to {len(dataset)} selected samples...")

    processed_dataset = dataset.map(preprocess_function, remove_columns=dataset.column_names)
    
    processed_dataset.set_format(type='torch', columns=['pixel_values', 'input_ids'])
    all_pixel_values = processed_dataset['pixel_values'][:]
    all_input_ids = processed_dataset['input_ids'][:]
    # all_pixel_values = torch.stack([item['pixel_values'] for item in processed_dataset])
    # all_input_ids = torch.stack([item['input_ids'] for item in processed_dataset])
    
    devset_images = torch.split(all_pixel_values, args.large_batch_size)
    devset_texts = torch.split(all_input_ids, args.large_batch_size)

    for lbi in range(len(devset_images)):
        model = LlavaForConditionalGeneration.from_pretrained(
            args.base_model,
            torch_dtype="auto",
            low_cpu_mem_usage=True
        )
        print(f'processing split {lbi}')

        # ==================================================================
        # Part 1: Vision Tower (CLIPVisionModel)
        # ==================================================================
        print("Part 1: Processing Vision Tower...")
        
        # Vision Tower의 입력은 이미지의 pixel_values
        vision_tower = model.vision_tower.to(device)
        pixel_values_split = devset_images[lbi].view(-1, args.batch_size, 3, 336, 336)
        
        # 초기 Embedding 계산
        hidden_states_list = []
        for i in range(len(pixel_values_split)):
            # CLIPVisionEmbeddings + pre_layrnorm
            initial_embeds = vision_tower.vision_model.embeddings(pixel_values_split[i].to(device))
            initial_embeds = vision_tower.vision_model.pre_layrnorm(initial_embeds)
            hidden_states_list.append(initial_embeds.cpu())
        
        vision_layer_index = 0
        while len(vision_tower.vision_model.encoder.layers) > 0:
            layer = vision_tower.vision_model.encoder.layers[0].to(device)
            save_pfx = f'/dev/shm/vision_{vision_layer_index}'
            
            # CLIPEncoderLayer 내부의 Linear 레이어들에 Hook 등록
            hook_fns = {
                'q': utils.register_input_H_hook(layer.self_attn.q_proj, f'{save_pfx}_q', gpu_id),
                'k': utils.register_input_H_hook(layer.self_attn.k_proj, f'{save_pfx}_k', gpu_id),
                'v': utils.register_input_H_hook(layer.self_attn.v_proj, f'{save_pfx}_v', gpu_id),
                'out': utils.register_input_H_hook(layer.self_attn.out_proj, f'{save_pfx}_out', gpu_id),
                'fc1': utils.register_input_H_hook(layer.mlp.fc1, f'{save_pfx}_fc1', gpu_id),
                'fc2': utils.register_input_H_hook(layer.mlp.fc2, f'{save_pfx}_fc2', gpu_id),
            }

            # 데이터 통과 및 다음 레이어 입력 준비
            for di in range(len(hidden_states_list)):
                # [핵심 수정] 누락된 attention_mask와 causal_attention_mask 인자를 None으로 전달합니다.
                layer_outputs = layer(
                    hidden_states=hidden_states_list[di].to(device),
                    attention_mask=None,
                    causal_attention_mask=None
                )
                hidden_states_list[di] = layer_outputs[0].cpu()
                utils.clean()
            
            # Hook 제거 및 결과 집계
            for key, fn in hook_fns.items():
                fn()
            
            dist.barrier()
            if gpu_id == 0:
                aggregate_and_save(args, f"vision_{vision_layer_index}", list(hook_fns.keys()))
            dist.barrier()

            layer = layer.cpu()
            del layer, vision_tower.vision_model.encoder.layers[0]
            utils.clean()
            print(f"Done processing vision tower layer {vision_layer_index}")
            vision_layer_index += 1
        
        # 마지막 LayerNorm 통과
        for di in range(len(hidden_states_list)):
            hidden_states_list[di] = vision_tower.vision_model.post_layernorm(hidden_states_list[di].to(device)).cpu()
        
        image_features = hidden_states_list
        del vision_tower, pixel_values_split
        utils.clean()

        # ==================================================================
        # Part 2: Multi-Modal Projector
        # ==================================================================
        print("\nPart 2: Processing Multi-Modal Projector...")
        projector = model.multi_modal_projector.to(device)
        save_pfx = '/dev/shm/projector'
        hook_fns = {
            'linear_1': utils.register_input_H_hook(projector.linear_1, f'{save_pfx}_linear_1', gpu_id),
            'linear_2': utils.register_input_H_hook(projector.linear_2, f'{save_pfx}_linear_2', gpu_id),
        }
        
        image_embeds = []
        for di in range(len(image_features)):
            image_embeds.append(projector(image_features[di].to(device)).cpu())
            utils.clean()

        for key, fn in hook_fns.items():
            fn()
        
        dist.barrier()
        if gpu_id == 0:
            aggregate_and_save(args, "projector", list(hook_fns.keys()))
        dist.barrier()
        
        del projector, image_features
        utils.clean()
        print("Done processing projector.")

        # ==================================================================
        # Part 3: Language Model (LlamaForCausalLM)
        # ==================================================================
        print("\nPart 3: Processing Language Model...")
        
        text_tokens_split = devset_texts[lbi].view(-1, args.batch_size, args.ctx_size)
        
        language_model_embed_layer = model.language_model.model.embed_tokens.to(device)
        text_embeds_list = [language_model_embed_layer(tokens.to(device)).cpu() for tokens in text_tokens_split]
        del language_model_embed_layer
        utils.clean()

        image_token_id = model.config.image_token_index

        final_input_embeds = []
        for img_emb, txt_emb, txt_tokens in zip(image_embeds, text_embeds_list, text_tokens_split):
            
            # 연산을 위해 현재 배치의 텐서들을 GPU로 보냅니다.
            # LLaVA 프로세서가 <image> 토큰을 여러 개로 확장했음을 가정합니다.
            txt_tokens_gpu = txt_tokens.to(device)
            txt_emb_gpu = txt_emb.to(device)
            img_emb_gpu = img_emb.to(device)

            special_image_mask = (txt_tokens_gpu == image_token_id).unsqueeze(-1).expand_as(txt_emb_gpu)

            img_emb_gpu = img_emb_gpu.to(txt_emb_gpu.device, txt_emb_gpu.dtype)

            combined_embeds = txt_emb_gpu.masked_scatter(special_image_mask, img_emb_gpu)
            
            final_input_embeds.append(combined_embeds.cpu())

        final_ctx_size = final_input_embeds[0].shape[1]
        position_ids = torch.arange(final_ctx_size, dtype=torch.long, device=device).unsqueeze(0).expand(args.batch_size, -1)
        attention_mask = _prepare_4d_causal_attention_mask(
            None, (args.batch_size, final_ctx_size), final_input_embeds[0], 0
        ).to(device)

        language_layer_index = 0
        while len(model.language_model.model.layers) > 0:
            layer = model.language_model.model.layers[0].to(device)
            save_pfx = f'/dev/shm/lang_{language_layer_index}'

            # LlamaDecoderLayer 내부의 Linear 레이어들에 Hook 등록
            hook_fns = {
                'q': utils.register_input_H_hook(layer.self_attn.q_proj, f'{save_pfx}_q', gpu_id),
                'k': utils.register_input_H_hook(layer.self_attn.k_proj, f'{save_pfx}_k', gpu_id),
                'v': utils.register_input_H_hook(layer.self_attn.v_proj, f'{save_pfx}_v', gpu_id),
                'o': utils.register_input_H_hook(layer.self_attn.o_proj, f'{save_pfx}_o', gpu_id),
                'gate': utils.register_input_H_hook(layer.mlp.gate_proj, f'{save_pfx}_gate', gpu_id),
                'up': utils.register_input_H_hook(layer.mlp.up_proj, f'{save_pfx}_up', gpu_id),
                'down': utils.register_input_H_hook(layer.mlp.down_proj, f'{save_pfx}_down', gpu_id),
            }

            for di in range(len(final_input_embeds)):
                final_input_embeds[di] = layer(
                    final_input_embeds[di].to(device),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False
                )[0].cpu()
                utils.clean()

            for key, fn in hook_fns.items():
                fn()

            dist.barrier()
            if gpu_id == 0:
                aggregate_and_save(args, f"lang_{language_layer_index}", list(hook_fns.keys()))
            dist.barrier()
            
            layer = layer.cpu()
            del layer, model.language_model.model.layers[0]
            utils.clean()
            print(f"Done processing language model layer {language_layer_index}")
            language_layer_index += 1

        del model, final_input_embeds, image_embeds, text_embeds_list
        utils.clean()

# 결과 집계 및 저장 로직을 함수로 분리하여 재사용성 높임
def aggregate_and_save(args, prefix, keys):
    """Aggregates Hessian data from all GPUs and saves it."""
    for key in keys:
        save_path = f"{args.save_path}/{prefix}_{key}.pt"
        if os.path.exists(save_path):
            data = torch.load(save_path, map_location=torch.device('cpu'))
            data['flatH'] = data['flatH'].to(torch.float64) * data['ct']
        else:
            data = None
        
        gi = 0
        gi_path = f"/dev/shm/{prefix}_{key}_{gi}.pt"
        while os.path.exists(gi_path):
            try:
                d2 = torch.load(gi_path, map_location=torch.device('cpu'))
                if data is not None:
                    data['flatH'] += utils.sym_to_flat(d2['H'])
                    data['ct'] += d2['ct']
                else:
                    data = d2
                    data['flatH'] = utils.sym_to_flat(data['H'])
                    del data['H']
                os.remove(gi_path)
            except (EOFError, RuntimeError) as e:
                print(f"Warning: Could not load or process {gi_path}. Skipping. Error: {e}")
            
            gi += 1
            gi_path = f"/dev/shm/{prefix}_{key}_{gi}.pt"
            
        if data and data['ct'] > 0:
            data['flatH'] /= data['ct']
            data['flatH'] = data['flatH'].float()
            torch.save(data, save_path)
        
        del data
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