import argparse
import json
import math
import os
import random

import datasets
import glog
import torch
from tqdm import tqdm

torch.set_grad_enabled(False)

import torch
import transformers
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, CLIPModel, AutoModel, SiglipModel
from tqdm import tqdm
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--hf_path', default='None', type=str)
# parser.add_argument('--model_id', default='openai/clip-vit-large-patch14', type=str)
parser.add_argument("--output_path", default=None, type=str)


# OpenAI CLIP 코드 참고해서 짜봄 
# 출처: https://github.com/openai/CLIP/blob/dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1/notebooks/Prompt_Engineering_for_ImageNet.ipynb
imagenet_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]

class ImageNet_valid_dataset(Dataset):
    def __init__(self, imagenet_class_index, transform, text_template_idx = 0,
                 image_folder_path = '/home/minkyu4506/NIC_fairness_ICCV_use_freq_mask/imagenet_validset'):
        
        self.image_folder_path = image_folder_path
        
        self.text_template = imagenet_templates[text_template_idx]
        
        self.image_path_list = []
        self.label_idx_list = []
        
        for cls_idx in range(1000):
            class_id = imagenet_class_index[str(cls_idx)][0]
            # class_name = imagenet_class_index[str(cls_idx)][1]

            image_name_list = os.listdir(f'{image_folder_path}/{class_id}')
            
            for image_name in image_name_list:
                
                self.image_path_list.append(f'{image_folder_path}/{class_id}/{image_name}')
                self.label_idx_list.append(cls_idx)
        
        self.transform = transform
    
    def __len__(self): 
        return len(self.image_path_list)
    
    def __getitem__(self, idx): 
        
        image_path = self.image_path_list[idx]
        
        image = Image.open(image_path).convert('RGB')
        
        # text = self.text_template.format(self.class_name_list[idx])
        
        input_data = self.transform(images=image, return_tensors="pt", padding=True)
        
        return input_data['pixel_values'].squeeze(0), self.label_idx_list[idx]

def main(args):  
    model = SiglipModel.from_pretrained(args.hf_path, force_download=False).to('cuda')
    processor = AutoProcessor.from_pretrained(args.hf_path, force_download=False)

    print("--- STARTING DEBUG MODE (Hugging Face Official Method) ---")

    # 1. 테스트할 이미지 경로와 텍스트 목록을 지정합니다.
    debug_image_path = "/data/ILSVRC2012/val/n01443537/ILSVRC2012_val_000006216.JPEG"
    texts = ["goldfish", "cat", "dog", "car", "computer"]
    texts = [f"a photo of {t}" for t in texts]

    # 2. 이미지를 불러옵니다.
    try:
        image = Image.open(debug_image_path).convert("RGB")
        print(f"✅ Successfully loaded image: {debug_image_path}")
    except FileNotFoundError:
        print(f"❌ ERROR: Debug image not found at {debug_image_path}.")
        print(f"   Please check if the file exists and update the path.")
        return # 이미지를 찾을 수 없으면 종료

    # 3. 공식 문서 방식대로 이미지와 텍스트를 함께 전처리합니다.
    inputs = processor(text=texts, images=image, padding="max_length", return_tensors="pt").to("cuda")
    print("✅ Successfully preprocessed image and texts together.")

    # 4. 모델의 forward pass를 통해 logits을 직접 얻습니다.
    with torch.no_grad():
        outputs = model(**inputs)

    # logits_per_image는 이미지와 각 텍스트 간의 유사도 점수(내적 값)입니다.
    logits_per_image = outputs.logits_per_image
    
    # SigLIP의 점수는 sigmoid 함수를 통과시켜 확률처럼 해석할 수 있습니다.
    probs = torch.sigmoid(logits_per_image.squeeze())

    # 5. 결과를 출력합니다.
    print("\n--- 📊 DEBUGGING RESULTS ---")
    print("Scores for the image vs. text prompts:")
    for i, text in enumerate(texts):
        logit = logits_per_image.squeeze()[i].item()
        prob = probs[i].item()
        print(f"- {text:<10}: Logit={logit:>8.4f}, Probability={prob:>7.2%}")

    print("\n👉 Expected result: 'goldfish' should have the highest Logit and a Probability close to 100%.")
    print("--- END OF DEBUGGING MODE ---")

    return # 디버깅 후 메인 함수 종료

    # =================================================================
    # =============== 여기서 디버깅 코드 종료 ===============
    # (기존의 데이터 로딩 및 평가 루프 코드는 이 아래에 위치)
    # =================================================================

    with open('/workspace/Weight_compression/qtip/eval/imagenet_class_index.json', 'r') as f:
        imagenet_class_index = json.load(f)    
    
    imagenet = ImageNet_valid_dataset(
        imagenet_class_index,
        processor,
        image_folder_path = '/data/ILSVRC2012/val'
    )
    
    imagenet_loader = DataLoader(
        imagenet,
        batch_size=256,
        num_workers=2,
        pin_memory=True,
        shuffle=False,
        drop_last=False
    )
    
    # SigLIP을 위한 텍스트 임베딩 생성 (템플릿 및 정규화 제거)
    with torch.no_grad():
        zeroshot_weights = []
        print("Generating text embeddings for 1000 classes...")
        for cls_idx in tqdm(range(1000)):
            class_name = imagenet_class_index[str(cls_idx)][1]
            
            # 템플릿 없이 클래스 이름만 사용
            texts_input = processor(text=[class_name], return_tensors="pt", padding=True).to('cuda')
            
            # 정규화 과정 없이 텍스트 특징 추출
            texts_embeds = model.get_text_features(**texts_input)
            
            zeroshot_weights.append(texts_embeds.squeeze())
        
    zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to("cuda")
    
    # 평가 루프
    top1, top5 = 0, 0
    total = 0

    print("Evaluating on ImageNet validation set...")
    for images, label_idx in tqdm(imagenet_loader):
    
        images = images.to('cuda')
        
        with torch.no_grad():
        
            # 이미지 특징 추출 (정규화 제거)
            image_features = model.get_image_features(images)
            
            # 정규화 없이 내적(dot product) 계산
            logits_per_text = torch.matmul(zeroshot_weights, image_features.t()) # [1000, b]
            
            top1_list = logits_per_text.topk(1, dim=0)[1]
            top5_list = logits_per_text.topk(5, dim=0)[1]
            
            for idx in range(images.size(0)):
                
                if label_idx[idx] in top1_list[:, idx]:
                    top1 += 1
                
                if label_idx[idx] in top5_list[:, idx]:
                    top5 += 1
                
        total += images.size(0)

    # 최종 결과 출력
    print(f"Top-1 Accuracy: {top1 / total * 100:.2f}%")
    print(f"Top-5 Accuracy: {top5 / total * 100:.2f}%")

    try:
        with open(f'{args.hf_path}_result.json', 'r') as f:
            comp_result= json.load(f)
    except:
        comp_result = {}
    comp_result['ppl'] = {'imagenet': (top1 / total, top5 / total)}
    os.makedirs(os.path.dirname(f'{args.output_path}_imagenet_result.json'), exist_ok = True)
    with open(f'{args.output_path}_imagenet_result.json', 'w') as f:
        json.dump(comp_result, f, indent=4)
        
if __name__ == '__main__':
    torch.set_grad_enabled(False)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    main(args)
