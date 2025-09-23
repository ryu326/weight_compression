import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForImageClassification
from tqdm import tqdm
import argparse
import json, os

def get_imagenet_dataloader(imagenet_val_dir, batch_size=32, num_workers=4):
    """ImageNet 검증 데이터셋을 위한 DataLoader를 생성합니다."""
    
    # DINOv2 모델이 학습된 방식과 동일한 전처리 파이프라인을 구성합니다.
    # 참조: https://huggingface.co/facebook/dinov2-large-imagenet1k-1-layer/blob/main/preprocessor_config.json
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    # torchvision.datasets.ImageNet을 사용하여 데이터셋 로드
    val_dataset = datasets.ImageNet(root=imagenet_val_dir, split='val', transform=val_transform)
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return val_loader

def evaluate(model, dataloader, device):
    """모델의 Top-1 및 Top-5 정확도를 평가합니다."""
    model.eval()  # 평가 모드
    
    top1_correct = 0
    top5_correct = 0
    total_samples = 0

    with torch.no_grad(): # 그래디언트 계산 비활성화
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            
            # 모델 추론
            outputs = model(images)
            logits = outputs.logits
            
            # Top-5 예측값 계산 (가장 확률이 높은 5개 클래스)
            _, top5_preds = torch.topk(logits, 5, dim=1)
            
            # 배치 내 총 샘플 수 업데이트
            total_samples += labels.size(0)
            
            # Top-1 정확도 계산
            top1_correct += (top5_preds[:, 0] == labels).sum().item()
            
            # Top-5 정확도 계산
            # labels.view(-1, 1).expand_as(top5_preds)는 각 정답 레이블을 5번 반복하여
            # top5 예측값 행렬과 비교할 수 있도록 형태를 맞추는 과정입니다.
            top5_correct += (top5_preds == labels.view(-1, 1).expand_as(top5_preds)).sum().item()

    top1_accuracy = 100 * top1_correct / total_samples
    top5_accuracy = 100 * top5_correct / total_samples
    
    return top1_accuracy, top5_accuracy

def main(args):
    # 장치 설정 (CUDA 사용 가능하면 GPU, 아니면 CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_name = args.hf_path
    # processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    model.to(device)
    
    # ImageNet 데이터로더 생성
    print("Loading ImageNet validation dataset...")
    val_loader = get_imagenet_dataloader(
        imagenet_val_dir=args.imagenet_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # 정확도 평가
    print("Starting evaluation...")
    top1_acc, top5_acc = evaluate(model, val_loader, device)
    
    print("\n--- Evaluation Results ---")
    print(f"Model: {model_name}")
    print(f"Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Top-5 Accuracy: {top5_acc:.2f}%")
    print("--------------------------")

    try:
        with open(f'{args.hf_path}_result.json', 'r') as f:
            comp_result= json.load(f)
    except:
        comp_result = {}
    comp_result['ppl'] = {'imagenet': (top1_acc, top5_acc)}
    os.makedirs(os.path.dirname(f'{args.output_path}_imagenet_result.json'), exist_ok = True)
    with open(f'{args.output_path}_imagenet_result.json', 'w') as f:
        json.dump(comp_result, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DINOv2 on ImageNet.")
    parser.add_argument('--imagenet_path', type=str, default='/data/ILSVRC2012',
                        help='Path to the ImageNet validation dataset directory.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for evaluation.')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loading workers.')
    parser.add_argument('--hf_path', type=str)
    parser.add_argument('--output_path', type=str)
    args = parser.parse_args()
    main(args)