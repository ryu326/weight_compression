import json

import torch
import torchvision
from einops import rearrange
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


class ImageNet_dataset(Dataset):
    def __init__(self, dataset_folder="/data/ILSVRC2012", split="train", image_size=(256, 256), slurm=False):

        if slurm == False:
            self.img_dir = f"{dataset_folder}/{split}"
        else:
            self.img_dir = dataset_folder

        if slurm == False:
            with open("./specific_images_list_imagenet_window_16.json", "r") as f:
                self.img_path_dict = json.load(f)
        else:  # 슬럼에서 돌리면
            with open("./specific_images_list_imagenet_window_16_slurm.json", "r") as f:
                self.img_path_dict = json.load(f)

        self.img_path_list = list(self.img_path_dict.keys())
        self.transform = transforms.Compose([transforms.CenterCrop(image_size), transforms.ToTensor()])

    def __len__(self):
        return len(self.img_path_list)

    def load_image(self, image_path):

        image = Image.open(self.img_dir + "/" + image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")

        return self.transform(image)

    def __getitem__(self, idx):
        img = self.load_image(self.img_path_list[idx])

        num_of_specific_patches = self.img_path_dict[self.img_path_list[idx]]

        return img, num_of_specific_patches


def inverse_fft(x_freq):
    x_freq = torch.fft.ifftshift(x_freq, dim=(-2, -1))

    # 이미지면 절대값을, latent면 실수부만 사용. 그렇게 해야 되더라. 신기.
    if x_freq.size(1) == 3:
        x_recon = torch.fft.ifftn(x_freq, dim=(-2, -1)).abs()
    else:
        x_recon = torch.fft.ifftn(x_freq, dim=(-2, -1)).real

    return x_recon


def get_high_freq(x, radius_denominator=8):
    x_freq = torch.fft.fftn(x, dim=(-2, -1))
    x_freq = torch.fft.fftshift(x_freq, dim=(-2, -1))

    B, C, H, W = x_freq.shape

    radius = (min(H, W) // 2) // radius_denominator

    mask = Image.new("RGB", (W, H))

    if W % 2 == 1:
        leftup_x = W // 2 - radius
        rightdown_x = W // 2 + radius
    else:
        leftup_x = W // 2 - radius - 1
        rightdown_x = W // 2 + radius

    if H % 2 == 1:
        leftup_y = H // 2 - radius
        rightdown_y = H // 2 + radius
    else:
        leftup_y = H // 2 - radius - 1
        rightdown_y = H // 2 + radius

    mask_draw = ImageDraw.Draw(mask)
    mask_draw.ellipse([leftup_x, leftup_y, rightdown_x, rightdown_y], fill=(255, 255, 255), outline="black")
    lpf = (
        torchvision.transforms.functional.pil_to_tensor(mask.convert("L"))[0].type(torch.float32) / 255.0
    )  # 0 아니면 1로

    lpf = lpf.view(1, 1, H, W).repeat(B, C, 1, 1)
    lpf = lpf.to(x.device)

    fft_x_high_freq = x_freq * (torch.ones_like(lpf) - lpf)

    return fft_x_high_freq


def separate_high_not_high(x, num_of_special_patch_list=None):

    patch_size = 16
    freq_masking_threshold = 0.1
    high_freq_decision_ratio_threshold = 0.5

    _, _, h, w = x.size()

    # 고주파 대역 추출
    fft_x_high_freq = get_high_freq(x)
    x_high_freq = inverse_fft(fft_x_high_freq)

    # 고주파 계수가 freq_threshold 이상이면 True, 아니면 False
    threshold_x_high_freq = (x_high_freq.mean(dim=1) > freq_masking_threshold).type(torch.float)

    # 평균 구한걸 접음
    threshold_x_high_freq = (
        threshold_x_high_freq.unfold(1, patch_size, patch_size)
        .unfold(2, patch_size, patch_size)
        .flatten(-2)
        .mean(dim=-1)
    )
    threshold_x_high_freq = threshold_x_high_freq.flatten(-2).contiguous()

    x_folding = rearrange(
        x,
        "b c (h1 h2) (w1 w2) -> b c h1 w1 h2 w2",
        h1=h // patch_size,
        w1=w // patch_size,
        h2=patch_size,
        w2=patch_size,
    ).contiguous()
    x_folding = rearrange(x_folding, "b c h1 w1 h2 w2 -> b c (h1 w1) h2 w2", h2=patch_size, w2=patch_size).contiguous()

    indices_per_image = []

    for i in range(x_folding.size(0)):

        topk_info = (threshold_x_high_freq[i]).topk(threshold_x_high_freq[i].size(-1), dim=-1)

        if num_of_special_patch_list == None:
            more_than_th = torch.nonzero(
                torch.where(topk_info.values > high_freq_decision_ratio_threshold, 1.0, 0.0)
            ).size(0)
        # 배치 단위로 돌리기 위함. 이미지 별 계수가 0.5 이상 넘어가는 패치 개수의 최소값을 씀. 말은 이렇게 어렵게 해놨지만 코드 보면 이해 갈거임.
        else:
            more_than_th = num_of_special_patch_list.min().item()

        topk_indices = topk_info.indices[:more_than_th]

        indices_per_image.append(topk_indices.tolist())

    return indices_per_image


train_dataset = ImageNet_dataset(dataset_folder="/data/ILSVRC2012", split="train", image_size=(256, 256), slurm=False)

dict_indices_per_image = {}

for idx, (img, num_of_specific_patches) in enumerate(tqdm(train_dataset)):

    indices_per_image = separate_high_not_high(img.unsqueeze(0))[0]

    dict_indices_per_image[train_dataset.img_path_list[idx]] = indices_per_image


with open("./imagenet_indices_per_image.json", "w") as f:
    json.dump(dict_indices_per_image, f, indent=4)
