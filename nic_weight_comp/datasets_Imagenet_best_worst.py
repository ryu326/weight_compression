import os

from torch.utils.data import Dataset
import PIL.Image as Image
from torchvision import transforms

# 이미지 한 장당 가지고 있어야 하는 정보
# 이미지 전체를 표현하는 캡션 하나
# 이미지 속 각 지역을 표현하는 캡션 n개 (n = 1,2,..)


class Imagenet_best_worst(Dataset):
    def __init__(self, best_group_folder: str, worst_group_folder: str):

        self.best_group_folder = best_group_folder
        self.worst_group_folder = worst_group_folder

        best_group_img_name_list = os.listdir(best_group_folder)
        best_group_img_name_list.sort()

        worst_group_img_name_list = os.listdir(worst_group_folder)
        worst_group_img_name_list.sort()

        self.best_group_img_name_list = []
        self.worst_group_img_name_list = []

        for image_name in best_group_img_name_list:
            if ".jpg" in image_name:
                self.best_group_img_name_list.append(image_name)

        for image_name in worst_group_img_name_list:
            if ".jpg" in image_name:
                self.worst_group_img_name_list.append(image_name)

        self.image_processor = transforms.ToTensor()

    def __len__(self):
        return len(self.best_group_img_name_list)

    def load_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")

        return self.image_processor(image)

    def __getitem__(self, idx):

        best_image_path = f"{self.best_group_folder}/{self.best_group_img_name_list[idx]}"
        best_image = self.load_image(best_image_path)

        worst_image_path = f"{self.worst_group_folder}/{self.worst_group_img_name_list[idx]}"
        worst_image = self.load_image(worst_image_path)

        return self.best_group_img_name_list[idx], best_image, self.worst_group_img_name_list[idx], worst_image
