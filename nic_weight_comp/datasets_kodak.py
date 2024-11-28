import os

from torch.utils.data import Dataset
import PIL.Image as Image
from torchvision import transforms

# 이미지 한 장당 가지고 있어야 하는 정보
# 이미지 전체를 표현하는 캡션 하나
# 이미지 속 각 지역을 표현하는 캡션 n개 (n = 1,2,..)

class Kodak_dataset(Dataset):
    def __init__(self, image_dataset_folder:str):

        self.image_dataset_folder = image_dataset_folder

        img_name_list = os.listdir(image_dataset_folder)
        img_name_list.sort()

        self.img_name_list = []

        for image_name in img_name_list:
            if '.png' in image_name:
                self.img_name_list.append(image_name)
        self.image_processor = transforms.ToTensor()
        
    def __len__(self): 
        return len(self.img_name_list)
    
    def load_image(self, image_path) :
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")

        return self.image_processor(image)

    def __getitem__(self, idx): 

        image_path = f'{self.image_dataset_folder}/{self.img_name_list[idx]}'
        image = self.load_image(image_path)
        
        return self.img_name_list[idx], image