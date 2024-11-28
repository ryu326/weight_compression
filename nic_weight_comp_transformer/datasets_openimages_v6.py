from torch.utils.data import Dataset
from torchvision import transforms

import PIL.Image as Image
import json, os

class Openimages_v6_dataset(Dataset):
    def __init__(self, dataset_folder='/public-dataset/open-images-v6/images', image_size = (256, 256)):
        
        self.img_dir = dataset_folder
            
        with open('./openimages_v6_indices_per_image.json', 'r') as f:
            self.img_path_dict = json.load(f)
                
        self.img_path_list = list(self.img_path_dict.keys())
        
        self.transform = transforms.Compose(
                [transforms.CenterCrop(image_size), transforms.ToTensor()])
            
    def __len__(self): 
        return len(self.img_path_list)

    def load_image(self, image_path) :
        
        image = Image.open(self.img_dir + '/' + image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")

        return self.transform(image)

    def __getitem__(self, idx): 
        img= self.load_image(self.img_path_list[idx])
        
        indices = self.img_path_dict[self.img_path_list[idx]]
        
        return img, indices
