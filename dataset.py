import os

import cv2
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import torchvision.transforms as T

class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.
        
        Note:
            Make sure to put the files as the following structure:
            <dataset name>
            ├── images
            |   ├── 0a7e06.jpg
            │   ├── 0aab0a.jpg
            │   ├── 0b1761.jpg
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                |
                ├── 1
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                ...
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        
        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))

        mask = []
        for i in range(self.num_classes):
            mask.append(cv2.imread(os.path.join(self.mask_dir, str(i),
                        img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
        mask = np.dstack(mask)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        
        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)
        
        return img, mask 
    


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self , img_ids , img_dir , mask_dir , img_ext , mask_ext , num_classes ):
        super(CustomDataset , self).__init__()
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = T.Compose([
            T.ToTensor()
        ])

        
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self,idx):
        image_path = os.path.join(self.img_dir , self.img_ids[idx] + self.img_ext)
        mask_path = os.path.join(self.mask_dir , self.img_ids[idx] + self.mask_ext)
        
        
        image = Image.open(image_path)
        mask = Image.open(mask_path)
        
        image = self.__resize_and_pad(image)
        mask = self.__resize_and_pad(mask)
        
        image = self.transform(image)
        mask = self.transform(mask)
        
        return image , mask
        
        
        
    def __resize_and_pad(self , image:Image , size=(512, 512)):
        
        image = np.array(image)
        
        h, w = image.shape[:2]
        scale = size[0] / max(h, w)
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized_image = cv2.resize(image, (new_w, new_h))
        
        delta_w = size[1] - new_w
        delta_h = size[0] - new_h
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        
        color = [0, 0, 0] 
        padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        
        return padded_image
        