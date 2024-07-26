import os

import cv2
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler , StandardScaler


class Dataset(torch.utils.data.Dataset):

    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
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
        
        img = cv2.imread(os.path.join(self.img_dir, img_id +'.' + self.img_ext),cv2.IMREAD_COLOR)
        mask = cv2.imread(os.path.join(self.mask_dir , img_id +'.'+self.mask_ext),cv2.IMREAD_GRAYSCALE)
        
        if img is None or mask is None:
            raise FileExistsError()
        
        
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32') / 255
        mask = np.expand_dims(mask , axis=0)
        
        if np.all( img == 0 ): # Is Image is 0?
            raise Exception(f"image All Elemnets is None")
        
        assert img.shape[1:] == mask.shape[1:], f"not same image shape: img shape {img.shape}, mask shape {mask.shape}"
        
        return img, mask 
    

from albumentations.core.composition import Compose
import albumentations as A
class Dataset_min_max(torch.utils.data.Dataset):
    """ Sklearn MIn max scaler"""
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = Compose([
            A.RandomRotate90(),
            A.Flip(),
            A.RandomCrop(height=512,width=512),
            A.Normalize( mean=[0.535]*3 ,std=[0.153]*3 )
        ])
        
        self.scaled = MinMaxScaler()
        self.clahe = cv2.createCLAHE(clipLimit=2.0 , tileGridSize=(8,8))
    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        
        img = cv2.imread(os.path.join(self.img_dir, img_id +'.' + self.img_ext),cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(self.mask_dir , img_id +'.'+self.mask_ext),cv2.IMREAD_GRAYSCALE)
        
        if img is None or mask is None:
            raise FileExistsError()
        
        
        if self.transform is not None:
            img = self.clahe.apply(img) # Histogram Equalization 
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
            img = self.__min_max_convert(img)

            
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32') / 255
        mask = np.expand_dims(mask , axis=0)
        
        if np.all( img == 0 ): 
            raise Exception(f"image All Elemnets is None")
        
        assert img.shape[1:] == mask.shape[1:], f"not same image shape: img shape {img.shape}, mask shape {mask.shape}"
        
        img = torch.Tensor(img)
        mask = torch.Tensor(mask)
        return img, mask 
    
    
    def __sobel_adative(self , image:np.array)-> np.array:
        dx = cv2.Sobel(image , cv2.CV_64F , 0 , 1 , ksize=3)
        dy = cv2.Sobel(image , cv2.CV_64F , 1 , 0 , ksize=3)
        img_mag = cv2.magnitude(dx , dy)
        img_mag = cv2.normalize(img_mag , None , 0 ,255 ,cv2.NORM_MINMAX)
        img_mag = np.uint8(img_mag)
        return img_mag
    
    def __min_max_convert(self,image:np.array):
        h,w,c = image.shape
        
        image_reshaped = image.reshape(-1,1)
        min_max_scaler = self.scaled
        image_scaled = min_max_scaler.fit_transform(image_reshaped)
        image_scaled = image_scaled.reshape(h,w,c)
        
        return image_scaled

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self , img_ids , img_dir , mask_dir , img_ext , mask_ext , num_classes ):
        super(CustomDataset , self).__init__()
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes


        
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self,idx):
        image_path = os.path.join(self.img_dir , self.img_ids[idx] + self.img_ext)
        mask_path = os.path.join(self.mask_dir , self.img_ids[idx] + self.mask_ext)
        
        
        image = Image.open(image_path)
        mask = Image.open(mask_path)

        image = self.__resize_and_pad(image)
        mask = self.__resize_and_pad(mask)

        return image , mask
        
        
        
    def __resize_and_pad(self , image : Image , size=(512, 512)):
        
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
        