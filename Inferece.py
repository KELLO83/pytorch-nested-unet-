import argparse
import os
from glob import glob

import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import albumentations as A
import archs
from dataset import  CustomDataset
from metrics import iou_score
from utils import AverageMeter
from losses import BCEDiceLoss
from metrics_evaluate import Eval_MODE 
from metrics import dice_coef
import copy
import random
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='models/data/CrackTree_IMAGE_NestedUNet_woDS/model.pth',
                        help='model name')
    parser.add_argument('--image-path' , default='CRKWH100_TEST', help="test data image path")
    parser.add_argument('--image-extension' , default='.png' , help="image extension")
    parser.add_argument('--mask-extension' , default='.bmp',help='mask extension')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    model = archs.__dict__['NestedUNet'](num_classes=1, input_channels=3, deep_supervision=False)
    model = model.cuda()


    img_ids = glob(os.path.join('inputs', args.image_path, 'images', '*.png'))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    _ , val_img_ids = train_test_split(img_ids, test_size=0.1, random_state=41)

    model.load_state_dict(torch.load(args.name))
    model.eval()



    val_dataset = CustomDataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs',args.image_path, 'images'),
        mask_dir=os.path.join('inputs',args.image_path, 'masks'),
        img_ext=args.image_extension,
        mask_ext=args.mask_extension,
        num_classes=1)
    
    # it = iter(val_dataset)
    # it_test = it.__next__()
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        drop_last=False)


    for i , target in enumerate(val_loader):
        batch = target
        if i==1:
            image_target = batch[0]
            mask_target = batch[1]
            #visualize_batch_cv2(image_target , mask_target)
            print("image shape :",image_target.shape)
            print("mask shape :",mask_target.shape)
        
        
    avg_meter = AverageMeter()
    e_mode = Eval_MODE()
    
    test_loss = e_mode(model , val_loader , device="cuda:0")
    print("Test dataset Loss : {} ".format(test_loss))
    
    tensor_image = image_target.detach().to("cuda:0")
    image = image_target.squeeze(dim=0).permute(1,2,0).detach().cpu().numpy()
    image = (image * 255).astype(np.uint8)
    mask = mask_target.squeeze(dim=0).permute(1,2,0).detach().cpu().numpy()
    
    with torch.no_grad():
        output = model(tensor_image)
        output = torch.sigmoid(output)
        print("output range : {} {}".format(torch.min(output), torch.max(output)))
        output_th = (output.squeeze(dim=0) > 0.45).float().permute(1,2,0).cpu().numpy()
        
        origin_win = 'origin'
        mask_win = 'mask'
        predict_win = 'predict'

        
        cv2.namedWindow(origin_win)
        cv2.namedWindow(mask_win)
        cv2.namedWindow(predict_win)
        cv2.moveWindow(origin_win , 500 , 1500)
        cv2.moveWindow(mask_win, 1500 , 1500)
        cv2.moveWindow(predict_win , 2000 , 1500)
        
        cv2.imshow(origin_win,image)
        cv2.imshow(mask_win , mask)
        cv2.imshow(predict_win , output_th)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    

    torch.cuda.empty_cache()

def visualize_batch_cv2(images, masks):
    batch_size = images.size(0)
    for i in range(batch_size):
        image = images[i].permute(1, 2, 0).cpu().numpy() 
        image = (image * 255).astype(np.uint8) 
        mask = masks[i].squeeze(0).cpu().numpy()  
        mask = (mask * 255).astype(np.uint8)  
        
        print("image shape " ,image.shape)
        print("mask shape " ,mask.shape)

        image_named = "image"
        mask_named = "mask"
        cv2.namedWindow(image_named)
        cv2.namedWindow(mask_named)
        cv2.moveWindow(image_named,1000,1000)
        cv2.moveWindow(mask_named,1500,1000)        
        cv2.imshow(image_named, image)
        cv2.imshow(mask_named, mask)
        cv2.waitKey()  
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
