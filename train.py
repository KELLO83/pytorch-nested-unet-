import argparse
import os
from collections import OrderedDict
from glob import glob

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from albumentations.augmentations import transforms
import torchvision.transforms as T
from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
import logging
import albumentations as A

import archs
import losses
from metrics import iou_score
from utils import AverageMeter, str2bool
from dataset import Dataset , CustomDataset ,Dataset_min_max
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy

ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')


def parse_args():
    parser = argparse.ArgumentParser()
    
    #mean - std
    parser.add_argument('--mean' , default=0.535, type=int , help='Dataset Average')
    parser.add_argument('--std' , default=0.153 , type=int , help='Dataset Standarad Divation')
    
    #pretrained
    parser.add_argument('--pretrained' , default='weight/Unet/CRACKTREE200/model.pt' , help='pretrain path')

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--save-dir',default='weight/Unet/Anorm/CRACKTREE200',help='pytorch model save directory')
    
    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='UNet',
                        choices=ARCH_NAMES,
                        help='model architecture: ' +
                        ' | '.join(ARCH_NAMES) +
                        ' (default: NestedUNet)')
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=512, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=512, type=int,
                        help='image height')
    
    # loss
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                        ' | '.join(LOSS_NAMES) +
                        ' (default: BCEDiceLoss)')
    
    # dataset
    parser.add_argument('--dataset', default='CRACKTREE200_INPUT',
                        help='dataset name')
    parser.add_argument('--img_ext', default='jpg',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='bmp',
                        help='mask file extension')

    # optimizer
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD']) +
                        ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-2, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-2, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    # scheduler
    parser.add_argument('--scheduler', default='ReduceLROnPlateau',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-12, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=10, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')
    parser.add_argument('--num_workers', default=4, type=int)

    config = parser.parse_args()


    return config


def train(config, train_loader, model, criterion, optimizer):
    logging.info(("\n" + "%12s" * 3) % ("Epoch", "GPU Mem", "Loss"))
        
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for i , (input, target) in enumerate(train_loader):
        input = input.cuda()
        target = target.cuda()

        # compute output
        if config['deep_supervision']:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            iou = iou_score(outputs[-1], target)
        else:
            output = model(input)
            loss = criterion(output, target)
            iou = iou_score(output, target)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        
        mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
        pbar.set_description(("%12s" * 2 + "%12.4g") % (f"{i + 1} / {len(train_loader)}", mem, loss))
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])


def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target  in val_loader:
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou = iou_score(outputs[-1], target)
            else:
                output = model(input)
                loss = criterion(output, target)
                iou = iou_score(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])


def main():
    config = vars(parse_args())

    writer = SummaryWriter('log_dir')
    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])
    os.makedirs('models/%s' % config['name'], exist_ok=True)

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open('models/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)

    # define loss function (criterion)
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.__dict__[config['loss']]().cuda()

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    # model = archs.__dict__[config['arch']](config['num_classes'],
    #                                        config['input_channels'])
    model = archs.UNet(num_classes=1 , input_channels=3 )
    
    if config['pretrained']:
        print("=================pretrained call================")
        w_call = torch.load(config['pretrained'])
        # print("pretrained call :",w_call)
        model.load_state_dict(w_call['model'].state_dict())
        # optimizer.load_state_dict(w_call['optimizer'])
        # print("Learing Rate {%e}:".format(optimizer.param_groups[-1]['lr']))


    device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
    model = model.to(device)

    params = filter(lambda p: p.requires_grad, model.parameters())
    
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
        
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
        
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=True, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])
        
    elif config['scheduler'] == 'ConstantLR':   
        scheduler = None
        
    else:
        raise NotImplementedError

    img_ids = glob(os.path.join('inputs',config['dataset'],'images','*' +'.'+config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.1, random_state=41)

    train_dataset = Dataset_min_max(
        img_ids=train_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform= None)
    
    train_transform = Compose([
        A.RandomRotate90(),
        A.Flip(),
        A.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(mean=[config['mean']]*3,std=[config['std']]*3)])
    

    val_transform = Compose([
        A.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(mean=[config['mean']],std=[config['std']])])
    
    
    # iterator_dataset = iter(train_dataset)
    # try:
    #     while True:
    #         image, mask = next(iterator_dataset)
    #         print(f'Image shape: {image.shape}')
    #         print(f'Mask shape: {mask.shape}')
    # except StopIteration:
    #     print("Stop Iteration")
        
    val_dataset = Dataset_min_max(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)
    

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)
    
    for k , i in enumerate(train_loader):
        batch = i
        image_ = batch[0]
        label_ = batch[1]
        visualize_batch_cv2(image_ , label_)
        break

    print("======================DEBUG===========================")
    
    log = OrderedDict([
        
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('val_loss', []),
        ('val_iou', []),
    ])


    best_iou = 0
    trigger = 0
    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer)
        writer.add_scalar('loss/train',train_log['loss'] , epoch)
        
        writer.add_scalar('iou/train' , train_log['iou'] , epoch)
        
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion)
        writer.add_scalar('loss/val',val_log['loss'] , epoch)
        writer.add_scalar('iou/val', val_log['iou'] , epoch)
        
        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])
 
        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f Learing Rate %e'
              % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou'] , scheduler.get_last_lr()[0]))

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])

        pd.DataFrame(log).to_csv('models/%s/log.csv' %
                                 config['name'], index=False)

        trigger += 1

        ckpt = {
            'train_best_loss' : train_log['loss'],
            'val_best_loss' : val_log['loss'],
            'model' : deepcopy(model).half(),
            'optimizer' : optimizer.state_dict(),
            'mean' : config['mean'],
            'std' : config['std'],
        }
        
        if val_log['iou'] > best_iou:
            if not os.path.isdir(config['save_dir']):
                os.makedirs(config['save_dir'],exist_ok=True)
                
            torch.save(ckpt, '%s/model.pt' %
                       config['save_dir'])
        
            best_iou = val_log['iou']
            print("=> saved best model")
            trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()

import cv2
import numpy as np
def visualize_batch_cv2(images, masks):
    batch_size = images.size(0)
    for i in range(batch_size):
        image = images[i].permute(1, 2, 0).cpu().numpy() 
        image = (image * 255).astype(np.uint8) 
        mask = masks[i].permute(1,2,0).cpu().numpy()   
        mask = (mask * 255).astype(np.uint8)  
        
        #denormalize_image = denormalize(images[i] , mean=[0.535,0.535,0.535],std=[0.153,0.153,0.153]) # image shape 3 512 512

        image_named = "image"
        mask_named = "mask"
        #denormalize_named = "denormalize image"
        cv2.namedWindow(image_named)
        cv2.namedWindow(mask_named)
        #cv2.namedWindow(denormalize_named)
        cv2.moveWindow(image_named,700,3500)
        cv2.moveWindow(mask_named,1500,3500)
        #cv2.moveWindow(denormalize_named,2000,1500)        
        cv2.imshow(image_named, image)
        cv2.imshow(mask_named, mask)
        #cv2.imshow(denormalize_named , denormalize_image)
        cv2.waitKey()  
        cv2.destroyAllWindows()

def denormalize(tensor , mean , std):
    tensor_copy = tensor.clone() # 3 512 512
    
    red_tensor = tensor_copy[0, : , : ]
    green_tensor = tensor_copy[1 , : , : ]
    blue_tensor = tensor_copy[2 , : , : ]
    
    red_tensor = red_tensor *  std[0] + mean[0]
    green_tensor = green_tensor * std[1] + mean[1]
    blue_tensor = blue_tensor * std[2] + mean[2]
    
    res_tensor = torch.stack([red_tensor, green_tensor, blue_tensor], dim=0)

    res_tensor = (res_tensor * 255).clamp(0, 255).byte()
    
    print(f"tensor max range {torch.max(res_tensor)} min range{torch.min(res_tensor)}")
    res  = res_tensor.permute(1, 2, 0).cpu().numpy()
    
    
    return res
    
    
if __name__ == '__main__':
    main()
