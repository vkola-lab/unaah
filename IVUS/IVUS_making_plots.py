#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from skimage.transform import rescale, resize, downscale_local_mean
from models.networks.backboned_unet import Unet
from models.networks.UNaah import UNaah
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import ConnectionPatch
import cv2
import numpy as np
import sys, glob, os
import scipy.ndimage 
import time
import math
import warnings
from statistics import mean
from models.utils import dice_coefficient, single_mask_color_img, double_mask_color_img_v2, iou
warnings.filterwarnings('ignore')


# In[2]:


rawpath = '/home/eikthedragonslayer/Desktop/CVS_data/CVS/raw'
maskpath = '/home/eikthedragonslayer/Desktop/CVS_data/CVS/mask'
annopath = '/home/eikthedragonslayer/Desktop/CVS_data/CVS/Annotated'
dataname1="IVUS_single_2"
dataname2="IVUS_single_4"
dataname3="IVUS"
gpuid=0
batch_size=1
width = 224
height = 224
phases=["test"]
backbone = 'resnet50'
predpath = '/home/eikthedragonslayer/Desktop/CVS_data/CVS/pred/combined_figure'
if not os.path.isdir(predpath):
    os.mkdir(predpath)


if(torch.cuda.is_available()):
    print(torch.cuda.get_device_properties(gpuid))
    torch.cuda.set_device(gpuid)
    device = torch.device(f'cuda:{gpuid}')
else:
    device = torch.device(f'cpu')

    
class Dataset(object):
    def __init__(self, fname, img_path, mask_path, anno_path, img_transform=None, mask_transform = None):
        
        self.fname=fname
        self.mask_path = mask_path
        self.img_path = img_path
        self.anno_path = anno_path
        
        self.img_transform=img_transform
        self.mask_transform = mask_transform
        
        infile = open(self.fname,'r')
        self.img_lines = infile.readlines()
        self.img_lines = [line.strip() for line in self.img_lines]
        infile.close()
        self.nitems = len(self.img_lines)
        
    def __getitem__(self, index):
        img = plt.imread(os.path.join(self.img_path,self.img_lines[index]+'.jpeg'))
        mask1 = np.load(os.path.join(self.mask_path,'2',self.img_lines[index]+'.npy'))
        mask1 = mask1.astype('uint8')
        mask2 = np.load(os.path.join(self.mask_path,'4',self.img_lines[index]+'.npy'))
        mask2 = mask2.astype('uint8')
        anno2 = plt.imread(os.path.join(self.anno_path,'4',self.img_lines[index]+'.jpeg'))
        anno1 = plt.imread(os.path.join(self.anno_path,'2',self.img_lines[index]+'.jpeg'))
            
        mask1 = mask1[:,:,None].repeat(3,axis=2)
        mask2 = mask2[:,:,None].repeat(3,axis=2)

        if self.img_transform is not None:
            img_new = self.img_transform(img)

        if self.mask_transform is not None:
            mask1_new = self.mask_transform(mask1)
            mask1_new = np.asarray(mask1_new)[:,:,0]
            mask2_new = self.mask_transform(mask2)
            mask2_new = np.asarray(mask2_new)[:,:,0]


        return img_new, mask1_new, mask2_new, anno1, anno2, self.img_lines[index]
    
    def __len__(self):
        return self.nitems
    
img_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((height,width)),
    transforms.ToTensor(),
    ])


mask_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((height,width)),
    ])

for k in range(5):
    start_time = time.time()
    dataset={}
    dataLoader={}
    for phase in phases: 
        dataset[phase]=Dataset(f"/home/eikthedragonslayer/Desktop/CVS_data/CVS/{phase}_{k}.txt",rawpath,maskpath,annopath, img_transform=img_transform, mask_transform = mask_transform)
        dataLoader[phase]=DataLoader(dataset[phase], batch_size=batch_size, 
                                    shuffle=False, num_workers=8, pin_memory=True)

    checkpoint1 = torch.load(f"../save/{dataname1}_{k}_unet_{backbone}_best_model.pt")
    model1 = Unet(backbone_name=backbone,pretrained=True,classes=checkpoint1["n_classes"]).to(device)
    model1.load_state_dict(checkpoint1["model_dict"])

    checkpoint2 = torch.load(f"../save/{dataname2}_{k}_unet_{backbone}_best_model.pt")
    model2 = Unet(backbone_name=backbone,pretrained=True,classes=checkpoint2["n_classes"]).to(device)
    model2.load_state_dict(checkpoint2["model_dict"])

    checkpoint3 = torch.load(f"../save/{dataname3}_{k}_unaah_{backbone}_best_model.pt")
    model3 = UNaah(backbone_name=backbone,pretrained=True,classes=checkpoint3["n_classes"]).to(device)
    model3.load_state_dict(checkpoint3["model_dict"])

    model1.eval()
    model2.eval()
    model3.eval()
    dices11_overall = []
    dices12_overall = []
    dices21_overall = []
    dices22_overall = []
    dices31_overall = []
    dices32_overall = []
    ious11 = []
    ious12 = []
    ious21 = []
    ious22 = []
    ious31 = []
    ious32 = []



    for ii , (X, y1, y2,z1, z2,name) in enumerate(dataLoader["test"]):
        X = X.to(device)
        output_m1 = model1(X)
        output_m2 = model2(X)
        o1,o2 = model3(X)
        
        dice11 = dice_coefficient(y1.squeeze(), np.argmax(np.moveaxis(output_m1.detach().squeeze().cpu().numpy(),0,-1),axis=2))
        dices11_overall.append(dice11)
        dice12 = dice_coefficient(y2.squeeze(), np.argmax(np.moveaxis(output_m1.detach().squeeze().cpu().numpy(),0,-1),axis=2))
        dices12_overall.append(dice12)
        
        dice21 = dice_coefficient(y1.squeeze(), np.argmax(np.moveaxis(output_m2.detach().squeeze().cpu().numpy(),0,-1),axis=2))
        dices21_overall.append(dice21)
        dice22 = dice_coefficient(y2.squeeze(), np.argmax(np.moveaxis(output_m2.detach().squeeze().cpu().numpy(),0,-1),axis=2))
        dices22_overall.append(dice22)
        
        dice31 = dice_coefficient(y1.squeeze(), np.argmax(np.moveaxis((o1+o2).detach().squeeze().cpu().numpy(),0,-1),axis=2))
        dices31_overall.append(dice31)
        dice32 = dice_coefficient(y2.squeeze(), np.argmax(np.moveaxis((o1+o2).detach().squeeze().cpu().numpy(),0,-1),axis=2))
        dices32_overall.append(dice32)
        
        iou11 = iou(output_m1.detach().cpu(), y1, ignore_background=False)
        ious11.append(iou11)
        iou12 = iou(output_m1.detach().cpu(), y2, ignore_background=False)
        ious12.append(iou12)
        
        iou21 = iou(output_m2.detach().cpu(), y1, ignore_background=False)
        ious21.append(iou21)
        iou22 = iou(output_m2.detach().cpu(), y2, ignore_background=False)
        ious22.append(iou22)
        
        iou31 = iou((o1+o2).detach().cpu(), y1, ignore_background=False)
        ious31.append(iou31)
        iou32 = iou((o1+o2).detach().cpu(), y2, ignore_background=False)
        ious32.append(iou32)
        
        raw = np.moveaxis(X.detach().squeeze().cpu().numpy(),0,-1)
        gt1 = z1.squeeze()
        gt2 = z2.squeeze()
        pred1 = single_mask_color_img(raw, np.argmax(np.moveaxis(output_m1.detach().squeeze().cpu().numpy(),0,-1),axis=2), alpha = 0.001)
        pred2 = single_mask_color_img(raw, np.argmax(np.moveaxis(output_m2.detach().squeeze().cpu().numpy(),0,-1),axis=2), color = [0,255,255], alpha = 0.005)
        pred3 = single_mask_color_img(raw, np.argmax(np.moveaxis((o1+o2).detach().squeeze().cpu().numpy(),0,-1),axis=2), color = [162,0,37], alpha = 0.005)
        
        fig = plt.figure(figsize=(18,9),constrained_layout=True)
        gs = fig.add_gridspec(6, 3)
        f_ax1 = fig.add_subplot(gs[:, 0])
        f_ax1.axis('off')
        f_ax1.imshow(raw)
        f_ax1.set_title('Original Patch')
        f_ax2 = fig.add_subplot(gs[0:3, 1])
        f_ax2.axis('off')
        f_ax2.imshow(gt1)
        f_ax2.set_title('Annotation 1')
        f_ax3 = fig.add_subplot(gs[3:6, 1])
        f_ax3.axis('off')
        f_ax3.imshow(gt2)
        f_ax3.set_title('Annotation 2')
        f_ax4 = fig.add_subplot(gs[0:2, 2])
        f_ax4.axis('off')
        f_ax4.imshow(pred1)
        f_ax4.set_title('UNet 1 Prediction')
        f_ax5 = fig.add_subplot(gs[4:6, 2])
        f_ax5.axis('off')
        f_ax5.imshow(pred2)
        f_ax5.set_title('UNet 2 Prediction')
        f_ax6 = fig.add_subplot(gs[2:4, 2])
        f_ax6.axis('off')
        f_ax6.imshow(pred3)
        f_ax6.set_title('UNaah Prediction')
        plt.savefig(os.path.join(predpath, name[0]+'.png'))
        
    print(f'\nround {k}:\n')
    print('ave overall dice coeff output1 vs. Anno 1: {}'.format(np.sum(dices11_overall)/len(dices11_overall)))
    print('ave overall dice coeff output1 vs. Anno 2: {}'.format(np.sum(dices12_overall)/len(dices12_overall)))
    print('ave iou corresponding output1 vs. Anno 1: {}'.format(np.sum(ious11)/len(ious11)))
    print('ave iou corresponding output1 vs. Anno 2: {}'.format(np.sum(ious12)/len(ious12)))
    print('----------------------------------------------')
    print('ave overall dice coeff output2 vs. Anno 1: {}'.format(np.sum(dices21_overall)/len(dices21_overall)))
    print('ave overall dice coeff output2 vs. Anno 2: {}'.format(np.sum(dices22_overall)/len(dices22_overall)))
    print('ave iou corresponding output2 vs. Anno 1: {}'.format(np.sum(ious21)/len(ious21)))
    print('ave iou corresponding output2 vs. Anno 2: {}'.format(np.sum(ious22)/len(ious22)))
    print('----------------------------------------------')
    print('ave overall dice coeff output3 vs. Anno 1: {}'.format(np.sum(dices31_overall)/len(dices31_overall)))
    print('ave overall dice coeff output3 vs. Anno 2: {}'.format(np.sum(dices32_overall)/len(dices32_overall)))
    print('ave iou corresponding output3 vs. Anno 1: {}'.format(np.sum(ious31)/len(ious31)))
    print('ave iou corresponding output3 vs. Anno 2: {}'.format(np.sum(ious32)/len(ious32)))
    print('----------------------------------------------')
    print('Time used: ', str(time.time()-start_time))


