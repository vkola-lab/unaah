#!/usr/bin/env python

dataname="IVUS"
rawpath = '/home/eikthedragonslayer/DATA/CVS_data/CVS/raw'
maskpath = '/home/eikthedragonslayer/DATA/CVS_data/CVS/mask'
ignore_index = -100
gpuid= 0

# --- unet params
n_classes= 2
backbone = 'resnet50'

# --- training params
patch_size=224
batch_size=16
num_epochs = 100
phases = ["train","val"]
validation_phases= ["val"]

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from skimage.transform import rescale, resize, downscale_local_mean
from models.networks.UNaah import UNaah
from models.loss import FocalLoss
from models.utils import iou, dice_coefficient, asMinutes, timeSince
import os
import matplotlib.pyplot as plt
import PIL
import cv2
import pyvips
import numpy as np
import sys, glob

from tensorboardX import SummaryWriter

import scipy.ndimage 

import time
import math

import random

from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings('ignore')

format_to_dtype = {
    'uchar': np.uint8,
    'char': np.int8,
    'ushort': np.uint16,
    'short': np.int16,
    'uint': np.uint32,
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
    'complex': np.complex64,
    'dpcomplex': np.complex128,
}


if(torch.cuda.is_available()):
    print(torch.cuda.get_device_properties(gpuid))
    torch.cuda.set_device(gpuid)
    device = torch.device(f'cuda:{gpuid}')
else:
    device = torch.device(f'cpu')

def read_this(image_file, gray_scale=False):
    image_src = cv2.imread(image_file)
    if gray_scale:
        image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)
    else:
        image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB)
    return image_src

def equalize_contrast(image_file, with_plot=False, gray_scale=False, bins=256):
    image_src = read_this(image_file=image_file, gray_scale=gray_scale)
    if not gray_scale:
        r_image = image_src[:, :, 0]
        g_image = image_src[:, :, 1]
        b_image = image_src[:, :, 2]

        r_image_eq = enhance_contrast(image_matrix=r_image)
        g_image_eq = enhance_contrast(image_matrix=g_image)
        b_image_eq = enhance_contrast(image_matrix=b_image)

        image_eq = np.dstack(tup=(r_image_eq, g_image_eq, b_image_eq))
        cmap_val = None
    else:
        image_eq = enhance_contrast(image_matrix=image_src)
        cmap_val = 'gray'

    if with_plot:
        fig = plt.figure(figsize=(10, 20))

        ax1 = fig.add_subplot(2, 2, 1)
        ax1.axis("off")
        ax1.title.set_text('Original')
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.axis("off")
        ax2.title.set_text("Equalized")

        ax1.imshow(image_src, cmap=cmap_val)
        ax2.imshow(image_eq, cmap=cmap_val)
        return True
    return image_eq

def enhance_contrast(image_matrix, bins=256):
    image_flattened = image_matrix.flatten()
    image_hist = np.zeros(bins)

    # frequency count of each pixel
    for pix in image_matrix:
        image_hist[pix] += 1

    # cummulative sum
    cum_sum = np.cumsum(image_hist)
    norm = (cum_sum - cum_sum.min()) * 255
    # normalization of the pixel values
    n_ = cum_sum.max() - cum_sum.min()
    uniform_norm = norm / n_
    uniform_norm = uniform_norm.astype('int')

    # flat histogram
    image_eq = uniform_norm[image_flattened]
    # reshaping the flattened matrix to its original shape
    image_eq = np.reshape(a=image_eq, newshape=image_matrix.shape)

    return image_eq


class Dataset(object):
    def __init__(self, fname, img_path, mask_path, img_transform=None, mask_transform = None, edge_weight= False):
        
        self.fname=fname
        self.mask_path = mask_path
        self.img_path = img_path
        self.edge_weight = edge_weight
        
        self.img_transform=img_transform
        self.mask_transform = mask_transform
        
        infile = open(self.fname,'r')
        self.img_lines = infile.readlines()
        self.img_lines = [line.strip() for line in self.img_lines]
        infile.close()
        self.nitems = len(self.img_lines)
        
    def __getitem__(self, index):
        #img = plt.imread(os.path.join(self.img_path,self.img_lines[index]+'.jpeg'))
        #img = pyvips.Image.new_from_file(os.path.join(self.img_path,self.img_lines[index]+'.jpeg'))
        #img = img.hist_equal()
        #np_3d = np.ndarray(buffer=img.write_to_memory(),
        #                   dtype=format_to_dtype[img.format],
        #                   shape=[img.height, img.width, img.bands])
        img = equalize_contrast(os.path.join(self.img_path,self.img_lines[index]+'.jpeg'))
        img = img.astype('uint8')
        mask1 = np.load(os.path.join(self.mask_path,'2',self.img_lines[index]+'.npy'))
        mask1 = mask1.astype('uint8')
        mask2 = np.load(os.path.join(self.mask_path,'4',self.img_lines[index]+'.npy'))
        mask2 = mask2.astype('uint8')
        
        mask1 = mask1[:,:,None].repeat(3,axis=2)
        mask2 = mask2[:,:,None].repeat(3,axis=2)
        
        seed = random.randrange(sys.maxsize) #get a random seed so that we can reproducibly do the transofrmations
        if self.img_transform is not None:
            random.seed(seed) # apply this seed to img transforms
            img_new = self.img_transform(img)

        if self.mask_transform is not None:
            random.seed(seed) 
            mask1_new = self.mask_transform(mask1)
            mask1_new = np.asarray(mask1_new)[:,:,0].squeeze()#[:,:,0:1]
            random.seed(seed)
            mask2_new = self.mask_transform(mask2)
            mask2_new = np.asarray(mask2_new)[:,:,0].squeeze()#[:,:,0:1]
            random.seed(seed)

        return img_new, mask1_new, mask2_new
    
    def __len__(self):
        return self.nitems


#note that since we need the transofrmations to be reproducible for both masks and images
#we do the spatial transformations first, and afterwards do any color augmentations
img_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((patch_size,patch_size)),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(size=(patch_size,patch_size),pad_if_needed=True), #these need to be in a reproducible order, first affine transforms and then color
    transforms.RandomResizedCrop(size=patch_size),
    transforms.RandomRotation(180),
    transforms.ColorJitter(brightness=0, contrast=0.3, saturation=0, hue=0),
    transforms.RandomGrayscale(),
    transforms.ToTensor()
    ])


mask_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((patch_size,patch_size)),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(size=(patch_size,patch_size),pad_if_needed=True), #these need to be in a reproducible order, first affine transforms and then color
    transforms.RandomResizedCrop(size=patch_size,interpolation=PIL.Image.NEAREST),
    transforms.RandomRotation(180),
    ])

for k in range(5):
    model = UNaah(backbone_name=backbone,pretrained=True,classes=n_classes).to(device)
    print(f"total params: \t{sum([np.prod(p.size()) for p in model.parameters()])}")
    dataset={}
    dataLoader={}
    for phase in phases: 
        dataset[phase]=Dataset(f"/home/eikthedragonslayer/DATA/CVS_data/CVS/{phase}_{k}.txt",rawpath,maskpath, img_transform=img_transform, mask_transform = mask_transform)
        dataLoader[phase]=DataLoader(dataset[phase], batch_size=batch_size, 
                                    shuffle=True, num_workers=16, pin_memory=True) 


    optim = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion1 = nn.functional.cross_entropy
    criterion2 = FocalLoss()
    writer=SummaryWriter()
    best_loss_on_test = np.Infinity
    start_time = time.time()
    for epoch in range(num_epochs):
        #zero out epoch based performance variables 
        all_acc = {key: 0 for key in phases} 
        all_loss = {key: torch.zeros(0).to(device) for key in phases}
        cmatrix = {key: np.zeros((2,2)) for key in phases}

        for phase in phases: #iterate through both training and validation states

            if phase == 'train':
                model.train()
            else:
                model.eval()

            for ii , (X, y1, y2) in enumerate(dataLoader[phase]): #for each of the batches
                X = X.to(device)  # [Nbatch, 3, H, W]
                y1 = y1.type('torch.LongTensor').to(device)  # [Nbatch, H, W] with class indices (0, 1)
                y2 = y2.type('torch.LongTensor').to(device)
                gt_dice = dice_coefficient(y1.cpu().numpy(), y2.cpu().numpy())
                weight = gt_dice / 2
                with torch.set_grad_enabled(phase == 'train'): 
                                                               
                    p1, p2 = model(X)  # [N, Nclass, H, W]
                    
                    loss_matrix1 = criterion1(p1+p2, y1)
                    loss_matrix2 = criterion1(p2+p1, y2)
                    loss1 = loss_matrix1.mean()
                    loss2 = loss_matrix2.mean()
                    #loss2 = criterion2(p2+p1, y2)
                    loss = weight * loss1 + (1-weight) * loss2
                    
                    if phase=="train": #back propogation
                        optim.zero_grad()
                        loss.backward()
                        optim.step()
                        train_loss = loss


                    all_loss[phase]=torch.cat((all_loss[phase],loss.detach().view(1,-1)))

                    if phase in validation_phases: #if this phase is part of validation, compute confusion matrix
                        p=(p1+p2)[:,:,:,:].detach().cpu().numpy()
                        cpredflat=np.argmax(p,axis=1).flatten()
                        yflat=y1.cpu().numpy().flatten()

                        cmatrix[phase]=cmatrix[phase]+confusion_matrix(yflat,cpredflat,labels=range(n_classes))

            all_acc[phase]=(cmatrix[phase]/cmatrix[phase].sum()).trace()
            all_loss[phase] = all_loss[phase].cpu().numpy().mean()
            
            #save metrics to tensorboard
            writer.add_scalar(f'{phase}/loss', all_loss[phase], epoch)
            if phase in validation_phases:
                writer.add_scalar(f'{phase}/acc', all_acc[phase], epoch)
            

        print('%s ([%d/%d] %d%%), train loss: %.4f test loss: %.4f' % (timeSince(start_time, (epoch+1) / num_epochs), 
                                                     epoch+1, num_epochs ,(epoch+1) / num_epochs * 100, all_loss["train"], all_loss["val"]),end="")    

        #if current loss is the best we've seen, save model state with all variables
        #necessary for recreation
        if all_loss["val"] < best_loss_on_test:
            best_loss_on_test = all_loss["val"]
            print("  **")
            state = {'epoch': epoch + 1,
             'model_dict': model.state_dict(),
             'optim_dict': optim.state_dict(),
             'best_loss_on_test': all_loss,
             'n_classes': n_classes,}


            torch.save(state, f"./save/{dataname}_{k}_unaah_{backbone}_L_model.pt")
        else:
            print("")



