dataname="IVUS_single_2"
rawpath = '/home/eikthedragonslayer/Desktop/CVS_data/CVS/raw'
maskpath = '/home/eikthedragonslayer/Desktop/CVS_data/CVS/mask/2'
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
from models.networks.backboned_unet import Unet
from models.utils import asMinutes, timeSince
import os

import PIL
import cv2

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

if(torch.cuda.is_available()):
    print(torch.cuda.get_device_properties(gpuid))
    torch.cuda.set_device(gpuid)
    device = torch.device(f'cuda:{gpuid}')
else:
    device = torch.device(f'cpu')


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
        img = plt.imread(os.path.join(self.img_path,self.img_lines[index]+'.jpeg'))
        mask1 = np.load(os.path.join(self.mask_path,self.img_lines[index]+'.npy'))
        mask1 = mask1.astype('uint8')
        
        mask1 = mask1[:,:,None].repeat(3,axis=2)
        
        seed = random.randrange(sys.maxsize) #get a random seed so that we can reproducibly do the transofrmations
        if self.img_transform is not None:
            random.seed(seed) # apply this seed to img transforms
            img_new = self.img_transform(img)

        if self.mask_transform is not None:
            random.seed(seed) 
            mask1_new = self.mask_transform(mask1)
            mask1_new = np.asarray(mask1_new)[:,:,0].squeeze()#[:,:,0:1]
            random.seed(seed)

        return img_new, mask1_new
    
    def __len__(self):
        return self.nitems

img_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((patch_size,patch_size)),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(size=(patch_size,patch_size),pad_if_needed=True), #these need to be in a reproducible order, first affine transforms and then color
    transforms.RandomResizedCrop(size=patch_size),
    transforms.RandomRotation(180),
    transforms.ColorJitter(brightness=0.6, contrast=0, saturation=0, hue=0),
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


for k in range(5): # cross_validation running times
    model = Unet(backbone_name=backbone,pretrained=True,classes=n_classes).to(device)
    print(f"total params: \t{sum([np.prod(p.size()) for p in model.parameters()])}")

    dataset={}
    dataLoader={}
    for phase in phases: 
        dataset[phase]=Dataset(f"/home/eikthedragonslayer/Desktop/CVS_data/CVS/{phase}_%s.txt"%k,rawpath,maskpath, img_transform=img_transform, mask_transform = mask_transform)
        dataLoader[phase]=DataLoader(dataset[phase], batch_size=batch_size, 
                                    shuffle=True, num_workers=16, pin_memory=True) 


    optim = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss(ignore_index = ignore_index, reduce = False)
    writer=SummaryWriter() 
    best_loss_on_test = np.Infinity
    start_time = time.time()
    for epoch in range(num_epochs): 
        all_acc = {key: 0 for key in phases} 
        all_loss = {key: torch.zeros(0).to(device) for key in phases}
        cmatrix = {key: np.zeros((2,2)) for key in phases}

        for phase in phases: 
            if phase == 'train':
                model.train()  
            else: 
                model.eval() 
                
            for ii , (X, y) in enumerate(dataLoader[phase]): 
                X = X.to(device)
                y = y.type('torch.LongTensor').to(device)
                with torch.set_grad_enabled(phase == 'train'):                                                             
                    prediction = model(X)
                    loss_matrix = criterion(prediction, y)
                    loss = loss_matrix.mean()
                
                    if phase=="train":
                        optim.zero_grad()
                        loss.backward()
                        optim.step()
                        train_loss = loss
                        


                    all_loss[phase]=torch.cat((all_loss[phase],loss.detach().view(1,-1)))

                    if phase in validation_phases: #if this phase is part of validation, compute confusion matrix
                        p=prediction[:,:,:,:].detach().cpu().numpy()
                        cpredflat=np.argmax(p,axis=1).flatten()
                        yflat=y.cpu().numpy().flatten()

                        cmatrix[phase]=cmatrix[phase]+confusion_matrix(yflat,cpredflat,labels=range(n_classes))

            all_acc[phase]=(cmatrix[phase]/cmatrix[phase].sum()).trace()
            all_loss[phase] = all_loss[phase].cpu().numpy().mean()
            
            #save metrics to tensorboard
            writer.add_scalar(f'{phase}/loss', all_loss[phase], epoch)
            if phase in validation_phases:
                writer.add_scalar(f'{phase}/acc', all_acc[phase], epoch)
            

        print('%s ([%d/%d] %d%%), train loss: %.4f test loss: %.4f' % (timeSince(start_time, (epoch+1) / num_epochs), 
                                                     epoch+1, num_epochs ,(epoch+1) / num_epochs * 100, all_loss["train"], all_loss["val"]),end="")    


        if all_loss["val"] < best_loss_on_test:
            best_loss_on_test = all_loss["val"]
            print("  **")
            state = {'epoch': epoch + 1,
             'model_dict': model.state_dict(),
             'optim_dict': optim.state_dict(),
             'best_loss_on_test': all_loss,
             'n_classes': n_classes,}


            torch.save(state, f"../save/{dataname}_{k}_unet_{backbone}_best_model.pt")
        else:
            print("")

