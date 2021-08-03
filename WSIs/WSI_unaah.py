dataname="WSI"
ignore_index = -100
gpuid= 0

# --- unet params
n_classes= 2
backbone = 'resnet50'
alpha = 0.5
beta = 0.5 # hyper parameter for loss

# --- training params
patch_size=224
batch_size=16
num_epochs = 30
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

import PIL
from PIL import Image
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
    def __init__(self, fname, img_transform=None, mask_transform = None, edge_weight= False):
        
        self.fname=fname
        self.edge_weight = edge_weight
        
        self.img_transform=img_transform
        self.mask_transform = mask_transform
        
        infile = open(self.fname,'r')
        self.img_lines = infile.readlines()
        self.img_lines = [line.strip() for line in self.img_lines]
        infile.close()
        self.nitems = len(self.img_lines)
        
    def __getitem__(self, index):
        img = Image.open(self.img_lines[index])
        path = self.img_lines[index].split('/')
        path[0] = '/'
        basename = path[-3]
        path[-2] = basename+'_annotations'
        mask1 = plt.imread(os.path.join(*path))
        path[-2] = '2_Roelofs'
        mask2 = plt.imread(os.path.join(*path))
        
        mask1 = mask1[:,:,1]-mask1[:,:,0]
        mask2 = mask2[:,:,1]-mask2[:,:,0]
        
        mask1 = mask1.astype('uint8')
        mask2 = mask2.astype('uint8')
        
        mask1 = mask1[:,:,None].repeat(3,axis=2)
        mask2 = mask2[:,:,None].repeat(3,axis=2)
        
        seed = random.randrange(sys.maxsize) 
        if self.img_transform is not None:
            random.seed(seed) 
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


img_transform = transforms.Compose([
    #transforms.ToPILImage(),
    transforms.Resize((patch_size,patch_size)),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(size=(patch_size,patch_size),pad_if_needed=True),
    transforms.RandomResizedCrop(size=patch_size),
    transforms.RandomRotation(180),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0),
    transforms.RandomGrayscale(),
    transforms.ToTensor()
    ])


mask_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((patch_size,patch_size)),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(size=(patch_size,patch_size),pad_if_needed=True),
    transforms.RandomResizedCrop(size=patch_size,interpolation=PIL.Image.NEAREST),
    transforms.RandomRotation(180),
    ])


for k in range(5):
    model = UNaah(backbone_name=backbone,pretrained=True,classes=n_classes).to(device)
    print(f"total params: \t{sum([np.prod(p.size()) for p in model.parameters()])}")
    dataset={}
    dataLoader={}
    for phase in phases: 
        dataset[phase]=Dataset(f"/home/eikthedragonslayer/Desktop/CVS_data/kidney/{phase}_{k}.txt", img_transform=img_transform, mask_transform = mask_transform)
        dataLoader[phase]=DataLoader(dataset[phase], batch_size=batch_size, 
                                    shuffle=True, num_workers=16, pin_memory=True) 

    optim = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = FocalLoss()

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

            for ii , (X, y1, y2) in enumerate(dataLoader[phase]):
                X = X.to(device) 
                y1 = y1.type('torch.LongTensor').to(device)
                y2 = y2.type('torch.LongTensor').to(device)
                #gt_dice = dice_coefficient(y1.cpu().numpy(), y2.cpu().numpy())
                #weight = gt_dice / 2
                with torch.set_grad_enabled(phase == 'train'): 
                                                               
                    p1,p2 = model(X)  # [N, Nclass, H, W]
                    loss_matrix1 = criterion((p1+p2), y1)
                    loss_matrix2 = criterion((p1+p2), y2)
                    loss1 = loss_matrix1.mean()
                    loss2 = loss_matrix2.mean()
                    loss = weight*loss1 + (1-weight)*loss2
                    if phase=="train": #back propogation
                        optim.zero_grad()
                        loss.backward()
                        optim.step()
                        train_loss = loss

                    all_loss[phase]=torch.cat((all_loss[phase],loss.detach().view(1,-1)))

                    if phase in validation_phases: 
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


            torch.save(state, f"../save/{dataname}_{k}_unaah_{backbone}_best_model.pt")
        else:
            print("")




