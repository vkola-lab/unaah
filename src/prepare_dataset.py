'''
Functions are used to prepare train, val, and test datasets and creating corresponding masks.
'''
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt
import sys
import os
import cv2
from skimage import data, filters
from skimage.segmentation import flood, flood_fill
from PIL import Image as im

def consectutive(data, stepsize=8):
    '''
    a helper function for creating binary mask
    find consecutive indexes in the matrix
    '''
    return np.split(data, np.where(np.diff(data) >= stepsize)[0]+1)

def create_binary_mask(inpath, outpath):
    '''
    inpath: path of images with segmentation annotation
    outpath: path of images with binary masks
    '''
    for img in os.listdir(inpath):
        image = plt.imread(os.path.join(inpath, img))
        image = image[0:499,0:499]
        plt.imshow(image)
        plt.show()
        mask = image[:,:,0]-image[:,:,1]
        mask[mask > 230] = 0 # to reduce red inconsistence noise and find the pure red circle
        mask[mask < 50] = 0
        result = mask.copy()
        # create binary mask
        seed = (0,0)
        count = 0
        for i in range(result.shape[0]):
            if np.sum(result[i]) > 0:
                count += 1
                if count > 30:
                    row = np.nonzero(result[i])[0]
                    if row.shape[0] > 1:
                        groups = consecutive(row)
                        if len(groups) > 1:
                            seed = (i,groups[-1][0]-10)
                            break
        result = flood_fill(result, seed, 255)
        result = result-mask
        result[result < 255] = 0
        result = result / 255
        fn = img.split('.')[0][10:]+'.npy'
        out = os.path.join(outpath, fn)
        with open(out,'wb') as f:
            np.save(f,result)
        
if __name__ == '__main__':
    for case in ['1','2','3']:
        inpath = os.path.join('./Annotated/',case)
        outpath = os.path.join('./mask/', case)
        create_binary_mask(inpath,outpath)
     
    # get patient accessions for split
    imgs = os.listdir('./raw')
    patients = ['2653397','5030769','3338282','3032919','3369519','3971349','2908522','3005470','2814484'] #25 images each
    temp = ['2927340'] # this dataset has the most images, it has to be in the training dataset, 234 images 
    # initialize lists
    train = []
    val = []
    test = []
    while len(train)+len(val)+len(test) != 10:
        tmp = random.choices(patients, k = 5)
        train = temp + tmp
        val = random.choices(list(set(patients)-set(tmp)), k = 2)
        test = list(set(patients)-set(tmp)-set(val))
    train_imgs = []
    val_imgs = []
    test_imgs = []
    for img in imgs:
        base = img.split('.')[0]
        acc = base.split('_')[2]
        if acc in train:
            train_imgs.append(base)
        elif acc in val:
            val_imgs.append(base)
        else:
            test_imgs.append(base)
    print('patients in train set: ')
    for p in train:
        print(p)
    print('total number of imgs: ', str(len(train_imgs)))
    train_out = open('train.txt','w')
    for i in train_imgs:
        train_out.write(i+'\n')
    train_out.close()
    
    print('patients in val set: ')
    for p in val:
        print(p)
    print('total number of imgs: ', str(len(val_imgs)))
    val_out = open('val.txt','w')
    for i in val_imgs:
        val_out.write(i+'\n')
    val_out.close()
    
    print('patients in test set: ')
    for p in test:
        print(p)
    print('total number of imgs: ', str(len(test_imgs)))
    test_out = open('test.txt','w')
    for i in test_imgs:
        test_out.write(i+'\n')
    test_out.close()
    
