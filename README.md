# U-Net-and-a-half: Convolutional network for biomedical image segmentation using multiple expert-driven annotations
This work is published in #TODO

## Introduction

This repo contains a PyTorch implementation of a deep learning framework that applies multiple experts' annotations in the learning stage and makes predciton with both accuracy and generalization. Our framework links a backboned encoder and 2 U-Net decoders where each decoder learns from a different annotation and updates paramaters back to the shared encoder. The structure is shown below. 

<img src="plots/ivus.png" width="425"/> 

The model was developed on two different datasets: whole scan images (WSIs) and intravascular ultrasound scans where each datasets involves 10 patients and during each model development, a 5-round cross-validation strategy was applied to ensure the reliablity.  
<img src="plots/glom1.png" width="600"/> 
<img src="plots/IVUS_dataset.jpg" width="395"/>

The performance of the final global prediction from each UNaah model was compared with 2 experts' annotations and along with original U-Net models trained using each expert's annotation only. Example results are shown below..

<img src="plots/WSI_unaah5.jpg" width="395"/>
<img src="plots/IVUS_result3.jpg" width="395"/>

Please refer to our paper for more details. 

## How to use

### Code dependencies
we recommend using conda/anaconda to manage your environments. All essential packages have been listed in the environment file.
```
conda env create -f unaah_environment.yml
```
Please note that the dependencies may require Python 3.6 or greater. It is recommemded to install and maintain all packages by using [`conda`](https://www.anaconda.com/) or [`pip`](https://pypi.org/project/pip/). For the installation of GPU accelerated PyTorch, additional effort may be required. Please check the official websites of [PyTorch](https://pytorch.org/get-started/locally/) and [CUDA](https://developer.nvidia.com/cuda-downloads) for detailed instructions.

These instructions will help you properly configure and use the tool.

### Data
Please contact us for requiring raw data due to the size of data and privacy.

### Preprocessing
* **Making Binary Masks** . 

    Clean bianry masks are critical in supervised learning. The pipeline of making binary masks is not restricted. If using our raw data, binary masks will be provided along with raw images and train, validation, test groups.


### Train, validate and test UNaah models 

```
python IVUS_unaah.py
python IVUS_single_unet.py 
python IVUS_making_plots.py
```
### You may need to move trainer files to the parent directory (the 'unaah' directory) and execute scripts.

Model weights and predicted raw scores on each subjects will be saved in:

```
./save/...
```

### You are free to modify and customize your own trainers based on your data type. Hyper-parameters used in IVUS trainers and WSI trainers fit its corresponding dataset only
