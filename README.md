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

recommend git clone only depth==1 to save space and cloning time (.git folder is too large, will clean .git later)
```
git clone https://github.com/vkola-lab/brain2020.git --depth 1
```

These instructions will help you properly configure and use the tool.

### Data
Please contact us for requiring raw data due to the size of data and privacy.

### Preprocessing
#### 1. preprocessing steps for FCN model:

* **step1: Linear registration using FSL FLIRT function** (need FSL to be installed). 

    We provided this bash pipeline (Data_Preprocess/registration.sh) to perform this step. To run the registration.sh on a single case:
    ```
    bash registation.sh folder_of_raw_nifti/ filename.nii output_folder_for_processed_data/
    ```
    To register all data in a folder, you can use the python script (Data_Preprocess/registration.py) in which calls the registration.sh.
    ```
    python registration.py folder_of_raw_data/ folder_for_processed_data/
    ```

* **step2: convert nifit into numpy and perform z-score voxel normalization** 

    "(scan-scan.mean())/scan.std()"        

* **step3: clip out the intensity outliers (voxel<-1 or voxel>2.5)** 

    "np.clip(scan, -1, 2.5)"   
    
    To run step 2 and 3 together:
    ```
    python intensity_normalization_and_clip.py folder_for_step1_outcomes/
    ```
    
* **step4: background removal** 
    
    Background signals outside the skull exist in the MRI. We set all background voxels with the same intensity (value=-1) to decrease the incluence of background signals. The general idea of doing background removal is using the Depth First Search with corners as starting points, then gradually filling out the searched background regions, until it reach outer bright sphere signals from skull fat. To run this step:
    
    ```
    python back_remove.py folder_for_prev_outcome_after_step123/ folder_for_final_output_of_step4/
    ```
    The background mask looks like below:
    
    <img src="plot/background_mask.jpg" width="400"/>


#### 2. processing step for post-analysis on regional correlation between neuropath outcome and FCN prediction:
    
  * We performed subcortical segmentation using FreeSurfer (need to be installed) on those 11 FHS cases where neuropath data is available. To do the subcortical segmentation, you need to firstly do "recon-all" step using the freesurfer and then run the bash script below to get the final outcome: 
      ```
      bash segment_combine_label.sh
      ``` 

### Code dependencies

The tool was developped based on the following packages:

1. PyTorch (1.1 or greater).
2. NumPy (1.16 or greater).
3. matplotlib (3.0.3 or greater)
4. tqdm (4.31 or greater).
5. FSL 

Please note that the dependencies may require Python 3.6 or greater. It is recommemded to install and maintain all packages by using [`conda`](https://www.anaconda.com/) or [`pip`](https://pypi.org/project/pip/). For the installation of GPU accelerated PyTorch, additional effort may be required. Please check the official websites of [PyTorch](https://pytorch.org/get-started/locally/) and [CUDA](https://developer.nvidia.com/cuda-downloads) for detailed instructions.

### Configuration file

The configuration file is a json file which allows you conveniently change hyperparameters of models used in this study. 

```json
{
    "repeat_time":              5,                 # how many times you want to do random data split between training and validation 
    "fcn":{
        "fil_num":              20,                # filter number of the first convolution layer in FCN
        "drop_rate":            0.5,
        "patch_size":           47,                # 47 has to be fixed, otherwise the FCN model has to change accordingly
        "batch_size":           10,
        "balanced":             1,                 # to solve data imbalance issue, we provdided two solution: set value to 0 (weighted cross entropy loss), set value to 1 (pytorch sampler samples data with probability according to the category)
        "Data_dir":             "/data_dir/ADNI/", # change the path according to you folder name
        "learning_rate":        0.0001,
        "train_epochs":         3000
    },
    "mlp_A": {
        "imbalan_ratio":        1.0,               # imbalanced weight in weighted corss entropy loss
        "fil_num":              100,               # first dense layer's output size 
        "drop_rate":            0.5,
        "batch_size":           8,
        "balanced":             0,
        "roi_threshold":        0.6,                
        "roi_count":            200,
        "choice":               "count",           # if choice == 'count', then select top #roi_count as ROI
                                                   # if choice == 'thres', then select value > roi_threshold as ROI
        "learning_rate":        0.01,
        "train_epochs":         300
    }, 
    
    ....
    ....
    
    "cnn": {
        "fil_num":              20,
        "drop_rate":            0.137,
        "batch_size":           2,
        "balanced":             0,
        "Data_dir":             "/data_dir/ADNI/",
        "learning_rate":        0.0001,
        "train_epochs":         200
    }
}
```

### Train, validate and test FCN and CNN models 

```
python main.py
```

In the main.py, run function 'fcn_main' will do number of repeat time indepentent FCN model training on random splitted data. Model performance is thus evaluated on all runs as mean +/- std. Disease probability maps will be automatically generated for each independent run in the following folders:

```
DPMs/fcn_exp0/
DPMs/fcn_exp1/
...
DPMs/fcn_expN/
```

Model weights and predicted raw scores on each subjects will be saved in:

```
ckeckpoint_dir/fcn_exp0/
ckeckpoint_dir/fcn_exp1/
...
ckeckpoint_dir/fcn_expN/
```

Similarly, run function 'cnn_main' will do number of repeat time indepentent CNN model training on random splitted data. Model performance is thus evaluated on all runs as mean +/- std. Results will be saved in the similiar way as FCN.  

### Train, validate and test MLP models 

```
python mlp_classifiers.py
```
Inside mlp_classifiers.py, various MLP models will be trained on different type of features, for more details, please see the comments in the script and refer to our paper. 
