# CAT-Net
This is the code of following paper "Few Shot Medical Image Segmentation with Cross Attention Transformer".
### Dependencies
```
dcm2nii
json5==0.8.5
jupyter==1.0.0
nibabel==2.5.1
numpy==1.22.0
opencv-python==4.5.5.62
Pillow>=8.1.1
sacred==0.8.2
scikit-image==0.18.3
SimpleITK==1.2.3
torch==1.10.2
torchvision=0.11.2
tqdm==4.62.3
```

### Datasets and pre-processing

Download:  
1. **Abdominal MRI CT**  [Synapse Multi-atlas Abdominal Segmentation dataset](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789)
2. **Abdominal MRI**  [Combined Healthy Abdominal Organ Segmentation dataset](https://chaos.grand-challenge.org/)  
3. **Cardiac MRI** [Multi-sequence Cardiac MRI Segmentation dataset (bSSFP fold)](http://www.sdspeople.fudan.edu.cn/zhuangxiahai/0/mscmrseg/)  

**Pre-processing** is according to [Ouyang et al.](https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation.git) and we follow their pre-processing pipeline. Please refer to [Ouyang et al.](https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation.git) for details.


| Model/Modification       | Hard Match Precision | Hard Match Recall | Hard Match F1 | Soft Match Precision | Soft Match Recall | Soft Match F1 |
|--------------------------|----------------------|-------------------|---------------|----------------------|-------------------|---------------|
| ReactionDataExtractor    | 4.1                  | 1.3               | 1.9           | 19.4                 | 5.9               | 9.0           |
| OChemR                   | 4.4                  | 2.8               | 3.4           | 12.4                 | 7.9               | 9.6           |
| RxnScribe                | 72.3                 | 66.2              | 69.1          | 83.8                 | 76.5              | 80.0          |
| - No pretraining         | 66.4                 | 59.4              | 62.7          | 80.4                 | 71.3              | 75.5          |
| - No compositional augmentation | 67.1         | 60.7              | 63.8          | 78.2                 | 70.2              | 74.0          |
| - Random reaction order  | 72.0                 | 64.2              | 67.9          | 83.9                 | 74.3              | 78.8          |
| - No postprocessing      | 70.8                 | 66.0              | 68.3          | 82.1                 | 76.4              | 79.1          |
| Ours                     | 59.24                | 60.32             | 59.78         | 74.30                | 74.96             | 74.63         |
| new                      | 60.02                | 61.56             | 60.29        | 74.68                | 75.35             | 75.01         |


### Training  
1. Download pre-trained [ResNet-101 weights](https://download.pytorch.org/models/resnet101-63fe2227.pth) and put into your own backbone folder.
2. Run `./exps/train_Abd.sh` or `./exps/train_CMR.sh`

### Testing
Run `./exp/validation.sh`

### Acknowledgement
This code is based on [Q-Net](https://github.com/zjlab-ammi/q-net), [PFENet](https://github.com/dvlab-research/PFENet)
