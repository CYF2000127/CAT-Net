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
| RxnScribe(1333x1333)     | 72.3                 | 66.2              | 69.1          | 83.8                 | 76.5              | 80.0          |
|Baseline                | 60.02                | 61.56             | 60.29        | 73.68                | 74.35             | 74.01         |
| ChemRxnGPT (llama-7b)(336X336)      | 70.11                | 64.81             | 67.36      | 82.67                | 78.04           | 80.30         |
| ChemRxnGPT (llama-7b)(448X448)      | 71.33                | 66.21             | 68.67      | 83.24                | 79.42           | 81.29        |
| ChemRxnGPT (llama2-13b)(448X448)    | 71.84                | 66.88             | 69.27       | 83.91                | 80.12          | 81.97         |
| ChemRxnGPT (llama2-13b)(1024X1024)  | 73.52               | 69.08             | 71.23       | 85.95                | 82.54           | 84.21         |
| ChemRxnGPT(New)(llama2-13b)(1024X1024)  | 74.23               | 69.63             | 71.86       | 86.65                | 82.42           | 84.48         |
| ChemRxnGPT(New)(llama2-7b)(1024X1024)  | 73.98               | 69.58             | 71.71       | 86.63                | 82.39           | 84.46         |
| ChemRxnGPT(New)(llama2-7b)(1333X1333)  | 74.67               | 69.67             | 72.08       | 86.91                | 82.77          | 84.79         |

| Number of Image tokens   |Soft Match Precision | Soft Match Recall | Soft Match F1 |
|--------------------------|----------------------|-------------------|---------------|
| 100                   | 4.1                  | 1.3               | 1.9           |
| 200                   | 4.4                  | 2.8               | 3.4           |
| 300                   | 86.91                | 82.77              | 84.79          |
| 400                   | 72.3                 | 66.2              | 69.1          |

|  w/BERT   |Freeze   |Soft Match Precision | Soft Match Recall | Soft Match F1 |
|--------------------------|----------------------|-------------------|---------------|---------------|
| \cmark    |          | 4.1                  | 1.3               | 1.9           |
|                   |           | 4.4                  | 2.8               | 3.4           |
|      |          | 72.3                 | 66.2              | 69.1          |

|  position representation    |Soft Match Precision | Soft Match Recall | Soft Match F1 |
|--------------------------|----------------------|-------------------|---------------|
| Vocab                       |                   |               |           |
| Numerical                  |                   |                |            |



### Training  
1. Download pre-trained [ResNet-101 weights](https://download.pytorch.org/models/resnet101-63fe2227.pth) and put into your own backbone folder.
2. Run `./exps/train_Abd.sh` or `./exps/train_CMR.sh`

### Testing
Run `./exp/validation.sh`

### Acknowledgement
This code is based on [Q-Net](https://github.com/zjlab-ammi/q-net), [PFENet](https://github.com/dvlab-research/PFENet)
