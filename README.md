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


| Model/real      | Hard Match Precision | Hard Match Recall | Hard Match F1 | Soft Match Precision | Soft Match Recall | Soft Match F1 |
|--------------------------|----------------------|-------------------|---------------|----------------------|-------------------|---------------|
| ReactionDataExtractor    | 4.10                  | 1.30               | 1.90           | 19.40                 | 5.90               | 9.00           |
| OChemR                   | 4.40                  | 2.80               | 3.40           | 12.40                 | 7.90               | 9.60           |
| RxnScribe     | 72.32                 | 66.23              | 69.12          | 83.83                 | 76.51              | 80.04          |
| ReactionImgMLLM     | 74.67               | 69.67             | 72.08       | 86.91                | 82.77          | 84.79         |


| Model/systic      | Hard Match Precision | Hard Match Recall | Hard Match F1 | Soft Match Precision | Soft Match Recall | Soft Match F1 |
|--------------------------|----------------------|-------------------|---------------|----------------------|-------------------|---------------|
| ReactionDataExtractor    | 8.40                  | 6.90              | 7.60           | 22.60                 | 11.40               | 15.20           |
| OChemR                   | 8.10                  | 7.50              | 7.80           | 15.90                 | 12.80              | 14.20           |
| RxnScribe    | 78.54                | 75.63              |     77.06      | 87.62                | 83.95              |      85.75    |
| ReactionImgMLLM   | 86.41             | 85.92            |   86.16     | 91.55              | 90.81          |     91.18     |



| Model        | OCR Accuracy | Role Identification Accuracy  |
|--------------------------|----------------------|-------------------|
| ReactionImgMLLM  | 94.91               | 93.62            |


| Number of Image tokens   |Soft Match Precision | Soft Match Recall | Soft Match F1 | OCR Accuracy | Role Identification Accuracy  |
|--------------------------|----------------------|-------------------|---------------|----------------------|-------------------|
| 100                   | 85.62                 | 82.11               | 83.83          |93.68|92.85|
| 200                   | 86.31                 | 82.36               | 84.29           |94.26|93.02|
| 300                   | 86.91                | 82.77              | 84.79          |94.91            |93.62                |
| 400                   | 86.42                | 82.20              | 84.26          |95.02    |93.14          |

|  w/BERT   |Freeze   |Soft Match Precision | Soft Match Recall | Soft Match F1 |OCR Accuracy | Role Identification Accuracy  |
|--------------------------|----------------------|-------------------|---------------|---------------|----------------------|-------------------|
| ✗    |       ✗          | 82.21                  | 78.55              | 80.34           |94.74| 93.41|
|   ✓   |       ✗         | 86.91                  | 82.77              | 84.79          |94.91|93.62|
|    ✓   |      ✓         | 71.65                  | 70.02              | 69.1          |94.21|  81.73|

|  position representation    |Soft Match Precision | Soft Match Recall | Soft Match F1 |
|--------------------------|----------------------|-------------------|---------------|
| Vocab                       |         86.15          |          81.23     |      83.62     |
| Numerical                  |       86.91            |        82.77        |      84.79      |

| separate training   |Soft Match Precision | Soft Match Recall | Soft Match F1 | OCR Accuracy | Role Identification Accuracy  |
|--------------------------|----------------------|-------------------|---------------|----------------------|-------------------|
| ✓                  | 86.84                 | 82.64               |    84.68       |94.78|93.39|
| ✗                  | 86.91                | 82.77              | 84.79          |94.91            |93.62                |




### Training  
1. Download pre-trained [ResNet-101 weights](https://download.pytorch.org/models/resnet101-63fe2227.pth) and put into your own backbone folder.
2. Run `./exps/train_Abd.sh` or `./exps/train_CMR.sh`

### Testing
Run `./exp/validation.sh`

### Acknowledgement
This code is based on [Q-Net](https://github.com/zjlab-ammi/q-net), [PFENet](https://github.com/dvlab-research/PFENet)
