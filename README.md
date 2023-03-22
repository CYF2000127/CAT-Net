# CAT-Net
This is the code of following paper "Few Shot Medical Image Segmentation with Cross Attention Transformer".
### Dependencies

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
 

### Training  
1. Download pre-trained ResNet-101 weights [vanilla version](https://download.pytorch.org/models/resnet101-63fe2227.pth) and put into your own backbone folder.
2. Run `exps/train_Abd.sh` or `exps/train_CMR.sh`

### Testing
Run `exp/validation.sh`

### Acknowledgement
This code is based on [Q-Net](https://github.com/zjlab-ammi/q-net),[PFENet](https://github.com/dvlab-research/PFENet) and [Mask2Former](https://github.com/facebookresearch/Mask2Former). 
