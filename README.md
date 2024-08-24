# jssc-CVDepth
# Self-Supervised Monocular Depth Estimation Using Hybrid CNN-VMamba Architecture

## Introduction
This is the official implementation of the paper: *Self-Supervised Monocular Depth Estimation Using Hybrid CNN-VMamba Architecture*.

## Reference Code
This project is built upon and inspired by the following repositories:
- [ManyDepth](https://github.com/nianticlabs/manydepth)
- [ManyDepthFormer](https://github.com/fogfog2/manydepthformer)
- [MonoDepth2](https://github.com/nianticlabs/monodepth2)

## Installation
Ensure you have the following environment:
- `pytorch==1.13.0`
- `Python==3.10`
- `cuda==11.7`

To install the required dependencies, run:
```bash
pip install -r requirements.txt
pip install causal_conv1d-1.0.0+cu118torch1.13cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install mamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Baidu Netdisk Downloads  
  
## KITTI_640x192  
- Pretrained Model:  
  - Link: [Click here to download](https://pan.baidu.com/s/1ZLopzwK2FZmzHmIJdjFldQ?pwd=5ag5)  
  - Extraction Code: 5ag5  
- Pretrained Results (npy):  
  - Link: [Click here to download](https://pan.baidu.com/s/19eugxlOksA5nuVJUdlUTJg?pwd=623h)  
  - Extraction Code: 623h  
  
## KITTI_640x192_r50  
- Pretrained Model:  
  - Link: [Click here to download](https://pan.baidu.com/s/1lWBkDkfQ4fUPOqgrp7EcrQ?pwd=l1e5)  
  - Extraction Code: l1e5  
- Pretrained Results (npy):  
  - Link: [Click here to download](https://pan.baidu.com/s/1r5YjBCOQt9DO9KYf36gIyQ?pwd=2846)  
  - Extraction Code: 2846  
  
## KITTI_1024x320  
- Pretrained Model:  
  - Link: [Click here to download](https://pan.baidu.com/s/11g5_uw5V8rfDLzlHVV-Yfw?pwd=kdka)  
  - Extraction Code: kdka  
- Pretrained Results (npy):  
  - Link: [Click here to download](https://pan.baidu.com/s/1DyWDgnoa1Te6bxZxl_rkhQ?pwd=dgv7)  
  - Extraction Code: dgv7  
  
## Ablation Experiment Results  
  
### Full-Sobel  
- Pretrained Model:  
  - Link: [Click here to download](https://pan.baidu.com/s/1RDSOLfqp-orpDZp1rh5tKw?pwd=ppuu)  
  - Extraction Code: ppuu  
- Pretrained Results (npy):  
  - Link: [Click here to download](https://pan.baidu.com/s/1QVSUqtw8nuhBqveaosUjOw?pwd=ag09)  
  - Extraction Code: ag09  
  
### Full-MAAM  
- Pretrained Model:  
  - Link: [Click here to download](https://pan.baidu.com/s/13Qsic5pOIVThClpjrMDrlQ?pwd=4pf0)  
  - Extraction Code: 4pf0  
- Pretrained Results (npy):  
  - Link: [Click here to download](https://pan.baidu.com/s/1Qok_LFAKHEoK0g2rYzizIw?pwd=55t4)  
  - Extraction Code: 55t4  
  
### Full-VMamba  
- Pretrained Model:  
  - Link: [Click here to download](https://pan.baidu.com/s/1WbAXsgTI-HPVaZx-o3kFYQ?pwd=nifj)  
  - Extraction Code: nifj  
- Pretrained Results (npy):  
  - Link: [Click here to download](https://pan.baidu.com/s/1K66qYeKgxT3ElO2LbIM87A?pwd=456s)  
  - Extraction Code: 456s  
  
### Baseline  
- Pretrained Model:  
  - Link: [Click here to download](https://pan.baidu.com/s/1UyrhhbM_cffnKqG76bUeQA?pwd=ujfp)  
  - Extraction Code: ujfp  
- Pretrained Results (npy):  
  - Link: [Click here to download](https://pan.baidu.com/s/1fcmO2SXBuqY8GVLrnDvk8A?pwd=zgna)  
  - Extraction Code: zgna

## Model Training and Evaluation  
  
### Training  
To train the model, run the following command in your terminal or command prompt:  
  
```bash  
python train.py

### Evaluation
To evaluate the depth estimation performance of the trained model, run the following command:
  
```bash  
python evaluate_depth.py




