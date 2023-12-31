# Semi-supervised medical image segmentation via feature similarity and reliable-region enhancement

by Jianwu long,  Chengxin Yang*, Yan Ren and  Ziqin Zeng. 

### News
```
<09.25.2023> We released the codes;
```
### Introduction
This repository is for our CIBM 2023 paper: '[Semi-supervised medical image segmentation via feature similarity and reliable-region enhancement]'.

### Requirements
This repository is based on PyTorch 1.12.0, CUDA 11.3 and Python 3.7.13. All experiments in our paper were conducted on a single NVIDIA 3090 GPU with an identical experimental setting.

### Usage
1. Clone the repo.;
```
git clone https://github.com/gdghds0/FSRENet-master.git
```
2. Put the data in './FSRENet-master/data';

3. Train the model;
```
cd FSRENet-master

sh train_FSRENet_2d.sh
sh train_FSRENet_3d.sh
```
4. Test the model;
```
cd FSRENet-master/code

python test_2D_ACDC.py
python test_3D_LA.py 
```


### Acknowledgements:
Our code is adapted from [MC-Net](https://github.com/ycwu1997/MC-Net), [SS-Net](), and [SSL4MIS](https://doi.org/10.1007/978-3-031-16443-9_4). Thanks for these authors for their valuable works and hope our model can promote the relevant research as well.

### Questions
If any questions, feel free to contact me at '871001000@qq.com'

