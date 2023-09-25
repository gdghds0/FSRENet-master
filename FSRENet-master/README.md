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
git clone https://github.com/ycwu1997/SS-Net.git
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
cd SS-Net/code

python test_2D_ACDC.py 
python test_3D_LA.py 
```

### Citation
If our SS-Net model is useful for your research, please consider citing:

      @inproceedings{wu2022exploring,
        title={Exploring Smoothness and Class-Separation for Semi-supervised Medical Image Segmentation},
        author={Wu, Yicheng and Wu, Zhonghua and Wu, Qianyi and Ge, Zongyuan and Cai, Jianfei},
        booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
        pages={34--43},
        volume={13435},
        year={2022},    
        doi={10.1007/978-3-031-16443-9\_4},
        organization={Springer, Cham}
        }

### Acknowledgements:
Our code is adapted from [MC-Net](https://github.com/ycwu1997/MC-Net), [SemiSeg-Contrastive](https://github.com/Shathe/SemiSeg-Contrastive), [VAT](https://github.com/lyakaap/VAT-pytorch), and [SSL4MIS](https://github.com/HiLab-git/SSL4MIS). Thanks for these authors for their valuable works and hope our model can promote the relevant research as well.

### Questions
If any questions, feel free to contact me at '871001000@qq.com'

