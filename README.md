# LDM-RSIC (2025--Accepted by TGRS)
### 📖[**Arxiv**](https://arxiv.org/abs/2406.03961) | 🖼️[**TGRS**](https://ieeexplore.ieee.org/document/10980206)

PyTorch codes for "[Exploring Distortion Prior with Latent Diffusion Models for Remote Sensing Image Compression]([https://arxiv.org/abs/2406.03961](https://ieeexplore.ieee.org/document/10980206))", **TGRS**, 2025.

- Authors: Junhui Li, Jutao Li, Xingsong Hou, and Huake Wang <br>


## Abstract
> Learning-based image compression algorithms typically focus on designing encoding and decoding networks and improving the accuracy of entropy model estimation to enhance the rate-distortion (RD) performance. However, few algorithms leverage the compression distortion prior from existing compression algorithms to improve RD performance. In this paper, we propose a latent diffusion model-based remote sensing image compression (LDM-RSIC) method, which aims to enhance the final decoding quality of RS images by utilizing the generated distortion prior from a LDM. Our approach consists of two stages. In Stage I, a self-encoder learns prior from the high-quality input image. In Stage II, the prior is generated through a LDM, conditioned on the decoded image of an existing learning-based image compression algorithm, to be used as auxiliary information for generating the texture-rich enhanced images. To better utilize the prior, a channel attention and gate-based dynamic feature attention module (DFAM) is embedded into a Transformer-based multi-scale enhancement network (MEN) for image enhancement. Extensive experimental results demonstrate the proposed LDM-RSIC outperforms existing state-of-the-art traditional and learning-based image compression algorithms in terms of both subjective perception and objective metrics.

## Network
![image](/figs/Method.png)
 
## 🧩 Install
```
git clone https://github.com/mlkk518/LDM-RSIC.git
```

## Environment
 > * CUDA 11.7
 > * Python 3.7.12
 > * PyTorch 1.13.1
 > * Torchvision 0.14.1

## 🎁 Dataset
Please download the following remote sensing benchmarks:
Experimental Datasets:
  [DOTA-v1.5](https://captain-whu.github.io/DOTA/dataset.html) | [UC-M](http://weegee.vision.ucmerced.edu/datasets/landuse.html) 

Testing set  (Baidu Netdisk) [DOTA:Download](https://pan.baidu.com/s/1R52rO-gxZH1jG-amwUCO-g) Code：ldc1 | [UC_M:Download](https://pan.baidu.com/s/1KJAy2cPVnj6VfqrlR5XPCg)  Code：pvf3 


## 🧩 Test
[Download Pre-trained Model](https://pan.baidu.com/s/1OsPSjPp34RHasHi9YM5rHg) (Baidu Netdisk) Code：v72j
- **Step I.**  Change the roots of ./ELIC/scripts/test.sh to your data and Use the pretrained models of [ELIC] to generate the initial decoded images.

- **Step II.**  Refer to test_DiffRS2_lambda.yml to set the data roots and pretrained models of [LDM], and run sh ./scriptEn/test.sh Lambada Gpu_ID. Here lambda belongs to [0.0004, 0.0008, 0.0032, 0.01, 0.045] 

```

sh ./ELIC/scripts/test.sh 0.0008 0

sh ./scriptEn/test.sh 0.0008 0
```

## 🧩 Train
- **Step II.** Learning the compression distortion prior.   
- **Step II.**  Using LDM to generate distortion prior, which is then fed into MEN for improved images.   
```
sh ./scriptEn/trainS1.sh 0.0008 0

sh ./scriptEn/trainS2.sh 0.0008 0

```

### Qualitative results 1
 ![image](/figs/DOTA_vis.png)
 
### Quantitative results 2
 ![image](/figs/UC_vis.png)

### Quantitative results 3
 ![image](/figs/UC_com_SOTA_vis.png)
 
#### More details can be found in our paper!

## Contact
If you have any questions or suggestions, feel free to contact me. 😊  
Email: mlkkljh@stu.xjtu.edu.cn



## Citation
If you find our work helpful in your research, please consider citing it. We appreciate your support！😊


## Acknowledgment: 

This work was supported by:  
- [BasicSR](https://github.com/xinntao/BasicSR)
- [DiffIR](https://github.com/Zj-BinXia/DiffIR)
- [HI-Diff](https://github.com/zhengchen1999/HI-Diff)
- [LDM](https://github.com/CompVis/latent-diffusion)
- [ELiC](https://github.com/VincentChandelier/ELiC-ReImplemetation)



```
@ARTICLE{10980206,
  author={Li, Junhui and Li, Jutao and Hou, Xingsong and Wang, Huake},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Exploring Distortion Prior with Latent Diffusion Models for Remote Sensing Image Compression}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Image coding;Entropy;Distortion;Transformers;Decoding;Adaptation models;Remote sensing;Accuracy;Discrete wavelet transforms;Diffusion models;Image compression;latent diffusion models;remote sensing image;image enhancement},
  doi={10.1109/TGRS.2025.3565259}}

```
