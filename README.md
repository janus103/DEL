# DEL: Domain Embedding Layer (Official Project, NeurIPS 2023)
This repository provides the official PyTorch implementation of the following paper:

> **Abstract:** 
*Single-domain generalization (SDG) is known as the most challenging scenario for addressing the domain shift problem, in which a model is trained on a single source domain and evaluated for robustness on unseen domains. The key point for improving SDG performance lies in the ability of the model to distinguish between domain-invariant content information and domain-specific style information. Traditionally, extrinsic factors, such as augmentation policies and adversarial representations, have been leveraged in training to achieve this goal. Although these methodologies have demonstrated remarkable domain generalization performances, they tend to increase the number of model parameters and implementation complexity. This paper proposes an SDG solution based on a domain embedding layer (DEL), which replaces a portion of the existing network architecture without requiring additional training. The potential for improving domain generalization was demonstrated without relying on external factors by replacing the first layer of a deep neural network (DNN), which is known for learning semantic features, with a domain embedding layer (DEL) and training it end-to-end in the frequency domain, which can serve as an inherent factor. The frequency domain can be distinguished into low- and high- frequency components, associated with content and style information, respectively. DEL employs discrete wavelet transform (DWT) to enable learning from low- and high-frequency domains combined. DWT allows the frequency components to be utilized in the spatial domain without any additional transformations, enabling the utilization of the inductive bias from convolution operations without additional computational overhead. Moreover, DEL can be used in conjunction with other methodologies that leverage extrinsic factors, owing to its high compatibility. Experimental results show that by applying DEL enabled average improvements of 1.51\% and 1.85\% on the corrupted CIFAR-10-C  and Digits datasets, respectively, compared with  the generative model-based progressive domain expansion network.*

<p align="center">
  <img src="assets/DEL_Overview_white.PNG" />
</p>

## Pytorch Implementation
### Installation

Clone this repository.
```
 => git clone https://github.com/janus103/DEL.git
 => cd DEL
```

Easy conda environment setting: you only need to import del.yaml.

```
 => conda create --name DEL python=3.9
 => conda env export > del.yaml
```

## Train & Validation

### Train
You can easily perform the learning with the bash command below.
Only by changing the code in brackets

+ modify
    + dataset path 
        + ex) /home/users/~
    + epochs
        + ex) 100
    + checkpoint path
        + ex) output
    + 7 7 7: Kernel size of first convolution layer
        + ex) 7 7 7
    + 2 2 2: Level of DWT -> you can only change 1 to 3
        + ex) 2 2 2
```
CUDA_VISIBLE_DEVICES=0 train.py [dataset path] --model [models] --lr 0.03 --warmup-epochs 5 --epochs [epochs] --sched cosine -b 128 -j 32 --val-split val --input-size 3 224 224 --output [checkpoint path] --no-prefetcher --num-classes 10 --pretrained --dwt-kernel-size [7 7 7] --dwt_level [2 2 2] --dwt_bn 0 1 --sfn [Train_id]
```

You can also add the following arguments for automatic augmentation or scale mode.

+ Example of Auto-Augment arguments
    + --aa 'v0r'
    + --aa 'v0'
    + --aa 'original'
    + --aa 'originalr'

+ Example of Scale Mode
    + --dwt_quant 0
        + Without Scale
    + --dwt_quant 1
        + Adaptive Scale
    + --dwt_quant 2
        + Only Scale Level 1 LL 
    + --dwt_quant 3
        + Only Low Frequency domain
    
### Validation
Also, you can easily perform the validation with the bash command below.
Only by changing the code in brackets

**please check your wegiht(.pth) file**

+ --corrupted
    + FORMAT: '{Corruption}_{Sevirity Level}'
    + By adding this argument, you can test with corrupted image without additional storage space.
    
```
CUDA_VISIBLE_DEVICES=0 python validate.py [dataset path] --model [models] --input-size 3 224 224 --checkpoint [.pth file] --no-prefetch -b 1 --corrupted [Fog_1] --sfn [Train_id] --results-file [result.csv] --dwt-kernel-size [7 7 7] --dwt_level [2 2 2] --dwt_bn 0 1 --dwt_quant [0-3]
```

## Acknowledgments
Our pytorch implementation is heavily derived from [TIMM](https://github.com/huggingface/pytorch-image-models), [Pytorch Wavelets](https://github.com/fbcotter/pytorch_wavelets) and [PDEN](https://github.com/lileicv/PDEN).

Thanks to the TIMM, Pytorch_Wavelets and PDEN implementations.
