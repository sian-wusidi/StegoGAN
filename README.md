# Official code for StegoGAN: Leveraging Steganography for Non-bijective Image-to-Image Translation

PyTorch implementation of "StegoGAN: Leveraging Steganography for Non-bijective Image-to-Image Translation".

Author: Sidi Wu, Yizi Chen, Loic Landrieu, Nicolas Gonthier, Samuel Mermet, Lorenz Hurni, Konrad Schindler

## Abstract
Image translation models traditionally postulate the existence of a unique correspondence between the semantic classes of the source and target domains. However, this assumption does not always hold in real-world scenarios due to divergent distributions, different class sets, and asymmetrical information representation. %This is particularly prevalent for abstract representations such as maps or labelled images.
As conventional GANs attempt to generate images matching the distribution of the target domain, they may hallucinate spurious instances of classes absent from the source domain, thereby diminishing their interpretability and reliability. 
CycleGAN-based methods are also known to hide information in the generated images to bypass cycle-consistency objectives, a process known as steganography.
In response to these challenges, we introduce StegoGAN, a novel model that leverages steganography to prevent spurious features in generated images. Our approach enhances the semantic consistency of the translated images without requiring additional feature detection, inpainting steps, or supervision. 
Our experimental evaluations demonstrate that StegoGAN outperforms existing GAN-based models across various non-bijective image-to-image translation tasks, both qualitatively and quantitatively.

<img src="img/problem.png" width="500"/>

So, we propose such pipline to solve the problem:

<img src="img/pipline.png" width="800"/>

If you find this code useful in your research, please consider to cite:

```
After ARXIV
```

### Project Structure

Structure of this repository:

```
|
├── data                         <- Data loader
├── dataset                      <- Dataset for training
├── img                          <- Images
│   ├── brats                    <- Brats dataset
│   ├── IGN                      <- IGN dataset
|   ├── google                   <- Google dataset
├── model                        <- Model
│   ├── base_model.py            <- Base model
│   ├── Networks.py              <- Networks
|   ├── stego_gan_model.py       <- Stego gan model
├── env_stego_gan.yml            <- Conda environment .yml file
├── train.py                     <- Training codes for Stego-GAN
├── test.py                      <- Testing codes for Stego-GAN
└── README.md
```

## Installation 🌠

### 1. Create and activate conda environment

```
conda env create -f env_stego_gan.yml
conda activate env_stego_gan
```

### 2. Download spatial-temporal historical maps dataset (training dataset)

The training dataset could be download from:

* [BRATS] (To be relased...)
* [IGN] (To be relased...)
* [google] (To be relased...)

And it should be placed within the 'dataset/' directory.

### 3. Download weights for inference or pre-training

The pre-train weights could be download from:
* [BRATS](To be relased...)
* [IGN](To be relased...)
* [google](To be relased...)

## How to use 🚀

### 1. Train models

* For example: training Stego-GAN with Brats dataset (training with pre-trained)
```
python train.py --dataroot dataset/brats/0.6 \
                --name brats_stego_0.6 \
                --model stego_gan \
                --gpu_ids 0 \
                --lambda_reg 0.3 \
                --lambda_consistency 1 \ 
                --resnet_layer 9 \
                --batch_size 12  \
                --no_flip \
                --n_epochs 100
```
Training results and weights are saved at `checkpoints/<name>`.

### 2. Testing models
```
python test.py --dataroot dataset/brats/0.6 \ 
               --name brats_stego_0.6 \ 
               --model stego_gan \
               --phase test \
               --no_dropout \
               --resnet_layer 9
```
Inferencing results will be saved at `results/<model_name>/test_latest`.

## Qualitative results 🥰

<img src="img/results.png" width="800"/>

## Acknowledgement
We appreciate the open source code from:  
* Public code of [Cycle-GAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) (for base architecture)
* Public code of [RANSAC-FLOW](https://github.com/XiSHEN0220/RANSAC-Flow) (for mask generation)
