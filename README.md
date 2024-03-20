# Official code for StegoGAN: Leveraging Steganography for Non-bijective Image-to-Image Translation

PyTorch implementation of "StegoGAN: Leveraging Steganography for Non-bijective Image-to-Image Translation".

Author: Sidi Wu, Yizi Chen, Samuel Mermet, Lorenz Hurni, Konrad Schindler, Nicolas Gonthier, Loic Landrieu

## Abstract
Most image-to-image translation models postulate bijective mapping --- a unique correspondence exists between the semantic classes of the source and target domains. However, this assumption does not always hold in real-world scenarios due to divergent distributions, different class sets, and asymmetrical information representation. 
As conventional GANs attempt to generate images matching the distribution of the target domain, they may hallucinate spurious instances of classes absent from the source domain, thereby diminishing the interpretability and reliability of translated images. 
CycleGAN-based methods are also known to hide the mismatched information in the generated images to bypass cycle consistency objectives, a process known as steganography.
In response to the challenge of non-bijective image translation, we introduce StegoGAN, a novel model that leverages steganography to prevent spurious features in generated images. Our approach enhances the semantic consistency of the translated images without requiring additional postprocessing or supervision. 
Our experimental evaluations demonstrate that StegoGAN outperforms existing GAN-based models across various non-bijective image-to-image translation tasks, both qualitatively and quantitatively.

<img src="img/problem.png" width="500"/>

So, we propose StegoGAN, 
- A model that, instead of disabling steganography, leverages this phenomenon to detect and mitigate semantic misalignment between domains; 
- In settings where the domain mapping is non-bijective, StegoGAN experimentally demonstrates superior semantic consistency over other GAN-based models both visually and quantitatively, without requiring detection or inpainting steps;
- We publish three datasets from open-access sources as a benchmark for evaluating non-bijective image translation models.

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
│   ├── BRATS_mismatch           <- Brats_mismatch dataset
│   ├── PlanIGN                  <- PlanIGN dataset
|   ├── Google_mismatch          <- Google_mismatch dataset
├── model                        <- Model
│   ├── base_model.py            <- Base model
│   ├── Networks.py              <- Networks
|   ├── stego_gan_model.py       <- StegoGAN model
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

### 2. Download datasets

We propose three datasets for benchmarking non-bijective image-to-image translation, and the datasets can be downloaded from Zenodo (To be relased...):

* [PlanIGN] This dataset was constructed from the French National Mapping Agency (IGN), comprising 1900 aerial images (ortho-imagery) at 3m spatial resolution and two versions of their corresponding maps -- one with toponyms and one without toponyms (_TU). We divided them into training (1000 images) and testing (900 images). In our experiment, we use trainA & trainB, testA & testB_TU for training and testing, respectively.
* [Google_mismatch] We created non-bijective datasets from the [maps dataset](https://github.com/phillipi/pix2pix?tab=readme-ov-file) by seperating the samples with highways from those without. We excluded all satellite images (trainA) featuring highways and subsampled maps (trainB) with varying proportions of highways from 0% to 65%. For the test set, we selected 898 pairs without highways. 
* [BRATS_mismatch] We used two modalities from [Brats2018](https://www.med.upenn.edu/sbia/brats2018/data.html) -- T1 and FLAIR. We selected transverse slices from the 60&deg to 100&deg. Each scan was classified as tumorous if more than 1% of its pixels were labelled as such and as healthy if it contained no tumor pixels. We provide "generate_mismatched_datasets.py" so users can generate datasets with varying proportions of tumorous samples during training. In our default seeting, we have 800 training samples with source images (T1) being healthy and target images (FLAIR) comprising 60% tumorous scans. The test set contains 335 paired scans of healthy brains. 

And it should be placed within the 'dataset/' directory.

### 3. Download weights for inference or pre-training

The pre-train weights could be download from:
* [BRATS](To be relased...)
* [IGN](To be relased...)
* [google](To be relased...)

## How to use 🚀

### 1. Train models

* For example: training StegoGAN with Google_mismatch dataset
```
python train.py --dataroot ./dataset/Google_mismatch/0.65 \
                --name google_stego_0.65 \
                --model stego_gan \
                --gpu_ids 0 \
                --lambda_reg 0.3 \
                --lambda_consistency 1 \ 
                --resnet_layer 8 \
                --batch_size 1  \
                --no_flip \
                --n_epochs 100
```
Training results and weights are saved at `checkpoints/<name>`.

### 2. Testing models
```
python test.py --dataroot ./dataset/Google_mismatch \ 
               --name google_stego_0.65 \ 
               --model stego_gan \
               --phase test \
               --no_dropout \
               --resnet_layer 8
```
Inferencing results will be saved at `results/<model_name>/test_latest`.

### 2. Evaluating results
For Google_mismatch
```
python evaluation/evaluate_google.py \
       --gt_path ./results/google_stego_0.65/test_latest/images/real_B \
       --pred_path ./results/google_stego_0.65/test_latest/images/fake_B_clean \
       --output_path ./results/google_stego_0.65/test_latest \
       --dataset Google \
       --method StegoGAN
```
For PlanIGN
```
python evaluation/evaluate_IGN.py \
       --gt_path_TU ./dataset/PlanIGN/testB_TU \ 
       --gt_path_T ./dataset/PlanIGN/testB \
       --pred_path ./results/PlanIGN/test_latest/images/fake_B_clean \
       --pred_path_mask ./results/PlanIGN/test_latest/images/latent_real_B_mask_upsampled \
       --output_path ./results/PlanIGN/test_latest \
       --dataset PlanIGN \
       --method StegoGAN 
```
For Brats_mismatch
```
python evaluation/evaluate_brats.py \
       --gt_path ./results/Brats/test_latest/images/real_B \
       --pred_path ./results/Brats/test_latest/images/fake_B_clean \
       --output_path ./results/Brats/test_latest \
       --seg_save_path ./results/Brats/test_latest/images/fake_B_tumor \
       --dataset Brats \
       --method StegoGAN
```

## Qualitative results 🥰

<img src="img/results.png" width="800"/>

## Citation
If you use our code or our datasets, please cite our [paper](http:..)
```
@inproceedings{...,
  title={StegoGAN: Leveraging Steganography for Non-Bijective Image-to-Image Translation},
  ...
}
```

If you want to use Google_mismatch, please also cite the following paper:
```
@inproceedings{isola2017image,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2017}
}
```
If you want to use Brats_mismatch, please also cite the following papers:
```
@article{menze2014multimodal,
  title={The multimodal brain tumor image segmentation benchmark (BRATS)},
  author={Menze, Bjoern H and Jakab, Andras and Bauer, Stefan and Kalpathy-Cramer, Jayashree and Farahani, Keyvan and Kirby, Justin and Burren, Yuliya and Porz, Nicole and Slotboom, Johannes and Wiest, Roland and others},
  journal={IEEE transactions on medical imaging},
  volume={34},
  number={10},
  pages={1993--2024},
  year={2014}
}
@article{bakas2017brats17,
  title={Advancing the cancer genome atlas glioma MRI collections with expert segmentation labels and radiomic features},
  author={Bakas, Spyridon and Akbari, Hamed and Sotiras, Aristeidis and Bilello, Michel and Rozycki, Martin and Kirby, Justin S and Freymann, John B and Farahani, Keyvan and Davatzikos, Christos},
  journal={Scientific data},
  volume={4},
  number={1},
  pages={1--13},
  year={2017}
}
@article{bakas2018ibrats17,
  title={Identifying the best machine learning algorithms for brain tumor segmentation, progression assessment, and overall survival prediction in the BRATS challenge},
  author={Bakas, Spyridon and Reyes, Mauricio and Jakab, Andras and Bauer, Stefan and Rempfler, Markus and Crimi, Alessandro and Shinohara, Russell Takeshi and Berger, Christoph and Ha, Sung Min and Rozycki, Martin and others},
  journal={arXiv preprint arXiv:1811.02629},
  year={2018}
}
```


## Acknowledgement
We appreciate the open source code from [Cycle-GAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) (for base architecture).
