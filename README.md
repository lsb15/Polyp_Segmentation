# Polyp_Segmentation
Individual Project


# Polyp Segmentation with UNet++

This repository contains code for polyp segmentation using the NestedUNet architecture. Polyp segmentation is a critical task in medical image analysis, particularly in diagnosing gastrointestinal diseases. The NestedUNet architecture, inspired by the popular U-Net model, is known for its effectiveness in semantic segmentation tasks.

The objective of this project is to accurately segment polyps from endoscopic images, aiding in early diagnosis and treatment planning for gastrointestinal disorders. The dataset used for training and evaluation is sourced from Kaggle, specifically from the Kvasir-SEG dataset.

## Installation

To set up the environment for running the code, follow these steps:

1. Install required dependencies:

```bash
pip install -r requirements.txt
```

2. Install specific versions of `torch` and `torchvision`:

```bash
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
```

3. Install Albumentations and its compatible version:

```bash
pip uninstall albumentations
pip install albumentations==0.5.1
```

## Usage

### Training

To train the polyp segmentation model, run the `train.py` script. You can specify various parameters such as dataset directory, architecture (NestedUNet), number of epochs, batch size, optimizer, etc.

```bash
python train.py --dataset polyp --arch NestedUNet --name polyp_segmentation --epochs 150 --batch_size 8 --input_w 384 --input_h 384 --img_ext jpg --mask_ext jpg --optimizer Adam 
```

### Validation

After training, you can evaluate the trained model using the `val.py` script. This script calculates evaluation metrics and provides insights into the model's performance.

```bash
python val.py --name polyp_segmentation
```

## Model Architecture

The NestedUNet architecture, based on U-Net, is designed to effectively capture contextual information while maintaining high-resolution features. It consists of encoding and decoding paths with skip connections for precise localization.

You can find the implementation of NestedUNet in the `archs.py` file. For more details on the architecture, refer to the original paper or the GitHub repository: [NestedUNet GitHub](https://github.com/4uiiurz1/pytorch-nested-unet).

## Dataset

The dataset used for this project is sourced from Kaggle, specifically from the Kvasir-SEG dataset. It contains endoscopic images with corresponding ground truth masks for polyp segmentation.

You can download the dataset from the following link: [Kvasir-SEG Dataset](https://www.kaggle.com/datasets/debeshjha1/kvasirseg)
