# Segmentation with U-Net

The goal of this project is to compare the performance of different models on the Image Classification task, trained on the CIFAR-10 dataset. These models are a MLP, a 1-layer CNN and a ResNet. 

To understand the background of this exercise you can:

Read the ResNet publication: https://arxiv.org/abs/1512.03385


Note that we will implement the same ideas as in these papers, but will deviate from the implementation details and conduct different and fewer experiments.

## Table of Contents

- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Setup](#setup)
  - [Prerequisites](#prerequisites)
  - [Cloning the Repository](#cloning-the-repository)
  - [Downloading the Data](#downloading-the-data)
  - [Installing Dependencies](#installing-dependencies)
- [Running the Notebook](#running-the-notebook)
- [Project Details](#project-details)
  - [Importing libraries](#importing-libraries)
  - [Data Exploration](#data-exploration)
  - [Implement PyTorch Dataset](#implement-pytorch-dataset)
  - [Implement the U-Net](#implement-the-u-net)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Training with Boundary Channel](#training-with-boundary-channel)
  - [Training with Dice Loss](#training-with-dice-loss)
- [Results](#results)
- [Acknowledgments](#acknowledgments)

## Project Overview

In this project, we will implement the U-Net architecture for two main tasks: segmentation and denoising. The primary focus is on segmenting nuclei in fluorescence microscopy images.

## Directory Structure

```plaintext
Segmentation-Denoising-UNet/
│
├── README.md
├── .gitignore
├── notebooks/
│   └── Segmentation_Denoising_UNet.ipynb
├── src/
│   ├── __init__.py
│   ├── data_exploration.py
│   ├── dataset.py
│   ├── unet_model.py
│   ├── training.py
│   ├── evaluation.py
│   ├── train_with_boundary.py
│   ├── train_with_dice_loss.py
├── data/
│   ├── dsb2018/
│       ├── train/
│       ├── test/
└── scripts/
    └── download_data.py
```
## Setup

### Prerequisites

- Python 3.7+
- Jupyter Notebook
- Git

### Cloning the Repository

First, clone the repository to your local machine:

```sh
git clone https://github.com/yourusername/Segmentation-UNet.git
cd Segmentation-UNet
```
## Downloading the Data

To keep the repository lightweight, the data is not included. You can download the data by running the provided script.

```sh
chmod +x scripts/download_data.sh
./scripts/download_data.sh

```

## Installing Dependencies

Install the required Python packages:

```sh
pip install -r requirements.txt
```

## Running the Notebook
After downloading the data, you can start the Jupyter notebook:



```sh
jupyter notebook notebooks/Segmentation_UNet.ipynb
```

## Project Details

### Importing libraries
First of all we import all required python libraries for completing the task.

- Script: import_libraries.py

### Data Exploration
In this step, we explore the dataset to understand its structure and visualize some examples. This is done in the data_exploration.py script.

- Script: data_exploration.py

### Implement PyTorch Dataset
We implement a custom PyTorch dataset to handle data loading and preprocessing. This implementation can be found in the dataset.py script.

- Script: dataset.py

### Implement the U-Net
We implement the U-Net architecture as described in Ronneberger et al. The code for this is in the unet_model.py script.

- Script: unet_model.py

### Training
We define the training functions and train the model, plotting the results of loss and metrics. The training code is in the training.py script.

- Script: training.py

### Evaluation
We evaluate the model on the test data to assess its performance. This evaluation is done in the evaluation.py script.

- Script: evaluation.py

### Training with Boundary Channel
To avoid merges of touching nuclei, we add a boundary channel to the learning objective and retrain the model. This process is handled in the train_with_boundary.py script.

- Script: train_with_boundary.py

### Training with Dice Loss
For robustness against class imbalance, we will use the Dice coefficient as loss

- Script: train_with_dice_loss.py

## Results
The results of the segmentation task jupyter notebook will be stored in the results/ directory as a pdf.


## Acknowledgments
- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention (pp. 234-241). Springer, Cham.
- Kaggle for providing the dataset.
