# Image Classification with ResNet

The goal of this project is to compare the performance of different models on the Image Classification task, trained on the CIFAR-10 dataset. 
These are mainly convolutional neural networks (CNNs) and residual neural networks (ResNets)

To understand the background of this exercise you can read the ResNet publication: https://arxiv.org/abs/1512.03385



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
  - [Loading Data](#loading-data)
  - [1_layer_CNN](#1-layer-CNN)
  - [1_layer_CNN_training](#1-layer-CNN-training)
  - [4_layer_CNN](#4-layer-CNN)
  - [4_layer_CNN_training](#4-layer-CNN-training)
  - [ResNet](#ResNet)
  - [ResNet_training](#ResNet-training)
  - [plain_ResNet](#plain-ResNet)
  - [plain_ResNet_training](#plain-ResNet-training)
  - [ResNet_training_scheduler](#ResNet-training-scheduler)
- [Results](#results)
- [Acknowledgments](#acknowledgments)

## Project Overview

In this project, we will implement three different architectures for the image classification task. 
More specifically two CNNs consisting of 1 and 4 layers, and a ResNet. In the last part of the project we will also test how the ResNet responds to learning rate scheduling. 

## Directory Structure

```plaintext
Image-Classification-with-ResNet/
│
├── README.md
├── .gitignore
├── notebooks/
│   └── Image_Classification_with_ResNet.ipynb
├── src/
│   ├── import_libraries.py
│   ├── load_data.py
│   ├── 1_layer_CNN.py
│   ├── 1_layer_CNN_training.py
│   ├── 4_layer_CNN.py
│   ├── 4_layer_CNN_training.py
│   ├── ResNet.py
│   ├── ResNet_training.py
│   ├── plain_ResNet.py
│   ├── plain_ResNet_training.py
│   ├── ResNet_training_scheduler.py
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
git clone https://github.com/PabloJM21/Image-Classification-with-ResNet.git
cd Image-Classification-with-ResNet
```
## Downloading the Data

To keep the repository lightweight, the data is not included. You can download the data by running the provided script.

```sh
pip install requests
python download_data.py

```

## Installing Dependencies

Install the required Python packages:

```sh
pip install -r requirements.txt
```

## Running the Notebook

After downloading the data, you can start the Jupyter notebook:



```sh
jupyter notebook notebooks/Image_Classification_with_ResNet.ipynb
```

## Project Details
[Loading Data](#loading-data)
  - [1_layer_CNN](#1-layer-CNN)
  - [1_layer_CNN_training](#1-layer-CNN-training)
  - [4_layer_CNN](#4-layer-CNN)
  - [4_layer_CNN_training](#4-layer-CNN-training)
  - [ResNet](#ResNet)
  - [ResNet_training](#ResNet-training)
  - [plain_ResNet](#plain-ResNet)
  - [plain_ResNet_training](#plain-ResNet-training)
  - [ResNet_training_scheduler](#ResNet-training-scheduler)

### Importing libraries
First of all we import all required python libraries for completing the task.

- Script: import_libraries.py

### Loading Data

In this step, we will load the data from the dataset. 

- Script: load_data.py

### 1_layer_CNN

We define a 1-layer convolutional neural network

- Script: 1_layer_CNN.py

### 1_layer_CNN_training

We define the training functions and train the model, plotting the results of loss and metrics. 

- Script: 1_layer_CNN_training.py

### 4_layer_CNN

We define a 4-layer convolutional neural network

- Script: 4_layer_CNN.py

### 4_layer_CNN_training

We define the training functions and train the model, plotting the results of loss and metrics. 

- Script: 4_layer_CNN_training.py

### ResNet

We define a residual neural network

- Script: Resnet.py

### ResNet_training

We define the training functions and train the model, plotting the results of loss and metrics. 

- Script: ResNet_training.py

### plain_ResNet

We define a plain residual neural network (that is, with residual connections disabled)

- Script: plain_Resnet.py

### plain_ResNet_training

We define the training functions and train the model, plotting the results of loss and metrics. 

- Script: plain_ResNet_training.py

### ResNet_training_scheduler

First we incorporate a scheduler in the training function. Then we train the model with different parameter values, plotting the results of loss and metrics. 

- Script: plain_ResNet_training.py

## Results
The results of the task can be viewed in the jupyter notebook, in the notebooks folder. 


## Acknowledgments
- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention (pp. 234-241). Springer, Cham.
- Kaggle for providing the dataset.
