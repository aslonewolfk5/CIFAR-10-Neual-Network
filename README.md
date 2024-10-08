# CIFAR-10 Image Classification with Keras

This repository contains a deep learning project for classifying images from the CIFAR-10 dataset using Keras. The project demonstrates the process of building and training neural network models, as well as evaluating their performance.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Data Description](#data-description)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Visualization](#visualization)
- [License](#license)

## Overview

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. The project aims to classify these images using neural networks built with Keras.

## Installation

To run this project, you need to have the following libraries installed:

```bash
pip install numpy tensorflow keras matplotlib scikit-learn
Data Description
The CIFAR-10 dataset contains the following classes:

Plane
Car
Bird
Cat
Deer
Dog
Frog
Horse
Ship
Truck
The dataset is split into 50,000 training images and 10,000 test images.

Model Architecture
Three different neural network models were created using Keras:

Model 1: A basic feedforward neural network.
Model 2: A network with dropout for regularization.
Model 3: An enhanced version of Model 2 with additional dropout layers.
Training
The models were trained on the training set with the following parameters:

Batch Size: 1000
Epochs: 100-150 depending on the model
TensorBoard was used for visualization during training.

Evaluation
The trained models were evaluated on the test dataset to obtain the loss and accuracy metrics.

Results
The performance of Model 2 resulted in:

Test Loss: 1.42
Test Accuracy: 49.1%
The confusion matrix and metrics such as recall, precision, and F1 score were calculated to further analyze the model's performance.

Visualization
The confusion matrix is visualized using Matplotlib.

License
This project is licensed under the MIT License. See the LICENSE file for more details.
