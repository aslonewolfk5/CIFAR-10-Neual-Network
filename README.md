# CIFAR-10 Image Classification with Keras

This project is a deep learning-based image classification system using the CIFAR-10 dataset. It demonstrates the creation, training, and evaluation of neural networks to classify images into one of 10 categories.

---

## Features

- **CIFAR-10 Dataset**: A well-known dataset containing 60,000 images (32x32 resolution) across 10 categories.
- **Neural Network Models**: Implementation of three models, including dropout regularization for improved performance.
- **Training and Evaluation**: Focused on hyperparameter tuning and performance evaluation using metrics like precision, recall, and F1 score.
- **Visualization Tools**: TensorBoard and Matplotlib were utilized for tracking and visualizing training progress and evaluation results.

---

## Technologies Used

- **Python 3.x**
- **Keras**: For constructing and training deep learning models.
- **TensorFlow**: Backend for Keras.
- **NumPy**: For numerical computations.
- **Matplotlib**: To create visualizations, including confusion matrices.
- **Scikit-learn**: For evaluation metrics like precision, recall, and F1 score.

---

## Setup Instructions

### 1. Clone the Repository

Clone this repository to your local system:

```bash
git clone https://github.com/yourusername/cifar10-image-classification.git
```

### 2. Install Dependencies

Install the required Python libraries:

```bash
pip install numpy tensorflow keras matplotlib scikit-learn
```

---

## Data Description

The **CIFAR-10 dataset** includes:

- **Training Set**: 50,000 images.
- **Test Set**: 10,000 images.

The images belong to one of the following 10 classes:

- **Plane**
- **Car**
- **Bird**
- **Cat**
- **Deer**
- **Dog**
- **Frog**
- **Horse**
- **Ship**
- **Truck**

---

## Model Architecture

Three neural network models were created using Keras:

1. **Model 1**: A basic feedforward neural network.
2. **Model 2**: Enhanced with dropout for regularization.
3. **Model 3**: Further improvements with additional dropout layers.

---

## Training

The models were trained on the CIFAR-10 training set using the following parameters:

- **Batch Size**: 1000
- **Epochs**: 100-150 (depending on the model)
- **Visualization**: Training progress was tracked using TensorBoard.

---

## Evaluation

The models were evaluated on the test dataset to calculate:

- **Loss**
- **Accuracy**

Additionally, a confusion matrix was generated, and metrics like precision, recall, and F1 score were calculated using Scikit-learn.

---

## Results

The performance of **Model 2** yielded the following results:

- **Test Loss**: 1.42
- **Test Accuracy**: 49.1%

Detailed analysis was performed using the confusion matrix and other evaluation metrics.

---

## Visualization

- **Confusion Matrix**: Visualized using Matplotlib to better understand model performance.

---
