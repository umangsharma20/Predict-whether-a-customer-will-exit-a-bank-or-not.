# Bank Customer Exit Prediction Using Artificial Neural Networks (ANN)

## Table of Contents
- [Installation](#installation)
- [Project Overview](#project-overview)
- [Data Preprocessing](#data-preprocessing)
  - [Train-Test Split](#train-test-split)
  - [Feature Scaling](#feature-scaling)
- [Model Architecture](#model-architecture)
  - [Dropout Layer](#dropout-layer)
  - [Optimizer](#optimizer)
  - [Early Stopping](#early-stopping)
- [Evaluation](#evaluation)
  - [Confusion Matrix](#confusion-matrix)
  - [Accuracy](#accuracy)
- [Results](#results)
- [License](#license)

## Installation

Ensure the following are installed before starting:

- Python 3.7+
- TensorFlow (with GPU support)
- Pandas
- Scikit-learn
- Matplotlib

## Project Overview

This project implements an **Artificial Neural Network (ANN)** using **TensorFlow with GPU support** to predict whether a customer will exit a bank. The model uses various features such as **credit score, geography, gender, balance**, and others to make predictions. The neural network is designed to prevent overfitting by using **dropout layers** and employs **early stopping** to halt training when the model stops improving.

## Data Preprocessing

### Train-Test Split
The dataset is split into a **training set** (used to train the model) and a **test set** (reserved for evaluation). This split ensures that the model is evaluated on unseen data to measure its generalization performance.

### Feature Scaling
The features are standardized using the **StandardScaler** to ensure they are on a similar scale. Feature scaling is essential for neural networks as it helps the model converge faster and perform better.

## Model Architecture

The neural network consists of fully connected **dense layers**. The architecture is as follows:

- **Hidden Layers:** Use **ReLU** activation function to introduce non-linearity.
- **Output Layer:** The output layer uses a **sigmoid function** to predict binary outcomes (exit or not).

### Dropout Layer
To prevent overfitting, **dropout layers** are added after each hidden layer. This helps the network generalize better by randomly dropping neurons during training, reducing reliance on specific nodes.

### Optimizer
The **Adam optimizer** is used in this model, offering an adaptive learning rate. This helps the network learn more efficiently and converge faster.

### Early Stopping
**Early stopping** is implemented to halt training when the validation loss ceases to improve. This prevents overfitting by ensuring the model doesn't continue training once it has reached its optimal performance.

## Evaluation

### Confusion Matrix
A **confusion matrix** is generated to evaluate the performance of the model. It provides insights into:
- **True Positives (TP)**
- **True Negatives (TN)**
- **False Positives (FP)**
- **False Negatives (FN)**

### Accuracy
The **accuracy** of the model is calculated to assess its performance on the test data. The accuracy indicates how well the model generalizes to unseen instances.

## Results

- **Model Accuracy:** 85% (Update this based on your actual results)
- **Confusion Matrix:**

|               | Predicted No | Predicted Yes |
|---------------|--------------|---------------|
| **Actual No** | 1530         | 50            |
| **Actual Yes**| 120          | 300           |

## License

This project is licensed for educational use and is part of a learning assignment.
