This project implements an Artificial Neural Network (ANN) using TensorFlow with GPU support to predict whether a customer will exit a bank or not. The project includes steps for loading the dataset, performing a train-test split, feature scaling, and building the neural network model using dropout layers to prevent overfitting. Early stopping is employed to halt training when the model stops improving.

Table of Contents
Installation
Project Overview
Data Preprocessing
Train-Test Split
Feature Scaling
Model Architecture
Dropout Layer
Optimizer
Early Stopping
Evaluation
Confusion Matrix
Accuracy
Results
License
Installation
Ensure the following are installed before starting:

Python 3.7+
TensorFlow (with GPU support)
Pandas
Scikit-learn
Matplotlib
Project Overview
The aim is to build a model that predicts whether a bank customer will leave, based on various features like credit score, geography, gender, balance, and others. The neural network is built using TensorFlow with GPU acceleration for faster training. Dropout layers are added to reduce overfitting, the Adam optimizer is used to minimize the loss, and early stopping is applied to halt training when validation loss does not improve.

Data Preprocessing
Train-Test Split
The dataset is split into training and testing sets. The training set is used to train the model, and the test set is reserved for evaluation.

Feature Scaling
A StandardScaler is applied to transform the features to ensure they are on a similar scale, helping the neural network converge faster and perform better.

Model Architecture
The model is built with fully connected dense layers. The ReLU activation function is used for hidden layers, while the output layer uses a sigmoid function.

Dropout Layer
Dropout layers are added after each hidden layer to prevent overfitting by randomly dropping neurons during training, encouraging the network to generalize better.

Optimizer
The Adam optimizer is used in this model, offering an adaptive learning rate that helps the network learn faster and more efficiently.

Early Stopping
Early stopping is implemented to halt the training process once the validation loss ceases to improve, ensuring the model does not overfit to the training data.

Evaluation
Confusion Matrix
A confusion matrix is generated to evaluate model performance. It provides insights into true positives, true negatives, false positives, and false negatives.

Accuracy
The accuracy of the model is calculated to assess how well it generalizes to unseen test data.

Results
Model Accuracy: 85% (example; update based on your actual results)
Confusion Matrix:
Predicted No	Predicted Yes
Actual No	1530	50
Actual Yes	120	300
