# A B Keshav Kumar - AE24S021
# Assignment 1 - Fashion MNIST


## 🔗 Links:
- **Weights & Biases (WandB) Link**: [https://wandb.ai/ae24s021-indian-institute-of-technology-madras/fashion_mnist_sweep_clean/reports/DA6401-Assignment-1-Submission-AE24S021-A-B-Keshav-Kumar---VmlldzoxMTY4MDM2NA]
- **GitHub Repository**: [https://github.com/Keshav-iitm/assignment-1]



##  Instructions:
- **Question 7**: The code is standalone as it implements the **confusion matrix with the best model**.
- **All other questions**: They work normally using the **argparse functions** provided in the assignment.
- **Question 3**: For each **backpropagation model**, the **accuracy will be printed**.

# Fashion-MNIST Classification with WandB Hyperparameter Sweeps

## Overview

This project implements a **fully connected neural network** for classifying the **Fashion-MNIST** dataset. The model is trained from scratch using multiple optimization algorithms and activation functions. **We use Weights & Biases (wandb) to track experiments and perform hyperparameter tuning.**

## Features

- Implements multiple optimizers: **SGD, Momentum, NAG, RMSprop, Adam, Nadam**
- Supports various activation functions: **Sigmoid, Tanh, ReLU**
- Hyperparameter tuning using **wandb sweeps**
- Model evaluation using **confusion matrix and accuracy tracking**


## Installation
Ensure you have Python installed and install the required dependencies:
 '''bash
pip install numpy keras wandb matplotlib scikit-learn

## Dataset
The Fashion-MNIST dataset is automatically downloaded using Keras:
- 10 classes of clothing items
- 60,000 training images and 10,000 test images
- Each image is 28x28 grayscale

## Training the Model
1. Initialize WandB
Before training, log in to Weights & Biases:
'''bash
wandb login
2. Run the Training Script
'''bash
python train.py

This will:
Load the dataset
Train the model using the selected optimizer
Log the training loss, validation loss, and accuracy to wandb

3. Hyperparameter Sweeps
To optimize hyperparameters using wandb sweeps, run:
'''bash
python sweep.py

This will:
Create a sweep with different hyperparameters
Train multiple models and track results in wandb

4. Evaluating the Model
  - Confusion Matrix
  - Access Results in WandB
All logs, including losses, accuracy, and hyperparameter comparisons, can be viewed on the wandb project dashboard.

## Contributer:
A B Keshav Kumar (AE24S021)
Indian Institute of Technology, Madras.
