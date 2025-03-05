# NAME :  A B KESHAV KUMAR
# REG  :  AE24S021
# DA6401 - INTRODUCTION TO DEEP LEARNING (JAN-MAY 2025)
#ASSIGNMENT - 1  
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import argparse
import numpy as np
import wandb
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist  # Using this for data loading, which is allowed

#----------------------------------------ARGUMENT PARSING----------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, default="fashion_mnist", choices=["mnist", "fashion_mnist"])
parser.add_argument("-e", "--epochs", type=int, default=10)
parser.add_argument("-b", "--batch_size", type=int, default=32)
parser.add_argument("-l", "--loss", type=str, default="cross_entropy", choices=["mean_squared_error", "cross_entropy"])
parser.add_argument("-o", "--optimizer", type=str, default="sgd", choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"])
parser.add_argument("-lr", "--learning_rate", type=float, default=0.01)
parser.add_argument("-m", "--momentum", type=float, default=0.9)
parser.add_argument("-beta", "--beta", type=float, default=0.9)
parser.add_argument("-beta1", "--beta1", type=float, default=0.9)
parser.add_argument("-beta2", "--beta2", type=float, default=0.999)
parser.add_argument("-eps", "--epsilon", type=float, default=1e-8)
parser.add_argument("-w_d", "--weight_decay", type=float, default=0.0)
parser.add_argument("-w_i", "--weight_init", type=str, default="random", choices=["random", "Xavier"])
parser.add_argument("-nhl", "--num_layers", type=int, default=2)
parser.add_argument("-sz", "--hidden_size", type=int, default=128)
parser.add_argument("-a", "--activation", type=str, default="sigmoid", choices=["identity", "sigmoid", "tanh", "ReLU"])

args = parser.parse_args()

# Data Preprocessing:
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

#------------------------------------------Question 1 - Visualizing sample images------------------------------------------------------------------------------------------------------
def visualize_samples(X, y, num_samples=5):
    plt.figure(figsize=(12, 3))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i+1)
        plt.imshow(X[i], cmap='gray')
        plt.title(f"Label: {y[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


visualize_samples(X_train, y_train)

X_train = X_train.reshape(X_train.shape[0], -1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], -1).astype('float32') / 255

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

#-------------------------------------------Question 2 - Feedforward NeuralNetwork-------------------------------------------------------------------------------------------

class FeedforwardNeuralNetwork:
    def __init__(self, input_size, num_classes, num_layers, hidden_size, activation):
        self.input_size = input_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.activation = activation
        
        self.weights = []
        self.biases = []
        
        # Input layer to first hidden layer
        self.weights.append(np.random.randn(hidden_size, input_size) * 0.01)
        self.biases.append(np.zeros((hidden_size, 1)))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.weights.append(np.random.randn(hidden_size, hidden_size) * 0.01)
            self.biases.append(np.zeros((hidden_size, 1)))
        
        # Last hidden layer to output layer
        self.weights.append(np.random.randn(num_classes, hidden_size) * 0.01)
        self.biases.append(np.zeros((num_classes, 1)))

    def forward(self, X):
        self.Z = []
        self.A = [X.T]  # X is now (batch_size, 784)
        for i in range(self.num_layers):
            z = np.dot(self.weights[i], self.A[-1]) + self.biases[i]
            a = self.activate(z)
            self.Z.append(z)
            self.A.append(a)
    
        # Output layer
        z = np.dot(self.weights[-1], self.A[-1]) + self.biases[-1]
        self.Z.append(z)
        self.A.append(self.softmax(z))
    
        return self.A[-1].T


    def activate(self, Z):
        if self.activation == "sigmoid":
            return 1 / (1 + np.exp(-Z))
        elif self.activation == "tanh":
            return np.tanh(Z)
        elif self.activation == "ReLU":
            return np.maximum(0, Z)
        else:  # identity
            return Z

    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
    