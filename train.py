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
    
    #--------------------------------------------Question 3 : Backpropogation-------------------------------------------------------------------------------------------------------    
    def backward(self, X, y):
        m = X.shape[0]
        y_encoded = self.one_hot_encode(y)
        
        dW = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]
        
        # Output layer
        dZ = self.A[-1] - y_encoded.T
        dW[-1] = (1/m) * np.dot(dZ, self.A[-2].T)
        db[-1] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        
        # Hidden layers
        for l in reversed(range(self.num_layers)):
            dA = np.dot(self.weights[l+1].T, dZ)
            dZ = dA * self.activate_derivative(self.Z[l])
            dW[l] = (1/m) * np.dot(dZ, self.A[l].T)
            db[l] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        
        return dW, db

    def activate_derivative(self, Z):
        if self.activation == "sigmoid":
            return self.activate(Z) * (1 - self.activate(Z))
        elif self.activation == "tanh":
            return 1 - np.tanh(Z)**2
        elif self.activation == "ReLU":
            return (Z > 0).astype(float)
        else:  # identity
            return np.ones_like(Z)

    def one_hot_encode(self, y):
        encoded = np.zeros((y.shape[0], self.num_classes))
        encoded[np.arange(y.shape[0]), y] = 1
        return encoded

# Creating neural network
nn = FeedforwardNeuralNetwork(input_size=784, num_classes=10, num_layers=args.num_layers, 
                              hidden_size=args.hidden_size, activation=args.activation)

# Testing forward pass
output = nn.forward(X_train[:args.batch_size])
print("Forward pass output shape:", output.shape)
print("Sample output probabilities:")
print(output[:5])  # Print first 5 samples

# Testing backward pass
dW, db = nn.backward(X_train[:args.batch_size], y_train[:args.batch_size])
print("Backward pass output shapes:")
print("dW:", [w.shape for w in dW])
print("db:", [b.shape for b in db])

print("Basic implementation complete. Ready for optimizer implementation.")

#......................................Optimisation functions............................................................................................

#SDG
import numpy as np

class SGD:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update(self, weights, biases, gradients):
        dW, db = gradients
        for i in range(len(weights)):
            weights[i] -= self.learning_rate * dW[i]
            biases[i] -= self.learning_rate * db[i]
        return weights, biases

class Momentum:
    def __init__(self, learning_rate, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None

    def update(self, weights, biases, gradients):
        if self.velocity is None:
            self.velocity = [(np.zeros_like(w), np.zeros_like(b)) for w, b in zip(weights, biases)]

        dW, db = gradients
        for i, ((w, b), (dw, db)) in enumerate(zip(zip(weights, biases), zip(dW, db))):
            self.velocity[i] = (
                self.momentum * self.velocity[i][0] + self.learning_rate * dw,
                self.momentum * self.velocity[i][1] + self.learning_rate * db
            )
            weights[i] -= self.velocity[i][0]
            biases[i] -= self.velocity[i][1]
        return weights, biases
    
#NAG
class NAG:
    def __init__(self, learning_rate, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None

    def update(self, weights, biases, gradients):
        if self.velocity is None:
            self.velocity = [(np.zeros_like(w), np.zeros_like(b)) for w, b in zip(weights, biases)]

        dW, db = gradients
        for i, ((w, b), (dw, db)) in enumerate(zip(zip(weights, biases), zip(dW, db))):
            v_w, v_b = self.velocity[i]
            v_w_new = self.momentum * v_w - self.learning_rate * dw
            v_b_new = self.momentum * v_b - self.learning_rate * db
            weights[i] += -self.momentum * v_w + (1 + self.momentum) * v_w_new
            biases[i] += -self.momentum * v_b + (1 + self.momentum) * v_b_new
            self.velocity[i] = (v_w_new, v_b_new)
        return weights, biases

#RMSPROP :
class RMSprop:
    def __init__(self, learning_rate, beta=0.9, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.square_grad = None

    def update(self, weights, biases, gradients):
        if self.square_grad is None:
            self.square_grad = [(np.zeros_like(w), np.zeros_like(b)) for w, b in zip(weights, biases)]

        dW, db = gradients
        for i, ((w, b), (dw, db)) in enumerate(zip(zip(weights, biases), zip(dW, db))):
            self.square_grad[i] = (
                self.beta * self.square_grad[i][0] + (1 - self.beta) * np.square(dw),
                self.beta * self.square_grad[i][1] + (1 - self.beta) * np.square(db)
            )
            weights[i] -= self.learning_rate * dw / (np.sqrt(self.square_grad[i][0] + self.epsilon))
            biases[i] -= self.learning_rate * db / (np.sqrt(self.square_grad[i][1] + self.epsilon))
        return weights, biases

#ADAM :
class Adam:
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, weights, biases, gradients):
        if self.m is None:
            self.m = [(np.zeros_like(w), np.zeros_like(b)) for w, b in zip(weights, biases)]
            self.v = [(np.zeros_like(w), np.zeros_like(b)) for w, b in zip(weights, biases)]

        self.t += 1
        dW, db = gradients
        for i, ((w, b), (dw, db)) in enumerate(zip(zip(weights, biases), zip(dW, db))):
            self.m[i] = (
                self.beta1 * self.m[i][0] + (1 - self.beta1) * dw,
                self.beta1 * self.m[i][1] + (1 - self.beta1) * db
            )
            self.v[i] = (
                self.beta2 * self.v[i][0] + (1 - self.beta2) * np.square(dw),
                self.beta2 * self.v[i][1] + (1 - self.beta2) * np.square(db)
            )
            m_hat = (self.m[i][0] / (1 - self.beta1**self.t), self.m[i][1] / (1 - self.beta1**self.t))
            v_hat = (self.v[i][0] / (1 - self.beta2**self.t), self.v[i][1] / (1 - self.beta2**self.t))
            weights[i] -= self.learning_rate * m_hat[0] / (np.sqrt(v_hat[0]) + self.epsilon)
            biases[i] -= self.learning_rate * m_hat[1] / (np.sqrt(v_hat[1]) + self.epsilon)
        return weights, biases

#NADAM :
class Nadam:
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, weights, biases, gradients):
        if self.m is None:
            self.m = [(np.zeros_like(w), np.zeros_like(b)) for w, b in zip(weights, biases)]
            self.v = [(np.zeros_like(w), np.zeros_like(b)) for w, b in zip(weights, biases)]

        self.t += 1
        dW, db = gradients
        for i, ((w, b), (dw, db)) in enumerate(zip(zip(weights, biases), zip(dW, db))):
            self.m[i] = (
                self.beta1 * self.m[i][0] + (1 - self.beta1) * dw,
                self.beta1 * self.m[i][1] + (1 - self.beta1) * db
            )
            self.v[i] = (
                self.beta2 * self.v[i][0] + (1 - self.beta2) * np.square(dw),
                self.beta2 * self.v[i][1] + (1 - self.beta2) * np.square(db)
            )
            m_hat = (self.m[i][0] / (1 - self.beta1**self.t), self.m[i][1] / (1 - self.beta1**self.t))
            v_hat = (self.v[i][0] / (1 - self.beta2**self.t), self.v[i][1] / (1 - self.beta2**self.t))
            m_hat_next = (
                m_hat[0] * self.beta1 + (1 - self.beta1) * dw / (1 - self.beta1**self.t),
                m_hat[1] * self.beta1 + (1 - self.beta1) * db / (1 - self.beta1**self.t)
            )
            weights[i] -= self.learning_rate * m_hat_next[0] / (np.sqrt(v_hat[0]) + self.epsilon)
            biases[i] -= self.learning_rate * m_hat_next[1] / (np.sqrt(v_hat[1]) + self.epsilon)
        return weights, biases

# Function to train the model and calculate accuracy
def train_and_evaluate(nn, X_train, y_train, X_test, y_test, optimizer, epochs, batch_size):
    for epoch in range(epochs):
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            
            _ = nn.forward(X_batch)
            gradients = nn.backward(X_batch, y_batch)
            nn.weights, nn.biases = optimizer.update(nn.weights, nn.biases, gradients)
        
        train_accuracy = calculate_accuracy(nn, X_train, y_train)
        print(f"Epoch {epoch+1}/{epochs}, Training Accuracy: {train_accuracy:.2f}%")
    
    test_accuracy = calculate_accuracy(nn, X_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    return test_accuracy

def calculate_accuracy(nn, X, y):
    predictions = np.argmax(nn.forward(X), axis=1)
    return np.mean(predictions == y) * 100

# Testing each optimizer
optimizers = {
    "SGD": SGD(args.learning_rate),
    "Momentum": Momentum(args.learning_rate, args.momentum),
    "NAG": NAG(args.learning_rate, args.momentum),
    "RMSprop": RMSprop(args.learning_rate, args.beta),
    "Adam": Adam(args.learning_rate, args.beta1, args.beta2),
    "Nadam": Nadam(args.learning_rate, args.beta1, args.beta2)
}

results = {}

#Checking accuracy for each optimizer.
for name, optimizer in optimizers.items():
    print(f"\nTraining with {name} optimizer:")
    nn = FeedforwardNeuralNetwork(input_size=784, num_classes=10, num_layers=args.num_layers, 
                                  hidden_size=args.hidden_size, activation=args.activation)
    accuracy = train_and_evaluate(nn, X_train, y_train, X_test, y_test, optimizer, args.epochs, args.batch_size)
    results[name] = accuracy

print("\nFinal Test Accuracies:")
for name, accuracy in results.items():
    print(f"{name}: {accuracy:.2f}%")