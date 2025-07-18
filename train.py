# A B KESHAV KUMAR
# DA6401 - DEEP LEARNING (JAN-MAY 2025)

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

#------------------------------------------Visualizing sample images------------------------------------------------------------------------------------------------------
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

#-------------------------------------------Feedforward NeuralNetwork-------------------------------------------------------------------------------------------

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
    
    #--------------------------------------------Backpropogation-------------------------------------------------------------------------------------------------------    
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



#--------------------------------------------------WANDB----------------------------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split

# Defining a dictionary for optimizers with lower-case keys
optimizers_sweep = {
    "sgd": SGD,
    "momentum": Momentum,
    "nag": NAG,
    "rmsprop": RMSprop,
    "adam": Adam,
    "nadam": Nadam
}

# Auxiliary function for computing cross-entropy loss
def cross_entropy_loss(nn, X, y):
    outputs = nn.forward(X)  # shape (m, num_classes)
    m = X.shape[0]
    # One-hot encode labels
    y_encoded = np.eye(nn.num_classes)[y]
    loss = -np.sum(y_encoded * np.log(outputs + 1e-8)) / m
    return loss

# Sweep training function
def sweep_train():
    run = wandb.init()   # Initializing Wandb for this run
    config = wandb.config

    #meaningful run name (e.g., hl_3_bs_16_ac_tanh)
    run.name = f"hl_{config.num_layers}_bs_{config.batch_size}_ac_{config.activation}"
    run.save()

    # Loading data from Fashion MNIST and spliting into train/validation/test
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )

    # Preprocess: flatten and normalize
    X_train = X_train.reshape(X_train.shape[0], -1).astype("float32") / 255
    X_val = X_val.reshape(X_val.shape[0], -1).astype("float32") / 255
    X_test = X_test.reshape(X_test.shape[0], -1).astype("float32") / 255

    # Creating a new neural network instance with hyperparameters from wandb.config
    sweep_nn = FeedforwardNeuralNetwork(
        input_size=784,
        num_classes=10,
        num_layers=config.num_layers,
        hidden_size=config.hidden_size,
        activation=config.activation,
    )

    # Creating the optimizer instance using our sweep dictionary (keys in lower-case)
    chosen_opt = config.optimizer.lower()
    optimizer_class = optimizers_sweep.get(chosen_opt)
    if optimizer_class is None:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")
    sweep_optimizer = optimizer_class(config.learning_rate)

    # Training loop
    for epoch in range(config.epochs):
        for i in range(0, len(X_train), config.batch_size):
            X_batch = X_train[i : i + config.batch_size]
            y_batch = y_train[i : i + config.batch_size]

            _ = sweep_nn.forward(X_batch)
            gradients = sweep_nn.backward(X_batch, y_batch)
            sweep_nn.weights, sweep_nn.biases = sweep_optimizer.update(
                sweep_nn.weights, sweep_nn.biases, gradients
            )

        # Computing metrics after each epoch
        train_acc = calculate_accuracy(sweep_nn, X_train, y_train)
        val_acc = calculate_accuracy(sweep_nn, X_val, y_val)
        train_loss = cross_entropy_loss(sweep_nn, X_train, y_train)
        val_loss = cross_entropy_loss(sweep_nn, X_val, y_val)

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
        })

# Sweep configuration (with a meaningful name format)
sweep_config = {
    "method": "bayes",
    "metric": {"name": "val_accuracy", "goal": "maximize"},
    "parameters": {
        "epochs": {"values": [5, 10]},
        "num_layers": {"values": [3, 4, 5]},
        "hidden_size": {"values": [32, 64, 128]},
        "weight_decay": {"values": [0, 0.0005, 0.5]},
        "learning_rate": {"values": [1e-3, 1e-4]},
        "optimizer": {"values": ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]},
        "batch_size": {"values": [16, 32, 64]},
        "weight_init": {"values": ["random", "Xavier"]},
        "activation": {"values": ["sigmoid", "tanh", "ReLU"]},
    },
    "name": "hl_{num_layers}_bs_{batch_size}_ac_{activation}",
}

# Initializing the sweep 
sweep_id = wandb.sweep(sweep_config, project="fashion_mnist_sweep_clean")

# Launch the sweep agent, 60 trials
wandb.agent(sweep_id, sweep_train, count=10)
#-------------------------------------------------------Confusion Matrix----------------------------------------------------------------------------



matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from keras.datasets import fashion_mnist


sweep_config = {
    "method": "grid",
    "name": "hl_5_bs_64_ac_ReLU",  
    "metric": {"name": "test_accuracy", "goal": "maximize"},
    "parameters": {
        "epochs": {"values": [5]},
        "batch_size": {"values": [16]},
        "learning_rate": {"values": [0.001]},
        "num_layers": {"values": [5]},
        "hidden_size": {"values": [64]},  # Best model: hl_5_bs_64_ac_ReLU
        "activation": {"values": ["ReLU"]},
        "optimizer": {"values": ["nadam"]}  
    }
}

# Initializing the sweep
sweep_id = wandb.sweep(sweep_config, project="fashion_mnist_sweep")

# Defining training function for sweep
def sweep_train():
    run_name = f"hl_5_bs_64_ac_ReLU"  
    run = wandb.init(name=run_name)
    config = wandb.config

    # Loading Fashion MNIST 
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    # Preprocess data
    X_train = X_train.reshape(X_train.shape[0], -1).astype("float32") / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1).astype("float32") / 255.0

    # Instantiating model with current hyperparameters
    model = FeedforwardNeuralNetwork(
        input_size=784,
        num_classes=10,
        num_layers=config.num_layers,
        hidden_size=config.hidden_size,
        activation=config.activation
    )

    optimizer = Nadam(config.learning_rate)

    # Training loop with backpropagation
    num_samples = X_train.shape[0]
    for epoch in range(config.epochs):
        permutation = np.random.permutation(num_samples)
        X_train_shuffled = X_train[permutation]
        y_train_shuffled = y_train[permutation]

        for i in range(0, num_samples, config.batch_size):
            X_batch = X_train_shuffled[i:i + config.batch_size]
            y_batch = y_train_shuffled[i:i + config.batch_size]
            _ = model.forward(X_batch)
            gradients = model.backward(X_batch, y_batch)
            model.weights, model.biases = optimizer.update(model.weights, model.biases, gradients)

        # Calculating training accuracy after each epoch, logging in WANDB
        train_acc = calculate_accuracy(model, X_train, y_train)
        print(f"Epoch {epoch + 1}/{config.epochs}: Train Accuracy: {train_acc:.2f}%")
        wandb.log({"Epoch": epoch + 1, "Train Accuracy": train_acc})

    # Evaluating on test set
    test_output = model.forward(X_test)
    predictions = np.argmax(test_output, axis=1)
    test_acc = accuracy_score(y_test, predictions)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    wandb.log({"Test Accuracy": test_acc})

    cm = confusion_matrix(y_test, predictions)

    wandb_table = wandb.Table(columns=["Predicted Label", "Actual Label", "Count"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            wandb_table.add_data(str(i), str(j), cm[i, j])

    wandb.log({"Confusion Matrix Table": wandb_table})

    #Confusion matrix for the table
    class_names = [f'Class {i}' for i in range(cm.shape[0])]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, cmap="viridis", values_format="d")
    ax.set_title(f"Confusion Matrix (Run: {run.name})", fontsize=16)
    plt.tight_layout()
    wandb.log({"Confusion Matrix": wandb.Image(fig)})
    plt.close(fig)

wandb.agent(sweep_id, function=sweep_train)
