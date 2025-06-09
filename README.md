# Basic Feed forward and Backpropogation
This repository was submitted as a part of Assignment 1 - DA6401, Deep Learning, IIT Madras.


## 🔗 Links:
- **Detailed report Weights & Biases (WandB) Link**: [[https://wandb.ai/ae24s021-indian-institute-of-technology-madras/fashion_mnist_sweep_clean/reports/DA6401-Assignment-1-Submission-AE24S021-A-B-Keshav-Kumar---VmlldzoxMTY4MDM2NA](https://api.wandb.ai/links/ae24s021-indian-institute-of-technology-madras/opeulyn5)]
- **GitHub Repository**: [https://github.com/Keshav-iitm/assignment-1]


This project implements a custom feedforward neural network for the Fashion MNIST dataset, with support for multiple optimizers and hyperparameter tuning using Weights & Biases (WandB).

## Features

- **Custom Feedforward Neural Network**: Implemented from scratch in Python.
- **Multiple Optimizers**: Supports SGD, Momentum, NAG, RMSprop, Adam, and Nadam.
- **Hyperparameter Tuning**: Utilizes WandB for sweeping and tracking experiments.
- **Visualization**: Includes sample image visualization, accuracy tracking, and confusion matrix plotting.
- **Command-Line Arguments**: Allows flexible configuration of model parameters.

## Requirements

- **Python 3.7+**
- **Required Libraries**:
  - `numpy`
  - `matplotlib`
  - `keras` (for dataset loading)
  - `scikit-learn`
  - `wandb`

Install the dependencies using:

pip install numpy matplotlib keras scikit-learn wandb

text

## Usage

1. **Clone the repository or copy the script** (e.g., `main.py`).

2. **Run the script**:

python main.py

text

By default, this will train a neural network on the Fashion MNIST dataset with default parameters.

3. **Customize training**:

You can specify arguments such as the optimizer, learning rate, batch size, etc. For example:

python main.py --optimizer adam --learning_rate 0.001 --batch_size 64 --num_layers 3 --hidden_size 128

text

For a full list of arguments, see the script's argument parser.

4. **Hyperparameter Sweep with WandB**:

To launch a WandB sweep (requires a WandB account and login):

wandb login
python main.py

text

The sweep configuration is defined in the script. You can adjust the parameters in the `sweep_config` dictionary.

5. **Outputs**:
- **Sample images**: Displayed at the start of training.
- **Training logs**: Printed to the console.
- **Confusion matrix**: Generated and logged to WandB if a sweep is active.

## Example Command
python main.py --optimizer nadam --learning_rate 0.001 --batch_size 64 --num_layers 5 --hidden_size 64 --activation ReLU
text

## WandB Sweep Configuration

The script supports WandB sweeps for hyperparameter optimization. The sweep configuration is set in the script:
sweep_config = {
"method": "bayes",
"metric": {"name": "val_accuracy", "goal": "maximize"},
"parameters": {
"epochs": {"values": [5,
"num_layers": {"values": },
"hidden_size": {"values": },
"weight_decay": {"values": [0, 0.0005, 0.5]},
"learning_rate": {"values": [1e-3, 1e-4]},
"optimizer": {"values": ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]},
"batch_size": {"values": },
"weight_init": {"values": ["random", "Xavier"]},
"activation": {"values": ["sigmoid", "tanh", "ReLU"]},
},
"name": "hl_{num_layers}bs{batch_size}ac{activation}",
}

text

## File Structure
- **main.py**: Main script for training and evaluation.
- **README.md**: This file.

## Notes
- **WandB Integration**: To use WandB sweeps, ensure you have logged in and have a project created.
- **Dataset**: Uses the Fashion MNIST dataset, loaded via `keras.datasets.fashion_mnist`.
- **Confusion Matrix**: Generated for the best model and logged to WandB.

## Contributer:
A B Keshav Kumar 
Indian Institute of Technology, Madras.
