# Neural Network from Scratch with NumPy

## Project Overview
This project involves implementing a fully-connected feedforward neural network (FFN) from scratch using only NumPy. The goal is to understand the fundamentals of deep learning by building all components manually.

## Team
Group 71 - Deep Learning Course

## Objectives
- Implement forward and backward propagation from scratch
- Train on Fashion-MNIST and CIFAR-10 datasets
- Experiment with different optimizers, activation functions, and hyperparameters
- Track experiments using Weights & Biases (WandB)
- Analyze model performance through visualizations and metrics

## Project Structure
```
DeepLearningGroup71/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
├── data/                    # Dataset storage (not tracked in git)
├── src/                     # Source code
│   ├── __init__.py
│   ├── neural_network.py    # Main FFNN class
│   ├── layers.py            # Layer implementations
│   ├── activations.py       # Activation functions
│   ├── losses.py            # Loss functions
│   ├── optimizers.py        # Optimization algorithms
│   ├── initializers.py      # Weight initialization methods
│   ├── data_loader.py       # Dataset loading utilities
│   └── utils.py             # Helper functions
├── experiments/             # Experiment scripts
│   ├── train.py            # Main training script
│   ├── evaluate.py         # Model evaluation
│   └── sweep_config.py     # WandB sweep configurations
├── notebooks/              # Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_testing.ipynb
│   └── 03_results_analysis.ipynb
├── configs/                # Configuration files
│   └── default_config.yaml
└── results/                # Saved models and outputs
    ├── models/
    ├── plots/
    └── logs/
```

## Datasets
- **Fashion-MNIST**: 60,000 training images, 10,000 test images (28x28 grayscale)
- **CIFAR-10**: 50,000 training images, 10,000 test images (32x32 RGB)

## Key Components to Implement

### 1. Neural Network Class
- Configurable architecture (number of layers, units per layer)
- Forward propagation
- Backward propagation
- Training loop with mini-batch gradient descent

### 2. Activation Functions
- ReLU
- Sigmoid
- Tanh
- Softmax (output layer)

### 3. Loss Functions
- Mean Squared Error (MSE)
- Cross-Entropy Loss
- L2 Regularization

### 4. Optimizers
- Vanilla Gradient Descent (SGD)
- SGD with Momentum
- RMSprop
- Adam

### 5. Weight Initialization
- Random initialization
- Xavier/Glorot initialization
- He initialization

## Hyperparameters to Experiment With
- Number of epochs
- Number of hidden layers
- Number of hidden units per layer
- Learning rate
- Batch size
- L2 regularization coefficient
- Optimizer choice
- Activation function choice
- Weight initialization method

## Getting Started

### Requirements
- Python 3.12+ (recommended) or Python 3.10+
- pip (latest version)
- Git

### Installation
```bash
# Create a virtual environment with Python 3.12
python3.12 -m venv deeplearning

# Activate the virtual environment
source deeplearning/bin/activate  # On Windows: deeplearning\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Install Jupyter kernel
python -m ipykernel install --user --name=deeplearning --display-name="Deep Learning (3.12)"
```

### Setting up Weights & Biases
```bash
# Login to WandB
wandb login

# Your API key will be saved for future runs
```

### Running Experiments
```bash
# Train a model with default configuration
python experiments/train.py

# Run a hyperparameter sweep
wandb sweep experiments/sweep_config.py
wandb agent <sweep_id>
```

## Experiment Tracking
All experiments will be logged to Weights & Biases including:
- Training and validation loss curves
- Accuracy metrics
- Confusion matrices
- Parameter histograms
- Gradient statistics
- Hyperparameter comparisons

## Evaluation Metrics
- Accuracy
- Precision, Recall, F1-Score
- Confusion Matrix
- Loss curves (train/validation)

## Resources
- [Fashion-MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Weights & Biases Documentation](https://docs.wandb.ai/)
- [NumPy Documentation](https://numpy.org/doc/)

## Notes
- This project uses **only NumPy** for neural network implementation
- No TensorFlow, PyTorch, or similar libraries are allowed for the core implementation
- Focus on understanding the mathematics and algorithms behind neural networks

## TODO
- [ ] Implement neural network class
- [ ] Implement activation functions
- [ ] Implement loss functions
- [ ] Implement optimizers
- [ ] Implement weight initializers
- [ ] Create data loading pipeline
- [ ] Build training loop
- [ ] Add evaluation metrics
- [ ] Set up WandB logging
- [ ] Run baseline experiments
- [ ] Conduct hyperparameter sweeps
- [ ] Analyze and document results

## License
MIT License - Feel free to use for educational purposes

