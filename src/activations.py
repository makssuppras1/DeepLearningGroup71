# Activaton functons for neural networks
import numpy as np

def relu(x):
    # ReLU: max(0, x)
    return np.maximum(0, x)

def relu_derivative(x):
    # Derivitive of ReLU: 1 if x > 0, else 0
    return (x > 0).astype(float)

def sigmoid(x):
    # Sigmoid: 1 / (1 + exp(-x)), cliped for stability
    x_clipped = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x_clipped))

def sigmoid_derivative(x):
    # Derivative: sigmoid(x) * (1 - sigmoid(x))
    return sigmoid(x) * (1 - sigmoid(x))

def tanh(x):
    # Tanh activaton
    return np.tanh(x)

def tanh_derivative(x):
    # Derivative: 1 - tanh(x)^2
    return 1 - np.tanh(x)**2

def softmax(x):
    # Softmax for output layer, subtract max for numrical stability
    x_shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def softmax_derivative(x):
    # Derivative of softmax
    return softmax(x) - np.square(softmax(x))

ACTIVATION_FUNCTIONS = {
    'relu': relu,
    'sigmoid': sigmoid,
    'tanh': tanh,
    'softmax': softmax
}

ACTIVATION_DERIVATIVES = {
    'relu': relu_derivative,
    'sigmoid': sigmoid_derivative,
    'tanh': tanh_derivative,
    'softmax': softmax_derivative
}

def get_activation(name):
    # Get activaton function by name
    if name not in ACTIVATION_FUNCTIONS:
        raise ValueError(f"Unknown activation function: {name}")
    return ACTIVATION_FUNCTIONS[name]

def get_activation_derivative(name):
    # Get derivitive by name
    if name not in ACTIVATION_DERIVATIVES:
        raise ValueError(f"Unknown activation function: {name}")
    return ACTIVATION_DERIVATIVES[name]
