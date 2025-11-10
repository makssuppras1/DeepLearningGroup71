"""
Activation functions and their derivatives.

Implement various activation functions used in neural networks.
Each function should have both forward and backward (derivative) computations.
"""

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    """
    Rectified Linear Unit (ReLU) activation function.
    Formula: f(x) = max(0, x)
    """
    return np.maximum(0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    """
    Derivative of ReLU activation.
    Returns 1 if x > 0, otherwise 0.
    Formula: f'(x) = 1 if x > 0, else 0
    Args: x: Input numpy array
    Returns: Derivative of ReLU activation
    """
    return (x > 0).astype(float)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation function.
    Formula: f(x) = 1 / (1 + np.exp(-x))
    Args: x: Input numpy array
    Returns: Sigmoid activation function
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """
    Derivative of sigmoid activation.
    Returns the product of the sigmoid function and its complement.
    Formula: f'(x) = f(x) * (1 - f(x))
    """
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x: np.ndarray) -> np.ndarray:
    """
    Tanh activation function.
    Formula: f(x) = np.tanh(x)
    """
    return np.tanh(x)


def tanh_derivative(x: np.ndarray) -> np.ndarray:
    """
    Derivative of tanh activation.
    Returns 1 - tanh(x)^2
    Formula: f'(x) = 1 - tanh(x)^2
    """
    return 1 - np.tanh(x)**2


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Softmax activation function for output layer.
    Returns a probability distribution over classes.
    Formula: f(x_i) = exp(x_i) / sum(exp(x_j)) for each class i.
    """
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


def softmax_derivative(x: np.ndarray) -> np.ndarray:
    """
    Derivative of softmax activation.
    Returns the difference between the softmax function and its square.
    Note: Usually combined with cross-entropy loss for simpler gradient.
    Formula: f'(x) = f(x) - f(x)^2
    """
    return softmax(x) - np.square(softmax(x))

# Dictionary mapping activation names to functions
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


def get_activation(name: str):
    """
    Get activation function by name.
    Args: name: Name of activation function
    Returns: Activation function by name.
    """
    if name not in ACTIVATION_FUNCTIONS:
        raise ValueError(f"Unknown activation function: {name}")
    return ACTIVATION_FUNCTIONS[name]


def get_activation_derivative(name: str):
    """ 
    Get activation derivative by name.
    Returns: Activation derivative function by name.
    """
    if name not in ACTIVATION_DERIVATIVES:
        raise ValueError(f"Unknown activation function: {name}")
    return ACTIVATION_DERIVATIVES[name]

