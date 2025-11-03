"""
Activation functions and their derivatives.

Implement various activation functions used in neural networks.
Each function should have both forward and backward (derivative) computations.
"""

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    """
    ReLU activation function.
    
    Formula: f(x) = max(0, x)
    
    Args:
        x: Input array
        
    Returns:
        Activated output
        
    TODO: Implement ReLU activation
    """
    pass


def relu_derivative(x: np.ndarray) -> np.ndarray:
    """
    Derivative of ReLU activation.
    
    Formula: f'(x) = 1 if x > 0, else 0
    
    Args:
        x: Input array (pre-activation values)
        
    Returns:
        Derivative values
        
    TODO: Implement ReLU derivative
    """
    pass


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation function.
    
    Formula: f(x) = 1 / (1 + exp(-x))
    
    Args:
        x: Input array
        
    Returns:
        Activated output
        
    TODO: Implement sigmoid activation
    Hint: Be careful with numerical stability for large negative values
    """
    pass


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """
    Derivative of sigmoid activation.
    
    Formula: f'(x) = f(x) * (1 - f(x))
    
    Args:
        x: Input array (can be pre-activation or post-activation)
        
    Returns:
        Derivative values
        
    TODO: Implement sigmoid derivative
    """
    pass


def tanh(x: np.ndarray) -> np.ndarray:
    """
    Tanh activation function.
    
    Formula: f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    
    Args:
        x: Input array
        
    Returns:
        Activated output
        
    TODO: Implement tanh activation
    Hint: You can use np.tanh()
    """
    pass


def tanh_derivative(x: np.ndarray) -> np.ndarray:
    """
    Derivative of tanh activation.
    
    Formula: f'(x) = 1 - f(x)^2
    
    Args:
        x: Input array
        
    Returns:
        Derivative values
        
    TODO: Implement tanh derivative
    """
    pass


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Softmax activation function for output layer.
    
    Formula: f(x_i) = exp(x_i) / sum(exp(x_j))
    
    Args:
        x: Input array of shape (batch_size, num_classes)
        
    Returns:
        Probability distribution over classes
        
    TODO: Implement softmax activation
    Hint: Subtract max(x) for numerical stability
    """
    pass


def softmax_derivative(x: np.ndarray) -> np.ndarray:
    """
    Derivative of softmax activation.
    
    Note: Usually combined with cross-entropy loss for simpler gradient.
    
    Args:
        x: Input array
        
    Returns:
        Derivative values
        
    TODO: Implement softmax derivative (optional, may not be used directly)
    """
    pass


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
    
    Args:
        name: Name of activation function
        
    Returns:
        Activation function
    """
    if name not in ACTIVATION_FUNCTIONS:
        raise ValueError(f"Unknown activation function: {name}")
    return ACTIVATION_FUNCTIONS[name]


def get_activation_derivative(name: str):
    """
    Get activation derivative by name.
    
    Args:
        name: Name of activation function
        
    Returns:
        Derivative function
    """
    if name not in ACTIVATION_DERIVATIVES:
        raise ValueError(f"Unknown activation function: {name}")
    return ACTIVATION_DERIVATIVES[name]

