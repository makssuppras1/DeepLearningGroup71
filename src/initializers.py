"""
Weight initialization methods for neural networks.

Different initialization strategies can significantly impact training.
"""

import numpy as np


def random_initialization(shape: tuple, seed: int = None) -> np.ndarray:
    """
    Random initialization with small values.
    
    Samples from a uniform distribution [-0.01, 0.01]
    
    Args:
        shape: Shape of weight matrix (input_size, output_size)
        seed: Random seed for reproducibility
        
    Returns:
        Initialized weight matrix
    """
    if seed is not None:
        np.random.seed(seed)
    return np.random.uniform(-0.01, 0.01, size=shape)


def xavier_initialization(shape: tuple, seed: int = None) -> np.ndarray:
    """
    Xavier/Glorot initialization.
    
    Good for sigmoid and tanh activations.
    Samples from uniform distribution with variance = 1/n_in
    
    Formula: U(-sqrt(6/(n_in + n_out)), sqrt(6/(n_in + n_out)))
    
    Args:
        shape: Shape of weight matrix (input_size, output_size)
        seed: Random seed
        
    Returns:
        Initialized weight matrix
    """
    if len(shape) < 2:
        raise ValueError("Xavier initialization requires at least 2D shape")
    
    n_in = shape[0]
    n_out = shape[1] if len(shape) > 1 else 1
    
    limit = np.sqrt(6.0 / (n_in + n_out))
    
    if seed is not None:
        np.random.seed(seed)
    return np.random.uniform(-limit, limit, size=shape)


def he_initialization(shape: tuple, seed: int = None) -> np.ndarray:
    """
    He initialization.
    
    Good for ReLU activations.
    Samples from normal distribution with variance = 2/n_in
    
    Formula: N(0, sqrt(2/n_in))
    
    Args:
        shape: Shape of weight matrix (input_size, output_size)
        seed: Random seed
        
    Returns:
        Initialized weight matrix
    """
    if len(shape) < 2:
        raise ValueError("He initialization requires at least 2D shape")
    
    n_in = shape[0]
    std = np.sqrt(2.0 / n_in)
    
    if seed is not None:
        np.random.seed(seed)
    return np.random.normal(0.0, std, size=shape)


def zeros_initialization(shape: tuple) -> np.ndarray:
    """
    Zero initialization (typically used for biases).
    
    Args:
        shape: Shape of array
        
    Returns:
        Zero-initialized array
    """
    return np.zeros(shape)


# Dictionary mapping initialization names to functions
INITIALIZERS = {
    'random': random_initialization,
    'xavier': xavier_initialization,
    'glorot': xavier_initialization,  # Xavier and Glorot are the same
    'he': he_initialization,
    'zeros': zeros_initialization
}


def get_initializer(name: str):
    """
    Get initializer function by name.
    
    Args:
        name: Name of initialization method
        
    Returns:
        Initializer function
    """
    if name not in INITIALIZERS:
        raise ValueError(f"Unknown initializer: {name}")
    return INITIALIZERS[name]


def initialize_weights(
    input_size: int,
    output_size: int,
    method: str = 'xavier',
    seed: int = None
) -> tuple:
    """
    Initialize weights and biases for a layer.
    
    Args:
        input_size: Number of input units
        output_size: Number of output units
        method: Initialization method
        seed: Random seed
        
    Returns:
        Tuple of (weights, biases)
    """
    # Get the initializer function
    initializer = get_initializer(method)
    
    # Initialize weights with the specified method
    weight_shape = (input_size, output_size)
    # Handle zeros initialization which doesn't take seed parameter
    if method == 'zeros':
        weights = initializer(weight_shape)
    else:
        weights = initializer(weight_shape, seed=seed)
    
    # Initialize biases to zeros
    bias_shape = (output_size,)
    biases = zeros_initialization(bias_shape)
    
    return weights, biases

