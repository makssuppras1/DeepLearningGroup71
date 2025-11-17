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


def xavier_initialization(shape: tuple, alpha: float = 1.0, seed: int = None) -> np.ndarray:
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

    if seed is not None:
        np.random.seed(seed)

    n_in, n_out = shape
    std = np.sqrt(2 * alpha / (n_in + n_out))

    return np.random.normal(0.0, std, size=shape) #uniform(-std, std, size=shape)


def he_initialization(shape: tuple, alpha: float = 2.0, seed: int = None) -> np.ndarray:
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
    
    if seed is not None:
        np.random.seed(seed)
    n_in, _ = shape
    std = np.sqrt(alpha / n_in)

    return np.random.normal(0, std, size=shape)


def zeros_initialization(shape: tuple) -> np.ndarray:
    """
    Zero initialization (typically used for biases).
    
    Args:
        shape: Shape of array
        
    Returns:
        Zero-initialized array
    """
    
    return np.zeros(shape)


def get_alpha_from_activation(activation: str) -> float:
    """
    Get alpha scaling constant based on activation function.
    α is used to scale the variance of the initialization distribution.
    Args:
        activation: Name of the activation function
    Returns:
        Alpha scaling constant
    """
    activation = activation.lower()
    if activation in ["tanh", "sigmoid"]:
        return 1.0
    elif activation in ["relu"]:
        return 2.0
    else:
        # Default safe choice
        return 1.0
    

# Dictionary mapping initialization names to functions
INITIALIZERS = {
    'random': random_initialization,
    'xavier': xavier_initialization, # Xavier and Glorot are the same
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
    method: str = 'xavier', #using xavier if nothing else is stated
    activation: str = 'relu', 
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
    alpha = get_alpha_from_activation(activation)
    init_fn = get_initializer(method)

    import inspect
    sig = inspect.signature(init_fn)

    kwargs = {}
    if 'alpha' in sig.parameters:
        kwargs['alpha'] = alpha
    if 'seed' in sig.parameters:
        kwargs['seed'] = seed

    W = init_fn((input_size, output_size), **kwargs)
    b = zeros_initialization((output_size,))
    return W, b

