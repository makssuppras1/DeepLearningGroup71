"""
Optimization algorithms for training neural networks.

Implement various optimizers: SGD, Momentum, RMSprop, and Adam.
"""

import numpy as np
from typing import Dict, List


class Optimizer:
    """
    Base class for all optimizers.
    """
    
    def __init__(self, learning_rate: float = 0.01):
        """
        Initialize optimizer.
        
        Args:
            learning_rate: Learning rate for parameter updates
        """
        self.learning_rate = learning_rate
    
    def update(self, params: Dict, grads: Dict) -> Dict:
        """
        Update parameters using gradients.
        
        Args:
            params: Dictionary of parameters (weights and biases)
            grads: Dictionary of gradients
            
        Returns:
            Updated parameters
        """
        raise NotImplementedError


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer.
    
    Update rule: W = W - learning_rate * gradient
    
    TODO: Implement SGD update rule
    """
    
    def __init__(self, learning_rate: float = 0.01):
        super().__init__(learning_rate)
    
    def update(self, params: Dict, grads: Dict) -> Dict:
        """
        Perform SGD update.
        
        Args:
            params: Current parameters
            grads: Gradients
            
        Returns:
            Updated parameters
            
        TODO: Implement SGD parameter update
        """
        pass


class MomentumSGD(Optimizer):
    """
    SGD with Momentum optimizer.
    
    Momentum helps accelerate SGD in relevant directions and dampens oscillations.
    
    Update rule:
        v = beta * v - learning_rate * gradient
        W = W + v
    
    TODO: Implement Momentum SGD
    """
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        """
        Initialize Momentum optimizer.
        
        Args:
            learning_rate: Learning rate
            momentum: Momentum coefficient (typically 0.9)
        """
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = {}
    
    def update(self, params: Dict, grads: Dict) -> Dict:
        """
        Perform Momentum SGD update.
        
        TODO: Implement momentum update
        Hint: Initialize velocity dictionary on first call
        """
        pass


class RMSprop(Optimizer):
    """
    RMSprop optimizer.
    
    RMSprop adapts the learning rate for each parameter based on recent gradients.
    
    Update rule:
        cache = decay_rate * cache + (1 - decay_rate) * gradient^2
        W = W - learning_rate * gradient / (sqrt(cache) + epsilon)
    
    TODO: Implement RMSprop
    """
    
    def __init__(
        self,
        learning_rate: float = 0.001,
        decay_rate: float = 0.9,
        epsilon: float = 1e-8
    ):
        """
        Initialize RMSprop optimizer.
        
        Args:
            learning_rate: Learning rate
            decay_rate: Decay rate for moving average (typically 0.9)
            epsilon: Small value to prevent division by zero
        """
        super().__init__(learning_rate)
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.cache = {}
    
    def update(self, params: Dict, grads: Dict) -> Dict:
        """
        Perform RMSprop update.
        
        TODO: Implement RMSprop update
        """
        pass


class Adam(Optimizer):
    """
    Adam (Adaptive Moment Estimation) optimizer.
    
    Adam combines ideas from Momentum and RMSprop.
    
    Update rule:
        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * gradient^2
        m_hat = m / (1 - beta1^t)
        v_hat = v / (1 - beta2^t)
        W = W - learning_rate * m_hat / (sqrt(v_hat) + epsilon)
    
    TODO: Implement Adam optimizer
    """
    
    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8
    ):
        """
        Initialize Adam optimizer.
        
        Args:
            learning_rate: Learning rate (default: 0.001)
            beta1: Exponential decay rate for first moment (default: 0.9)
            beta2: Exponential decay rate for second moment (default: 0.999)
            epsilon: Small value to prevent division by zero
        """
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # First moment estimate
        self.v = {}  # Second moment estimate
        self.t = 0   # Time step
    
    def update(self, params: Dict, grads: Dict) -> Dict:
        """
        Perform Adam update.
        
        TODO: Implement Adam update
        Hint: Remember to increment time step and apply bias correction
        """
        pass


# Dictionary mapping optimizer names to classes
OPTIMIZERS = {
    'sgd': SGD,
    'momentum': MomentumSGD,
    'rmsprop': RMSprop,
    'adam': Adam
}


def get_optimizer(name: str, **kwargs):
    """
    Get optimizer by name.
    
    Args:
        name: Name of optimizer
        **kwargs: Optimizer-specific parameters
        
    Returns:
        Optimizer instance
    """
    if name not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer: {name}")
    return OPTIMIZERS[name](**kwargs)

