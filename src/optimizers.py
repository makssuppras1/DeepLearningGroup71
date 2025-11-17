# Optimization algorithms for training neural networks: SGD, Momentum, RMSprop, Adam

import numpy as np
from typing import Dict, List


class Optimizer:
    # Base class for all optimizers
    
    def __init__(self, learning_rate: float = 0.01):
        # Initialize optimizer with learning rate
        self.learning_rate = learning_rate
    
    def update(self, params: Dict, grads: Dict) -> Dict:
        # Update parameters using gradients (must be implemented by subclasses)
        # params: dictionary of parameters, grads: dictionary of gradients
        raise NotImplementedError


class SGD(Optimizer):
    # Stochastic Gradient Descent: W = W - learning_rate * gradient
    
    def __init__(self, learning_rate: float = 0.01):
        super().__init__(learning_rate)
    
    def update(self, params: Dict, grads: Dict) -> Dict:
        # SGD update: param = param - learning_rate * gradient
        updated_params = {}
        
        for key in params:
            if key not in grads:
                raise ValueError(f"Gradient for parameter '{key}' not found")
            
            # Apply SGD update rule
            updated_params[key] = params[key] - self.learning_rate * grads[key]
        
        return updated_params


class MomentumSGD(Optimizer):
    # SGD with Momentum: v = beta*v - lr*grad, W = W + v
    # TODO: Implement Momentum SGD
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        # Initialize with learning rate and momentum coefficient (typically 0.9)
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = {}  # Store velocity for each parameter
    
    def update(self, params: Dict, grads: Dict) -> Dict:
        # TODO: Implement momentum update (initialize velocity on first call)
        pass


class RMSprop(Optimizer):
    # RMSprop: adapts learning rate per parameter
    # cache = decay_rate * cache + (1-decay_rate) * grad^2
    # W = W - lr * grad / (sqrt(cache) + epsilon)
    # TODO: Implement RMSprop
    
    def __init__(
        self,
        learning_rate: float = 0.001,
        decay_rate: float = 0.9,
        epsilon: float = 1e-8
    ):
        # Initialize with learning rate, decay_rate (typically 0.9), and epsilon
        super().__init__(learning_rate)
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.cache = {}  # Store moving average of squared gradients
    
    def update(self, params: Dict, grads: Dict) -> Dict:
        # TODO: Implement RMSprop update
        pass


class Adam(Optimizer):
    # Adam optimizer: combines Momentum and RMSprop
    # m = beta1*m + (1-beta1)*grad, v = beta2*v + (1-beta2)*grad^2
    # m_hat = m/(1-beta1^t), v_hat = v/(1-beta2^t)
    # W = W - lr * m_hat / (sqrt(v_hat) + epsilon)
    # TODO: Implement Adam optimizer
    
    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8
    ):
        # Initialize with learning rate, beta1 (0.9), beta2 (0.999), epsilon
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # First moment estimate (momentum)
        self.v = {}  # Second moment estimate (RMSprop)
        self.t = 0   # Time step for bias correction
    
    def update(self, params: Dict, grads: Dict) -> Dict:
        # TODO: Implement Adam update (remember bias correction and increment t)
        pass


# Dictionary mapping optimizer names to classes
OPTIMIZERS = {
    'sgd': SGD,
    'momentum': MomentumSGD,
    'rmsprop': RMSprop,
    'adam': Adam
}


def get_optimizer(name: str, **kwargs):
    # Get optimizer instance by name
    # name: optimizer name, **kwargs: optimizer-specific parameters
    if name not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer: {name}")
    return OPTIMIZERS[name](**kwargs)

