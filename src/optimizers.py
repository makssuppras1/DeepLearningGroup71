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
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        # Initialize with learning rate and momentum coefficient (typically 0.9)
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = {}  # Store velocity for each parameter
    
    def update(self, params: Dict, grads: Dict) -> Dict:
        # Momentum update: v = beta*v - lr*grad, W = W + v
        updated_params = {}
        
        for key in params:
            if key not in grads:
                raise ValueError(f"Gradient for parameter '{key}' not found")
            
            # Initialize velocity for this parameter if not exists
            if key not in self.velocity:
                self.velocity[key] = np.zeros_like(params[key])
            
            # Update velocity: v = beta * v - learning_rate * gradient
            self.velocity[key] = self.momentum * self.velocity[key] - self.learning_rate * grads[key]
            
            # Update parameters: W = W + v
            updated_params[key] = params[key] + self.velocity[key]
        
        return updated_params


class RMSprop(Optimizer):
    # RMSprop: adapts learning rate per parameter
    # cache = decay_rate * cache + (1-decay_rate) * grad^2
    # W = W - lr * grad / (sqrt(cache) + epsilon)
    
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
        # RMSprop update: cache = decay_rate*cache + (1-decay_rate)*grad^2
        # W = W - lr * grad / (sqrt(cache) + epsilon)
        updated_params = {}
        
        for key in params:
            if key not in grads:
                raise ValueError(f"Gradient for parameter '{key}' not found")
            
            # Initialize cache for this parameter if not exists
            if key not in self.cache:
                self.cache[key] = np.zeros_like(params[key])
            
            # Update cache: moving average of squared gradients
            self.cache[key] = self.decay_rate * self.cache[key] + (1 - self.decay_rate) * grads[key] ** 2
            
            # Update parameters: W = W - lr * grad / (sqrt(cache) + epsilon)
            updated_params[key] = params[key] - self.learning_rate * grads[key] / (np.sqrt(self.cache[key]) + self.epsilon)
        
        return updated_params


class Adam(Optimizer):
    # Adam optimizer: combines Momentum and RMSprop
    # m = beta1*m + (1-beta1)*grad, v = beta2*v + (1-beta2)*grad^2
    # m_hat = m/(1-beta1^t), v_hat = v/(1-beta2^t)
    # W = W - lr * m_hat / (sqrt(v_hat) + epsilon)
    
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
        # Adam update with bias correction
        # Increment time step
        self.t += 1
        
        updated_params = {}
        
        for key in params:
            if key not in grads:
                raise ValueError(f"Gradient for parameter '{key}' not found")
            
            # Initialize moment estimates for this parameter if not exists
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
            if key not in self.v:
                self.v[key] = np.zeros_like(params[key])
            
            # Update biased first moment estimate (momentum)
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            
            # Update biased second moment estimate (RMSprop)
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grads[key] ** 2
            
            # Compute bias-corrected moment estimates
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            
            # Update parameters: W = W - lr * m_hat / (sqrt(v_hat) + epsilon)
            updated_params[key] = params[key] - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return updated_params


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

