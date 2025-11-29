# Optimizaton algorithms: SGD, Momentum, RMSprop, Adam
import numpy as np

class Optimizer:
    # Base class for optimizers
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    
    def update(self, params, grads):
        # Update params using grads (must be implmented by subclasses)
        raise NotImplementedError

class SGD(Optimizer):
    # SGD: W = W - lr * grad
    def __init__(self, learning_rate=0.01):
        super().__init__(learning_rate)
    
    def update(self, params, grads):
        # SGD update rule
        updated_params = {}
        for key in params:
            if key not in grads:
                raise ValueError(f"Gradient for parameter '{key}' not found")
            updated_params[key] = params[key] - self.learning_rate * grads[key]
        return updated_params

class MomentumSGD(Optimizer):
    # SGD with momentum: v = beta*v - lr*grad, W = W + v
    def __init__(self, learning_rate=0.01, momentum=0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = {}  # Velocity for each param
    
    def update(self, params, grads):
        # Momentum update
        updated_params = {}
        for key in params:
            if key not in grads:
                raise ValueError(f"Gradient for parameter '{key}' not found")
            if key not in self.velocity:
                self.velocity[key] = np.zeros_like(params[key])
            # Update velocity
            self.velocity[key] = self.momentum * self.velocity[key] - self.learning_rate * grads[key]
            # Update params
            updated_params[key] = params[key] + self.velocity[key]
        return updated_params

class RMSprop(Optimizer):
    # RMSprop: adapts lr per param
    def __init__(self, learning_rate=0.001, decay_rate=0.9, epsilon=1e-8):
        super().__init__(learning_rate)
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.cache = {}  # Moving average of squared grads
    
    def update(self, params, grads):
        # RMSprop update
        updated_params = {}
        for key in params:
            if key not in grads:
                raise ValueError(f"Gradient for parameter '{key}' not found")
            if key not in self.cache:
                self.cache[key] = np.zeros_like(params[key])
            # Update cache
            self.cache[key] = self.decay_rate * self.cache[key] + (1 - self.decay_rate) * grads[key] ** 2
            # Update params
            updated_params[key] = params[key] - self.learning_rate * grads[key] / (np.sqrt(self.cache[key]) + self.epsilon)
        return updated_params

class Adam(Optimizer):
    # Adam: combines Momentum and RMSprop
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # First moment (momentum)
        self.v = {}  # Second moment (RMSprop)
        self.t = 0   # Time step for bias corection
    
    def update(self, params, grads):
        # Adam update with bias corection
        self.t += 1
        updated_params = {}
        for key in params:
            if key not in grads:
                raise ValueError(f"Gradient for parameter '{key}' not found")
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
            if key not in self.v:
                self.v[key] = np.zeros_like(params[key])
            # Update moments
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grads[key] ** 2
            # Bias corection
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            # Update params
            updated_params[key] = params[key] - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return updated_params

OPTIMIZERS = {
    'sgd': SGD,
    'momentum': MomentumSGD,
    'rmsprop': RMSprop,
    'adam': Adam
}

def get_optimizer(name, **kwargs):
    # Get optimizer instnce by name
    if name not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer: {name}")
    return OPTIMIZERS[name](**kwargs)
