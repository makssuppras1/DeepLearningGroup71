# Dense layer implementaton
import numpy as np
from .initializers import initialize_weights
from .activations import get_activation, get_activation_derivative

class DenseLayer:
    # Dense layer: holds weights, biases, and activaton
    def __init__(self, input_size, output_size, activation='relu',
                 weight_init='xavier', seed=None):
        # Initilize weights and biases
        self.W, self.b = initialize_weights(
            input_size,
            output_size,
            method=weight_init,
            seed=seed
        )
        self.activation = activation
        # Cache for forward pass values
        self.activation_cache = {
            'A_prev': None,
            'Z': None,
            'A': None
        }
        # Gradients for backward pass
        self.dW = None
        self.db = None
