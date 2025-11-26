# Layer implementations for the neural network
# This module contains classes for different layer types

import numpy as np
from typing import Optional
from .initializers import initialize_weights
from .activations import get_activation, get_activation_derivative

class DenseLayer:
    # Holds W, b, activation, and caches. Does NOT perform forward/backward
    def __init__(self, input_size, output_size, activation='relu',
                 weight_init='xavier', seed=None):

        self.W, self.b = initialize_weights(
            input_size,
            output_size,
            method=weight_init,
            seed=seed
        )

        self.activation = activation
        self.activation_cache = {
            'A_prev': None,
            'Z': None,
            'A': None
        }

        # Gradients
        self.dW = None
        self.db = None
