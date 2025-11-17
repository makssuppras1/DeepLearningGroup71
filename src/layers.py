"""
Layer implementations for the neural network.

This module contains classes for different layer types.
"""

import numpy as np
from typing import Optional
from .initializers import initialize_weights
from .activations import get_activation, get_activation_derivative


class DenseLayer:
    """
    Fully connected (dense) layer.
    This layer performs the operation: output = activation(X @ W + b)
    """
    
    def __init__(
            self, 
            input_size: int, 
            output_size: int, 
            activation: str = 'relu',
            weight_init: str = 'xavier',
            seed: Optional[int] = None):
        
        """
        Initialize a dense layer.
        
        Args:
            input_size: Number of input features
            output_size: Number of output features
            activation: Activation function to use
            weight_init: Weight initialization method
            seed: Random seed for reproducibility
        """

        # Initialize weights and biases using the initializer from initializers.py
        self.W, self.b = initialize_weights(input_size, output_size, method=weight_init, seed=seed)
        self.activation = activation
        
        # Cache for forward pass (store Z and A for backprop)
        self.activation_cache = {
            'Z': None,  # pre-activation
            'A': None   # post-activation
        }

        # Gradients placeholders (to be filled in backward) evt?
        #self.dW = None
        #self.db = None
        pass
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass through the layer. 
        Args:
            X: Input data of shape (batch_size, input_size)    
        Returns:
            Output of shape (batch_size, output_size)
        """

         # Store input for backprop
        self.activation_cache['A_prev'] = X

        # Linear step: Z = X @ W + b
        Z = X @ self.W + self.b
        self.activation_cache['Z'] = Z  # store pre-activation

        # Activation step
        activation_func = get_activation(self.activation)
        A = activation_func(Z)
        self.activation_cache['A'] = A  # store activation

        return A

    
    def backward(self, dA: np.ndarray) -> np.ndarray:
        """
        Backward pass through the layer.
        
        Args:
            dA: Gradient of loss with respect to layer output
            
        Returns:
            Gradient of loss with respect to layer input
        """
        # calculate dZ = dA  multiplied by activation_derivative(Z)
        activation_grad_func = get_activation_derivative(self.activation)
        dZ = dA * activation_grad_func(self.activation_cache['Z'])

        # Compute gradients for weights and biases
        self.dW = self.activation_cache['A_prev'].T @ dZ 
        self.db = np.sum(dZ, axis=0) #keepdims=True hvis fejl i shapes

        # Compute gradient with respect to layer input (previous layer)
        dA_prev = dZ @ self.W.T
        return dA_prev

