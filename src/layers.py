"""
Layer implementations for the neural network.

This module contains classes for different layer types.
"""

import numpy as np
from typing import Optional


class DenseLayer:
    """
    Fully connected (dense) layer.
    
    This layer performs the operation: output = activation(X @ W + b)
    
    TODO: Implement this class with:
    - Forward pass
    - Backward pass
    - Parameter storage (weights and biases)
    """
    
    def __init__(self, input_size: int, output_size: int, activation: str = 'relu'):
        """
        Initialize a dense layer.
        
        Args:
            input_size: Number of input features
            output_size: Number of output features
            activation: Activation function to use
        """
        # TODO: Initialize weights and biases
        pass
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass through the layer.
        
        Args:
            X: Input data of shape (batch_size, input_size)
            
        Returns:
            Output of shape (batch_size, output_size)
        """
        # TODO: Implement forward pass
        pass
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass through the layer.
        
        Args:
            grad_output: Gradient of loss with respect to layer output
            
        Returns:
            Gradient of loss with respect to layer input
        """
        # TODO: Implement backward pass
        pass

