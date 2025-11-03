"""
Fully-Connected Feedforward Neural Network (FFNN) Implementation

This module contains the main neural network class that will be implemented from scratch.
"""

import numpy as np
from typing import List, Tuple, Optional


class NeuralNetwork:
    """
    A flexible fully-connected feedforward neural network implemented with NumPy.
    
    This class should support:
    - Configurable number of layers and units
    - Multiple activation functions
    - Different optimizers
    - L2 regularization
    - Mini-batch training
    
    Attributes:
        TODO: Define your attributes here
        
    Example usage:
        model = NeuralNetwork(
            input_size=784,
            hidden_layers=[128, 64],
            output_size=10,
            activation='relu',
            learning_rate=0.01,
            optimizer='adam'
        )
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_layers: List[int],
        output_size: int,
        activation: str = 'relu',
        output_activation: str = 'softmax',
        learning_rate: float = 0.01,
        optimizer: str = 'sgd',
        weight_init: str = 'xavier',
        l2_lambda: float = 0.0,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the neural network.
        
        Args:
            input_size: Number of input features
            hidden_layers: List containing number of units in each hidden layer
            output_size: Number of output classes
            activation: Activation function for hidden layers ('relu', 'sigmoid', 'tanh')
            output_activation: Activation function for output layer ('softmax', 'sigmoid')
            learning_rate: Learning rate for optimization
            optimizer: Optimizer to use ('sgd', 'momentum', 'rmsprop', 'adam')
            weight_init: Weight initialization method ('random', 'xavier', 'he')
            l2_lambda: L2 regularization coefficient
            random_seed: Random seed for reproducibility
        """
        # TODO: Initialize network parameters
        pass
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Perform forward propagation through the network.
        
        Args:
            X: Input data of shape (batch_size, input_size)
            
        Returns:
            Output predictions of shape (batch_size, output_size)
            
        TODO: Implement forward pass
        - Store intermediate values (pre-activations and activations) for backprop
        - Apply activation functions
        - Return final output
        """
        pass
    
    def backward(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Perform backward propagation to compute gradients.
        
        Args:
            X: Input data of shape (batch_size, input_size)
            y: True labels of shape (batch_size, output_size)
            
        TODO: Implement backward pass
        - Compute gradients for all weights and biases
        - Include L2 regularization in gradient computation
        - Store gradients for optimizer to use
        """
        pass
    
    def update_weights(self) -> None:
        """
        Update weights using the selected optimizer.
        
        TODO: Implement weight updates
        - Use computed gradients from backward pass
        - Apply optimizer-specific updates (SGD, Momentum, RMSprop, Adam)
        - Update all weights and biases
        """
        pass
    
    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute the loss function.
        
        Args:
            y_pred: Predicted values of shape (batch_size, output_size)
            y_true: True labels of shape (batch_size, output_size)
            
        Returns:
            Loss value (scalar)
            
        TODO: Implement loss computation
        - Compute cross-entropy or MSE loss
        - Add L2 regularization term
        - Return total loss
        """
        pass
    
    def train_step(self, X_batch: np.ndarray, y_batch: np.ndarray) -> float:
        """
        Perform one training step (forward pass, backward pass, weight update).
        
        Args:
            X_batch: Mini-batch of input data
            y_batch: Mini-batch of labels
            
        Returns:
            Loss value for this batch
            
        TODO: Implement one complete training step
        """
        pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on input data.
        
        Args:
            X: Input data of shape (batch_size, input_size)
            
        Returns:
            Predicted class labels of shape (batch_size,)
            
        TODO: Implement prediction
        - Perform forward pass
        - Convert probabilities to class labels
        """
        pass
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Input data of shape (batch_size, input_size)
            
        Returns:
            Prediction probabilities of shape (batch_size, output_size)
            
        TODO: Implement probability prediction
        """
        pass
    
    def get_params(self) -> dict:
        """
        Get current model parameters.
        
        Returns:
            Dictionary containing all weights and biases
            
        TODO: Return all model parameters in a dictionary
        """
        pass
    
    def set_params(self, params: dict) -> None:
        """
        Set model parameters.
        
        Args:
            params: Dictionary containing weights and biases
            
        TODO: Load parameters into the model
        """
        pass


# Additional helper functions can be added below

