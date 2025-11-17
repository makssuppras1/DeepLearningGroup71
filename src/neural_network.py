"""
Fully-Connected Feedforward Neural Network (FFNN) Implementation

This module contains the main neural network class that will be implemented from scratch.
"""

import numpy as np
from typing import List, Tuple, Optional
from layers import *
from losses import *


class NeuralNetwork:
    
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
        loss_function: str = 'cross_entropy_loss', # we need to couple this to losses.py
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
            loss_function: Loss function to use ('mse', 'cross_entropy', 'binary_cross_entropy')
            random_seed: Random seed for reproducibility
        """
            
        self.layers = []
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.activation = activation
        self.output_activation = output_activation
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.weight_init = weight_init
        self.l2_lambda = l2_lambda
        self.loss_function = loss_function

        input_dim = input_size

        #Hidden layers
        for hidden_units in hidden_layers:
            layer = DenseLayer(
                input_size=input_dim,
                output_size=hidden_units,
                activation=activation,
                weight_init=weight_init,
                seed=random_seed
            )
            self.layers.append(layer)
            input_dim = hidden_units 

        # Output layer
        self.layers.append(DenseLayer(
            input_size=input_dim,
            output_size=output_size,
            activation=output_activation,
            weight_init=weight_init,
            seed=random_seed
        ))

        pass
    
    def forward_whole_network(self, X: np.ndarray) -> np.ndarray:
        """
        Perform forward propagation through the network.     
        Args: X: Input data of shape (batch_size, input_size)        
        Returns: Output predictions of shape (batch_size, output_size)  
        """
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A
    
    
    def backward(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Perform backward propagation to compute gradients.
        Args: X: Input data of shape (batch_size, input_size), y: True labels of shape (batch_size, output_size)
        """
        y_pred = self.forward_whole_network(X)
        dA = get_loss_derivative(self.loss_function)(y_pred, y)

        for layer in enumerate(reversed(self.layers)):
                dA_prev = layer.backward(dA)

                # Add L2 regularization to weight gradients
                layer.dW += self.l2_lambda * layer.W
                dA = dA_prev
    

    def update_weights(self) -> None:
        """
        Update weights using the selected optimizer.
        Use computed gradients from backward pass, Apply optimizer-specific updates, Update all weights and biases
        """
        grads = {}
        for i, layer in enumerate(self.layers, start=1):
            grads[f'W{i}'] = layer.dW
            grads[f'b{i}'] = layer.db

        #self.params = self.optimizer.update(self.params, grads) 
        #self.set_params(self.params)
        
    
    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute the loss function. 
        Args: y_pred: Predicted values of shape (batch_size, output_size), y_true: True labels of shape (batch_size, output_size) 
        Returns: Loss value (scalar)
        """
        loss = get_loss_function(self.loss_function)(y_pred, y_true)
        return loss 
    ## What about L2 regularization term? 
    
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
        y_pred = self.forward_whole_network(X_batch)
        loss = self.compute_loss(y_pred, y_batch)
        self.backward(X_batch, y_batch)
        self.update_weights()
        return loss
    
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

