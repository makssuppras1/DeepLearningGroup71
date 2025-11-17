"""
Fully-Connected Feedforward Neural Network (FFNN) Implementation

This module contains the main neural network class that will be implemented from scratch.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from layers import DenseLayer
from activations import get_activation
from losses import get_loss_function, get_loss_derivative, cross_entropy_loss, l2_regularization
from optimizers import get_optimizer


class NeuralNetwork:
    # Fully-connected feedforward neural network implemented with NumPy
    # Supports: configurable layers, multiple activations, optimizers, L2 regularization
    
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
        # Initialize network: create layers, set up optimizer, store hyperparameters
        self.layers = []
        input_dim = input_size

        # Create hidden layers
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
        
        # Store hyperparameters
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.activation = activation
        self.output_activation = output_activation
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.l2_lambda = l2_lambda
        
        # Initialize optimizer
        self.optimizer = get_optimizer(optimizer, learning_rate=learning_rate)
        
        # Store loss function (default to cross_entropy for classification)
        self.loss_function = 'cross_entropy'
        
        # Store last predictions and loss for debugging
        self.last_predictions = None
        self.last_loss = None
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        # Forward pass: propagate input through all layers
        # X: (batch_size, input_size) -> returns: (batch_size, output_size)
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A
    
    
    def backward(self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray = None) -> None:
        # Backward pass: compute gradients for all weights and biases
        # X: input data, y: true labels, y_pred: optional pre-computed predictions
        
        # Get predictions if not provided
        if y_pred is None:
            y_pred = self.forward(X)
        
        # Compute initial gradient from loss function
        loss_derivative = get_loss_derivative(self.loss_function)
        dA = loss_derivative(y_pred, y)
        
        # Backpropagate through layers in reverse order
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            dA = layer.backward(dA)  # Compute gradients for this layer
            
            # Add L2 regularization term to weight gradients
            if self.l2_lambda > 0:
                layer.dW += self.l2_lambda * layer.W
    
    def update_weights(self) -> None:
        # Update all weights and biases using the optimizer
        
        # Collect parameters and gradients from all layers
        params = {}
        grads = {}
        for i, layer in enumerate(self.layers):
            params[f'W{i+1}'] = layer.W
            params[f'b{i+1}'] = layer.b
            grads[f'W{i+1}'] = layer.dW
            grads[f'b{i+1}'] = layer.db
        
        # Apply optimizer update rule
        updated_params = self.optimizer.update(params, grads)
        
        # Update layer weights and biases with new values
        for i, layer in enumerate(self.layers):
            layer.W = updated_params[f'W{i+1}']
            layer.b = updated_params[f'b{i+1}']
    
    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        # Compute total loss: data loss + L2 regularization
        # y_pred: predictions, y_true: true labels -> returns: scalar loss
        
        # Compute data loss (e.g., cross-entropy)
        loss_func = get_loss_function(self.loss_function)
        data_loss = loss_func(y_pred, y_true)
        
        # Add L2 regularization term if specified
        if self.l2_lambda > 0:
            weights = [layer.W for layer in self.layers]
            reg_loss = l2_regularization(weights, self.l2_lambda)
            total_loss = data_loss + reg_loss
        else:
            total_loss = data_loss
        
        return total_loss
    
    def train_step(self, X_batch: np.ndarray, y_batch: np.ndarray) -> float:
        # One complete training step: forward -> backward -> update
        # Returns loss value for this batch
        
        # Forward pass: compute predictions
        y_pred = self.forward(X_batch)
        
        # Compute loss
        loss = self.compute_loss(y_pred, y_batch)
        
        # Backward pass: compute gradients (pass y_pred to avoid redundant forward)
        self.backward(X_batch, y_batch, y_pred=y_pred)
        
        # Update weights using computed gradients
        self.update_weights()
        
        # Store for debugging/monitoring
        self.last_predictions = y_pred
        self.last_loss = loss
        
        return loss
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        # Predict class labels: get probabilities and return argmax
        # X: (batch_size, input_size) -> returns: (batch_size,) class indices
        probabilities = self.predict_proba(X)
        predictions = np.argmax(probabilities, axis=1)
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Get prediction probabilities (forward pass without training)
        # X: (batch_size, input_size) -> returns: (batch_size, output_size) probabilities
        probabilities = self.forward(X)
        return probabilities
    
    def get_params(self) -> dict:
        # Get all model parameters (weights and biases) as dictionary
        # Returns: {'W1': weights, 'b1': biases, 'W2': ..., ...}
        params = {}
        for i, layer in enumerate(self.layers):
            params[f'W{i+1}'] = layer.W.copy()
            params[f'b{i+1}'] = layer.b.copy()
        return params
    
    def set_params(self, params: dict) -> None:
        # Set model parameters from dictionary
        # params: {'W1': weights, 'b1': biases, 'W2': ..., ...}
        for i, layer in enumerate(self.layers):
            layer.W = params[f'W{i+1}'].copy()
            layer.b = params[f'b{i+1}'].copy()


# Additional helper functions can be added below

