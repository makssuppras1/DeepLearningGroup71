"""
PyTorch equivalent of the NumPy neural network for comparison testing.
This module provides a PyTorch implementation that matches the NumPy version's behavior.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional


class PyTorchNeuralNetwork(nn.Module):
    """
    PyTorch equivalent of the NumPy NeuralNetwork class.
    Matches the architecture and behavior of the NumPy implementation.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_layers: List[int],
        output_size: int,
        activation: str = 'relu',
        output_activation: str = 'softmax',
        learning_rate: float = 0.001,
        optimizer: str = 'adam',
        weight_init: str = 'he',
        l2_lambda: float = 0.0,
        dropout_rate: float = 0.0,
        random_seed: Optional[int] = None
    ):
        super().__init__()
        
        # Store hyperparameters
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.activation = activation
        self.output_activation = output_activation
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.l2_lambda = l2_lambda
        self.dropout_rate = dropout_rate
        
        # Set random seed for reproducibility
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
        
        # Build layers
        layers = []
        input_dim = input_size
        
        # Create hidden layers
        for hidden_units in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_units))
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_units
        
        # Output layer
        layers.append(nn.Linear(input_dim, output_size))
        
        self.layers = nn.ModuleList(layers)
        
        # Initialize weights
        self._initialize_weights(weight_init, random_seed)
        
        # Setup optimizer
        self._setup_optimizer()
        
        # Store loss function
        self.loss_function = 'cross_entropy'
    
    def _initialize_weights(self, weight_init: str, seed: Optional[int] = None):
        """Initialize weights using the specified method."""
        if seed is not None:
            torch.manual_seed(seed)
        
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                if weight_init == 'he':
                    # He initialization for ReLU
                    nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                    nn.init.zeros_(layer.bias)
                elif weight_init == 'xavier':
                    # Xavier/Glorot initialization
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
                elif weight_init == 'random':
                    # Random uniform initialization
                    nn.init.uniform_(layer.weight, -0.01, 0.01)
                    nn.init.zeros_(layer.bias)
                else:
                    # Default: use PyTorch default initialization
                    pass
    
    def _setup_optimizer(self):
        """Setup the optimizer."""
        if self.optimizer_name == 'adam':
            self.optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=0.0  # We'll handle L2 manually to match NumPy version
            )
        elif self.optimizer_name == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=0.0
            )
        elif self.optimizer_name == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=0.0
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        X: (batch_size, input_size) tensor
        Returns: (batch_size, output_size) tensor
        """
        A = X
        
        # Process all layers except the last one
        layer_idx = 0
        for i in range(len(self.layers) - 1):
            layer = self.layers[i]
            if isinstance(layer, nn.Linear):
                A = layer(A)
                # Apply activation
                if self.activation == 'relu':
                    A = F.relu(A)
                elif self.activation == 'sigmoid':
                    A = torch.sigmoid(A)
                elif self.activation == 'tanh':
                    A = torch.tanh(A)
                else:
                    raise ValueError(f"Unknown activation: {self.activation}")
                layer_idx += 1
            elif isinstance(layer, nn.Dropout):
                # Dropout is handled automatically by PyTorch based on training mode
                A = layer(A)
        
        # Output layer
        output_layer = self.layers[-1]
        Z = output_layer(A)
        
        # Apply output activation
        if self.output_activation == 'softmax':
            A = F.softmax(Z, dim=1)
        elif self.output_activation == 'sigmoid':
            A = torch.sigmoid(Z)
        else:
            A = Z
        
        return A
    
    def compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute loss: data loss + L2 regularization.
        y_pred: (batch_size, output_size) predictions
        y_true: (batch_size, output_size) one-hot encoded labels
        Returns: scalar loss tensor
        """
        # Data loss (cross-entropy)
        if self.loss_function == 'cross_entropy':
            # PyTorch's CrossEntropyLoss expects class indices, not one-hot
            # So we'll compute manually to match NumPy version
            eps = 1e-12
            y_pred_clipped = torch.clamp(y_pred, eps, 1.0 - eps)
            data_loss = -torch.sum(y_true * torch.log(y_pred_clipped)) / y_pred.shape[0]
        else:
            raise ValueError(f"Unknown loss function: {self.loss_function}")
        
        # L2 regularization
        if self.l2_lambda > 0:
            l2_sum = 0.0
            for layer in self.layers:
                if isinstance(layer, nn.Linear):
                    l2_sum += torch.sum(layer.weight ** 2)
            reg_loss = (self.l2_lambda / 2) * l2_sum
            total_loss = data_loss + reg_loss
        else:
            total_loss = data_loss
        
        return total_loss
    
    def train_step(self, X_batch: torch.Tensor, y_batch: torch.Tensor) -> float:
        """
        One complete training step: forward -> backward -> update.
        Returns loss value for this batch.
        """
        self.train()  # Set to training mode (enables dropout)
        
        # Forward pass
        y_pred = self.forward(X_batch)
        
        # Compute loss
        loss = self.compute_loss(y_pred, y_batch)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Apply L2 regularization to gradients (to match NumPy implementation)
        if self.l2_lambda > 0:
            m = X_batch.shape[0]
            for layer in self.layers:
                if isinstance(layer, nn.Linear):
                    layer.weight.grad += (self.l2_lambda / m) * layer.weight
        
        # Update weights
        self.optimizer.step()
        
        return loss.item()
    
    def predict(self, X: torch.Tensor) -> np.ndarray:
        """Predict class labels."""
        probabilities = self.predict_proba(X)
        predictions = torch.argmax(probabilities, dim=1)
        return predictions.cpu().numpy()
    
    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """Get prediction probabilities."""
        self.eval()  # Disable dropout for inference
        with torch.no_grad():
            probabilities = self.forward(X)
        return probabilities
    
    def get_params(self) -> dict:
        """Get all model parameters as dictionary (for comparison with NumPy version)."""
        params = {}
        layer_idx = 0
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                layer_idx += 1
                # Make explicit copies to avoid views that update when weights change
                params[f'W{layer_idx}'] = layer.weight.detach().cpu().numpy().T.copy()  # Transpose to match NumPy format
                params[f'b{layer_idx}'] = layer.bias.detach().cpu().numpy().copy()
        return params
    
    def set_params(self, params: dict) -> None:
        """Set model parameters from dictionary (for comparison with NumPy version)."""
        layer_idx = 0
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                layer_idx += 1
                W_key = f'W{layer_idx}'
                b_key = f'b{layer_idx}'
                if W_key in params:
                    # Transpose to match PyTorch format (PyTorch uses (out_features, in_features))
                    layer.weight.data = torch.from_numpy(params[W_key].T).float()
                if b_key in params:
                    layer.bias.data = torch.from_numpy(params[b_key]).float()
        
        # Note: Optimizer state reset is handled separately to avoid issues
        # The state will be reset when needed (e.g., in comparison tests)
    
    def reset_optimizer_state(self):
        """Reset optimizer state to initial values (for comparison testing)."""
        # Initialize state if it doesn't exist by doing a dummy backward pass
        if len(self.optimizer.state) == 0:
            dummy_input = torch.zeros(1, self.input_size, requires_grad=False)
            dummy_output = self.forward(dummy_input)
            dummy_loss = dummy_output.sum()
            self.optimizer.zero_grad()
            dummy_loss.backward()
            self.optimizer.zero_grad()  # Clear gradients
        
        # Reset the optimizer state
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                if param in self.optimizer.state:
                    state = self.optimizer.state[param]
                    if 'exp_avg' in state:
                        state['exp_avg'].zero_()
                    if 'exp_avg_sq' in state:
                        state['exp_avg_sq'].zero_()
                    if 'step' in state:
                        state['step'] = torch.tensor(0, dtype=torch.int32, device=param.device)

