import numpy as np
from typing import Optional, List

from .initializers import initialize_weights
from .layers import DenseLayer
from .activations import get_activation, get_activation_derivative
from .losses import get_loss_function, get_loss_derivative, l2_regularization
from .optimizers import get_optimizer


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
        learning_rate: float = 0.001,
        optimizer: str = 'adam',
        weight_init: str = 'he',
        l2_lambda: float = 0.0,
        dropout_rate: float = 0.0,
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
        self.dropout_rate = dropout_rate
        
        # Training mode flag (for dropout)
        self.training = True
        
        # Initialize optimizer
        self.optimizer = get_optimizer(optimizer, learning_rate=learning_rate)
        
        # Store loss function (default to cross_entropy for classification)
        self.loss_function = 'cross_entropy'
        
        # Store last predictions and loss for debugging
        self.last_predictions = None
        self.last_loss = None
        
        # Store dropout masks for backward pass
        self.dropout_masks = []
    
    
    # ------------------------------------------------------------------
    # FORWARD
    # ------------------------------------------------------------------

    def forward(self, X: np.ndarray) -> np.ndarray:
        # Forward pass with optional dropout during training
        # Dropout is applied to hidden layers only (not input or output)
        
        A = X
        self.dropout_masks = []  # Clear masks from previous forward pass

        for i, layer in enumerate(self.layers):
            # Cache previous activation
            layer.activation_cache['A_prev'] = A

            # Linear
            Z = A @ layer.W + layer.b
            layer.activation_cache['Z'] = Z

            # Activation
            act = get_activation(layer.activation)
            A = act(Z)

            # Apply dropout to hidden layers during training (not output layer)
            is_hidden_layer = (i < len(self.layers) - 1)
            if self.training and is_hidden_layer and self.dropout_rate > 0.0:
                # Create dropout mask: randomly set neurons to 0 with probability dropout_rate
                dropout_mask = (np.random.random(A.shape) > self.dropout_rate).astype(float)
                # Scale by (1 - dropout_rate) to maintain expected value during training
                dropout_mask /= (1.0 - self.dropout_rate)
                A = A * dropout_mask
                self.dropout_masks.append(dropout_mask)
            else:
                self.dropout_masks.append(None)  # No dropout for this layer

            # Cache activation
            layer.activation_cache['A'] = A

        return A

    # ------------------------------------------------------------------
    # BACKWARD
    # ------------------------------------------------------------------

    #----------------------------------------
    # OLD BACKWARDS FUNCTION
    #----------------------------------------
    # def backward(self, X: np.ndarray, y: np.ndarray, y_pred=None):

    #     if y_pred is None:
    #         y_pred = self.forward(X)

    #     loss_deriv_fn = get_loss_derivative(self.loss_function)
    #     dA = loss_deriv_fn(y_pred, y)

    #     # Backprop layers in reverse
    #     for layer in reversed(self.layers):
    #         m = X.shape[0]
            
    #         Z = layer.activation_cache['Z']
    #         A_prev = layer.activation_cache['A_prev']

    #         # dZ = dA * activation'(Z)
    #         activation_grad = get_activation_derivative(layer.activation)
    #         dZ = dA * activation_grad(Z)

    #         # Gradients
    #         layer.dW = (A_prev.T @ dZ) / m
    #         layer.db = np.sum(dZ, axis=0) / m

    #         # L2 regularization
    #         if self.l2_lambda > 0:
    #             layer.dW += (self.l2_lambda / m) * layer.W

    #         # Next gradient
    #         dA = dZ @ layer.W.T
    
    #----------------------------------------
    # NEW BACKWARDS FUNCTION from gpt (adds special casing for softmax to avoid multiplying with the softmax activation derivative in the output layer)
    #----------------------------------------
    def backward(self, X: np.ndarray, y: np.ndarray, y_pred=None):

        if y_pred is None:
            y_pred = self.forward(X)

        loss_deriv_fn = get_loss_derivative(self.loss_function)
        dA = loss_deriv_fn(y_pred, y)

        m = X.shape[0]
        n_layers = len(self.layers)
        for idx in range(n_layers - 1, -1, -1):
            layer = self.layers[idx]
            Z = layer.activation_cache['Z']
            A_prev = layer.activation_cache['A_prev']

            # If using softmax + cross-entropy and the loss derivative already returns dZ,
            # do not multiply again by the softmax Jacobian on the output layer.
            is_output = (idx == n_layers - 1)
            if is_output and self.output_activation == 'softmax' and self.loss_function == 'cross_entropy':
                dZ = dA
            else:
                activation_grad = get_activation_derivative(layer.activation)
                dZ = dA * activation_grad(Z)

            # Gradients (averaged over batch)
            layer.dW = (A_prev.T @ dZ) / m
            layer.db = np.sum(dZ, axis=0, keepdims=True) / m
            if layer.b.shape != layer.db.shape:
                layer.db = layer.db.reshape(layer.b.shape)

            # L2 regularization (scale consistent with averaging)
            if self.l2_lambda > 0:
                layer.dW += (self.l2_lambda / m) * layer.W

            # Apply dropout mask to gradient if dropout was used during forward pass
            if idx < len(self.dropout_masks) - 1:  # Not output layer
                dropout_mask = self.dropout_masks[idx]
                if dropout_mask is not None:
                    dA = dA * dropout_mask
            
            # propagate to previous layer
            dA = dZ @ layer.W.T

    # ------------------------------------------------------------------
    # UPDATE
    # ------------------------------------------------------------------

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
    
    
    # ------------------------------------------------------------------
    # COMPUTE LOSS
    # ------------------------------------------------------------------

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
    

    # ------------------------------------------------------------------
    # TRAINING STEP
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # PREDICT
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Predict class labels: get probabilities and return argmax
        # X: (batch_size, input_size) -> returns: (batch_size,) class indices
        probabilities = self.predict_proba(X)
        predictions = np.argmax(probabilities, axis=1)
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Get prediction probabilities (forward pass without training/dropout)
        # X: (batch_size, input_size) -> returns: (batch_size, output_size) probabilities
        was_training = self.training
        self.training = False  # Disable dropout for inference
        probabilities = self.forward(X)
        self.training = was_training  # Restore previous training state
        return probabilities
    
    def train(self):
        """Set model to training mode (enables dropout)."""
        self.training = True
    
    def eval(self):
        """Set model to evaluation mode (disables dropout)."""
        self.training = False
    
    
    # ------------------------------------------------------------------
    # PARAMETERS
    # ------------------------------------------------------------------

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
