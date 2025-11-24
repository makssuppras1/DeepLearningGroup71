import numpy as np
from typing import Optional, List

from initializers import initialize_weights
from layers import DenseLayer
from activations import get_activation, get_activation_derivative
from losses import get_loss_function, get_loss_derivative, l2_regularization
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
    
    
    # ------------------------------------------------------------------
    # FORWARD
    # ------------------------------------------------------------------

    def forward(self, X: np.ndarray) -> np.ndarray:

        A = X

        for layer in self.layers:

            # Cache previous activation
            layer.activation_cache['A_prev'] = A

            # Linear
            Z = A @ layer.W + layer.b
            layer.activation_cache['Z'] = Z

            # Activation
            act = get_activation(layer.activation)
            A = act(Z)

            # Cache activation
            layer.activation_cache['A'] = A

        return A

    # ------------------------------------------------------------------
    # BACKWARD
    # ------------------------------------------------------------------

    def backward(self, X: np.ndarray, y: np.ndarray, y_pred=None):

        if y_pred is None:
            y_pred = self.forward(X)

        loss_deriv_fn = get_loss_derivative(self.loss_function)
        dA = loss_deriv_fn(y_pred, y)

        # Backprop layers in reverse
        for layer in reversed(self.layers):

            Z = layer.activation_cache['Z']
            A_prev = layer.activation_cache['A_prev']

            # dZ = dA * activation'(Z)
            activation_grad = get_activation_derivative(layer.activation)
            dZ = dA * activation_grad(Z)

            # Gradients
            layer.dW = A_prev.T @ dZ
            layer.db = np.sum(dZ, axis=0)

            # L2 regularization
            if self.l2_lambda > 0:
                layer.dW += self.l2_lambda * layer.W

            # Next gradient
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
        # Get prediction probabilities (forward pass without training)
        # X: (batch_size, input_size) -> returns: (batch_size, output_size) probabilities
        probabilities = self.forward(X)
        return probabilities
    
    
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
