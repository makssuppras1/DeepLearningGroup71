# Neural network implementaton from scratch
import numpy as np
from .initializers import initialize_weights
from .layers import DenseLayer
from .activations import get_activation, get_activation_derivative
from .losses import get_loss_function, get_loss_derivative, l2_regularization
from .optimizers import get_optimizer

class NeuralNetwork:
    # Feedforward neural network with NumPy
    def __init__(self, input_size, hidden_layers, output_size, activation='relu',
                 output_activation='softmax', learning_rate=0.001, optimizer='adam',
                 weight_init='he', l2_lambda=0.0, dropout_rate=0.0, random_seed=None):
        # Initilize network layers
        self.layers = []
        input_dim = input_size
        # Create hidden layers
        for hidden_units in hidden_layers:
            layer = DenseLayer(input_size=input_dim, output_size=hidden_units,
                             activation=activation, weight_init=weight_init, seed=random_seed)
            self.layers.append(layer)
            input_dim = hidden_units
        # Output layer
        self.layers.append(DenseLayer(input_size=input_dim, output_size=output_size,
                                    activation=output_activation, weight_init=weight_init, seed=random_seed))
        # Store hyperparams
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.activation = activation
        self.output_activation = output_activation
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.l2_lambda = l2_lambda
        self.dropout_rate = dropout_rate
        self.training = True
        # Setup optimzer
        self.optimizer = get_optimizer(optimizer, learning_rate=learning_rate)
        self.loss_function = 'cross_entropy'
        self.last_predictions = None
        self.last_loss = None
        self.dropout_masks = []
    
    def forward(self, X):
        # Forward pass with dropout during training
        A = X
        self.dropout_masks = []
        for i, layer in enumerate(self.layers):
            layer.activation_cache['A_prev'] = A
            # Linear transform
            Z = A @ layer.W + layer.b
            layer.activation_cache['Z'] = Z
            # Activaton
            act = get_activation(layer.activation)
            A = act(Z)
            # Apply dropout to hidden layers
            is_hidden_layer = (i < len(self.layers) - 1)
            if self.training and is_hidden_layer and self.dropout_rate > 0.0:
                dropout_mask = (np.random.random(A.shape) > self.dropout_rate).astype(float)
                dropout_mask /= (1.0 - self.dropout_rate)
                A = A * dropout_mask
                self.dropout_masks.append(dropout_mask)
            else:
                self.dropout_masks.append(None)
            layer.activation_cache['A'] = A
        return A
    
    def backward(self, X, y, y_pred=None):
        # Backward pass: compute gradients
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
            # Skip softmax deriv if using softmax+crossentropy
            is_output = (idx == n_layers - 1)
            if is_output and self.output_activation == 'softmax' and self.loss_function == 'cross_entropy':
                dZ = dA
            else:
                activation_grad = get_activation_derivative(layer.activation)
                dZ = dA * activation_grad(Z)
            # Compute gradients (already averaged by loss deriv)
            layer.dW = A_prev.T @ dZ
            layer.db = np.sum(dZ, axis=0, keepdims=True)
            if layer.b.shape != layer.db.shape:
                layer.db = layer.db.reshape(layer.b.shape)
            # L2 regulariation
            if self.l2_lambda > 0:
                layer.dW += (self.l2_lambda / m) * layer.W
            # Apply dropout mask to gradient
            if idx < len(self.dropout_masks) - 1:
                dropout_mask = self.dropout_masks[idx]
                if dropout_mask is not None:
                    dA = dA * dropout_mask
            # Propagate to prev layer
            dA = dZ @ layer.W.T
    
    def update_weights(self):
        # Update weights using optimzer
        params = {}
        grads = {}
        for i, layer in enumerate(self.layers):
            params[f'W{i+1}'] = layer.W
            params[f'b{i+1}'] = layer.b
            grads[f'W{i+1}'] = layer.dW
            grads[f'b{i+1}'] = layer.db
        updated_params = self.optimizer.update(params, grads)
        for i, layer in enumerate(self.layers):
            layer.W = updated_params[f'W{i+1}']
            layer.b = updated_params[f'b{i+1}']
    
    def compute_loss(self, y_pred, y_true):
        # Compute total loss: data loss + L2
        loss_func = get_loss_function(self.loss_function)
        data_loss = loss_func(y_pred, y_true)
        if self.l2_lambda > 0:
            weights = [layer.W for layer in self.layers]
            reg_loss = l2_regularization(weights, self.l2_lambda)
            total_loss = data_loss + reg_loss
        else:
            total_loss = data_loss
        return total_loss
    
    def train_step(self, X_batch, y_batch):
        # One training step: forward -> backward -> update
        y_pred = self.forward(X_batch)
        loss = self.compute_loss(y_pred, y_batch)
        self.backward(X_batch, y_batch, y_pred=y_pred)
        self.update_weights()
        self.last_predictions = y_pred
        self.last_loss = loss
        return loss
    
    def predict(self, X):
        # Predict class labels
        probabilities = self.predict_proba(X)
        predictions = np.argmax(probabilities, axis=1)
        return predictions
    
    def predict_proba(self, X):
        # Get prediction probabilites (no dropout)
        was_training = self.training
        self.training = False
        probabilities = self.forward(X)
        self.training = was_training
        return probabilities
    
    def train(self):
        # Set to training mode
        self.training = True
    
    def eval(self):
        # Set to eval mode
        self.training = False
    
    def get_params(self):
        # Get all params as dict
        params = {}
        for i, layer in enumerate(self.layers):
            params[f'W{i+1}'] = layer.W.copy()
            params[f'b{i+1}'] = layer.b.copy()
        return params
    
    def set_params(self, params):
        # Set params from dict
        for i, layer in enumerate(self.layers):
            layer.W = params[f'W{i+1}'].copy()
            layer.b = params[f'b{i+1}'].copy()
