# Loss functions for training neural networks
# Implement various loss functions and their derivatives for backpropagation

import numpy as np

def min_log_likelihood(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    # Minimum Log-Likelihood loss
    # Formula: MLL = -sum(log(Pr(y_true | y_pred)))
    # y_pred: Predicted probabilities of shape (batch_size, num_classes)
    # y_true: True labels (one-hot encoded) of shape (batch_size, num_classes)
    # Returns: MLL loss value (scalar)
    eps = 1e-12
    y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)
    
    # Ensure batch dimension
    if y_pred_clipped.ndim == 1:
        y_pred_clipped = y_pred_clipped.reshape(1, -1)
        y_true = y_true.reshape(1, -1)
    
    # Compute negative log-likelihood: -sum(y_true * log(y_pred))
    MLL = -np.sum(y_true * np.log(y_pred_clipped))
    return float(MLL)

def mean_squared_error(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    MSE = np.mean(np.square(y_pred - y_true))
    return float(MSE)

def mse_derivative(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    n = y_pred.size
    dMSE = (2/n) * (y_pred - y_true)
    return dMSE

def cross_entropy_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    # Cross-entropy loss for classification
    # Formula: CE = -(1/n) * sum(y_true * log(y_pred))
    # y_pred: Predicted probabilities of shape (batch_size, num_classes)
    # y_true: True labels (one-hot encoded) of shape (batch_size, num_classes)
    # Returns: Cross-entropy loss value (scalar)
    # Hint: Add small epsilon to prevent log(0)
    
    eps = 1e-12
    y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)

    # Ensure batch dimension
    if y_pred_clipped.ndim == 1:
        y_pred_clipped = y_pred_clipped.reshape(1, -1)
        y_true = y_true.reshape(1, -1)

    loss = -np.sum(y_true * np.log(y_pred_clipped)) / y_pred_clipped.shape[0]
    return float(loss)

def cross_entropy_derivative(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Compute derivative of cross-entropy loss with respect to logits (before softmax).
    For softmax + cross-entropy, this simplifies to: (y_pred - y_true) / n
    where y_pred is the softmax output and n is the batch size.
    This returns averaged gradients, matching PyTorch's behavior.
    """
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(1, -1)
        y_true = y_true.reshape(1, -1)

    n = y_pred.shape[0]
    # Return averaged gradient: (y_pred - y_true) / n
    # This matches PyTorch's behavior where loss.backward() computes gradients
    # from the averaged loss, so gradients are automatically averaged
    return (y_pred - y_true) / n

def binary_cross_entropy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    # Formula: BCE = -(1/n) * sum(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
    eps = 1e-12
    y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)
    loss = -np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
    return float(loss)

def l2_regularization(weights: list, lambda_: float) -> float:
    # Formula: L2 = (lambda/2) * sum(W^2)
    l2_sum = 0.0
    for W in weights:
        l2_sum += np.sum(np.square(W))
    
    L2 = (lambda_ / 2) * l2_sum
    return L2

def l2_regularization_derivative(weight: np.ndarray, lambda_: float) -> np.ndarray:
    # Formula: dL2/dW = lambda * W
    return lambda_ * weight


# Dictionary mapping loss names to functions
LOSS_FUNCTIONS = {
    'mse': mean_squared_error,
    'cross_entropy': cross_entropy_loss,
    'binary_cross_entropy': binary_cross_entropy
}

LOSS_DERIVATIVES = {
    'mse': mse_derivative,
    'cross_entropy': cross_entropy_derivative,
    'binary_cross_entropy': cross_entropy_derivative  # Same as cross_entropy
}

def get_loss_function(name: str):
    if name not in LOSS_FUNCTIONS:
        raise ValueError(f"Unknown loss function: {name}")
    return LOSS_FUNCTIONS[name]

def get_loss_derivative(name: str):
    if name not in LOSS_DERIVATIVES:
        raise ValueError(f"Unknown loss function: {name}")
    return LOSS_DERIVATIVES[name]