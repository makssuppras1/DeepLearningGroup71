# Loss functons for training
import numpy as np

def min_log_likelihood(y_pred, y_true):
    # Min log likelihood loss
    eps = 1e-12
    y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)
    if y_pred_clipped.ndim == 1:
        y_pred_clipped = y_pred_clipped.reshape(1, -1)
        y_true = y_true.reshape(1, -1)
    MLL = -np.sum(y_true * np.log(y_pred_clipped))
    return float(MLL)

def mean_squared_error(y_pred, y_true):
    # MSE loss
    MSE = np.mean(np.square(y_pred - y_true))
    return float(MSE)

def mse_derivative(y_pred, y_true):
    # Derivative of MSE
    n = y_pred.size
    dMSE = (2/n) * (y_pred - y_true)
    return dMSE

def cross_entropy_loss(y_pred, y_true):
    # Cross-entropy loss for classificaton
    eps = 1e-12
    y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)
    if y_pred_clipped.ndim == 1:
        y_pred_clipped = y_pred_clipped.reshape(1, -1)
        y_true = y_true.reshape(1, -1)
    loss = -np.sum(y_true * np.log(y_pred_clipped)) / y_pred_clipped.shape[0]
    return float(loss)

def cross_entropy_derivative(y_pred, y_true):
    # Derivative of cross-entropy (simplified for softmax)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(1, -1)
        y_true = y_true.reshape(1, -1)
    n = y_pred.shape[0]
    # Averaged gradient to match PyTorch behavoir
    return (y_pred - y_true) / n

def binary_cross_entropy(y_pred, y_true):
    # Binary cross-entropy loss
    eps = 1e-12
    y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)
    loss = -np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
    return float(loss)

def l2_regularization(weights, lambda_):
    # L2 regulariation: (lambda/2) * sum(W^2)
    l2_sum = 0.0
    for W in weights:
        l2_sum += np.sum(np.square(W))
    L2 = (lambda_ / 2) * l2_sum
    return L2

def l2_regularization_derivative(weight, lambda_):
    # Derivative: lambda * W
    return lambda_ * weight

LOSS_FUNCTIONS = {
    'mse': mean_squared_error,
    'cross_entropy': cross_entropy_loss,
    'binary_cross_entropy': binary_cross_entropy
}

LOSS_DERIVATIVES = {
    'mse': mse_derivative,
    'cross_entropy': cross_entropy_derivative,
    'binary_cross_entropy': cross_entropy_derivative
}

def get_loss_function(name):
    # Get loss functon by name
    if name not in LOSS_FUNCTIONS:
        raise ValueError(f"Unknown loss function: {name}")
    return LOSS_FUNCTIONS[name]

def get_loss_derivative(name):
    # Get loss derivitive by name
    if name not in LOSS_DERIVATIVES:
        raise ValueError(f"Unknown loss function: {name}")
    return LOSS_DERIVATIVES[name]
