"""
Loss functions for training neural networks.

Implement various loss functions and their derivatives for backpropagation.
"""

import numpy as np

def min_log_likelihood(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Minimum Log-Likelihood loss.
    
    Formula: MLL = -sum(log(Pr(y_true | y_pred)))
    
    Args:
        y_pred: Predicted probabilities of shape (batch_size, num_classes)
        y_true: True labels (one-hot encoded) of shape (batch_size, num_classes)
        
    Returns:
        MLL loss value (scalar)
    """
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
    """
    Mean Squared Error loss.
    
    Formula: MSE = (1/n) * sum((y_pred - y_true)^2)
    
    Args:
        y_pred: Predicted values of shape (batch_size, output_size)
        y_true: True values of shape (batch_size, output_size)
        
    Returns:
        MSE loss value (scalar)
        
    TODO: Implement MSE loss
    """
    MSE = np.mean(np.square(y_pred - y_true))
    return float(MSE)



def mse_derivative(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Derivative of Mean Squared Error loss.
    
    Formula: dMSE/dy_pred = (2/n) * (y_pred - y_true)
    
    Args:
        y_pred: Predicted values
        y_true: True values
        
    Returns:
        Gradient of MSE with respect to predictions
        
    TODO: Implement MSE derivative
    """
    n = y_pred.size
    dMSE = (2/n) * (y_pred - y_true)
    return dMSE


def cross_entropy_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Cross-entropy loss for classification.
    
    Formula: CE = -(1/n) * sum(y_true * log(y_pred))
    
    Args:
        y_pred: Predicted probabilities of shape (batch_size, num_classes)
        y_true: True labels (one-hot encoded) of shape (batch_size, num_classes)
        
    Returns:
        Cross-entropy loss value (scalar)
        
    TODO: Implement cross-entropy loss
    Hint: Add small epsilon to prevent log(0)
    """
    
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
    Derivative of cross-entropy loss (combined with softmax).
    
    When used with softmax activation, the gradient simplifies to: y_pred - y_true
    
    Args:
        y_pred: Predicted probabilities
        y_true: True labels (one-hot encoded)
        
    Returns:
        Gradient of cross-entropy with respect to predictions
        
    TODO: Implement cross-entropy derivative
    """
    # Ensure batch dimension
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(1, -1)
        y_true = y_true.reshape(1, -1)

    n = y_pred.shape[0]
    return (y_pred - y_true) / n


def binary_cross_entropy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Binary cross-entropy loss.
    
    Formula: BCE = -(1/n) * sum(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
    
    Args:
        y_pred: Predicted probabilities
        y_true: True binary labels
        
    Returns:
        Binary cross-entropy loss value
        
    TODO: Implement binary cross-entropy loss
    """
    eps = 1e-12
    y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)
    loss = -np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
    return float(loss)


def l2_regularization(weights: list, lambda_: float) -> float:
    """
    Compute L2 regularization term.
    
    Formula: L2 = (lambda/2) * sum(W^2)
    
    Args:
        weights: List of weight matrices from all layers
        lambda_: Regularization coefficient
        
    Returns:
        L2 regularization term
        
    TODO: Implement L2 regularization
    Hint: Sum the squared values of all weights (not biases)
    """
    l2_sum = 0.0
    for W in weights:
        l2_sum += np.sum(np.square(W))
    return (lambda_ / 2) * l2_sum



def l2_regularization_derivative(weight: np.ndarray, lambda_: float) -> np.ndarray:
    """
    Derivative of L2 regularization term.
    
    Formula: dL2/dW = lambda * W
    
    Args:
        weight: Weight matrix
        lambda_: Regularization coefficient
        
    Returns:
        Gradient of L2 term with respect to weights
        
    TODO: Implement L2 regularization derivative
    """
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
    """
    Get loss function by name.
    
    Args:
        name: Name of loss function
        
    Returns:
        Loss function
    """
    if name not in LOSS_FUNCTIONS:
        raise ValueError(f"Unknown loss function: {name}")
    return LOSS_FUNCTIONS[name]


def get_loss_derivative(name: str):
    """
    Get loss derivative by name.
    
    Args:
        name: Name of loss function
        
    Returns:
        Derivative function
    """
    if name not in LOSS_DERIVATIVES:
        raise ValueError(f"Unknown loss function: {name}")
    return LOSS_DERIVATIVES[name]