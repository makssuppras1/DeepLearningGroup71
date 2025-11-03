"""
Data loading and preprocessing utilities.

Functions to download, load, and preprocess Fashion-MNIST and CIFAR-10 datasets.
"""

import numpy as np
import os
from typing import Tuple, Optional
import gzip
import pickle


def download_fashion_mnist(data_dir: str = './data') -> None:
    """
    Download Fashion-MNIST dataset.
    
    Args:
        data_dir: Directory to save the dataset
        
    TODO: Implement Fashion-MNIST download
    Hint: You can use keras.datasets or download from official source
    """
    pass


def download_cifar10(data_dir: str = './data') -> None:
    """
    Download CIFAR-10 dataset.
    
    Args:
        data_dir: Directory to save the dataset
        
    TODO: Implement CIFAR-10 download
    """
    pass


def load_fashion_mnist(data_dir: str = './data') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load Fashion-MNIST dataset.
    
    Args:
        data_dir: Directory containing the dataset
        
    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
        
    TODO: Implement Fashion-MNIST loading
    - Load training and test data
    - Flatten images if needed
    - Normalize pixel values to [0, 1]
    """
    pass


def load_cifar10(data_dir: str = './data') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load CIFAR-10 dataset.
    
    Args:
        data_dir: Directory containing the dataset
        
    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
        
    TODO: Implement CIFAR-10 loading
    - Load training and test batches
    - Flatten images if needed
    - Normalize pixel values to [0, 1]
    """
    pass


def preprocess_data(
    X: np.ndarray,
    y: np.ndarray,
    num_classes: int = 10,
    flatten: bool = True,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess input data and labels.
    
    Args:
        X: Input data
        y: Labels
        num_classes: Number of classes for one-hot encoding
        flatten: Whether to flatten images
        normalize: Whether to normalize to [0, 1]
        
    Returns:
        Tuple of (X_processed, y_processed)
        
    TODO: Implement data preprocessing
    - Flatten images if needed
    - Normalize pixel values
    - One-hot encode labels
    """
    pass


def create_mini_batches(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True
) -> list:
    """
    Create mini-batches for training.
    
    Args:
        X: Input data
        y: Labels
        batch_size: Size of each mini-batch
        shuffle: Whether to shuffle data before batching
        
    Returns:
        List of (X_batch, y_batch) tuples
        
    TODO: Implement mini-batch creation
    - Shuffle data if requested
    - Split into batches
    - Handle last batch if it's smaller
    """
    pass


def train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    val_split: float = 0.2,
    random_seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and validation sets.
    
    Args:
        X: Input data
        y: Labels
        val_split: Fraction of data to use for validation
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_val, y_train, y_val)
        
    TODO: Implement train/validation split
    """
    pass


# Dataset information
FASHION_MNIST_CLASSES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def get_class_names(dataset: str) -> list:
    """
    Get class names for a dataset.
    
    Args:
        dataset: Dataset name ('fashion_mnist' or 'cifar10')
        
    Returns:
        List of class names
    """
    if dataset.lower() == 'fashion_mnist':
        return FASHION_MNIST_CLASSES
    elif dataset.lower() == 'cifar10':
        return CIFAR10_CLASSES
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

