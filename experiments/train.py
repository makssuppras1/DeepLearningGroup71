"""
Main training script for neural network experiments.

This script handles the complete training pipeline with WandB logging.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.neural_network import NeuralNetwork
from src.data_loader import load_fashion_mnist, load_cifar10, preprocess_data, create_mini_batches, train_val_split
from src.utils import accuracy_score, plot_training_curves, set_random_seed
import wandb


def train_epoch(model, X_train, y_train, batch_size):
    """
    Train for one epoch.
    
    Args:
        model: Neural network model
        X_train: Training data
        y_train: Training labels
        batch_size: Batch size
        
    Returns:
        Average loss and accuracy for the epoch
        
    TODO: Implement one epoch of training
    - Create mini-batches
    - Train on each batch
    - Compute average loss and accuracy
    """
    pass


def evaluate(model, X_val, y_val):
    """
    Evaluate model on validation set.
    
    Args:
        model: Neural network model
        X_val: Validation data
        y_val: Validation labels
        
    Returns:
        Validation loss and accuracy
        
    TODO: Implement model evaluation
    - Make predictions
    - Compute loss and accuracy
    """
    pass


def train(config):
    """
    Complete training pipeline.
    
    Args:
        config: Dictionary containing hyperparameters
        
    TODO: Implement full training pipeline
    - Initialize WandB
    - Load and preprocess data
    - Create model
    - Training loop
    - Log metrics to WandB
    - Save best model
    """
    
    # Initialize Weights & Biases
    # TODO: Initialize wandb.init() with config
    
    # Set random seed
    # TODO: Set random seed from config
    
    # Load dataset
    # TODO: Load dataset based on config['dataset']
    
    # Preprocess data
    # TODO: Preprocess and split into train/val
    
    # Create model
    # TODO: Initialize NeuralNetwork with config parameters
    
    # Training loop
    # TODO: Implement training loop
    # - For each epoch:
    #   - Train on training set
    #   - Evaluate on validation set
    #   - Log metrics to WandB
    #   - Save best model
    #   - Print progress
    
    pass


def main():
    """
    Main function to run training.
    
    TODO: Set up configuration and run training
    """
    
    # Default configuration
    config = {
        # Dataset
        'dataset': 'fashion_mnist',  # or 'cifar10'
        
        # Model architecture
        'input_size': 784,  # 28*28 for Fashion-MNIST
        'hidden_layers': [128, 64],
        'output_size': 10,
        
        # Activation and loss
        'activation': 'relu',
        'output_activation': 'softmax',
        'loss': 'cross_entropy',
        
        # Training hyperparameters
        'num_epochs': 50,
        'batch_size': 32,
        'learning_rate': 0.001,
        
        # Optimization
        'optimizer': 'adam',  # 'sgd', 'momentum', 'rmsprop', 'adam'
        
        # Regularization
        'l2_lambda': 0.0001,
        
        # Initialization
        'weight_init': 'xavier',  # 'random', 'xavier', 'he'
        
        # Other
        'val_split': 0.2,
        'random_seed': 42,
        'project_name': 'neural-network-numpy',
        'experiment_name': 'baseline'
    }
    
    # TODO: Parse command line arguments to override config if needed
    
    # Run training
    train(config)
    
    print("Training completed!")


if __name__ == '__main__':
    main()

