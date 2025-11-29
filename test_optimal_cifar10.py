#!/usr/bin/env python3
"""
Quick test script to verify optimal CIFAR-10 configuration.
This uses the settings that achieved 80.9% validation accuracy in your best run.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.train_simple import train, init_wandb

# Optimal configuration based on your best results (Run 101: 80.9% val_acc)
optimal_config = {
    'dataset': 'cifar10',
    'hidden_layers': [256],  # Single layer - best result!
    'output_size': 10,
    'activation': 'tanh',  # Tanh, NOT ReLU!
    'output_activation': 'softmax',
    'num_epochs': 150,  # Best runs needed 67-113 epochs
    'batch_size': 64,  # Optimal range: 24-80
    'learning_rate': 0.003,  # Optimal range: 0.0026-0.0049 (10x higher than default!)
    'optimizer': 'sgd',  # SGD worked better than Adam for CIFAR-10
    'l2_lambda': 0.003,  # Optimal range: 0.002-0.005 (30x higher than default!)
    'weight_init': 'xavier',  # Xavier for tanh activation!
    'dropout_rate': 0.1,  # Optimal range: 0.059-0.129
    'val_split': 0.2,
    'random_seed': 42,
    'project_name': 'neural-network-numpy',
    'experiment_name': 'cifar10_optimal_test',
    'use_wandb': True,
    'entity': 'makssuppras1-danmarks-tekniske-universitet-dtu'
}

if __name__ == '__main__':
    print("=" * 70)
    print("Testing Optimal CIFAR-10 Configuration")
    print("=" * 70)
    print("\nKey differences from default:")
    print("  - Activation: tanh (not ReLU)")
    print("  - Weight init: xavier (not he)")
    print("  - Learning rate: 0.003 (not 0.0003)")
    print("  - L2 lambda: 0.003 (not 0.0001)")
    print("  - Optimizer: sgd (not adam)")
    print("  - Architecture: [256] single layer (not [1024, 512, 256])")
    print("\nExpected: 65-81% validation accuracy")
    print("=" * 70)
    print()
    
    # Initialize WandB
    init_wandb(optimal_config)
    
    # Train
    train(optimal_config)

