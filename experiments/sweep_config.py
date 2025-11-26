"""
Weights & Biases sweep configuration for hyperparameter tuning.

This file defines sweep configurations for different experiments.
"""

import wandb
import os
import sys


# Sweep configuration for random search
sweep_config_random = {
    'method': 'random',  # Random search
    'metric': {
        'name': 'val_acc',  # Must match the metric name logged in training scripts
        'goal': 'maximize'
    },
    'parameters': {
        # Fixed parameters (required by train.py)
        'dataset': {'value': 'cifar10'},  # Focus on CIFAR-10
        'output_size': {'value': 10},
        'output_activation': {'value': 'softmax'},
        'num_epochs': {'value': 150},  # More epochs for CIFAR-10
        'val_split': {'value': 0.2},
        'random_seed': {'value': 42},
        
        # Swept parameters
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 0.0001,
            'max': 0.1
        },
        'batch_size': {
            'values': [16, 32, 64, 128]
        },
        'optimizer': {
            'values': ['sgd', 'momentum', 'rmsprop', 'adam']
        },
        'hidden_layers': {
            'values': [
                [256],
                [512],
                [1024],
                [512, 256],
                [1024, 512],
                [1024, 512, 256],
                [512, 256, 128]
            ]
        },
        'activation': {
            'values': ['relu', 'sigmoid', 'tanh']
        },
        'weight_init': {
            'values': ['random', 'xavier', 'he']
        },
        'l2_lambda': {
            'distribution': 'log_uniform_values',
            'min': 0.00001,
            'max': 0.01
        },
        'dropout_rate': {
            'values': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]  # Dropout regularization
        }
    }
}


# Sweep configuration for Bayesian optimization
sweep_config_bayes = {
    'method': 'bayes',  # Bayesian optimization
    'metric': {
        'name': 'val_acc',  # Must match the metric name logged in training scripts
        'goal': 'maximize'
    },
    'parameters': {
        # Fixed parameters (required by train.py)
        'dataset': {'value': 'cifar10'},  # Focus on CIFAR-10
        'output_size': {'value': 10},
        'output_activation': {'value': 'softmax'},
        'num_epochs': {'value': 150},  # More epochs for CIFAR-10
        'val_split': {'value': 0.2},
        'random_seed': {'value': 42},
        'activation': {'value': 'relu'},
        'weight_init': {'value': 'he'},
        
        # Swept parameters
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 0.0001,
            'max': 0.1
        },
        'batch_size': {
            'values': [32, 64, 128]
        },
        'optimizer': {
            'values': ['adam', 'rmsprop']
        },
        'hidden_layers': {
            'values': [
                [512, 256],
                [1024, 512],
                [1024, 512, 256]
            ]
        },
        'l2_lambda': {
            'distribution': 'log_uniform_values',
            'min': 0.00001,
            'max': 0.01
        },
        'dropout_rate': {
            'values': [0.0, 0.2, 0.3, 0.4]  # Dropout regularization (focused range for Bayesian)
        }
    }
}


# Sweep configuration for grid search (activation functions)
sweep_config_activations = {
    'method': 'grid',  # Grid search
    'metric': {
        'name': 'val_acc',  # Must match the metric name logged in training scripts
        'goal': 'maximize'
    },
    'parameters': {
        # Fixed parameters (required by train.py)
        'dataset': {'value': 'cifar10'},  # Focus on CIFAR-10
        'output_size': {'value': 10},
        'output_activation': {'value': 'softmax'},
        'num_epochs': {'value': 150},  # More epochs for CIFAR-10
        'val_split': {'value': 0.2},
        'random_seed': {'value': 42},
        
        # Swept parameters
        'activation': {
            'values': ['relu', 'sigmoid', 'tanh']
        },
        'weight_init': {
            'values': ['random', 'xavier', 'he']
        },
        'learning_rate': {
            'value': 0.001
        },
        'optimizer': {
            'value': 'adam'
        },
        'batch_size': {
            'value': 64
        },
        'hidden_layers': {
            'value': [1024, 512]  # Larger network for CIFAR-10
        },
        'l2_lambda': {
            'value': 0.0001
        },
        'dropout_rate': {
            'value': 0.0  # Fixed value for activation function comparison
        }
    }
}


def create_sweep(sweep_config, project_name='neural-network-numpy'):
    """
    Create a WandB sweep.
    
    Args:
        sweep_config: Sweep configuration dictionary
        project_name: WandB project name
        
    Returns:
        Sweep ID
        
    TODO: Initialize sweep and return sweep ID
    """
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    return sweep_id


def run_sweep_agent(sweep_id, train_function, count=10):
    """
    Run a sweep agent.
    
    Args:
        sweep_id: Sweep ID from create_sweep
        train_function: Training function to run for each configuration
        count: Number of runs to execute
        
    TODO: Run wandb agent for the sweep
    """
    wandb.agent(sweep_id, function=train_function, count=count)


if __name__ == '__main__':
    """
    Example usage:
    
    1. Create a sweep:
       python sweep_config.py
    
    2. Run agent:
       wandb agent <sweep_id>
    """
    import sys
    
    # Import train function
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from experiments.train import train
    
    # Example: Create and run a sweep
    if len(sys.argv) > 1:
        config_name = sys.argv[1]
        if config_name == 'random':
            sweep_config = sweep_config_random
        elif config_name == 'bayes':
            sweep_config = sweep_config_bayes
        elif config_name == 'activations':
            sweep_config = sweep_config_activations
        else:
            print(f"Unknown config: {config_name}")
            print("Available: random, bayes, activations")
            sys.exit(1)
        
        project_name = sys.argv[2] if len(sys.argv) > 2 else 'neural-network-numpy'
        sweep_id = create_sweep(sweep_config, project_name)
        print(f"Created sweep: {sweep_id}")
        print(f"Run with: wandb agent {sweep_id}")
    else:
        print("Sweep configurations defined.")
        print("\nUsage:")
        print("  python sweep_config.py <config_name> [project_name]")
        print("\nConfigs: random, bayes, activations")
        print("\nExample:")
        print("  python sweep_config.py random")
        print("  wandb agent <sweep_id>")

