"""
Weights & Biases sweep configuration for hyperparameter tuning.

This file defines sweep configurations for different experiments.
"""

import wandb


# Sweep configuration for random search
sweep_config_random = {
    'method': 'random',  # Random search
    'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
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
                [64],
                [128],
                [256],
                [128, 64],
                [256, 128],
                [512, 256],
                [128, 64, 32]
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
        }
    }
}


# Sweep configuration for Bayesian optimization
sweep_config_bayes = {
    'method': 'bayes',  # Bayesian optimization
    'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
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
                [128, 64],
                [256, 128],
                [256, 128, 64]
            ]
        },
        'l2_lambda': {
            'distribution': 'log_uniform_values',
            'min': 0.00001,
            'max': 0.01
        }
    }
}


# Sweep configuration for grid search (activation functions)
sweep_config_activations = {
    'method': 'grid',  # Grid search
    'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
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
    
    # TODO: Add command line interface to create sweeps
    print("Sweep configurations defined.")
    print("To create a sweep, use wandb.sweep() in your training script")
    print("Or modify this file to create sweeps directly")

