#!/usr/bin/env python3
"""
Create WandB sweep for HPC using the configuration from train3.ipynb
Usage: python create_hpc_sweep.py
"""

import wandb
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Sweep configuration (from train3.ipynb)
# Note: Remove 'entity' if using personal account, or use correct entity name
sweep_config = {
    'program': 'experiments/train_simple.py',
    'method': 'bayes',
    'project': 'neural-network-numpy',
    'name': 'HPC_sweep',

    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 2
    },
    'metric': {
        'name': 'val_acc',
        'goal': 'maximize'   
    },
    'parameters': {
        'optimizer': {
            'values': ['adam', 'sgd']
        },
        'hidden_layers': {
            'values': [
                [32],
                [64],
                [128],
                [256],
                [128, 64],
                [256, 128],
                [512, 256],
                [128, 64, 32],
                [1024, 512, 256],
                [2048, 1024, 512],
                [4096, 2048, 1024]
            ]
        },
        'l2_lambda': {
            'distribution': 'uniform',
            'min': 0.000001,
            'max': 0.01
        },
        'learning_rate': {
            'distribution': 'uniform',
            'min': 0.0000001,
            'max': 0.1
        },
        'batch_size': {
            'distribution': 'q_log_uniform_values',
            'q': 8,
            'min': 8,
            'max': 128,
        },
        'activation': {
            'values': ['relu', 'sigmoid', 'tanh']
        },
        'output_activation': {
            'value': 'softmax'  # Keep fixed - softmax is standard for multi-class
        },
        'num_epochs': {
            'distribution': 'int_uniform',
            'min': 100,
            'max': 200
        },
        'weight_init': {
            'values': ['random', 'xavier', 'he']
        },
        'dropout_rate': {
            'distribution': 'uniform',
            'min': 0.0,
            'max': 0.5
        },
        # Fixed parameters
        'dataset': {'value': 'cifar10'},
        'output_size': {'value': 10},
        'val_split': {'value': 0.2},
        'random_seed': {'value': 42},
        'use_wandb': {'value': True}
        # Note: entity will be set automatically based on your WandB login
    }
}

if __name__ == '__main__':
    # Check WandB login
    try:
        api_key = wandb.api.api_key
        if api_key is None:
            print("ERROR: WandB not logged in. Run 'wandb login' first.")
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: WandB not logged in. Run 'wandb login' first.")
        print(f"Error details: {e}")
        sys.exit(1)
    
    print("Creating WandB sweep...")
    print(f"Project: {sweep_config['project']}")
    if 'entity' in sweep_config:
        print(f"Entity: {sweep_config['entity']}")
    else:
        print("Entity: (using default/personal account)")
    print(f"Method: {sweep_config['method']}")
    print("")
    
    # Create sweep
    sweep_id = wandb.sweep(sweep_config)
    
    print("=" * 60)
    print(f"✅ Sweep created successfully!")
    print(f"Sweep ID: {sweep_id}")
    entity_part = f"{sweep_config.get('entity', 'your-username')}/" if 'entity' in sweep_config else ""
    print(f"Sweep URL: https://wandb.ai/{entity_part}{sweep_config['project']}/sweeps/{sweep_id}")
    print("=" * 60)
    print("")
    print("To run sweep agents on HPC:")
    print(f"  sbatch scripts/submit_sweep_agent.sh {sweep_id} 10")
    print("")
    print("Or submit multiple agents in parallel:")
    print(f"  for i in {{1..5}}; do sbatch scripts/submit_sweep_agent.sh {sweep_id} 10; done")
    print("")

