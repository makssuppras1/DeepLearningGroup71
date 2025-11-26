# Main training script for neural network experiments with WandB logging

import numpy as np
import sys
import os
import pickle
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.neural_network import NeuralNetwork
from src.data_loader import load_fashion_mnist, load_cifar10, preprocess_data, create_mini_batches, train_val_split
from src.utils import accuracy_score, set_random_seed
from src.hpc_utils import get_data_dir, get_results_dir, setup_hpc_directories
import wandb
from tqdm import tqdm

def get_project_root():
    # Get the project root directory (parent of experiments/)
    return os.path.dirname(os.path.dirname(__file__))

def labels_to_indices(y):
    # Convert one-hot encoded labels to class indices, or return as-is if already indices
    if y.ndim > 1 and y.shape[1] > 1:
        return np.argmax(y, axis=1)
    return y

def load_data(dataset_name, data_dir):
    # Load dataset and return data with input size
    dataset_configs = {
        'fashion_mnist': (load_fashion_mnist, 784),
        'cifar10': (load_cifar10, 3072)
    }
    
    if dataset_name not in dataset_configs:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(dataset_configs.keys())}")
    
    load_func, input_size = dataset_configs[dataset_name]
    X_train_full, y_train_full, X_test, y_test = load_func(data_dir)
    
    return X_train_full, y_train_full, X_test, y_test, input_size

def train_epoch(model, X_train, y_train, batch_size):
    # Train for one epoch and return average loss and accuracy
    batches = create_mini_batches(X_train, y_train, batch_size=batch_size, shuffle=True)
    losses = []
    predictions = []
    labels = []
    
    for X_batch, y_batch in batches:
        # Train on batch
        loss = model.train_step(X_batch, y_batch)
        losses.append(loss)
        
        # Get predictions for accuracy calculation
        preds = model.predict(X_batch)
        predictions.append(preds)
        labels.append(labels_to_indices(y_batch))
    
    avg_loss = np.mean(losses)
    all_preds = np.concatenate(predictions)
    all_labels = np.concatenate(labels)
    accuracy = accuracy_score(all_preds, all_labels)
    
    return avg_loss, accuracy


def evaluate(model, X_val, y_val):
    # Evaluate model on validation/test set and return loss and accuracy
    y_pred_proba = model.predict_proba(X_val)
    y_pred = model.predict(X_val)
    loss = model.compute_loss(y_pred_proba, y_val)
    accuracy = accuracy_score(y_pred, labels_to_indices(y_val))
    
    return loss, accuracy


def init_wandb(config):
    # Initialize WandB if enabled. Returns run object if WandB is active, None otherwise
    use_wandb = config.get('use_wandb', True)
    
    if not use_wandb:
        return False
    
    try:
        wandb.init(
            project=config.get('project_name', 'neural-network-numpy'),
            name=config.get('experiment_name', 'baseline'),
            entity=config.get('entity', None),
            config=config
        )
        return True
    except Exception as e:
        print(f"Warning: WandB init failed ({e}). Continuing without WandB.")
        return False


def get_default_config():
    # Return default configuration dictionary
    return {
        'dataset': 'cifar10',
        'hidden_layers': [1024, 512, 256],
        'output_size': 10,
        'activation': 'relu',
        'output_activation': 'softmax',
        'num_epochs': 150,
        'batch_size': 64,
        'learning_rate': 0.0003,
        'optimizer': 'adam',
        'l2_lambda': 0.0001,
        'weight_init': 'he',
        'dropout_rate': 0.0,
        'val_split': 0.2,
        'random_seed': 42,
        'project_name': 'neural-network-numpy',
        'experiment_name': 'cifar10_baseline',
        'use_wandb': True,
        'entity': 'makssuppras1-danmarks-tekniske-universitet-dtu'
    }


def train(config=None):
    # Main training function. Compatible with WandB sweeps.
    # If config is None, uses wandb.config (for sweeps). Otherwise uses provided config dict.
    
    # Handle WandB sweep mode (config comes from wandb.init())
    if config is None:
        wandb.init()
        config = dict(wandb.config)
        use_wandb = True
    else:
        use_wandb = init_wandb(config)
    
    # Setup
    set_random_seed(config.get('random_seed', 42))
    setup_hpc_directories()
    
    # Load and preprocess data
    project_root = get_project_root()
    data_dir = get_data_dir(project_root)
    X_train_full, y_train_full, X_test, y_test, input_size = load_data(config['dataset'], data_dir)
    
    num_classes = config.get('output_size', 10)
    X_train_full, y_train_full = preprocess_data(X_train_full, y_train_full, num_classes=num_classes, flatten=True, normalize=True)
    X_test, y_test = preprocess_data(X_test, y_test, num_classes=num_classes, flatten=True, normalize=True)
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_val_split(
        X_train_full, y_train_full,
        val_split=config.get('val_split', 0.2),
        random_seed=config.get('random_seed', 42)
    )
    
    # Create model
    model = NeuralNetwork(
        input_size=input_size,
        hidden_layers=config['hidden_layers'],
        output_size=num_classes,
        activation=config.get('activation', 'relu'),
        output_activation=config.get('output_activation', 'softmax'),
        learning_rate=config['learning_rate'],
        optimizer=config.get('optimizer', 'adam'),
        weight_init=config.get('weight_init', 'he'),
        l2_lambda=config.get('l2_lambda', 0.0),
        dropout_rate=config.get('dropout_rate', 0.0),
        random_seed=config.get('random_seed', 42)
    )
    
    # Training loop
    num_epochs = config.get('num_epochs', 50)
    batch_size = config.get('batch_size', 64)
    best_val_acc = 0.0
    best_model_params = None
    
    for epoch in tqdm(range(num_epochs), desc="Training"):
        train_loss, train_acc = train_epoch(model, X_train, y_train, batch_size)
        val_loss, val_acc = evaluate(model, X_val, y_val)
        
        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_params = model.get_params()
        
        # Log metrics to WandB
        if use_wandb:
            try:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_acc': train_acc,
                    'val_acc': val_acc
                })
            except Exception as e:
                print(f"Warning: Failed to log metrics: {e}")
    
    # Evaluate best model on test set
    model.set_params(best_model_params)
    test_loss, test_acc = evaluate(model, X_test, y_test)
    
    if use_wandb:
        wandb.log({'test_loss': test_loss, 'test_acc': test_acc})
    
    # Save model
    results_dir = get_results_dir(project_root)
    models_dir = os.path.join(results_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    experiment_name = config.get('experiment_name', 'baseline')
    model_path = os.path.join(models_dir, f"{experiment_name}_best.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(best_model_params, f)
    
    if use_wandb:
        wandb.finish()


def parse_args():
    # Parse command line arguments and return parsed args
    parser = argparse.ArgumentParser(description='Train neural network')
    parser.add_argument('--dataset', type=str, help='Dataset name (cifar10 or fashion_mnist)')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--optimizer', type=str, help='Optimizer name')
    parser.add_argument('--name', type=str, help='Experiment name')
    parser.add_argument('--hidden-layers', type=str, help='Comma-separated hidden layer sizes (e.g., "1024,512,256")')
    parser.add_argument('--no-wandb', action='store_true', help='Disable WandB logging')
    return parser.parse_args()


def update_config_from_args(config, args):
    # Update config dictionary with command line arguments (only if provided)
    if args.dataset:
        config['dataset'] = args.dataset
    if args.epochs:
        config['num_epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.lr:
        config['learning_rate'] = args.lr
    if args.optimizer:
        config['optimizer'] = args.optimizer
    if args.name:
        config['experiment_name'] = args.name
    if args.hidden_layers:
        config['hidden_layers'] = [int(x.strip()) for x in args.hidden_layers.split(',')]
    if args.no_wandb:
        config['use_wandb'] = False


def main():
    # Main function for standalone runs
    config = get_default_config()
    args = parse_args()
    update_config_from_args(config, args)
    train(config)


if __name__ == '__main__':
    # Check if running as WandB sweep agent (program mode)
    import os
    import sys
    
    # When WandB runs a program for sweeps, it typically:
    # 1. Sets WANDB_SWEEP_ID environment variable, OR
    # 2. Calls the program and expects wandb.init() to get config from sweep
    # 
    # If no command line arguments (or just --wandb flag), likely a sweep run
    # If wandb.run is already initialized, we're in a sweep
    is_sweep_run = (
        os.environ.get('WANDB_SWEEP_ID') or 
        os.environ.get('WANDB_PROJECT') or
        (len(sys.argv) == 1) or  # No args = likely sweep
        wandb.run is not None     # Already initialized
    )
    
    if is_sweep_run:
        # Running as sweep agent - call train() without config
        # train() will call wandb.init() which gets config from sweep
        train()
    else:
        # Normal standalone run - use main()
        main()
