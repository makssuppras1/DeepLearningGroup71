# Main training script for neural network experiments with WandB logging

import numpy as np
import sys
import os
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.neural_network import NeuralNetwork
from src.data_loader import load_fashion_mnist, load_cifar10, preprocess_data, create_mini_batches, train_val_split
from src.utils import accuracy_score, set_random_seed
from src.hpc_utils import get_data_dir, get_results_dir, setup_hpc_directories
import wandb
from tqdm import tqdm


def load_data(dataset_name, data_dir):
    """Load and return dataset."""
    if dataset_name == 'fashion_mnist':
        data = load_fashion_mnist(data_dir)
        input_size = 784
    elif dataset_name == 'cifar10':
        data = load_cifar10(data_dir)
        input_size = 3072
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    X_train_full, y_train_full, X_test, y_test = data
    return X_train_full, y_train_full, X_test, y_test, input_size


def train_epoch(model, X_train, y_train, batch_size):
    """Train for one epoch."""
    batches = create_mini_batches(X_train, y_train, batch_size=batch_size, shuffle=True)
    losses = []
    predictions = []
    labels = []
    
    for X_batch, y_batch in batches:
        loss = model.train_step(X_batch, y_batch)
        losses.append(loss)
        
        preds = model.predict(X_batch)
        predictions.append(preds)
        
        if y_batch.ndim > 1:
            y_batch_indices = np.argmax(y_batch, axis=1)
        else:
            y_batch_indices = y_batch
        labels.append(y_batch_indices)
    
    avg_loss = np.mean(losses)
    all_preds = np.concatenate(predictions)
    all_labels = np.concatenate(labels)
    accuracy = accuracy_score(all_preds, all_labels)
    
    return avg_loss, accuracy


def evaluate(model, X_val, y_val):
    """Evaluate model on validation/test set."""
    y_pred_proba = model.predict_proba(X_val)
    y_pred = model.predict(X_val)
    
    loss = model.compute_loss(y_pred_proba, y_val)
    
    if y_val.ndim > 1 and y_val.shape[1] > 1:
        y_val_indices = np.argmax(y_val, axis=1)
    else:
        y_val_indices = y_val
    
    accuracy = accuracy_score(y_pred, y_val_indices)
    return loss, accuracy


def train(config=None):
    """
    Main training function. Compatible with WandB sweeps.
    
    If config is None, uses wandb.config (for sweeps).
    Otherwise uses provided config dict (for standalone runs).
    """
    # Initialize WandB - gets config from sweep if running as agent
    use_wandb = False
    if config is None:
        # Running as sweep agent - wandb.init() gets config from sweep
        wandb.init()
        config = dict(wandb.config)
        use_wandb = True  # Sweep mode always uses wandb
    else:
        # Standalone run - initialize WandB with provided config
        use_wandb = config.get('use_wandb', True)
        if use_wandb:
            try:
                wandb.init(
                    project=config.get('project_name', 'neural-network-numpy'),
                    name=config.get('experiment_name', 'baseline'),
                    entity=config.get('entity', None),
                    config=config
                )
            except Exception as e:
                print(f"Warning: WandB init failed ({e}). Continuing without WandB.")
                use_wandb = False
    
    # Set random seed
    set_random_seed(config.get('random_seed', 42))
    
    # Setup HPC directories if available
    setup_hpc_directories()
    
    # Load data
    data_dir = get_data_dir(os.path.dirname(os.path.dirname(__file__)))
    X_train_full, y_train_full, X_test, y_test, input_size = load_data(
        config['dataset'], data_dir
    )
    
    # Preprocess
    X_train_full, y_train_full = preprocess_data(
        X_train_full, y_train_full,
        num_classes=config.get('output_size', 10),
        flatten=True,
        normalize=True
    )
    X_test, y_test = preprocess_data(
        X_test, y_test,
        num_classes=config.get('output_size', 10),
        flatten=True,
        normalize=True
    )
    
    # Split train/val
    X_train, X_val, y_train, y_val = train_val_split(
        X_train_full, y_train_full,
        val_split=config.get('val_split', 0.2),
        random_seed=config.get('random_seed', 42)
    )
    
    # Create model
    model = NeuralNetwork(
        input_size=input_size,
        hidden_layers=config['hidden_layers'],
        output_size=config.get('output_size', 10),
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
    best_val_acc = 0.0
    best_model_params = None
    
    num_epochs = config.get('num_epochs', 50)
    batch_size = config.get('batch_size', 64)
    
    for epoch in tqdm(range(num_epochs), desc="Training"):
        # Train
        train_loss, train_acc = train_epoch(model, X_train, y_train, batch_size)
        
        # Validate
        val_loss, val_acc = evaluate(model, X_val, y_val)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_params = model.get_params()
        
        # Log metrics
        if use_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc
            })
    
    # Load best model and evaluate on test set
    if best_model_params is not None:
        model.set_params(best_model_params)
    
    test_loss, test_acc = evaluate(model, X_test, y_test)
    if use_wandb:
        wandb.log({'test_loss': test_loss, 'test_acc': test_acc})
    
    # Save model
    results_dir = get_results_dir(os.path.dirname(os.path.dirname(__file__)))
    models_dir = os.path.join(results_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    experiment_name = config.get('experiment_name', 'baseline')
    model_path = os.path.join(models_dir, f"{experiment_name}_best.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(best_model_params, f)
    
    if use_wandb:
        wandb.finish()


def main():
    """Main function for standalone runs."""
    config = {
        'dataset': 'cifar10',  # Focus on CIFAR-10
        'hidden_layers': [1024, 512, 256],  # Larger network for CIFAR-10 (3072 input features)
        'output_size': 10,
        'activation': 'relu',
        'output_activation': 'softmax',
        'num_epochs': 150,  # More epochs for CIFAR-10
        'batch_size': 64,
        'learning_rate': 0.0003,  # Slightly higher LR for CIFAR-10
        'optimizer': 'adam',
        'l2_lambda': 0.0001,
        'weight_init': 'he',
        'val_split': 0.2,
        'random_seed': 42,
        'project_name': 'neural-network-numpy',
        'experiment_name': 'cifar10_baseline',
        'use_wandb': True,
        'entity': 'makssuppras1-danmarks-tekniske-universitet-dtu'
    }
    
    # Parse command line args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=config['dataset'])
    parser.add_argument('--epochs', type=int, default=config['num_epochs'])
    parser.add_argument('--batch-size', type=int, default=config['batch_size'])
    parser.add_argument('--lr', type=float, default=config['learning_rate'])
    parser.add_argument('--optimizer', type=str, default=config['optimizer'])
    parser.add_argument('--name', type=str, default=config['experiment_name'])
    parser.add_argument('--hidden-layers', type=str, default=None)
    parser.add_argument('--no-wandb', action='store_true')
    
    args = parser.parse_args()
    
    config['dataset'] = args.dataset
    config['num_epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.lr
    config['optimizer'] = args.optimizer
    config['experiment_name'] = args.name
    
    if args.hidden_layers:
        config['hidden_layers'] = [int(x.strip()) for x in args.hidden_layers.split(',')]
    
    if args.no_wandb:
        config['use_wandb'] = False
    
    train(config)


if __name__ == '__main__':
    main()
