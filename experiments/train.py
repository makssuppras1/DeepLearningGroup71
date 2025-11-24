# Main training script for neural network experiments with WandB logging

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.neural_network import NeuralNetwork
from src.data_loader import load_fashion_mnist, load_cifar10, preprocess_data, create_mini_batches, train_val_split
from src.utils import accuracy_score, plot_training_curves, set_random_seed
import wandb
from tqdm import tqdm


def train_epoch(model, X_train, y_train, batch_size, show_batch_progress=False):
    # Train for one epoch: create mini-batches, train on each, compute average loss and accuracy
    batches = create_mini_batches(X_train, y_train, batch_size=batch_size, shuffle=True)
    
    epoch_losses = []
    epoch_predictions = []
    epoch_labels = []
    
    # Create progress bar for batches if requested
    batch_iter = tqdm(batches, desc="  Batches", leave=False, disable=not show_batch_progress) if show_batch_progress else batches
    
    for X_batch, y_batch in batch_iter:
        # Train on batch
        loss = model.train_step(X_batch, y_batch)
        epoch_losses.append(loss)
        
        # Get predictions for accuracy
        predictions = model.predict(X_batch)
        epoch_predictions.append(predictions)
        
        # Convert y_batch to class indices for accuracy calculation
        if y_batch.ndim > 1:
            y_batch_indices = np.argmax(y_batch, axis=1)
        else:
            y_batch_indices = y_batch
        epoch_labels.append(y_batch_indices)
        
        # Update batch progress bar if enabled
        if show_batch_progress:
            batch_iter.set_postfix({'loss': f'{loss:.4f}'})
    
    # Compute average loss and accuracy
    avg_loss = np.mean(epoch_losses)
    all_predictions = np.concatenate(epoch_predictions)
    all_labels = np.concatenate(epoch_labels)
    accuracy = accuracy_score(all_predictions, all_labels)
    
    return avg_loss, accuracy


def evaluate(model, X_val, y_val):
    # Evaluate model on validation set: make predictions, compute loss and accuracy
    # Get probability predictions and class predictions
    y_pred_proba = model.predict_proba(X_val)
    y_pred = model.predict(X_val)
    
    # Compute loss
    loss = model.compute_loss(y_pred_proba, y_val)
    
    # Compute accuracy
    # Convert y_val to class indices if one-hot encoded
    if y_val.ndim > 1 and y_val.shape[1] > 1:
        y_val_indices = np.argmax(y_val, axis=1)
    else:
        y_val_indices = y_val
    
    accuracy = accuracy_score(y_pred, y_val_indices)
    
    return loss, accuracy


def train(config):
    # Complete training pipeline: WandB init, load data, create model, train, log metrics
    
    # Initialize Weights & Biases for experiment tracking (optional)
    use_wandb = config.get('use_wandb', True)
    run = None
    if use_wandb:
        try:
            # Initialize WandB run
            run = wandb.init(
                project=config.get('project_name', 'neural-network-numpy'),
                name=config.get('experiment_name', 'baseline'),
                entity=config.get('entity', None),  # Optional: set if using team account
                config=config,
                resume='allow'  # Allow resuming if run already exists
            )
            print(f"✅ WandB initialized: {run.url}")
        except Exception as e:
            print(f"⚠️  Warning: WandB initialization failed ({e})")
            print("   Continuing without WandB logging. To fix: run 'wandb login'")
            use_wandb = False
            run = None
    
    # Set random seed
    set_random_seed(config['random_seed'])
    
    # Load dataset
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    if config['dataset'] == 'fashion_mnist':
        X_train_full, y_train_full, X_test, y_test = load_fashion_mnist(data_dir)
        input_size = 784  # 28*28
    elif config['dataset'] == 'cifar10':
        X_train_full, y_train_full, X_test, y_test = load_cifar10(data_dir)
        input_size = 3072  # 32*32*3
    else:
        raise ValueError(f"Unknown dataset: {config['dataset']}")
    
    # Preprocess data
    X_train_full, y_train_full = preprocess_data(
        X_train_full, y_train_full,
        num_classes=config['output_size'],
        flatten=True,
        normalize=True
    )
    X_test, y_test = preprocess_data(
        X_test, y_test,
        num_classes=config['output_size'],
        flatten=True,
        normalize=True
    )
    
    # Split into train/val
    X_train, X_val, y_train, y_val = train_val_split(
        X_train_full, y_train_full,
        val_split=config['val_split'],
        random_seed=config['random_seed']
    )
    
    print(f"Dataset: {config['dataset']}")
    print(f"Train samples: {X_train.shape[0]}, Val samples: {X_val.shape[0]}, Test samples: {X_test.shape[0]}")
    
    # Create model
    model = NeuralNetwork(
        input_size=input_size,
        hidden_layers=config['hidden_layers'],
        output_size=config['output_size'],
        activation=config['activation'],
        output_activation=config['output_activation'],
        learning_rate=config['learning_rate'],
        optimizer=config['optimizer'],
        weight_init=config['weight_init'],
        l2_lambda=config['l2_lambda'],
        random_seed=config['random_seed']
    )
    
    # Training loop
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0.0
    best_model_params = None
    
    print("\nStarting training...")
    print(f"Epochs: {config['num_epochs']}, Batch size: {config['batch_size']}, Learning rate: {config['learning_rate']}")
    print("-" * 80)
    
    # Create progress bar for epochs
    epoch_pbar = tqdm(range(config['num_epochs']), desc="Training", unit="epoch")
    
    for epoch in epoch_pbar:
        # Train for one epoch
        show_batch_pbar = config.get('show_batch_progress', False)
        train_loss, train_acc = train_epoch(model, X_train, y_train, config['batch_size'], show_batch_progress=show_batch_pbar)
        
        # Evaluate on validation set
        val_loss, val_acc = evaluate(model, X_val, y_val)
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_params = model.get_params()
        
        # Log to WandB
        if use_wandb and run is not None:
            run.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc
            })
        
        # Update progress bar with current metrics
        epoch_pbar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'train_acc': f'{train_acc:.4f}',
            'val_loss': f'{val_loss:.4f}',
            'val_acc': f'{val_acc:.4f}',
            'best_val': f'{best_val_acc:.4f}'
        })
    
    # Load best model
    if best_model_params is not None:
        model.set_params(best_model_params)
        print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    
    # Evaluate on test set
    test_loss, test_acc = evaluate(model, X_test, y_test)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    
    # Log test metrics
    if use_wandb and run is not None:
        run.log({
            'test_loss': test_loss,
            'test_acc': test_acc
        })
    
    # Save model
    os.makedirs('results/models', exist_ok=True)
    model_path = f"results/models/{config['experiment_name']}_best.pkl"
    import pickle
    with open(model_path, 'wb') as f:
        pickle.dump(best_model_params, f)
    print(f"Model saved to {model_path}")
    
    # Finish WandB run
    if use_wandb and run is not None:
        run.finish()
        print(f"✅ WandB run completed. View at: {run.url}")


def main():
    # Main function to run training
    
    # Default configuration - Optimized for highest accuracy
    config = {
        # Dataset
        'dataset': 'fashion_mnist',  # or 'cifar10'
        
        # Model architecture
        'input_size': 784,  # 28*28 for Fashion-MNIST (will be overridden based on dataset)
        'hidden_layers': [512, 384, 256, 128],  # Deeper/more units for better capacity
        'output_size': 10,
        
        # Activation and loss
        'activation': 'relu',  # Best for deep networks (non-saturating)
        'output_activation': 'softmax',  # Correct for multi-class classification
        'loss': 'cross_entropy',
        
        # Training hyperparameters
        'num_epochs': 50,
        'batch_size': 32,
        'learning_rate': 0.001,  # Good default for Adam
        
        # Optimization
        'optimizer': 'adam',  # Best optimizer (adaptive learning rates + momentum)
        
        # Regularization
        'l2_lambda': 0.0001,  # Prevents overfitting
        
        # Initialization
        'weight_init': 'he',  # Best for ReLU activations (He initialization)
        
        # Other
        'val_split': 0.2,
        'random_seed': 42,
        'project_name': 'neural-network-numpy',
        'experiment_name': 'baseline',
        'use_wandb': True,  # Set to False to disable WandB
        'entity': 'makssuppras1-danmarks-tekniske-universitet-dtu',  # Your WandB entity
        'show_batch_progress': False  # Set to True to show batch-level progress bars
    }
    
    # Parse command line arguments to override config if needed
    import argparse
    parser = argparse.ArgumentParser(description='Train neural network')
    parser.add_argument('--dataset', type=str, default=config['dataset'])
    parser.add_argument('--epochs', type=int, default=config['num_epochs'])
    parser.add_argument('--batch-size', type=int, default=config['batch_size'])
    parser.add_argument('--lr', type=float, default=config['learning_rate'])
    parser.add_argument('--optimizer', type=str, default=config['optimizer'])
    parser.add_argument('--name', type=str, default=config['experiment_name'])
    parser.add_argument('--no-wandb', action='store_true', help='Disable WandB logging')
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    config['dataset'] = args.dataset
    config['num_epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.lr
    config['optimizer'] = args.optimizer
    config['experiment_name'] = args.name
    if args.no_wandb:
        config['use_wandb'] = False
    
    # Run training
    train(config)
    
    print("Training completed!")


if __name__ == '__main__':
    main()

