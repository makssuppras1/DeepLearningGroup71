# Main training script for neural network experiments with WandB logging

import numpy as np
import sys
import os
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.neural_network import NeuralNetwork
from src.data_loader import load_fashion_mnist, load_cifar10, download_fashion_mnist, download_cifar10, preprocess_data, create_mini_batches, train_val_split
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
    # Downloads data if it doesn't exist
    dataset_configs = {
        'fashion_mnist': (load_fashion_mnist, download_fashion_mnist, 784),
        'cifar10': (load_cifar10, download_cifar10, 3072)
    }
    
    if dataset_name not in dataset_configs:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(dataset_configs.keys())}")
    
    load_func, download_func, input_size = dataset_configs[dataset_name]
    
    # Download data if it doesn't exist
    print(f"Checking for {dataset_name} data in {data_dir}...")
    download_func(data_dir)
    
    # Load the dataset
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


def train(config):
    # Main training function. Compatible with WandB sweeps.
    # Config should be a dict. WandB run should be initialized before calling this.
    
    # Get the wandb run object if it exists
    try:
        run = wandb.run if wandb.run is not None else None
    except Exception:
        run = None
    
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
        if run is not None:
            run.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc
            })
    
    # Evaluate best model on test set
    model.set_params(best_model_params)
    test_loss, test_acc = evaluate(model, X_test, y_test)
    
    if run is not None:
        run.log({'test_loss': test_loss, 'test_acc': test_acc})
    
    # Save model
    results_dir = get_results_dir(project_root)
    models_dir = os.path.join(results_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    experiment_name = config.get('experiment_name', 'baseline')
    model_path = os.path.join(models_dir, f"{experiment_name}_best.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(best_model_params, f)
    
    if run is not None:
        try:
            run.finish()
            print(f"✅ WandB run completed. View at: {run.url}")
        except Exception:
            pass


def main():
    # Main function - works for both standalone runs and WandB sweeps
    # Initialize wandb run (wandb.agent will provide wandb.config for sweep values)
    default_config = get_default_config()
    
    # Check if we're running as a sweep agent by checking environment
    import os
    is_sweep_agent = os.environ.get('WANDB_SWEEP_ID') is not None
    
    try:
        if is_sweep_agent:
            # Running as sweep agent - don't pass config, let wandb get it from sweep
            # wandb.init() will automatically get config from the sweep
            run = wandb.init(
                project=default_config.get('project_name', 'neural-network-numpy'),
                resume='allow'
            )
        else:
            # Standalone run - use default config
            run = wandb.init(
                project=default_config.get('project_name', 'neural-network-numpy'),
                config=default_config,
                resume='allow'
            )
    except Exception as e:
        # If wandb fails to initialize, still allow local runs
        print(f"Warning: WandB init failed ({e}). Continuing without WandB.")
        import traceback
        traceback.print_exc()
        run = None
    
    # Get overrides from wandb.config (for sweeps) or use defaults
    try:
        if run is not None:
            cfg_override = dict(wandb.config)
            print(f"Got config from WandB: {len(cfg_override)} parameters")
            if len(cfg_override) > 0:
                print(f"Sample config keys: {list(cfg_override.keys())[:5]}")
        else:
            cfg_override = {}
    except Exception as e:
        print(f"Warning: Could not get wandb.config: {e}")
        import traceback
        traceback.print_exc()
        cfg_override = {}
    
    # Start with defaults, then override from wandb.config (sweep)
    config = get_default_config()
    for k, v in cfg_override.items():
        config[k] = v
    
    print(f"Final config has {len(config)} parameters")
    print(f"Dataset: {config.get('dataset')}, Hidden layers: {config.get('hidden_layers')}")
    
    # Validate required config keys
    required_keys = ['dataset', 'hidden_layers', 'output_size', 'num_epochs', 'batch_size', 'learning_rate']
    missing_keys = [k for k in required_keys if k not in config]
    if missing_keys:
        raise ValueError(f"Missing required config keys: {missing_keys}")
    
    train(config)


if __name__ == '__main__':
    # Always call main() - it handles both sweep mode and standalone runs
    # When run by wandb agent, wandb.init() in main() will get config from sweep
    # When run standalone, wandb.init() uses default config
    main()
