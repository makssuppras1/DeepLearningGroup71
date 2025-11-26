"""
Script entrypoint for training. Designed to be runnable by Weights & Biases sweep agent
when `sweep_config['program'] = 'train3.py'` is used from a notebook.

This script mirrors the cleaned training logic in the notebook and safely resolves
the project root using `__file__` (falls back to cwd only when necessary).
"""
from pathlib import Path
import sys
import os
import numpy as np
import wandb
from tqdm import tqdm

try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    # Fallback for interactive environments where __file__ is not defined
    BASE_DIR = Path.cwd()

sys.path.append(str(BASE_DIR))

from src.neural_network import NeuralNetwork
from src.data_loader import load_fashion_mnist, load_cifar10, preprocess_data, create_mini_batches, train_val_split
from src.utils import accuracy_score, set_random_seed


def train_epoch(model, X_train, y_train, batch_size, show_batch_progress=False):
    batches = create_mini_batches(X_train, y_train, batch_size=batch_size, shuffle=True)
    epoch_losses, epoch_predictions, epoch_labels = [], [], []
    batch_iter = tqdm(batches, desc="  Batches", leave=False, disable=not show_batch_progress) if show_batch_progress else batches
    for X_batch, y_batch in batch_iter:
        loss = model.train_step(X_batch, y_batch)
        epoch_losses.append(loss)
        preds = model.predict(X_batch)
        epoch_predictions.append(preds)
        if y_batch.ndim > 1:
            labels = np.argmax(y_batch, axis=1)
        else:
            labels = y_batch
        epoch_labels.append(labels)
        if show_batch_progress:
            batch_iter.set_postfix({'loss': f'{loss:.4f}'})
    avg_loss = np.mean(epoch_losses)
    all_preds = np.concatenate(epoch_predictions)
    all_labels = np.concatenate(epoch_labels)
    acc = accuracy_score(all_preds, all_labels)
    return avg_loss, acc


def evaluate(model, X_val, y_val):
    y_pred_proba = model.predict_proba(X_val)
    y_pred = model.predict(X_val)
    loss = model.compute_loss(y_pred_proba, y_val)
    if y_val.ndim > 1 and y_val.shape[1] > 1:
        y_val_idx = np.argmax(y_val, axis=1)
    else:
        y_val_idx = y_val
    acc = accuracy_score(y_pred, y_val_idx)
    return loss, acc


DEFAULT_CONFIG = {
    'dataset': 'cifar10',
    'input_size': 3072,
    'hidden_layers': [512, 256],
    'output_size': 10,
    'activation': 'relu',
    'output_activation': 'softmax',
    'loss': 'cross_entropy',
    'num_epochs': 200,
    'batch_size': 32,
    'learning_rate': 0.00005557087831314196,
    'optimizer': 'adam',
    'l2_lambda': 0.003142735379161425,
    'weight_init': 'he',
    'val_split': 0.2,
    'random_seed': 42,
    'project_name': 'neural-network-numpy',
    'experiment_name': 'baseline',
    'entity': None,
    'show_batch_progress': True,
}


def train(config):
    if config is None:
        raise ValueError('train() requires a config dict')

    try:
        run = wandb.run if wandb.run is not None else None
    except Exception:
        run = None

    set_random_seed(config['random_seed'])

    data_dir = os.path.join(str(BASE_DIR), 'data')
    if config['dataset'] == 'fashion_mnist':
        X_train_full, y_train_full, X_test, y_test = load_fashion_mnist(data_dir)
        input_size = 784
    elif config['dataset'] == 'cifar10':
        X_train_full, y_train_full, X_test, y_test = load_cifar10(data_dir)
        input_size = 3072
    else:
        raise ValueError(f"Unknown dataset: {config['dataset']}")

    X_train_full, y_train_full = preprocess_data(X_train_full, y_train_full, num_classes=config['output_size'], flatten=True, normalize=True)
    X_test, y_test = preprocess_data(X_test, y_test, num_classes=config['output_size'], flatten=True, normalize=True)

    X_train, X_val, y_train, y_val = train_val_split(X_train_full, y_train_full, val_split=config['val_split'], random_seed=config['random_seed'])

    print(f"Dataset: {config['dataset']}")
    print(f"Train samples: {X_train.shape[0]}, Val samples: {X_val.shape[0]}, Test samples: {X_test.shape[0]}")

    model = NeuralNetwork(input_size=input_size, hidden_layers=config['hidden_layers'], output_size=config['output_size'], activation=config['activation'], output_activation=config['output_activation'], learning_rate=config['learning_rate'], optimizer=config['optimizer'], weight_init=config['weight_init'], l2_lambda=config['l2_lambda'], random_seed=config['random_seed'])

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val_acc = 0.0
    best_model_params = None

    epoch_pbar = tqdm(range(config['num_epochs']), desc="Training", unit="epoch")
    #patience = config.get('early_stopping_patience', 5)
    #early_stop_counter = 0

    for epoch in epoch_pbar:
        train_loss, train_acc = train_epoch(model, X_train, y_train, config['batch_size'], show_batch_progress=config.get('show_batch_progress', False))
        val_loss, val_acc = evaluate(model, X_val, y_val)
        train_losses.append(train_loss); val_losses.append(val_loss); train_accs.append(train_acc); val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_params = model.get_params()
    #        early_stop_counter = 0
    #    else:
    #        early_stop_counter += 1

    #    if early_stop_counter >= patience:
    #        print(f"\nEarly stopping triggered at epoch {epoch}")
    #        if best_model_params is not None:
    #            model.set_params(best_model_params)
    #        break

        if run is not None:
            run.log({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'train_acc': train_acc, 'val_acc': val_acc})

        epoch_pbar.set_postfix({'train_loss': f'{train_loss:.4f}', 'train_acc': f'{train_acc:.4f}', 'val_loss': f'{val_loss:.4f}', 'val_acc': f'{val_acc:.4f}', 'best_val': f'{best_val_acc:.4f}'})

    if best_model_params is not None:
        model.set_params(best_model_params)
        print(f"\nBest validation accuracy: {best_val_acc:.4f}")

    test_loss, test_acc = evaluate(model, X_test, y_test)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    if run is not None:
        run.log({'test_loss': test_loss, 'test_acc': test_acc})

    os.makedirs('results/models', exist_ok=True)
    model_path = f"results/models/{config['experiment_name']}_best.pkl"
    import pickle
    with open(model_path, 'wb') as f:
        pickle.dump(best_model_params, f)
    print(f"Model saved to {model_path}")

    if run is not None:
        try:
            run.finish()
            print(f"✅ WandB run completed. View at: {run.url}")
        except Exception:
            pass


def main():
    # Initialize wandb run (wandb.agent will provide wandb.config for sweep values)
    # Start with the defaults, then override from wandb.config
    try:
        run = wandb.init(project=DEFAULT_CONFIG.get('project_name', 'neural-network-numpy'), config=DEFAULT_CONFIG, resume='allow')
    except Exception:
        # If wandb fails to initialize, still allow local runs
        run = None

    # Get overrides from wandb.config
    try:
        cfg_override = dict(wandb.config) if run is not None else {}
    except Exception:
        cfg_override = {}

    config = DEFAULT_CONFIG.copy()
    for k, v in cfg_override.items():
        config[k] = v

    # Normalize common cases where sweep defines single int for hidden_layers
    h = config.get('hidden_layers')
    if isinstance(h, int):
        config['hidden_layers'] = [h]
    elif isinstance(h, str):
        # allow comma-separated string
        try:
            config['hidden_layers'] = [int(x.strip()) for x in h.split(',') if x.strip()]
        except Exception:
            pass

    train(config)


if __name__ == '__main__':
    main()
