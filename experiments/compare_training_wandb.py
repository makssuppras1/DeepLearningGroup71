"""
Compare NumPy and PyTorch neural networks by training both with identical hyperparameters
and logging to WandB to verify they converge to the same validation accuracy.
"""

import numpy as np
import torch
import sys
import os
import wandb
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.neural_network import NeuralNetwork
from src.pytorch_neural_network import PyTorchNeuralNetwork
from src.data_loader import load_fashion_mnist, load_cifar10, preprocess_data, create_mini_batches, train_val_split
from src.utils import accuracy_score, set_random_seed


def train_epoch(model, X_train, y_train, batch_size, is_pytorch=False):
    """Train for one epoch."""
    batches = create_mini_batches(X_train, y_train, batch_size=batch_size, shuffle=True)
    losses = []
    predictions = []
    labels = []
    
    for X_batch, y_batch in batches:
        if is_pytorch:
            X_batch_torch = torch.from_numpy(X_batch).float()
            y_batch_torch = torch.from_numpy(y_batch).float()
            loss = model.train_step(X_batch_torch, y_batch_torch)
            preds = model.predict(X_batch_torch)
        else:
            loss = model.train_step(X_batch, y_batch)
            preds = model.predict(X_batch)
        
        losses.append(loss)
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


def evaluate(model, X_val, y_val, is_pytorch=False):
    """Evaluate model on validation set."""
    if is_pytorch:
        X_val_torch = torch.from_numpy(X_val).float()
        y_pred_proba = model.predict_proba(X_val_torch).cpu().numpy()
        y_pred = model.predict(X_val_torch)
        loss = model.compute_loss(
            torch.from_numpy(y_pred_proba).float(),
            torch.from_numpy(y_val).float()
        ).item()
    else:
        y_pred_proba = model.predict_proba(X_val)
        y_pred = model.predict(X_val)
        loss = model.compute_loss(y_pred_proba, y_val)
    
    if y_val.ndim > 1 and y_val.shape[1] > 1:
        y_val_indices = np.argmax(y_val, axis=1)
    else:
        y_val_indices = y_val
    
    accuracy = accuracy_score(y_pred, y_val_indices)
    return loss, accuracy


def train_both_models(config):
    """Train both NumPy and PyTorch models with identical hyperparameters."""
    
    # Set random seed for reproducibility
    seed = config.get('random_seed', 42)
    set_random_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Load data
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    dataset_name = config.get('dataset', 'fashion_mnist')
    
    if dataset_name == 'fashion_mnist':
        X_train_full, y_train_full, X_test, y_test = load_fashion_mnist(data_dir)
        input_size = 784
    elif dataset_name == 'cifar10':
        X_train_full, y_train_full, X_test, y_test = load_cifar10(data_dir)
        input_size = 3072
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Preprocess
    num_classes = config.get('output_size', 10)
    X_train_full, y_train_full = preprocess_data(
        X_train_full, y_train_full,
        num_classes=num_classes, flatten=True, normalize=True
    )
    X_test, y_test = preprocess_data(
        X_test, y_test,
        num_classes=num_classes, flatten=True, normalize=True
    )
    
    # Split train/val
    X_train, X_val, y_train, y_val = train_val_split(
        X_train_full, y_train_full,
        val_split=config.get('val_split', 0.2),
        random_seed=seed
    )
    
    # Model configuration
    model_config = {
        'input_size': input_size,
        'hidden_layers': config.get('hidden_layers', [128, 64]),
        'output_size': num_classes,
        'activation': config.get('activation', 'relu'),
        'output_activation': config.get('output_activation', 'softmax'),
        'learning_rate': config.get('learning_rate', 0.001),
        'optimizer': config.get('optimizer', 'adam'),
        'weight_init': config.get('weight_init', 'he'),
        'l2_lambda': config.get('l2_lambda', 0.0),
        'dropout_rate': config.get('dropout_rate', 0.0),
        'random_seed': seed
    }
    
    # Create models
    print("Creating NumPy model...")
    numpy_model = NeuralNetwork(**model_config)
    
    print("Creating PyTorch model...")
    torch.manual_seed(seed)
    np.random.seed(seed)
    pytorch_model = PyTorchNeuralNetwork(**model_config)
    
    # Copy weights from NumPy to PyTorch to ensure identical initialization
    print("Copying weights to ensure identical initialization...")
    numpy_params = numpy_model.get_params()
    pytorch_model.set_params(numpy_params)
    
    # Training parameters
    num_epochs = config.get('num_epochs', 20)
    batch_size = config.get('batch_size', 64)
    
    # Track best validation accuracies
    best_numpy_val_acc = 0.0
    best_pytorch_val_acc = 0.0
    
    print(f"\nTraining both models for {num_epochs} epochs...")
    print(f"Dataset: {dataset_name}, Train size: {len(X_train)}, Val size: {len(X_val)}")
    
    for epoch in tqdm(range(num_epochs), desc="Training"):
        # Train NumPy model
        numpy_model.train()
        numpy_train_loss, numpy_train_acc = train_epoch(numpy_model, X_train, y_train, batch_size, is_pytorch=False)
        
        # Train PyTorch model
        pytorch_model.train()
        pytorch_train_loss, pytorch_train_acc = train_epoch(pytorch_model, X_train, y_train, batch_size, is_pytorch=True)
        
        # Evaluate NumPy model
        numpy_model.eval()
        numpy_val_loss, numpy_val_acc = evaluate(numpy_model, X_val, y_val, is_pytorch=False)
        
        # Evaluate PyTorch model
        pytorch_model.eval()
        pytorch_val_loss, pytorch_val_acc = evaluate(pytorch_model, X_val, y_val, is_pytorch=True)
        
        # Update best accuracies
        if numpy_val_acc > best_numpy_val_acc:
            best_numpy_val_acc = numpy_val_acc
        if pytorch_val_acc > best_pytorch_val_acc:
            best_pytorch_val_acc = pytorch_val_acc
        
        # Log to WandB
        wandb.log({
            'epoch': epoch,
            'numpy_train_loss': numpy_train_loss,
            'numpy_train_acc': numpy_train_acc,
            'numpy_val_loss': numpy_val_loss,
            'numpy_val_acc': numpy_val_acc,
            'pytorch_train_loss': pytorch_train_loss,
            'pytorch_train_acc': pytorch_train_acc,
            'pytorch_val_loss': pytorch_val_loss,
            'pytorch_val_acc': pytorch_val_acc,
            'val_acc_diff': abs(numpy_val_acc - pytorch_val_acc),
            'train_loss_diff': abs(numpy_train_loss - pytorch_train_loss),
        })
        
        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"\nEpoch {epoch + 1}/{num_epochs}:")
            print(f"  NumPy   - Train Loss: {numpy_train_loss:.4f}, Train Acc: {numpy_train_acc:.4f}, Val Loss: {numpy_val_loss:.4f}, Val Acc: {numpy_val_acc:.4f}")
            print(f"  PyTorch - Train Loss: {pytorch_train_loss:.4f}, Train Acc: {pytorch_train_acc:.4f}, Val Loss: {pytorch_val_loss:.4f}, Val Acc: {pytorch_val_acc:.4f}")
            print(f"  Difference - Val Acc: {abs(numpy_val_acc - pytorch_val_acc):.6f}, Train Loss: {abs(numpy_train_loss - pytorch_train_loss):.6f}")
    
    # Final evaluation on test set
    numpy_model.eval()
    numpy_test_loss, numpy_test_acc = evaluate(numpy_model, X_test, y_test, is_pytorch=False)
    
    pytorch_model.eval()
    pytorch_test_loss, pytorch_test_acc = evaluate(pytorch_model, X_test, y_test, is_pytorch=True)
    
    wandb.log({
        'numpy_test_loss': numpy_test_loss,
        'numpy_test_acc': numpy_test_acc,
        'pytorch_test_loss': pytorch_test_loss,
        'pytorch_test_acc': pytorch_test_acc,
        'test_acc_diff': abs(numpy_test_acc - pytorch_test_acc),
        'best_numpy_val_acc': best_numpy_val_acc,
        'best_pytorch_val_acc': best_pytorch_val_acc,
        'best_val_acc_diff': abs(best_numpy_val_acc - best_pytorch_val_acc),
    })
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Best Validation Accuracy:")
    print(f"  NumPy:   {best_numpy_val_acc:.6f}")
    print(f"  PyTorch: {best_pytorch_val_acc:.6f}")
    print(f"  Difference: {abs(best_numpy_val_acc - best_pytorch_val_acc):.6f}")
    print(f"\nTest Accuracy:")
    print(f"  NumPy:   {numpy_test_acc:.6f}")
    print(f"  PyTorch: {pytorch_test_acc:.6f}")
    print(f"  Difference: {abs(numpy_test_acc - pytorch_test_acc):.6f}")
    
    return {
        'best_numpy_val_acc': best_numpy_val_acc,
        'best_pytorch_val_acc': best_pytorch_val_acc,
        'numpy_test_acc': numpy_test_acc,
        'pytorch_test_acc': pytorch_test_acc,
    }


def main():
    """Main function."""
    # Simple hyperparameters for fast convergence
    config = {
        'dataset': 'fashion_mnist',  # Smaller dataset, faster training
        'hidden_layers': [128, 64],  # Small network
        'output_size': 10,
        'activation': 'relu',
        'output_activation': 'softmax',
        'num_epochs': 20,  # Few epochs for quick test
        'batch_size': 64,
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'weight_init': 'he',
        'l2_lambda': 0.0001,
        'dropout_rate': 0.0,
        'val_split': 0.2,
        'random_seed': 42,
        'project_name': 'numpy-vs-pytorch-comparison',
        'experiment_name': 'fashion_mnist_quick_test',
    }
    
    # Initialize WandB
    wandb.init(
        project=config.get('project_name', 'numpy-vs-pytorch-comparison'),
        name=config.get('experiment_name', 'comparison'),
        config=config,
        entity=config.get('entity', None)
    )
    
    try:
        results = train_both_models(config)
        
        # Log summary
        wandb.summary['best_val_acc_diff'] = abs(results['best_numpy_val_acc'] - results['best_pytorch_val_acc'])
        wandb.summary['test_acc_diff'] = abs(results['numpy_test_acc'] - results['pytorch_test_acc'])
        wandb.summary['models_match'] = abs(results['best_numpy_val_acc'] - results['best_pytorch_val_acc']) < 0.001
        
    finally:
        wandb.finish()


if __name__ == '__main__':
    main()


