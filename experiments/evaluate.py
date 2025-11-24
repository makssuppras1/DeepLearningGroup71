"""
Model evaluation script.

Load a trained model and evaluate it on test set with detailed metrics.
"""

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.neural_network import NeuralNetwork
from src.data_loader import load_fashion_mnist, load_cifar10, preprocess_data, get_class_names
from src.utils import (
    accuracy_score,
    plot_confusion_matrix,
    print_classification_report,
    visualize_predictions,
    load_model
)


def evaluate_model(model, X_test, y_test, dataset_name, save_dir='results/plots'):
    """
    Comprehensive model evaluation.
    
    Args:
        model: Trained neural network model
        X_test: Test data
        y_test: Test labels
        dataset_name: Name of dataset for class names
        save_dir: Directory to save plots
    """
    # Get class names
    class_names = get_class_names(dataset_name)
    
    # Make predictions
    print("Making predictions on test set...")
    y_pred_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    
    # Convert y_test to class indices if one-hot encoded
    if y_test.ndim > 1 and y_test.shape[1] > 1:
        y_test_indices = np.argmax(y_test, axis=1)
    else:
        y_test_indices = y_test
    
    # Compute accuracy
    accuracy = accuracy_score(y_pred, y_test_indices)
    print(f"\n{'='*60}")
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"{'='*60}\n")
    
    # Compute loss
    loss = model.compute_loss(y_pred_proba, y_test)
    print(f"Test Loss: {loss:.4f}\n")
    
    # Print classification report
    print_classification_report(y_test_indices, y_pred, class_names)
    
    # Generate confusion matrix
    os.makedirs(save_dir, exist_ok=True)
    cm_path = os.path.join(save_dir, 'confusion_matrix.png')
    print(f"Generating confusion matrix...")
    plot_confusion_matrix(y_test_indices, y_pred, class_names, save_path=cm_path)
    
    return accuracy, loss


def main():
    """
    Main evaluation function.
    """
    import argparse
    import pickle
    
    parser = argparse.ArgumentParser(description='Evaluate trained neural network model')
    parser.add_argument('--model-path', type=str, default='results/models/baseline_best.pkl',
                        help='Path to saved model file')
    parser.add_argument('--dataset', type=str, default='fashion_mnist',
                        choices=['fashion_mnist', 'cifar10'],
                        help='Dataset name')
    parser.add_argument('--config-path', type=str, default=None,
                        help='Path to config file (optional, for model architecture)')
    
    args = parser.parse_args()
    
    # Configuration
    model_path = args.model_path
    dataset = args.dataset
    results_dir = 'results/plots'
    
    print("="*60)
    print("Model Evaluation")
    print("="*60)
    
    # Load model parameters
    print(f"\nLoading model from {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        model_params = pickle.load(f)
    
    # Determine input size based on dataset
    if dataset == 'fashion_mnist':
        input_size = 784  # 28*28
    elif dataset == 'cifar10':
        input_size = 3072  # 32*32*3
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # Infer architecture from saved parameters
    num_layers = len([k for k in model_params.keys() if k.startswith('W')])
    hidden_layers = []
    for i in range(1, num_layers):
        hidden_size = model_params[f'W{i}'].shape[1]
        hidden_layers.append(hidden_size)
    output_size = model_params[f'W{num_layers}'].shape[1]
    
    print(f"Model architecture: {input_size} -> {' -> '.join(map(str, hidden_layers))} -> {output_size}")
    
    # Create model with same architecture (we'll load weights after)
    # Note: We need to know the original config. For now, use defaults.
    model = NeuralNetwork(
        input_size=input_size,
        hidden_layers=hidden_layers,
        output_size=output_size,
        activation='relu',  # Default, should match training
        output_activation='softmax',
        learning_rate=0.001,  # Not used for evaluation
        optimizer='adam',  # Not used for evaluation
        weight_init='he',  # Not used for evaluation
        l2_lambda=0.0001,  # Not used for evaluation
        random_seed=42
    )
    
    # Load saved parameters
    model.set_params(model_params)
    print("✅ Model loaded successfully!")
    
    # Load test dataset
    print(f"\nLoading {dataset} test data...")
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    
    if dataset == 'fashion_mnist':
        _, _, X_test, y_test = load_fashion_mnist(data_dir)
    elif dataset == 'cifar10':
        _, _, X_test, y_test = load_cifar10(data_dir)
    
    # Preprocess data
    X_test, y_test = preprocess_data(
        X_test, y_test,
        num_classes=output_size,
        flatten=True,
        normalize=True
    )
    
    print(f"✅ Test set loaded: {X_test.shape[0]} samples")
    
    # Evaluate model
    print("\n" + "="*60)
    print("Evaluating model...")
    print("="*60)
    accuracy, loss = evaluate_model(model, X_test, y_test, dataset, results_dir)
    
    print("\n" + "="*60)
    print("Evaluation completed!")
    print(f"Results saved to: {results_dir}")
    print("="*60)


if __name__ == '__main__':
    main()

