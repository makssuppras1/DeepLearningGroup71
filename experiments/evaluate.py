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


def evaluate_model(model, X_test, y_test, dataset_name):
    """
    Comprehensive model evaluation.
    
    Args:
        model: Trained neural network model
        X_test: Test data
        y_test: Test labels
        dataset_name: Name of dataset for class names
        
    TODO: Implement comprehensive evaluation
    - Make predictions
    - Compute accuracy
    - Generate confusion matrix
    - Print classification report
    - Visualize sample predictions
    """
    pass


def main():
    """
    Main evaluation function.
    
    TODO: Implement evaluation pipeline
    - Load saved model
    - Load test dataset
    - Evaluate model
    - Generate visualizations
    - Save results
    """
    
    # Configuration
    model_path = '../results/models/best_model.pkl'
    dataset = 'fashion_mnist'  # or 'cifar10'
    results_dir = '../results/plots/'
    
    print(f"Loading model from {model_path}")
    # TODO: Load model
    
    print(f"Loading {dataset} test data")
    # TODO: Load and preprocess test data
    
    print("Evaluating model...")
    # TODO: Run evaluation
    
    print("Evaluation completed! Check results in", results_dir)


if __name__ == '__main__':
    main()

