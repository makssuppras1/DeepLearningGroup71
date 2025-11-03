"""
Unit tests for activation functions.

Run with: python -m pytest tests/
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.activations import relu, relu_derivative, sigmoid, sigmoid_derivative, tanh, tanh_derivative, softmax


def test_relu_positive():
    """Test ReLU with positive values."""
    # TODO: Implement test
    # x = np.array([1, 2, 3])
    # result = relu(x)
    # expected = np.array([1, 2, 3])
    # assert np.allclose(result, expected)
    pass


def test_relu_negative():
    """Test ReLU with negative values."""
    # TODO: Implement test
    pass


def test_relu_derivative():
    """Test ReLU derivative."""
    # TODO: Implement test
    pass


def test_sigmoid_range():
    """Test that sigmoid outputs are in [0, 1]."""
    # TODO: Implement test
    # x = np.random.randn(10, 5)
    # result = sigmoid(x)
    # assert np.all(result >= 0) and np.all(result <= 1)
    pass


def test_sigmoid_zero():
    """Test that sigmoid(0) = 0.5."""
    # TODO: Implement test
    pass


def test_softmax_sum():
    """Test that softmax outputs sum to 1."""
    # TODO: Implement test
    # x = np.random.randn(5, 10)
    # result = softmax(x)
    # sums = np.sum(result, axis=1)
    # assert np.allclose(sums, np.ones(5))
    pass


def test_tanh_range():
    """Test that tanh outputs are in [-1, 1]."""
    # TODO: Implement test
    pass


if __name__ == '__main__':
    print("Run tests with: pytest tests/")

