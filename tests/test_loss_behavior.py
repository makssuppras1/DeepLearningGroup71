"""
Tests for loss function behavior.

This module tests that loss functions behave correctly:
- Loss decreases with correct predictions
- Loss increases with incorrect predictions
- Loss is zero/minimal at perfect predictions
"""
import sys
import os
import numpy as np
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from losses import (
    mean_squared_error,
    cross_entropy_loss,
    binary_cross_entropy,
    min_log_likelihood
)


class TestMSELoss:
    """Test Mean Squared Error loss behavior."""
    
    def test_mse_perfect_prediction(self):
        """Test MSE is zero for perfect predictions."""
        y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_pred = y_true.copy()
        
        loss = mean_squared_error(y_pred, y_true)
        assert np.isclose(loss, 0.0), f"MSE should be 0 for perfect prediction, got {loss}"
    
    def test_mse_increases_with_error(self):
        """Test MSE increases as predictions get worse."""
        y_true = np.array([[1.0, 2.0]])
        
        # Perfect prediction
        y_pred_perfect = y_true.copy()
        loss_perfect = mean_squared_error(y_pred_perfect, y_true)
        
        # Small error
        y_pred_small = y_true + 0.1
        loss_small = mean_squared_error(y_pred_small, y_true)
        
        # Large error
        y_pred_large = y_true + 1.0
        loss_large = mean_squared_error(y_pred_large, y_true)
        
        assert loss_perfect < loss_small < loss_large, \
            "MSE should increase with prediction error"
    
    def test_mse_symmetric(self):
        """Test MSE is symmetric (same error above or below true value)."""
        y_true = np.array([[1.0, 2.0]])
        y_pred_above = y_true + 0.5
        y_pred_below = y_true - 0.5
        
        loss_above = mean_squared_error(y_pred_above, y_true)
        loss_below = mean_squared_error(y_pred_below, y_true)
        
        assert np.isclose(loss_above, loss_below), \
            "MSE should be symmetric"


class TestCrossEntropyLoss:
    """Test Cross-Entropy loss behavior."""
    
    def test_cross_entropy_perfect_prediction(self):
        """Test cross-entropy is minimal for perfect predictions."""
        # Perfect prediction: predicted probability matches true label
        y_true = np.array([[0.0, 1.0], [1.0, 0.0]])  # One-hot encoded
        y_pred = np.array([[0.0, 1.0], [1.0, 0.0]])  # Perfect match
        
        loss = cross_entropy_loss(y_pred, y_true)
        # Should be very close to zero (numerical precision limits)
        assert loss < 1e-10, f"Cross-entropy should be ~0 for perfect prediction, got {loss}"
    
    def test_cross_entropy_increases_with_error(self):
        """Test cross-entropy increases as predictions get worse."""
        y_true = np.array([[0.0, 1.0]])  # True class is 1
        
        # Perfect prediction
        y_pred_perfect = np.array([[0.0, 1.0]])
        loss_perfect = cross_entropy_loss(y_pred_perfect, y_true)
        
        # Somewhat confident but wrong
        y_pred_wrong = np.array([[0.7, 0.3]])
        loss_wrong = cross_entropy_loss(y_pred_wrong, y_true)
        
        # Very wrong
        y_pred_very_wrong = np.array([[0.99, 0.01]])
        loss_very_wrong = cross_entropy_loss(y_pred_very_wrong, y_true)
        
        assert loss_perfect < loss_wrong < loss_very_wrong, \
            "Cross-entropy should increase with prediction error"
    
    def test_cross_entropy_uniform_prediction(self):
        """Test cross-entropy with uniform (random) predictions."""
        y_true = np.array([[0.0, 1.0, 0.0]])  # True class is 1
        y_pred_uniform = np.array([[1/3, 1/3, 1/3]])  # Uniform distribution
        
        loss_uniform = cross_entropy_loss(y_pred_uniform, y_true)
        
        # Should be positive (not zero)
        assert loss_uniform > 0, "Cross-entropy should be positive for uniform prediction"
        
        # Should be around -log(1/3) ≈ 1.099
        expected_loss = -np.log(1/3)
        assert np.isclose(loss_uniform, expected_loss, rtol=1e-3), \
            f"Cross-entropy for uniform prediction should be ~{expected_loss}, got {loss_uniform}"


class TestBinaryCrossEntropyLoss:
    """Test Binary Cross-Entropy loss behavior."""
    
    def test_bce_perfect_prediction(self):
        """Test BCE is minimal for perfect predictions."""
        y_true = np.array([1.0, 0.0, 1.0])
        y_pred = np.array([1.0, 0.0, 1.0])  # Perfect match
        
        loss = binary_cross_entropy(y_pred, y_true)
        assert loss < 1e-10, f"BCE should be ~0 for perfect prediction, got {loss}"
    
    def test_bce_increases_with_error(self):
        """Test BCE increases as predictions get worse."""
        y_true = np.array([1.0])
        
        # Perfect prediction
        y_pred_perfect = np.array([1.0])
        loss_perfect = binary_cross_entropy(y_pred_perfect, y_true)
        
        # Somewhat wrong
        y_pred_wrong = np.array([0.7])
        loss_wrong = binary_cross_entropy(y_pred_wrong, y_true)
        
        # Very wrong
        y_pred_very_wrong = np.array([0.1])
        loss_very_wrong = binary_cross_entropy(y_pred_very_wrong, y_true)
        
        assert loss_perfect < loss_wrong < loss_very_wrong, \
            "BCE should increase with prediction error"
    
    def test_bce_symmetric(self):
        """Test BCE is symmetric for positive and negative cases."""
        y_true_pos = np.array([1.0])
        y_true_neg = np.array([0.0])
        
        # Same prediction error magnitude
        y_pred_pos = np.array([0.3])  # Predicted 0.3, true is 1.0 (error 0.7)
        y_pred_neg = np.array([0.7])  # Predicted 0.7, true is 0.0 (error 0.7)
        
        loss_pos = binary_cross_entropy(y_pred_pos, y_true_pos)
        loss_neg = binary_cross_entropy(y_pred_neg, y_true_neg)
        
        # Should be similar (not necessarily equal due to log asymmetry)
        assert abs(loss_pos - loss_neg) < 0.1, \
            "BCE should be somewhat symmetric for similar error magnitudes"


class TestMinLogLikelihoodLoss:
    """Test Minimum Log-Likelihood loss behavior."""
    
    def test_mll_perfect_prediction(self):
        """Test MLL is minimal for perfect predictions."""
        y_true = np.array([[0.0, 1.0], [1.0, 0.0]])
        y_pred = np.array([[0.0, 1.0], [1.0, 0.0]])  # Perfect match
        
        loss = min_log_likelihood(y_pred, y_true)
        assert loss < 1e-10, f"MLL should be ~0 for perfect prediction, got {loss}"
    
    def test_mll_increases_with_error(self):
        """Test MLL increases as predictions get worse."""
        y_true = np.array([[0.0, 1.0]])
        
        # Perfect prediction
        y_pred_perfect = np.array([[0.0, 1.0]])
        loss_perfect = min_log_likelihood(y_pred_perfect, y_true)
        
        # Wrong prediction
        y_pred_wrong = np.array([[0.7, 0.3]])
        loss_wrong = min_log_likelihood(y_pred_wrong, y_true)
        
        # Very wrong prediction
        y_pred_very_wrong = np.array([[0.99, 0.01]])
        loss_very_wrong = min_log_likelihood(y_pred_very_wrong, y_true)
        
        assert loss_perfect < loss_wrong < loss_very_wrong, \
            "MLL should increase with prediction error"


class TestLossProperties:
    """Test general properties of loss functions."""
    
    def test_all_losses_non_negative(self):
        """Test all loss functions return non-negative values."""
        y_pred = np.array([[0.3, 0.7], [0.6, 0.4]])
        y_true = np.array([[0.0, 1.0], [1.0, 0.0]])
        
        mse_loss = mean_squared_error(y_pred, y_true)
        ce_loss = cross_entropy_loss(y_pred, y_true)
        
        assert mse_loss >= 0, "MSE should be non-negative"
        assert ce_loss >= 0, "Cross-entropy should be non-negative"
        
        y_pred_binary = np.array([0.3, 0.7])
        y_true_binary = np.array([0.0, 1.0])
        bce_loss = binary_cross_entropy(y_pred_binary, y_true_binary)
        
        assert bce_loss >= 0, "Binary cross-entropy should be non-negative"
    
    def test_loss_scales_with_batch_size(self):
        """Test that loss scales appropriately with batch size."""
        # Single sample
        y_pred_1 = np.array([[0.3, 0.7]])
        y_true_1 = np.array([[0.0, 1.0]])
        loss_1 = cross_entropy_loss(y_pred_1, y_true_1)
        
        # Two samples (same predictions)
        y_pred_2 = np.array([[0.3, 0.7], [0.3, 0.7]])
        y_true_2 = np.array([[0.0, 1.0], [0.0, 1.0]])
        loss_2 = cross_entropy_loss(y_pred_2, y_true_2)
        
        # Loss should be approximately double for double the batch size
        # (for cross-entropy, it's averaged, so should be similar)
        # Actually, cross-entropy is averaged, so should be similar
        assert np.isclose(loss_1, loss_2, rtol=1e-3), \
            f"Cross-entropy should average over batch: loss_1={loss_1}, loss_2={loss_2}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

