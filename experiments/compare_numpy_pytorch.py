"""
Comparison script to test if NumPy and PyTorch neural networks produce the same results.
This script initializes both networks with the same weights and compares:
1. Forward pass outputs
2. Loss values
3. Gradients
4. Weight updates after training steps
"""

import numpy as np
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.neural_network import NeuralNetwork
from src.pytorch_neural_network import PyTorchNeuralNetwork
from src.data_loader import load_cifar10, preprocess_data, create_mini_batches
from src.utils import set_random_seed


def compare_arrays(arr1, arr2, name="Arrays", rtol=1e-4, atol=1e-5):
    """Compare two numpy arrays and print results."""
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)
    
    if arr1.shape != arr2.shape:
        print(f"x {name}: Shape mismatch! {arr1.shape} vs {arr2.shape}")
        return False
    
    max_diff = np.max(np.abs(arr1 - arr2))
    mean_diff = np.mean(np.abs(arr1 - arr2))
    relative_diff = np.max(np.abs((arr1 - arr2) / (arr2 + 1e-10)))
    
    is_close = np.allclose(arr1, arr2, rtol=rtol, atol=atol)
    
    if is_close:
        print(f"✔ {name}: Match! (max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}, rel_diff={relative_diff:.2e})")
    else:
        print(f"x {name}: Mismatch! (max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}, rel_diff={relative_diff:.2e})")
        print(f"   First few values - NumPy: {arr1.flatten()[:5]}, PyTorch: {arr2.flatten()[:5]}")
    
    return is_close


def copy_weights_from_numpy_to_pytorch(numpy_model, pytorch_model):
    """Copy weights from NumPy model to PyTorch model."""
    numpy_params = numpy_model.get_params()
    pytorch_model.set_params(numpy_params)


def test_forward_pass(numpy_model, pytorch_model, X_test):
    """Test if forward pass produces the same outputs."""
    print("\n" + "="*60)
    print("TEST 1: Forward Pass Outputs")
    print("="*60)
    
    # NumPy forward pass
    numpy_model.eval()
    y_pred_numpy = numpy_model.predict_proba(X_test)
    
    # PyTorch forward pass
    X_torch = torch.from_numpy(X_test).float()
    y_pred_pytorch = pytorch_model.predict_proba(X_torch).cpu().numpy()
    
    return compare_arrays(y_pred_numpy, y_pred_pytorch, "Forward Pass Outputs")


def test_loss_computation(numpy_model, pytorch_model, X_test, y_test):
    """Test if loss computation produces the same values."""
    print("\n" + "="*60)
    print("TEST 2: Loss Computation")
    print("="*60)
    
    # NumPy loss
    numpy_model.eval()
    y_pred_numpy = numpy_model.predict_proba(X_test)
    loss_numpy = numpy_model.compute_loss(y_pred_numpy, y_test)
    
    # PyTorch loss
    X_torch = torch.from_numpy(X_test).float()
    y_torch = torch.from_numpy(y_test).float()
    y_pred_pytorch = pytorch_model.predict_proba(X_torch)
    loss_pytorch = pytorch_model.compute_loss(y_pred_pytorch, y_torch).item()
    
    return compare_arrays(np.array([loss_numpy]), np.array([loss_pytorch]), "Loss Values", rtol=1e-3)


def test_gradients(numpy_model, pytorch_model, X_batch, y_batch):
    """Test if gradients are similar after backward pass."""
    print("\n" + "="*60)
    print("TEST 3: Gradient Computation")
    print("="*60)
    
    # NumPy backward pass
    numpy_model.train()
    y_pred_numpy = numpy_model.forward(X_batch)
    numpy_model.backward(X_batch, y_batch, y_pred=y_pred_numpy)
    
    # PyTorch backward pass
    pytorch_model.train()
    X_torch = torch.from_numpy(X_batch).float()
    y_torch = torch.from_numpy(y_batch).float()
    
    y_pred_pytorch = pytorch_model.forward(X_torch)
    loss = pytorch_model.compute_loss(y_pred_pytorch, y_torch)
    pytorch_model.optimizer.zero_grad()
    loss.backward()
    
    # Apply L2 regularization to gradients if needed
    if numpy_model.l2_lambda > 0:
        m = X_batch.shape[0]
        for layer in pytorch_model.layers:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.grad += (numpy_model.l2_lambda / m) * layer.weight
    
    # Compare gradients
    all_match = True
    for i, (numpy_layer, pytorch_layer) in enumerate(zip(numpy_model.layers, 
                                                          [l for l in pytorch_model.layers if isinstance(l, torch.nn.Linear)])):
        # NumPy gradients
        dW_numpy = numpy_layer.dW
        db_numpy = numpy_layer.db
        
        # PyTorch gradients (transpose weight gradient to match NumPy format)
        dW_pytorch = pytorch_layer.weight.grad.T.detach().cpu().numpy()
        db_pytorch = pytorch_layer.bias.grad.detach().cpu().numpy()
        
        match_w = compare_arrays(dW_numpy, dW_pytorch, f"Layer {i+1} Weight Gradients (dW)", rtol=1e-3)
        match_b = compare_arrays(db_numpy, db_pytorch, f"Layer {i+1} Bias Gradients (db)", rtol=1e-3)
        
        if not (match_w and match_b):
            all_match = False
    
    return all_match


def test_training_step(numpy_model, pytorch_model, X_batch, y_batch):
    """Test if training step produces the same weight updates."""
    print("\n" + "="*60)
    print("TEST 4: Training Step (Weight Updates)")
    print("="*60)
    
    # Get initial weights FIRST (before any operations)
    numpy_params_before = numpy_model.get_params()
    pytorch_params_before = pytorch_model.get_params()
    
    # Reset optimizer state (this shouldn't modify weights)
    pytorch_model.reset_optimizer_state()
    
    # Training step
    numpy_model.train()
    loss_numpy = numpy_model.train_step(X_batch, y_batch)
    
    pytorch_model.train()
    X_torch = torch.from_numpy(X_batch).float()
    y_torch = torch.from_numpy(y_batch).float()
    loss_pytorch = pytorch_model.train_step(X_torch, y_torch)
    
    # Get updated weights
    numpy_params_after = numpy_model.get_params()
    pytorch_params_after = pytorch_model.get_params()
    
    # Compare weight updates
    all_match = True
    for key in numpy_params_before.keys():
        # Compute weight changes
        numpy_update = numpy_params_after[key] - numpy_params_before[key]
        pytorch_update = pytorch_params_after[key] - pytorch_params_before[key]
        
        match = compare_arrays(numpy_update, pytorch_update, f"Weight Update ({key})", rtol=1e-1, atol=1e-3)
        if not match:
            all_match = False
    
    return all_match


def test_multiple_training_steps(numpy_model, pytorch_model, X_train, y_train, num_steps=5):
    """Test multiple training steps to see if networks diverge."""
    print("\n" + "="*60)
    print("TEST 5: Multiple Training Steps")
    print("="*60)
    
    batches = create_mini_batches(X_train, y_train, batch_size=32, shuffle=False)
    
    all_match = True
    for step in range(min(num_steps, len(batches))):
        X_batch, y_batch = batches[step]
        
        # Synchronize random states before each step to ensure any random operations
        # (like dropout, if enabled) use the same random sequence
        torch.manual_seed(seed + step)  # Use step offset to avoid same sequence
        np.random.seed(seed + step)
        
        # Copy weights from NumPy to PyTorch before each step
        copy_weights_from_numpy_to_pytorch(numpy_model, pytorch_model)
        pytorch_model.reset_optimizer_state()  # Reset optimizer state
        
        # Training step
        numpy_model.train()
        loss_numpy = numpy_model.train_step(X_batch, y_batch)
        
        pytorch_model.train()
        X_torch = torch.from_numpy(X_batch).float()
        y_torch = torch.from_numpy(y_batch).float()
        loss_pytorch = pytorch_model.train_step(X_torch, y_torch)
        
        # Compare outputs after training step
        numpy_model.eval()
        y_pred_numpy = numpy_model.predict_proba(X_batch)
        
        pytorch_model.eval()
        y_pred_pytorch = pytorch_model.predict_proba(X_torch).cpu().numpy()
        
        match = compare_arrays(y_pred_numpy, y_pred_pytorch, f"Step {step+1} Outputs", rtol=1e-2)
        if not match:
            all_match = False
        
        # Compare weights
        numpy_params = numpy_model.get_params()
        pytorch_params = pytorch_model.get_params()
        
        for key in numpy_params.keys():
            match_w = compare_arrays(numpy_params[key], pytorch_params[key], 
                                    f"Step {step+1} Weights ({key})", rtol=1e-2, atol=1e-4)
            if not match_w:
                all_match = False
    
    return all_match


def main():
    """Main comparison function."""
    print("="*60)
    print("NumPy vs PyTorch Neural Network Comparison")
    print("="*60)
    
    # Set random seed for reproducibility
    # IMPORTANT: Set seeds for both NumPy and PyTorch to ensure both models
    # use the same random initialization. Even though weights are copied,
    # this ensures any random operations during training use the same sequence.
    seed = 42
    set_random_seed(seed)  # Sets NumPy seed
    torch.manual_seed(seed)  # Sets PyTorch seed
    np.random.seed(seed)  # Explicit NumPy seed (redundant but clear)
    
    # Also set CUDA seed if available (for GPU reproducibility)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Load a small subset of data for testing
    print("\nLoading data...")
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    X_train_full, y_train_full, X_test, y_test = load_cifar10(data_dir)
    
    # Use a small subset for faster testing
    X_train_subset = X_train_full[:1000]
    y_train_subset = y_train_full[:1000]
    X_test_subset = X_test[:100]
    y_test_subset = y_test[:100]
    
    # Preprocess data
    X_train_subset, y_train_subset = preprocess_data(
        X_train_subset, y_train_subset,
        num_classes=10, flatten=True, normalize=True
    )
    X_test_subset, y_test_subset = preprocess_data(
        X_test_subset, y_test_subset,
        num_classes=10, flatten=True, normalize=True
    )
    
    # Model configuration
    config = {
        'input_size': 3072,
        'hidden_layers': [128, 64],
        'output_size': 10,
        'activation': 'relu',
        'output_activation': 'softmax',
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'weight_init': 'he',
        'l2_lambda': 0.0001,
        'dropout_rate': 0.0,  # Disable dropout for easier comparison
        'random_seed': seed
    }
    
    # Create models
    # IMPORTANT: Both models use the same seed from config.
    # Note: NumPy and PyTorch use different RNG algorithms, so even with the same seed,
    # they would generate different random numbers. However, we copy weights from NumPy
    # to PyTorch to ensure they start with IDENTICAL weights, eliminating any initialization differences.
    print("\nCreating models...")
    print(f"Using random_seed: {seed} for both models")
    
    # Create NumPy model first
    numpy_model = NeuralNetwork(**config)
    
    # Reset seeds before creating PyTorch model to ensure same RNG state
    # (even though weights will be copied, this ensures any other random ops are synchronized)
    torch.manual_seed(seed)
    np.random.seed(seed)
    pytorch_model = PyTorchNeuralNetwork(**config)
    
    # Copy weights from NumPy to PyTorch to ensure they start with IDENTICAL weights
    # This guarantees both models have exactly the same initial parameters, regardless of RNG differences
    print("\nCopying weights from NumPy model to PyTorch model...")
    copy_weights_from_numpy_to_pytorch(numpy_model, pytorch_model)
    
    # Verify initial weights match
    # IMPORTANT: We copy weights from NumPy to PyTorch, so they should match exactly.
    # Even though both models use the same seed, NumPy and PyTorch use different RNG
    # algorithms, so they generate different random numbers. Copying weights ensures
    # both models start with IDENTICAL parameters for a fair comparison.
    print("\nVerifying initial weights match after copying...")
    numpy_params = numpy_model.get_params()
    pytorch_params = pytorch_model.get_params()
    initial_weights_match = True
    for key in numpy_params.keys():
        match = compare_arrays(numpy_params[key], pytorch_params[key], f"Initial {key}", rtol=1e-5)
        if not match:
            initial_weights_match = False
    
    if not initial_weights_match:
        print("\n!!  Warning: Initial weights don't match exactly after copying!")
        print("   This should not happen - weights should be identical.")
    else:
        print("   ✓ Initial weights match perfectly (as expected after copying)")
    
    # Run tests
    results = {}
    
    # Test 1: Forward pass
    results['forward_pass'] = test_forward_pass(numpy_model, pytorch_model, X_test_subset)
    
    # Test 2: Loss computation
    results['loss'] = test_loss_computation(numpy_model, pytorch_model, X_test_subset, y_test_subset)
    
    # Test 3: Gradients
    X_batch = X_train_subset[:32]
    y_batch = y_train_subset[:32]
    # Reset models to same state
    copy_weights_from_numpy_to_pytorch(numpy_model, pytorch_model)
    results['gradients'] = test_gradients(numpy_model, pytorch_model, X_batch, y_batch)
    
    # Test 4: Training step
    copy_weights_from_numpy_to_pytorch(numpy_model, pytorch_model)
    results['training_step'] = test_training_step(numpy_model, pytorch_model, X_batch, y_batch)
    
    # Test 5: Multiple training steps
    copy_weights_from_numpy_to_pytorch(numpy_model, pytorch_model)
    results['multiple_steps'] = test_multiple_training_steps(
        numpy_model, pytorch_model, X_train_subset, y_train_subset, num_steps=5
    )
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for test_name, passed in results.items():
        status = "✔ PASSED" if passed else "x FAILED"
        print(f"{test_name:20s}: {status}")
    
    all_passed = all(results.values())
    print("\n" + "="*60)
    if all_passed:
        print("(y) All tests passed! Networks produce similar results.")
    else:
        print("!!  Some tests failed. Networks may have differences.")
    print("="*60)
    
    return all_passed


if __name__ == '__main__':
    main()

