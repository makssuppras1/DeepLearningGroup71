#!/usr/bin/env python3
"""
Test script for data_loader.py
This script verifies that Fashion-MNIST and CIFAR-10 loading works correctly.
"""

import sys
import os

# Add src directory to path (go up one level from tests/ to root, then into src/)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import (
    download_fashion_mnist,
    download_cifar10,
    load_fashion_mnist,
    load_cifar10,
    preprocess_data,
    create_mini_batches,
    train_val_split
)

# Data directory (relative to project root)
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def test_fashion_mnist():
    """Test Fashion-MNIST loading."""
    print("=" * 60)
    print("Testing Fashion-MNIST")
    print("=" * 60)
    
    # Download dataset
    print("\n1. Downloading Fashion-MNIST...")
    download_fashion_mnist(DATA_DIR)
    
    # Load dataset
    print("\n2. Loading Fashion-MNIST...")
    X_train, y_train, X_test, y_test = load_fashion_mnist(DATA_DIR)
    
    # Verify shapes
    print("\n3. Verifying shapes...")
    print(f"   Train images: {X_train.shape} (expected: (60000, 28, 28))")
    print(f"   Train labels: {y_train.shape} (expected: (60000,))")
    print(f"   Test images:  {X_test.shape} (expected: (10000, 28, 28))")
    print(f"   Test labels:  {y_test.shape} (expected: (10000,))")
    
    assert X_train.shape == (60000, 28, 28), f"Wrong train images shape: {X_train.shape}"
    assert y_train.shape == (60000,), f"Wrong train labels shape: {y_train.shape}"
    assert X_test.shape == (10000, 28, 28), f"Wrong test images shape: {X_test.shape}"
    assert y_test.shape == (10000,), f"Wrong test labels shape: {y_test.shape}"
    
    # Verify data ranges
    print("\n4. Verifying data ranges...")
    print(f"   Image values: {X_train.min()}-{X_train.max()} (expected: 0-255)")
    print(f"   Label values: {y_train.min()}-{y_train.max()} (expected: 0-9)")
    
    assert X_train.min() >= 0 and X_train.max() <= 255, "Image values out of range"
    assert y_train.min() >= 0 and y_train.max() <= 9, "Label values out of range"
    
    # Test preprocessing
    print("\n5. Testing preprocessing...")
    X_proc, y_proc = preprocess_data(X_train[:100], y_train[:100], flatten=True, normalize=True)
    print(f"   Processed images: {X_proc.shape} (expected: (100, 784))")
    print(f"   Processed labels: {y_proc.shape} (expected: (100, 10))")
    print(f"   Image range: {X_proc.min():.3f}-{X_proc.max():.3f} (expected: 0.0-1.0)")
    
    assert X_proc.shape == (100, 784), f"Wrong processed shape: {X_proc.shape}"
    assert y_proc.shape == (100, 10), f"Wrong one-hot shape: {y_proc.shape}"
    assert 0 <= X_proc.min() and X_proc.max() <= 1, "Normalization failed"
    
    # Test mini-batches
    print("\n6. Testing mini-batch creation...")
    batches = create_mini_batches(X_proc, y_proc, batch_size=32)
    print(f"   Number of batches: {len(batches)} (expected: 4)")
    print(f"   First batch shape: {batches[0][0].shape}, {batches[0][1].shape}")
    
    print("\n✅ Fashion-MNIST tests passed!\n")
    return True

def test_cifar10():
    """Test CIFAR-10 loading."""
    print("=" * 60)
    print("Testing CIFAR-10")
    print("=" * 60)
    
    # Download dataset
    print("\n1. Downloading CIFAR-10 (this may take a while)...")
    download_cifar10(DATA_DIR)
    
    # Load dataset
    print("\n2. Loading CIFAR-10...")
    X_train, y_train, X_test, y_test = load_cifar10(DATA_DIR)
    
    # Verify shapes
    print("\n3. Verifying shapes...")
    print(f"   Train images: {X_train.shape} (expected: (50000, 32, 32, 3))")
    print(f"   Train labels: {y_train.shape} (expected: (50000,))")
    print(f"   Test images:  {X_test.shape} (expected: (10000, 32, 32, 3))")
    print(f"   Test labels:  {y_test.shape} (expected: (10000,))")
    
    assert X_train.shape == (50000, 32, 32, 3), f"Wrong train images shape: {X_train.shape}"
    assert y_train.shape == (50000,), f"Wrong train labels shape: {y_train.shape}"
    assert X_test.shape == (10000, 32, 32, 3), f"Wrong test images shape: {X_test.shape}"
    assert y_test.shape == (10000,), f"Wrong test labels shape: {y_test.shape}"
    
    # Verify data ranges
    print("\n4. Verifying data ranges...")
    print(f"   Image values: {X_train.min()}-{X_train.max()} (expected: 0-255)")
    print(f"   Label values: {y_train.min()}-{y_train.max()} (expected: 0-9)")
    
    assert X_train.min() >= 0 and X_train.max() <= 255, "Image values out of range"
    assert y_train.min() >= 0 and y_train.max() <= 9, "Label values out of range"
    
    # Test preprocessing (flatten RGB to vector)
    print("\n5. Testing preprocessing...")
    X_proc, y_proc = preprocess_data(X_train[:100], y_train[:100], flatten=True, normalize=True)
    print(f"   Processed images: {X_proc.shape} (expected: (100, 3072))")
    print(f"   Processed labels: {y_proc.shape} (expected: (100, 10))")
    print(f"   Image range: {X_proc.min():.3f}-{X_proc.max():.3f} (expected: 0.0-1.0)")
    
    assert X_proc.shape == (100, 3072), f"Wrong processed shape: {X_proc.shape}"
    assert y_proc.shape == (100, 10), f"Wrong one-hot shape: {y_proc.shape}"
    assert 0 <= X_proc.min() and X_proc.max() <= 1, "Normalization failed"
    
    # Test train/val split
    print("\n6. Testing train/validation split...")
    X_tr, X_val, y_tr, y_val = train_val_split(X_proc, y_proc, val_split=0.2, random_seed=42)
    print(f"   Train split: {X_tr.shape} (expected: (80, 3072))")
    print(f"   Val split:   {X_val.shape} (expected: (20, 3072))")
    
    assert X_tr.shape[0] == 80, f"Wrong train split size: {X_tr.shape[0]}"
    assert X_val.shape[0] == 20, f"Wrong val split size: {X_val.shape[0]}"
    
    print("\n✅ CIFAR-10 tests passed!\n")
    return True

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("DATA LOADER TEST SUITE")
    print("=" * 60 + "\n")
    
    try:
        # Test Fashion-MNIST
        test_fashion_mnist()
        
        # Test CIFAR-10
        test_cifar10()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("=" * 60 + "\n")
        print("Your data loaders are working correctly!")
        print("You can now use them in your neural network training.\n")
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ TEST FAILED")
        print("=" * 60)
        print(f"\nError: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

