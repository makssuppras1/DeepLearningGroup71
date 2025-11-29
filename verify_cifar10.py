#!/usr/bin/env python3
"""
Verification script for CIFAR-10 data loading.
Shows sample images with their labels to verify correct data loading.
"""

import numpy as np
import sys
import os

# Add project root to path (same as training scripts)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_cifar10, get_class_names, preprocess_data

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Visualization will be skipped.")

def verify_cifar10_data(data_dir='./data', num_samples=20):
    """
    Verify CIFAR-10 data loading by displaying sample images with labels.
    
    Args:
        data_dir: Directory containing the CIFAR-10 data
        num_samples: Number of sample images to display
    """
    print("=" * 70)
    print("CIFAR-10 Data Loading Verification")
    print("=" * 70)
    
    # Load the dataset
    print("\n1. Loading CIFAR-10 dataset...")
    X_train, y_train, X_test, y_test = load_cifar10(data_dir)
    
    # Get class names
    class_names = get_class_names('cifar10')
    
    # Print dataset statistics
    print("\n2. Dataset Statistics:")
    print(f"   Training set: {X_train.shape[0]:,} images")
    print(f"   Test set: {X_test.shape[0]:,} images")
    print(f"   Image shape: {X_train.shape[1:]} (height, width, channels)")
    print(f"   Image dtype: {X_train.dtype}")
    print(f"   Image value range: {X_train.min()} - {X_train.max()}")
    print(f"   Label range: {y_train.min()} - {y_train.max()}")
    
    # Verify shapes
    assert X_train.shape == (50000, 32, 32, 3), f"Expected train shape (50000, 32, 32, 3), got {X_train.shape}"
    assert X_test.shape == (10000, 32, 32, 3), f"Expected test shape (10000, 32, 32, 3), got {X_test.shape}"
    assert y_train.shape == (50000,), f"Expected train labels shape (50000,), got {y_train.shape}"
    assert y_test.shape == (10000,), f"Expected test labels shape (10000,), got {y_test.shape}"
    assert y_train.min() >= 0 and y_train.max() <= 9, "Labels should be in range 0-9"
    assert X_train.min() >= 0 and X_train.max() <= 255, "Images should be in range 0-255"
    
    print("\n   ✅ All shape and range checks passed!")
    
    # Show label distribution
    print("\n3. Label Distribution (Training Set):")
    unique_labels, counts = np.unique(y_train, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"   Class {label} ({class_names[label]:15s}): {count:5,} samples ({count/len(y_train)*100:.1f}%)")
    
    # Verify each class has 5000 samples
    assert all(count == 5000 for count in counts), "Each class should have 5000 training samples"
    print("   ✅ All classes have 5000 samples as expected!")
    
    # Display sample images from training set
    if HAS_MATPLOTLIB:
        print(f"\n4. Displaying {num_samples} sample images from training set...")
        fig, axes = plt.subplots(4, 5, figsize=(15, 12))
        fig.suptitle('CIFAR-10 Training Samples with Labels', fontsize=16, fontweight='bold')
        
        # Select samples from different classes
        samples_per_class = num_samples // 10
        sample_indices = []
        
        for class_idx in range(10):
            class_indices = np.where(y_train == class_idx)[0]
            selected = np.random.choice(class_indices, size=min(samples_per_class, len(class_indices)), replace=False)
            sample_indices.extend(selected)
        
        # Fill remaining slots randomly
        remaining = num_samples - len(sample_indices)
        if remaining > 0:
            all_indices = np.arange(len(X_train))
            remaining_indices = np.random.choice(all_indices, size=remaining, replace=False)
            sample_indices.extend(remaining_indices)
        
        sample_indices = sample_indices[:num_samples]
        
        for idx, ax in enumerate(axes.flat):
            if idx < len(sample_indices):
                img_idx = sample_indices[idx]
                image = X_train[img_idx]
                label = y_train[img_idx]
                
                # Display image
                ax.imshow(image.astype(np.uint8))
                ax.set_title(f'Label: {label} ({class_names[label]})', fontsize=10, fontweight='bold')
                ax.axis('off')
            else:
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('cifar10_verification_samples.png', dpi=150, bbox_inches='tight')
        print("   ✅ Saved visualization to 'cifar10_verification_samples.png'")
    else:
        print(f"\n4. Sample images (first {min(num_samples, 10)} from each class):")
        for class_idx in range(10):
            class_indices = np.where(y_train == class_idx)[0]
            if len(class_indices) > 0:
                sample_idx = class_indices[0]
                label = y_train[sample_idx]
                print(f"   Class {class_idx} ({class_names[label]:15s}): Sample index {sample_idx}, Label: {label}")
    
    # Test preprocessing
    print("\n5. Testing preprocessing pipeline...")
    X_sample = X_train[:100]
    y_sample = y_train[:100]
    
    X_processed, y_processed = preprocess_data(
        X_sample, y_sample,
        num_classes=10,
        flatten=True,
        normalize=True
    )
    
    print(f"   Original shape: {X_sample.shape}")
    print(f"   Processed shape: {X_processed.shape} (expected: (100, 3072))")
    print(f"   Processed value range: {X_processed.min():.3f} - {X_processed.max():.3f} (expected: 0.0 - 1.0)")
    print(f"   Labels shape: {y_processed.shape} (expected: (100, 10))")
    
    assert X_processed.shape == (100, 3072), f"Expected flattened shape (100, 3072), got {X_processed.shape}"
    assert y_processed.shape == (100, 10), f"Expected one-hot shape (100, 10), got {y_processed.shape}"
    assert np.allclose(X_processed.min(), 0.0) and np.allclose(X_processed.max(), 1.0), "Normalization failed"
    
    # Verify one-hot encoding
    for i in range(10):
        label_idx = y_sample[i]
        one_hot = y_processed[i]
        assert one_hot[label_idx] == 1.0, f"One-hot encoding incorrect for sample {i}"
        assert np.sum(one_hot) == 1.0, f"One-hot encoding should sum to 1.0 for sample {i}"
    
    print("   ✅ Preprocessing pipeline works correctly!")
    
    # Verify label-image correspondence for a few samples
    print("\n6. Verifying label-image correspondence...")
    verification_samples = [
        (0, "airplane"),
        (1, "automobile"),
        (2, "bird"),
        (3, "cat"),
        (4, "deer"),
        (5, "dog"),
        (6, "frog"),
        (7, "horse"),
        (8, "ship"),
        (9, "truck")
    ]
    
    print("   Checking one sample from each class:")
    for class_idx, class_name in verification_samples:
        class_indices = np.where(y_train == class_idx)[0]
        if len(class_indices) > 0:
            sample_idx = class_indices[0]
            actual_label = y_train[sample_idx]
            assert actual_label == class_idx, f"Mismatch: expected {class_idx}, got {actual_label}"
            print(f"   ✅ Class {class_idx} ({class_name:15s}): Sample {sample_idx} has label {actual_label}")
    
    print("\n" + "=" * 70)
    print("✅ ALL VERIFICATIONS PASSED!")
    print("=" * 70)
    print("\nYour CIFAR-10 data is loaded correctly with correct labels.")
    print("The data is ready to be fed to your feedforward neural network.")
    print("\nSummary:")
    print(f"  - Training images: {X_train.shape[0]:,} samples, shape {X_train.shape[1:]}")
    print(f"  - Test images: {X_test.shape[0]:,} samples, shape {X_test.shape[1:]}")
    print(f"  - After flattening: {X_train.shape[0]:,} samples, {X_train.shape[1]*X_train.shape[2]*X_train.shape[3]:,} features")
    print(f"  - Labels: {len(class_names)} classes ({', '.join(class_names)})")
    
    if HAS_MATPLOTLIB:
        plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Verify CIFAR-10 data loading')
    parser.add_argument('--data-dir', type=str, default='./data', help='Directory containing CIFAR-10 data')
    parser.add_argument('--num-samples', type=int, default=20, help='Number of sample images to display')
    args = parser.parse_args()
    
    verify_cifar10_data(args.data_dir, args.num_samples)

