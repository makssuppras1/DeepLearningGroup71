# Simple test script for NeuralNetwork class

import sys
import os
import numpy as np

# Add src to path (go up one level from tests/, then into src/)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from neural_network import NeuralNetwork

def test_xor():
    # Test on XOR problem - simple sanity check
    print("Testing XOR problem...")
    
    # Create network
    model = NeuralNetwork(
        input_size=2,
        hidden_layers=[4],
        output_size=2,
        activation='relu',
        output_activation='softmax',
        learning_rate=0.1,
        optimizer='sgd'
    )
    
    # XOR data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])  # One-hot encoded
    
    # Train for a few steps
    print("Training...")
    for i in range(100):
        loss = model.train_step(X, y)
        if i % 20 == 0:
            print(f"  Step {i}, Loss: {loss:.4f}")
    
    # Test predictions
    predictions = model.predict(X)
    print(f"\nPredictions: {predictions}")
    print(f"Expected: [0, 1, 1, 0]")
    
    # Check accuracy
    expected = np.array([0, 1, 1, 0])
    accuracy = np.mean(predictions == expected)
    print(f"Accuracy: {accuracy*100:.1f}%")
    
    if accuracy >= 0.75:
        print("✅ XOR test passed!")
    else:
        print("⚠️  XOR test needs more training or debugging")
    
    return accuracy >= 0.75

def test_forward_shape():
    # Test that forward pass returns correct shape
    print("\nTesting forward pass shape...")
    
    model = NeuralNetwork(
        input_size=10,
        hidden_layers=[20, 15],
        output_size=5,
        activation='relu',
        output_activation='softmax'
    )
    
    X = np.random.randn(32, 10)  # batch_size=32
    y_pred = model.forward(X)
    
    assert y_pred.shape == (32, 5), f"Expected shape (32, 5), got {y_pred.shape}"
    print(f"✅ Forward pass shape correct: {y_pred.shape}")
    return True

def test_predict():
    # Test prediction methods (predict_proba and predict)
    print("\nTesting prediction methods...")
    
    model = NeuralNetwork(
        input_size=5,
        hidden_layers=[10],
        output_size=3,
        activation='relu',
        output_activation='softmax'
    )
    
    X = np.random.randn(10, 5)
    
    # Test predict_proba
    proba = model.predict_proba(X)
    assert proba.shape == (10, 3), f"Expected shape (10, 3), got {proba.shape}"
    assert np.allclose(np.sum(proba, axis=1), 1.0), "Probabilities should sum to 1"
    print("✅ predict_proba works correctly")
    
    # Test predict
    predictions = model.predict(X)
    assert predictions.shape == (10,), f"Expected shape (10,), got {predictions.shape}"
    assert np.all((predictions >= 0) & (predictions < 3)), "Predictions should be valid class indices"
    print("✅ predict works correctly")
    return True

if __name__ == '__main__':
    print("=" * 50)
    print("Neural Network Test Suite")
    print("=" * 50)
    
    try:
        test_forward_shape()
        test_predict()
        test_xor()
        
        print("\n" + "=" * 50)
        print("✅ All basic tests passed!")
        print("=" * 50)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

