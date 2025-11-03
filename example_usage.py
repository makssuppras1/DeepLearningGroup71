"""
Example usage of the neural network implementation.

This script demonstrates how to use the neural network once implemented.
"""

import numpy as np
import sys
sys.path.append('.')

from src.neural_network import NeuralNetwork
from src.data_loader import load_fashion_mnist, preprocess_data, train_val_split
from src.utils import accuracy_score

# This is a template showing expected usage
# Uncomment and modify after implementing the classes

def main():
    print("=" * 60)
    print("Neural Network from Scratch - Example Usage")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n1. Loading Fashion-MNIST dataset...")
    # TODO: Uncomment after implementing data_loader
    # X_train, y_train, X_test, y_test = load_fashion_mnist()
    # print(f"   Training samples: {X_train.shape[0]}")
    # print(f"   Test samples: {X_test.shape[0]}")
    
    # Step 2: Preprocess data
    print("\n2. Preprocessing data...")
    # TODO: Uncomment after implementing preprocessing
    # X_train, y_train = preprocess_data(X_train, y_train, flatten=True, normalize=True)
    # X_test, y_test = preprocess_data(X_test, y_test, flatten=True, normalize=True)
    
    # Split into train and validation
    # X_train, X_val, y_train, y_val = train_val_split(X_train, y_train, val_split=0.2)
    # print(f"   Training set: {X_train.shape}")
    # print(f"   Validation set: {X_val.shape}")
    
    # Step 3: Create model
    print("\n3. Creating neural network...")
    # TODO: Uncomment after implementing NeuralNetwork
    # model = NeuralNetwork(
    #     input_size=784,              # 28*28 for Fashion-MNIST
    #     hidden_layers=[128, 64],     # Two hidden layers
    #     output_size=10,              # 10 classes
    #     activation='relu',
    #     learning_rate=0.001,
    #     optimizer='adam',
    #     l2_lambda=0.0001,
    #     random_seed=42
    # )
    # print("   Model created successfully!")
    # print(f"   Architecture: 784 -> 128 -> 64 -> 10")
    
    # Step 4: Train model
    print("\n4. Training model...")
    # TODO: Uncomment after implementing training
    # num_epochs = 10
    # batch_size = 32
    # 
    # for epoch in range(num_epochs):
    #     # Training
    #     train_loss = 0
    #     num_batches = len(X_train) // batch_size
    #     
    #     for i in range(0, len(X_train), batch_size):
    #         X_batch = X_train[i:i+batch_size]
    #         y_batch = y_train[i:i+batch_size]
    #         
    #         loss = model.train_step(X_batch, y_batch)
    #         train_loss += loss
    #     
    #     train_loss /= num_batches
    #     
    #     # Validation
    #     val_pred = model.forward(X_val)
    #     val_loss = model.compute_loss(val_pred, y_val)
    #     val_acc = accuracy_score(model.predict(X_val), np.argmax(y_val, axis=1))
    #     
    #     print(f"   Epoch {epoch+1}/{num_epochs} - "
    #           f"Train Loss: {train_loss:.4f}, "
    #           f"Val Loss: {val_loss:.4f}, "
    #           f"Val Acc: {val_acc:.4f}")
    
    # Step 5: Evaluate on test set
    print("\n5. Evaluating on test set...")
    # TODO: Uncomment after implementing evaluation
    # test_pred = model.predict(X_test)
    # test_acc = accuracy_score(test_pred, np.argmax(y_test, axis=1))
    # print(f"   Test Accuracy: {test_acc:.4f}")
    
    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)
    
    # Expected output format:
    print("\n[Expected output after implementation]")
    print("Epoch 1/10 - Train Loss: 0.8234, Val Loss: 0.7123, Val Acc: 0.7543")
    print("Epoch 2/10 - Train Loss: 0.6543, Val Loss: 0.6234, Val Acc: 0.7923")
    print("...")
    print("Test Accuracy: 0.8456")


if __name__ == '__main__':
    main()

