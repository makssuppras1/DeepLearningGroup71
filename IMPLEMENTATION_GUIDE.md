# Implementation Guide

This guide provides detailed instructions for implementing each component of the neural network.

## Table of Contents
1. [Activation Functions](#1-activation-functions)
2. [Weight Initialization](#2-weight-initialization)
3. [Loss Functions](#3-loss-functions)
4. [Forward Propagation](#4-forward-propagation)
5. [Backward Propagation](#5-backward-propagation)
6. [Optimizers](#6-optimizers)
7. [Training Loop](#7-training-loop)
8. [Evaluation](#8-evaluation)

---

## 1. Activation Functions

### ReLU (Rectified Linear Unit)

**Forward:**
```
f(x) = max(0, x)
```

**Derivative:**
```
f'(x) = 1 if x > 0, else 0
```

**Implementation hints:**
- Use `np.maximum(0, x)`
- For derivative, use boolean indexing or `(x > 0).astype(float)`

### Sigmoid

**Forward:**
```
f(x) = 1 / (1 + exp(-x))
```
**Derivative:**
```
f'(x) = f(x) * (1 - f(x))
```

**Implementation hints:**
- For numerical stability: `np.clip(x, -500, 500)` before exp
- Alternatively: use `scipy.special.expit`

### Tanh

**Forward:**
```
f(x) = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
```

**Derivative:**
```
f'(x) = 1 - tanh²(x)
```

**Implementation hints:**
- NumPy has `np.tanh()`

### Softmax (for output layer)

**Forward:**
```
f(x_i) = exp(x_i) / sum(exp(x_j))
```

**Implementation hints:**
- For numerical stability: subtract max before exp
```python
exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
return exp_x / np.sum(exp_x, axis=1, keepdims=True)
```

---

## 2. Weight Initialization

### Random Initialization
```python
W = np.random.randn(input_size, output_size) * 0.01
b = np.zeros(output_size)
```

### Xavier/Glorot Initialization
Best for sigmoid/tanh activations:
```python
limit = np.sqrt(6 / (input_size + output_size))
W = np.random.uniform(-limit, limit, (input_size, output_size))
```

### He Initialization
Best for ReLU activations:
```python
W = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
```

---

## 3. Loss Functions

### Cross-Entropy Loss

**Formula:**
```
L = -(1/n) * sum(y_true * log(y_pred))
```

**Implementation:**
```python
epsilon = 1e-8  # Prevent log(0)
loss = -np.mean(np.sum(y_true * np.log(y_pred + epsilon), axis=1))
```

**With L2 Regularization:**
```python
l2_term = (lambda / 2) * sum(np.sum(W**2) for W in weights)
total_loss = loss + l2_term
```

### Gradient (with softmax)

When using softmax + cross-entropy, the gradient simplifies to:
```
dL/dz = y_pred - y_true
```

This is a beautiful result and makes backprop easier!

---

## 4. Forward Propagation

### Algorithm

For each layer l from 1 to L:
1. Compute pre-activation: `z[l] = a[l-1] @ W[l] + b[l]`
2. Apply activation: `a[l] = activation(z[l])`

**Implementation tips:**
- Store all intermediate values (z and a) - needed for backprop
- Use `@` operator for matrix multiplication
- Last layer uses softmax activation

**Example:**
```python
def forward(self, X):
    self.cache = {}
    self.cache['a0'] = X
    
    for l in range(1, self.num_layers):
        # Linear transformation
        self.cache[f'z{l}'] = self.cache[f'a{l-1}'] @ self.W[l] + self.b[l]
        
        # Activation
        if l < self.num_layers - 1:
            self.cache[f'a{l}'] = relu(self.cache[f'z{l}'])
        else:
            self.cache[f'a{l}'] = softmax(self.cache[f'z{l}'])
    
    return self.cache[f'a{self.num_layers-1}']
```

---

## 5. Backward Propagation

### Algorithm

For each layer l from L to 1 (reverse order):

1. **Output layer (l = L):**
   ```
   dz[L] = y_pred - y_true  (when using softmax + cross-entropy)
   ```

2. **Hidden layers (l < L):**
   ```
   dz[l] = da[l] * activation'(z[l])
   ```

3. **Compute gradients:**
   ```
   dW[l] = (1/m) * a[l-1].T @ dz[l] + (lambda/m) * W[l]  (with L2)
   db[l] = (1/m) * sum(dz[l], axis=0)
   ```

4. **Backpropagate error:**
   ```
   da[l-1] = dz[l] @ W[l].T
   ```

**Implementation tips:**
- Start from the output layer and work backwards
- Don't forget to divide by batch size
- Add L2 regularization term to weight gradients

**Example:**
```python
def backward(self, X, y):
    m = X.shape[0]
    self.gradients = {}
    
    # Output layer
    dz = self.cache[f'a{self.num_layers-1}'] - y
    
    for l in range(self.num_layers - 1, 0, -1):
        # Gradients for weights and biases
        self.gradients[f'dW{l}'] = (1/m) * self.cache[f'a{l-1}'].T @ dz
        self.gradients[f'db{l}'] = (1/m) * np.sum(dz, axis=0)
        
        # Add L2 regularization to weights
        if self.l2_lambda > 0:
            self.gradients[f'dW{l}'] += (self.l2_lambda / m) * self.W[l]
        
        if l > 1:
            # Backpropagate to previous layer
            da = dz @ self.W[l].T
            dz = da * relu_derivative(self.cache[f'z{l-1}'])
```

---

## 6. Optimizers

### SGD (Stochastic Gradient Descent)

```python
W = W - learning_rate * dW
b = b - learning_rate * db
```

### Momentum

```python
# Initialize velocity
v_W = 0
v_b = 0

# Update
v_W = beta * v_W + (1 - beta) * dW
v_b = beta * v_b + (1 - beta) * db
W = W - learning_rate * v_W
b = b - learning_rate * v_b
```

Typical: `beta = 0.9`

### RMSprop

```python
# Initialize cache
s_W = 0
s_b = 0

# Update
s_W = beta * s_W + (1 - beta) * dW**2
s_b = beta * s_b + (1 - beta) * db**2
W = W - learning_rate * dW / (np.sqrt(s_W) + epsilon)
b = b - learning_rate * db / (np.sqrt(s_b) + epsilon)
```

Typical: `beta = 0.9`, `epsilon = 1e-8`

### Adam (Adaptive Moment Estimation)

Combines momentum and RMSprop:

```python
# Initialize
m_W = 0  # First moment
v_W = 0  # Second moment
t = 0    # Time step

# Update
t += 1
m_W = beta1 * m_W + (1 - beta1) * dW
v_W = beta2 * v_W + (1 - beta2) * dW**2

# Bias correction
m_W_corrected = m_W / (1 - beta1**t)
v_W_corrected = v_W / (1 - beta2**t)

# Parameter update
W = W - learning_rate * m_W_corrected / (np.sqrt(v_W_corrected) + epsilon)
```

Typical: `beta1 = 0.9`, `beta2 = 0.999`, `epsilon = 1e-8`

---

## 7. Training Loop

### Algorithm

```python
for epoch in range(num_epochs):
    # Shuffle data
    indices = np.random.permutation(len(X_train))
    X_shuffled = X_train[indices]
    y_shuffled = y_train[indices]
    
    # Create mini-batches
    for i in range(0, len(X_train), batch_size):
        X_batch = X_shuffled[i:i+batch_size]
        y_batch = y_shuffled[i:i+batch_size]
        
        # Forward pass
        y_pred = model.forward(X_batch)
        
        # Compute loss
        loss = model.compute_loss(y_pred, y_batch)
        
        # Backward pass
        model.backward(X_batch, y_batch)
        
        # Update weights
        model.update_weights()
    
    # Validation
    val_loss = evaluate(model, X_val, y_val)
    
    # Log metrics
    wandb.log({
        'train_loss': train_loss,
        'val_loss': val_loss,
        'epoch': epoch
    })
```

---

## 8. Evaluation

### Accuracy

```python
def accuracy(y_pred, y_true):
    # Convert one-hot to class labels
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_true, axis=1)
    return np.mean(y_pred_labels == y_true_labels)
```

### Confusion Matrix

Use `sklearn.metrics.confusion_matrix`:
```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true_labels, y_pred_labels)
```

---

## Common Issues and Debugging

### Issue: Gradients are zero
- Check if you're storing intermediate values correctly
- Verify activation derivatives
- Check if learning rate is too small

### Issue: Loss is NaN or Inf
- Add epsilon to log: `np.log(x + 1e-8)`
- Reduce learning rate
- Check for numerical overflow in softmax
- Gradient clipping: `np.clip(grad, -5, 5)`

### Issue: Loss not decreasing
- Verify backpropagation (use gradient checking)
- Try different learning rates
- Check if data is normalized
- Ensure weights are being updated

### Gradient Checking

Verify backprop implementation:
```python
def gradient_check(model, X, y, epsilon=1e-7):
    # Compute analytical gradients
    model.forward(X)
    model.backward(X, y)
    analytical_grad = model.gradients['dW1'].copy()
    
    # Compute numerical gradients
    numerical_grad = np.zeros_like(model.W[1])
    
    for i in range(model.W[1].shape[0]):
        for j in range(model.W[1].shape[1]):
            # W + epsilon
            model.W[1][i,j] += epsilon
            loss_plus = model.compute_loss(model.forward(X), y)
            
            # W - epsilon
            model.W[1][i,j] -= 2 * epsilon
            loss_minus = model.compute_loss(model.forward(X), y)
            
            # Restore original value
            model.W[1][i,j] += epsilon
            
            # Compute gradient
            numerical_grad[i,j] = (loss_plus - loss_minus) / (2 * epsilon)
    
    # Compare
    difference = np.linalg.norm(analytical_grad - numerical_grad)
    relative_error = difference / (np.linalg.norm(analytical_grad) + np.linalg.norm(numerical_grad))
    
    print(f"Relative error: {relative_error}")
    # Should be < 1e-7 for correct implementation
```

---

## Testing Strategy

1. **Unit tests**: Test each function individually
2. **Integration tests**: Test forward and backward pass together
3. **Overfit test**: Train on 10 samples, should reach 100% accuracy
4. **Known dataset**: Test on XOR or simple 2D data
5. **Gradient checking**: Verify backprop numerically
6. **Visualization**: Plot decision boundaries for 2D data

---

Good luck with your implementation! Remember to test each component before moving to the next one.

