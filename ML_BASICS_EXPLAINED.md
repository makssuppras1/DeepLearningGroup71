# ML Basics Explained (For Beginners)

This document explains fundamental machine learning concepts in simple terms, using analogies and examples from our codebase.

---

## 1. Batches (Mini-Batches)

### What is a Batch?

Think of training a neural network like studying for an exam:
- **Full dataset** = All 60,000 Fashion-MNIST images (your entire textbook)
- **Batch** = A small group of images you study at once (like reading 32 pages at a time)

### Why Use Batches?

**Without batches (processing all data at once):**
- You'd need to load all 60,000 images into memory at once (expensive!)
- One mistake affects everything
- Slow to update

**With batches (processing 32 images at a time):**
- Only need memory for 32 images
- Can learn from mistakes faster
- More efficient updates

### Example from Our Code

```python
# In train.py, we create batches:
batches = create_mini_batches(X_train, y_train, batch_size=32, shuffle=True)

# If we have 60,000 training images:
# - Batch size = 32
# - Number of batches = 60,000 ÷ 32 = 1,875 batches
# - Each batch contains 32 images and their labels
```

**Visual Example:**
```
Full Dataset (60,000 images):
[████████████████████████████████████████████████████████████████]

Split into batches (batch_size=32):
[████] [████] [████] [████] ... [████]  (1,875 batches total)
  ↑      ↑      ↑      ↑            ↑
Batch 1  Batch 2 Batch 3 Batch 4  Batch 1875
```

### Key Points:
- **Batch size** = How many examples you process together (typically 32, 64, or 128)
- Smaller batches = More updates, but noisier
- Larger batches = Smoother updates, but slower

---

## 2. Epochs

### What is an Epoch?

An **epoch** = One complete pass through your entire training dataset.

**Analogy:** Like reading your entire textbook once from cover to cover.

### Example

If you have 60,000 training images and batch size = 32:
- **1 epoch** = Process all 1,875 batches (one time through all 60,000 images)
- **50 epochs** = Process all batches 50 times (read the textbook 50 times)

### Why Multiple Epochs?

**First epoch:** The model sees everything once, but doesn't learn well yet
**After 50 epochs:** The model has seen each image 50 times and learned patterns

### From Our Training Code

```python
# In train.py:
for epoch in range(config['num_epochs']):  # e.g., 50 epochs
    train_loss, train_acc = train_epoch(model, X_train, y_train, batch_size=32)
    val_loss, val_acc = evaluate(model, X_val, y_val)
    print(f"Epoch {epoch+1}/50: Train Acc: {train_acc:.4f}")
```

**Visual Timeline:**
```
Epoch 1:  [Batch 1] → [Batch 2] → ... → [Batch 1875] ✅ Done with epoch 1
Epoch 2:  [Batch 1] → [Batch 2] → ... → [Batch 1875] ✅ Done with epoch 2
...
Epoch 50: [Batch 1] → [Batch 2] → ... → [Batch 1875] ✅ Done with epoch 50
```

### Key Points:
- **1 epoch** = One complete pass through all training data
- More epochs = More learning, but risk of overfitting
- Too few epochs = Underfitting (model hasn't learned enough)
- Too many epochs = Overfitting (model memorizes training data)

---

## 3. Learning Rate

### What is Learning Rate?

**Learning rate** = How big of a step you take when updating weights

**Analogy:** Like walking down a hill to find the bottom (lowest error):
- **High learning rate** = Big steps (fast, but might overshoot the bottom)
- **Low learning rate** = Small steps (slow, but more precise)

### Visual Example

```
Finding the minimum (lowest error):

High LR (0.1):     Low LR (0.001):
    ╱╲                 ╱╲
   ╱  ╲               ╱  ╲
  ╱    ╲             ╱    ╲
 ╱      ╲           ╱      ╲
╱        ╲         ╱        ╲
→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→
Big jumps, might miss!  Small steps, precise
```

### How It Works in SGD

```python
# In optimizers.py, SGD class:
updated_params[key] = params[key] - self.learning_rate * grads[key]
#                                 ↑
#                         This controls step size!
```

**Example:**
- Current weight: `W = 5.0`
- Gradient: `dW = 2.0` (tells us to decrease W)
- Learning rate: `lr = 0.001`

**Update:**
```
New W = 5.0 - 0.001 * 2.0 = 5.0 - 0.002 = 4.998
```

### Common Learning Rates:
- **0.1** = Very high (risky, might diverge)
- **0.01** = High (for SGD)
- **0.001** = Medium (good default for Adam)
- **0.0001** = Low (very safe, but slow)

### Key Points:
- Learning rate controls **how much** weights change
- Too high = Model might not converge (bounces around)
- Too low = Model learns very slowly
- Different optimizers use different learning rates (Adam typically uses 0.001)

---

## 4. L2 Regularization

### What is L2 Regularization?

**L2 regularization** = A penalty for having large weights (prevents overfitting)

**Analogy:** Like a speed limit on a highway:
- Without regularization = No speed limit (model can memorize training data)
- With regularization = Speed limit (keeps model simple and general)

### Why We Need It?

**Problem:** Model might memorize training data instead of learning patterns
**Solution:** Penalize large weights → Forces model to be simpler

### How It Works

**Formula:**
```
Total Loss = Data Loss + L2 Penalty
L2 Penalty = λ * (sum of all weights²)
```

Where:
- **λ (lambda)** = Regularization strength (e.g., 0.0001)
- **Weights²** = Square each weight, then sum them all

### Example

```python
# In neural_network.py, compute_loss():
if self.l2_lambda > 0:
    weights = [layer.W for layer in self.layers]
    reg_loss = l2_regularization(weights, self.l2_lambda)
    total_loss = data_loss + reg_loss
```

**Visual Example:**
```
Without L2:
Weights: [10.5, -8.2, 15.3, ...]  ← Large weights (overfitting risk)

With L2 (λ=0.0001):
Weights: [2.1, -1.5, 3.2, ...]    ← Smaller weights (simpler model)
```

### How It's Applied

```python
# In neural_network.py, backward():
if self.l2_lambda > 0:
    layer.dW += self.l2_lambda * layer.W
    #              ↑
    #    Adds penalty to gradient
```

This makes the gradient push weights toward zero!

### Key Points:
- **L2 regularization** = Penalty for large weights
- **λ (l2_lambda)** = How strong the penalty is
- Higher λ = Simpler model (less overfitting, but might underfit)
- Lower λ = More complex model (better fit, but risk of overfitting)
- Typical values: 0.0001 to 0.001

---

## 5. Forward Pass

### What is a Forward Pass?

**Forward pass** = Running data through the network to get predictions

**Analogy:** Like a factory assembly line:
- Input (raw materials) → Layer 1 → Layer 2 → Layer 3 → Output (finished product)

### Step-by-Step Process

```python
# In neural_network.py, forward():
def forward(self, X):
    A = X  # Start with input
    for layer in self.layers:
        A = layer.forward(A)  # Pass through each layer
    return A  # Final output (predictions)
```

### What Happens in Each Layer?

```python
# In layers.py, DenseLayer.forward():
# Step 1: Linear transformation
Z = X @ W + b  # Matrix multiplication + bias

# Step 2: Apply activation function
A = activation(Z)  # e.g., ReLU, sigmoid, softmax
```

### Visual Example

```
Input (784 pixels):
[0.2, 0.5, 0.1, ..., 0.8]
         ↓
    Layer 1 (256 neurons)
    [ReLU activation]
         ↓
    Layer 2 (128 neurons)
    [ReLU activation]
         ↓
    Output Layer (10 classes)
    [Softmax activation]
         ↓
Predictions (probabilities):
[0.05, 0.02, 0.01, 0.80, 0.03, 0.02, 0.04, 0.01, 0.01, 0.01]
 ↑      ↑      ↑      ↑      ↑      ↑      ↑      ↑      ↑      ↑
Class 0 Class 1 Class 2 Class 3 Class 4 Class 5 Class 6 Class 7 Class 8 Class 9
                              ↑
                        80% confident it's class 3 (Dress)
```

### Key Points:
- Forward pass = **Input → Network → Output**
- Each layer transforms the data
- Last layer gives probabilities (softmax)
- No learning happens here (just computation)

---

## 6. Do We Zero Gradients?

### Short Answer: **No, we don't explicitly zero gradients in our code**

### Why Not?

In our implementation, gradients are **computed fresh** for each batch:

```python
# In layers.py, backward():
def backward(self, dA):
    # Compute gradients (overwrites previous values)
    self.dW = self.activation_cache['A_prev'].T @ dZ 
    self.db = np.sum(dZ, axis=0)
```

**What happens:**
1. Each `backward()` call computes new gradients
2. Old gradients are **overwritten** (not accumulated)
3. Weights are updated immediately after each batch

### Why This Works

```python
# Training loop:
for X_batch, y_batch in batches:
    loss = model.train_step(X_batch, y_batch)
    # ↑ This does: forward → backward → update_weights
    # Each batch gets fresh gradients!
```

### When Would You Zero Gradients?

In frameworks like PyTorch, you might accumulate gradients across multiple batches:

```python
# PyTorch example (NOT our code):
for batch in batches:
    loss = model(batch)
    loss.backward()  # Accumulates gradients
    # ... more batches ...
optimizer.step()  # Update once
optimizer.zero_grad()  # Clear for next iteration
```

**But in our code:** Each batch updates immediately, so no accumulation = no need to zero!

### Key Points:
- **Our code:** Gradients are overwritten each batch (no zeroing needed)
- **PyTorch style:** Gradients accumulate, so you zero them
- Both approaches work, just different strategies

---

## 7. SGD Class: How Parameters Are Updated

### What is SGD?

**SGD** = Stochastic Gradient Descent (the simplest optimizer)

**Stochastic** = Random (we use random batches)
**Gradient** = Direction of steepest increase
**Descent** = Going down (minimizing error)

### The Update Rule

```python
# In optimizers.py, SGD.update():
updated_params[key] = params[key] - self.learning_rate * grads[key]
#                    ↑              ↑                    ↑
#              Old weight      Step size          How wrong we are
```

**Formula:**
```
New Weight = Old Weight - Learning Rate × Gradient
```

### Step-by-Step Example

**Scenario:** We're training on one batch

**Step 1: Forward Pass**
```python
X_batch = [32 images]  # Shape: (32, 784)
y_pred = model.forward(X_batch)  # Shape: (32, 10)
# Output: Probabilities for each class
```

**Step 2: Compute Loss**
```python
loss = compute_loss(y_pred, y_true)
# Example: loss = 0.8234 (we're wrong!)
```

**Step 3: Backward Pass (Compute Gradients)**
```python
model.backward(X_batch, y_batch, y_pred)
# This computes:
# - layer1.dW = gradient for layer 1 weights
# - layer1.db = gradient for layer 1 biases
# - layer2.dW = gradient for layer 2 weights
# - etc.
```

**Step 4: Update Weights (SGD)**
```python
# In update_weights():
params = {'W1': layer1.W, 'b1': layer1.b, 'W2': layer2.W, ...}
grads = {'W1': layer1.dW, 'b1': layer1.db, 'W2': layer2.dW, ...}

# SGD.update() does:
for key in params:
    updated_params[key] = params[key] - learning_rate * grads[key]
    
# Example:
# Old W1 = 0.5
# Gradient dW1 = 0.2 (tells us to increase W1)
# Learning rate = 0.001
# New W1 = 0.5 - 0.001 * 0.2 = 0.5 - 0.0002 = 0.4998
```

**Step 5: Apply Updates**
```python
# Update each layer:
layer1.W = updated_params['W1']
layer1.b = updated_params['b1']
layer2.W = updated_params['W2']
# ... etc
```

### Visual Example

```
Before Update:
Weight: 0.5
Gradient: 0.2 (positive = should increase weight)
Learning Rate: 0.001

Update:
New Weight = 0.5 - 0.001 × 0.2 = 0.4998

Wait, that decreased! Why?

Because gradient tells us direction:
- Positive gradient = Loss increases if we increase weight
- So we DECREASE weight to reduce loss
```

### Complete Training Flow

```
Batch 1:
  Forward → Loss: 0.8234
  Backward → Gradients computed
  SGD Update → Weights changed slightly
  ✅ Done with batch 1

Batch 2:
  Forward → Loss: 0.8156 (better!)
  Backward → Gradients computed
  SGD Update → Weights changed slightly
  ✅ Done with batch 2

... (repeat for all batches)

After 1 epoch:
  Loss decreased from 0.8234 → 0.6543
  Model learned something!
```

### Key Points:
- **SGD** = Simplest optimizer (just: weight - lr × gradient)
- **Gradient** = Tells us which direction to change weights
- **Learning rate** = How big of a step to take
- We update weights **after each batch** (not after each epoch)

---

## Summary: How Everything Fits Together

### Complete Training Loop

```python
# 1. Setup
model = NeuralNetwork(...)  # Create network
epochs = 50
batch_size = 32

# 2. Training loop
for epoch in range(epochs):  # 50 epochs
    for batch in batches:  # 1,875 batches per epoch
        
        # Forward pass: Get predictions
        y_pred = model.forward(X_batch)
        
        # Compute loss
        loss = model.compute_loss(y_pred, y_batch)
        
        # Backward pass: Compute gradients
        model.backward(X_batch, y_batch, y_pred)
        
        # Update weights (SGD)
        model.update_weights()  # Uses learning_rate
        
    # After all batches, evaluate on validation set
    val_acc = evaluate(model, X_val, y_val)
```

### Key Relationships:

- **1 Epoch** = Process all batches once
- **1 Batch** = Process batch_size images together
- **Forward Pass** = Get predictions (no learning)
- **Backward Pass** = Compute gradients (how to improve)
- **SGD Update** = Change weights using gradients × learning_rate
- **L2 Regularization** = Penalty to keep weights small

---

## Quick Reference

| Concept | What It Does | Typical Value |
|---------|--------------|---------------|
| **Batch** | Group of examples processed together | 32, 64, or 128 |
| **Epoch** | One complete pass through all data | 20-100 |
| **Learning Rate** | Step size for weight updates | 0.001 (Adam), 0.01 (SGD) |
| **L2 Lambda** | Regularization strength | 0.0001 to 0.001 |
| **Forward Pass** | Compute predictions | Always happens |
| **Gradient Zeroing** | Clear accumulated gradients | Not needed in our code |
| **SGD Update** | Update weights: `W = W - lr × grad` | After each batch |

---

## Questions?

If something is still unclear, think about it in terms of:
- **Batches** = Studying small chunks at a time
- **Epochs** = Reading the whole book multiple times
- **Learning Rate** = Step size when walking down a hill
- **L2 Regularization** = Speed limit to prevent going too fast
- **Forward Pass** = Assembly line (input → output)
- **SGD** = Taking steps in the direction that reduces error

Happy learning! 🚀

    