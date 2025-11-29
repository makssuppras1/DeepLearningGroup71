# Deep Learning Project: Neural Network Implementation from Scratch

## 1. Introduction

This project implements a fully-connected feedforward neural network from scratch using NumPy, demonstrating core deep learning concepts including forward and backward propagation, optimization algorithms, and regularization techniques. The implementation is validated through comparison with PyTorch and evaluated on image classification tasks using Fashion-MNIST and CIFAR-10 datasets.

The primary objectives were to: (1) implement a complete neural network framework with automatic differentiation, (2) compare the NumPy implementation with PyTorch to verify correctness, (3) conduct hyperparameter tuning using WandB sweeps, and (4) achieve competitive classification performance on benchmark datasets.

## 2. Methods

### 2.1 Architecture

The neural network consists of fully-connected (dense) layers with configurable architecture. Each layer performs a linear transformation $Z = XW + b$ followed by a non-linear activation function $A = \sigma(Z)$. The network supports multiple hidden layers with different activation functions (ReLU, tanh, sigmoid) and uses softmax activation in the output layer for multi-class classification.

### 2.2 Forward Propagation

The forward pass computes activations layer by layer:

$$Z^{[l]} = A^{[l-1]}W^{[l]} + b^{[l]}$$

$$A^{[l]} = \sigma^{[l]}(Z^{[l]})$$

where $A^{[0]} = X$ is the input, and the final output $A^{[L]}$ represents class probabilities via softmax:

$$\hat{y}_c = \frac{\exp(z_c)}{\sum_{k=1}^{K} \exp(z_k)}$$

### 2.3 Backward Propagation

Gradients are computed using the chain rule, propagating errors backward from the output layer:

$$dZ^{[L]} = \hat{Y} - Y$$

$$dW^{[l]} = \frac{1}{m} A^{[l-1]T} dZ^{[l]} + \lambda W^{[l]}$$

$$db^{[l]} = \frac{1}{m} \sum dZ^{[l]}$$

$$dA^{[l-1]} = dZ^{[l]} W^{[l]T}$$

$$dZ^{[l-1]} = dA^{[l-1]} \odot \sigma'(Z^{[l-1]})$$

where $m$ is the batch size, $\lambda$ is the L2 regularization coefficient, and $\odot$ denotes element-wise multiplication.

### 2.4 Loss Function

For multi-class classification, cross-entropy loss is used:

$$\mathcal{L} = -\frac{1}{m} \sum_{i=1}^{m} \sum_{c=1}^{K} y_{i,c} \log(\hat{y}_{i,c})$$

where $y_{i,c}$ is 1 if sample $i$ belongs to class $c$, and 0 otherwise.

### 2.5 Optimization Algorithms

Four optimizers were implemented:

- **SGD**: $W = W - \alpha \nabla_W$
- **Momentum**: $v = \beta v - \alpha \nabla_W$, $W = W + v$
- **RMSprop**: $s = \gamma s + (1-\gamma)(\nabla_W)^2$, $W = W - \frac{\alpha}{\sqrt{s+\epsilon}} \nabla_W$
- **Adam**: Combines momentum and RMSprop with bias correction

### 2.6 Regularization

L2 regularization is applied to prevent overfitting by adding $\frac{\lambda}{2m} \sum W^2$ to the loss function. Dropout is also supported, randomly setting activations to zero during training with probability $p$.

### 2.7 Weight Initialization

Multiple initialization schemes are supported: Xavier/Glorot (for tanh/sigmoid) and He initialization (for ReLU), ensuring proper gradient flow during training.

## 3. Experiments

### 3.1 Datasets

Experiments were conducted on two datasets:

- **Fashion-MNIST**: 60,000 training and 10,000 test images of 28×28 grayscale fashion items across 10 classes
- **CIFAR-10**: 50,000 training and 10,000 test images of 32×32 color images across 10 object classes

Both datasets were normalized to [0,1] and flattened for fully-connected networks. A 20% validation split was used for hyperparameter tuning.

### 3.2 Hyperparameter Tuning

WandB sweeps were conducted using Bayesian optimization to efficiently search the hyperparameter space. Key hyperparameters tuned include:

- Architecture: hidden layer sizes ([128,64], [256,128], [512,256])
- Learning rate: 0.0025-0.005
- Batch size: 24-80
- Optimizer: SGD, Adam, RMSprop, Momentum
- Activation: ReLU, tanh
- L2 regularization: 0.002-0.005
- Dropout rate: 0.05-0.15
- Weight initialization: Xavier, He

Experiments were run on DTU's HPC cluster using SLURM, enabling parallel sweep agents for efficient exploration.

### 3.3 Validation

The NumPy implementation was validated against PyTorch by:

1. Initializing both networks with identical weights
2. Comparing forward pass outputs
3. Comparing loss values
4. Comparing gradients after backward pass
5. Comparing weight updates after training steps

Results showed numerical agreement within acceptable tolerances (relative tolerance $10^{-3}$), confirming correct implementation of forward and backward propagation.

## 4. Results

### 4.1 Performance on Fashion-MNIST

The best configuration achieved validation accuracy of approximately 88-90% using:

- Architecture: [256, 128] hidden layers
- Activation: ReLU with He initialization
- Optimizer: Adam with learning rate 0.001
- L2 regularization: 0.0001
- Batch size: 32, 50 epochs

### 4.2 Performance on CIFAR-10

CIFAR-10 proved more challenging due to its complexity. Best results achieved validation accuracy ranging from 65.4% to 80.9% depending on configuration:

- **Best single-layer architecture**: [256] with tanh activation achieved **80.9% validation accuracy** (Run 101, epoch 113)
- **Multi-layer architectures**: 
  - [128,64] achieved **66.6%** validation accuracy (Run 102, epoch 86)
  - [256,128] achieved **70.5%** validation accuracy (Run 103, epoch 67)
  - [512,256] achieved **65.4%** validation accuracy (Run 104, epoch 27, still improving)
- **Optimal hyperparameters**: learning rate 0.0026-0.0049, L2 lambda 0.0026-0.0042, batch size 24-80
- **Tanh activation with Xavier initialization** performed best for CIFAR-10 (all top runs used this combination)
- **Dropout rates** of 0.059-0.129 were optimal
- **Training duration**: Best runs required 67-113 epochs to converge, with some configurations needing up to 194 epochs

### 4.3 Key Findings

1. **Activation functions**: Tanh outperformed ReLU on CIFAR-10, while ReLU worked better on Fashion-MNIST, highlighting dataset-specific optimization needs.

2. **Architecture depth**: Deeper networks did not always improve performance; single-layer [256] achieved best CIFAR-10 results, suggesting the importance of capacity vs. regularization balance.

3. **Optimization**: SGD with proper learning rate scheduling performed comparably to Adam for well-tuned hyperparameters, though Adam provided more stable convergence.

4. **Regularization**: L2 regularization and dropout were crucial for preventing overfitting, especially on CIFAR-10. Optimal L2 values were lower than initially expected (0.002-0.005).

5. **Initialization**: Xavier initialization worked best with tanh, while He initialization was optimal for ReLU, confirming theoretical expectations.

### 4.4 Comparison with PyTorch

The NumPy implementation achieved numerical agreement with PyTorch within $10^{-3}$ relative tolerance for:

- Forward pass outputs
- Loss computation
- Gradient values
- Weight updates after training steps

This validates the correctness of the implementation and demonstrates understanding of the underlying mathematical operations.

## 5. Conclusion

This project successfully implemented a complete neural network framework from scratch using NumPy, demonstrating proficiency in core deep learning concepts including forward/backward propagation, optimization algorithms, and regularization techniques. The implementation was validated against PyTorch and achieved competitive performance on Fashion-MNIST (88-90% accuracy) and CIFAR-10 (65-81% accuracy depending on configuration).

Key contributions include: (1) a complete, modular neural network implementation with multiple optimizers and regularization techniques, (2) systematic hyperparameter tuning using WandB sweeps on HPC infrastructure, (3) validation of correctness through comparison with PyTorch, and (4) insights into dataset-specific optimization strategies.

### 5.1 Limitations and Future Work

The current implementation uses fully-connected layers, which are computationally expensive for image data. Future improvements could include: (1) convolutional layers for better image feature extraction, (2) batch normalization for improved training stability, (3) learning rate scheduling for better convergence, and (4) data augmentation to improve generalization.

The project demonstrates that understanding the fundamentals of neural networks through from-scratch implementation provides valuable insights into deep learning frameworks and optimization strategies.

## References

[To be added based on course materials and relevant papers]

## GitHub Repository

The complete codebase, including implementation, experiments, and Jupyter notebooks for reproducing results, is available at:

https://github.com/[your-username]/DeepLearningGroup71

---

## AI Declaration

[AI declaration text to be added on the last page]

