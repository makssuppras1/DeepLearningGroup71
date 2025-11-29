# CIFAR-10 Accuracy Diagnosis

## Problem
You're getting ≤50% test accuracy on CIFAR-10, which is below the expected range of 65-81% based on your report.

## Key Issues Found

### 1. **Wrong Activation Function** ❌
- **Current**: ReLU with He initialization
- **Best for CIFAR-10**: Tanh with Xavier initialization
- **Impact**: Your report shows tanh achieved 80.9% vs ReLU's lower performance

### 2. **Learning Rate Too Low** ❌
- **Current default**: 0.0003
- **Optimal range**: 0.0026-0.0049 (8-16x higher!)
- **Impact**: Model learns too slowly, may not converge properly

### 3. **L2 Regularization Too Weak** ❌
- **Current default**: 0.0001
- **Optimal range**: 0.002-0.005 (20-50x higher!)
- **Impact**: Insufficient regularization, model may overfit or underfit

### 4. **Architecture May Be Too Complex** ⚠️
- **Current default**: [1024, 512, 256]
- **Best result**: Single layer [256] achieved 80.9%
- **Impact**: Deeper networks may be harder to train without proper tuning

### 5. **Optimizer Choice** ⚠️
- **Current**: Adam
- **Best for CIFAR-10**: SGD with proper learning rate
- **Impact**: Adam may not work as well with the current hyperparameters

## Recommended Configuration

Based on your best results (Run 101: 80.9% validation accuracy):

```python
config = {
    'dataset': 'cifar10',
    'hidden_layers': [256],  # Single layer worked best!
    'output_size': 10,
    'activation': 'tanh',  # NOT ReLU!
    'output_activation': 'softmax',
    'num_epochs': 150,  # Best runs needed 67-113 epochs
    'batch_size': 64,  # Optimal range: 24-80
    'learning_rate': 0.003,  # Optimal range: 0.0026-0.0049
    'optimizer': 'sgd',  # SGD worked better than Adam
    'l2_lambda': 0.003,  # Optimal range: 0.002-0.005
    'weight_init': 'xavier',  # For tanh activation!
    'dropout_rate': 0.1,  # Optimal range: 0.059-0.129
    'val_split': 0.2,
    'random_seed': 42
}
```

## Quick Fix Script

Run this to test the optimal configuration:

```python
# In your notebook or as a script
from experiments.train_simple import train, get_default_config

# Get default config
config = get_default_config()

# Override with optimal CIFAR-10 settings
config.update({
    'hidden_layers': [256],  # Single layer
    'activation': 'tanh',  # Tanh, not ReLU!
    'weight_init': 'xavier',  # Xavier for tanh
    'learning_rate': 0.003,  # Much higher!
    'l2_lambda': 0.003,  # Much higher!
    'optimizer': 'sgd',  # SGD instead of Adam
    'dropout_rate': 0.1,
    'num_epochs': 150
})

train(config)
```

## Why These Changes Matter

1. **Tanh + Xavier**: Better gradient flow for CIFAR-10's complex patterns
2. **Higher Learning Rate**: CIFAR-10 needs faster learning to escape local minima
3. **Stronger L2**: Prevents overfitting on the more complex dataset
4. **Simpler Architecture**: Single layer [256] reduces overfitting risk
5. **SGD**: More stable with the higher learning rates needed

## Expected Results

With these changes, you should see:
- **Validation accuracy**: 65-81% (depending on exact hyperparameters)
- **Test accuracy**: Should match validation accuracy closely
- **Training time**: 67-113 epochs to converge

## Additional Debugging Steps

If accuracy is still low after these changes:

1. **Check data preprocessing**: Verify images are normalized to [0,1] and labels are one-hot encoded
2. **Monitor training curves**: Look for signs of overfitting (train acc >> val acc) or underfitting (both low)
3. **Check gradient flow**: Ensure gradients aren't vanishing/exploding
4. **Verify loss function**: Cross-entropy should decrease steadily
5. **Check random seed**: Ensure reproducibility

