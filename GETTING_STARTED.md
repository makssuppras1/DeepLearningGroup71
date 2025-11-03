# Getting Started Guide

Welcome to the Neural Network from Scratch project! This guide will help you get started with the project, especially if you're new to GitHub, coding, or AI.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Setting Up Your Environment](#setting-up-your-environment)
3. [Understanding the Project Structure](#understanding-the-project-structure)
4. [Your First Steps](#your-first-steps)
5. [Working with Git and GitHub](#working-with-git-and-github)
6. [Common Issues and Solutions](#common-issues-and-solutions)
7. [Learning Resources](#learning-resources)

## Prerequisites

Before starting, make sure you have:
- Python 3.8 or higher installed
- A code editor (VS Code, PyCharm, or Jupyter Notebook)
- Basic understanding of Python programming
- A GitHub account

## Setting Up Your Environment

### Step 1: Clone the Repository

```bash
# Navigate to where you want to store the project
cd ~/Documents

# Clone the repository (if working with GitHub)
git clone <your-repository-url>

# Navigate into the project directory
cd DeepLearningGroup71
```

### Step 2: Create a Virtual Environment

A virtual environment keeps your project dependencies isolated.

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` at the beginning of your terminal prompt.

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs all necessary packages including NumPy, matplotlib, and Weights & Biases.

### Step 4: Set Up Weights & Biases

```bash
wandb login
```

You'll be prompted to enter your API key. Get it from https://wandb.ai/authorize

## Understanding the Project Structure

```
DeepLearningGroup71/
├── src/                    # Source code (where you'll implement)
│   ├── neural_network.py   # Main neural network class
│   ├── activations.py      # Activation functions (ReLU, Sigmoid, etc.)
│   ├── losses.py           # Loss functions
│   ├── optimizers.py       # Optimization algorithms
│   └── ...
├── experiments/            # Scripts to run experiments
│   ├── train.py           # Main training script
│   └── evaluate.py        # Evaluation script
├── notebooks/             # Jupyter notebooks for exploration
├── data/                  # Dataset storage (created automatically)
├── results/               # Saved models and plots
└── configs/               # Configuration files
```

## Your First Steps

### Step 1: Explore the Data

Start with the first notebook to understand the datasets:

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

This notebook will guide you through:
- Loading Fashion-MNIST and CIFAR-10
- Visualizing sample images
- Understanding data shapes and distributions

### Step 2: Implement Activation Functions

Open `src/activations.py` and implement the activation functions. Start with ReLU:

```python
def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def relu_derivative(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)
```

Test your implementations in `notebooks/02_model_testing.ipynb`

### Step 3: Build the Neural Network

Follow this order:
1. **activations.py** - Implement activation functions
2. **initializers.py** - Implement weight initialization
3. **losses.py** - Implement loss functions
4. **optimizers.py** - Implement optimizers (start with SGD)
5. **neural_network.py** - Put it all together

### Step 4: Train Your First Model

Once you've implemented the basics:

```bash
python experiments/train.py
```

This will train a model and log results to WandB.

## Working with Git and GitHub

### Basic Git Workflow

1. **Check status** - See what files have changed:
   ```bash
   git status
   ```

2. **Stage changes** - Add files you want to commit:
   ```bash
   git add src/activations.py
   # Or add all changes:
   git add .
   ```

3. **Commit changes** - Save your changes with a message:
   ```bash
   git commit -m "Implemented ReLU activation function"
   ```

4. **Push to GitHub** - Upload your changes:
   ```bash
   git push origin main
   ```

5. **Pull latest changes** - Download updates from teammates:
   ```bash
   git pull origin main
   ```

### Working in a Team

- **Create branches** for new features:
  ```bash
  git checkout -b feature/implement-adam-optimizer
  ```

- **Merge branches** when feature is complete:
  ```bash
  git checkout main
  git merge feature/implement-adam-optimizer
  ```

- **Resolve conflicts** - If two people edit the same file, you'll need to manually resolve conflicts. Your editor will show the conflicting sections.

## Common Issues and Solutions

### Issue: "Module not found" error

**Solution:** Make sure you're in the project directory and the virtual environment is activated:
```bash
cd DeepLearningGroup71
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### Issue: Dimension mismatch in matrix operations

**Solution:** Use `print(array.shape)` frequently to debug. Common issues:
- Forgetting to transpose matrices
- Mixing row and column vectors
- Batch dimensions not matching

### Issue: NaN or Inf in loss

**Solution:** This usually means:
- Learning rate is too high (try 0.001 or lower)
- Numerical instability in softmax/log (add small epsilon: `np.log(x + 1e-8)`)
- Exploding gradients (check your backprop implementation)

### Issue: Model not learning

**Solution:** Check:
- Are gradients being computed correctly? (Use gradient checking)
- Is the loss decreasing at all?
- Try a smaller learning rate
- Make sure you're updating weights after computing gradients

## Learning Resources

### NumPy Basics
- [NumPy Quickstart Tutorial](https://numpy.org/doc/stable/user/quickstart.html)
- [100 NumPy Exercises](https://github.com/rougier/numpy-100)

### Neural Networks Theory
- [3Blue1Brown Neural Networks Series](https://www.youtube.com/watch?v=aircAruvnKk) (Videos)
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) (Online book)
- [Stanford CS231n](http://cs231n.stanford.edu/) (Course notes)

### Mathematics
- Matrix multiplication: Dimensions must match `(m, n) @ (n, p) = (m, p)`
- Backpropagation: Chain rule of calculus
- Vector/matrix derivatives

### Git and GitHub
- [GitHub Git Handbook](https://guides.github.com/introduction/git-handbook/)
- [Git Cheat Sheet](https://education.github.com/git-cheat-sheet-education.pdf)

### Weights & Biases
- [WandB Quickstart](https://docs.wandb.ai/quickstart)
- [WandB Python Library](https://docs.wandb.ai/guides/track)

## Implementation Tips

### 1. Start Small
- Test with tiny datasets (10-100 samples) first
- Use small networks (1 hidden layer, 8 units)
- Make sure the model can overfit a small dataset (this validates your implementation)

### 2. Debug Systematically
- Test each component individually
- Use gradient checking to verify backpropagation
- Print shapes frequently during development

### 3. Vectorize Everything
- Avoid Python loops when possible
- Use NumPy broadcasting
- Process mini-batches, not individual samples

### 4. Normalize Your Data
- Scale pixel values to [0, 1]
- Or standardize: `(x - mean) / std`

### 5. Monitor Training
- Plot loss curves
- Check if training loss decreases
- Compare train vs validation loss (to detect overfitting)

## Getting Help

- **Documentation**: Check function docstrings in the code
- **Team**: Ask your teammates (use Discord, Slack, or GitHub Issues)
- **Debugging**: Use `print()` statements liberally
- **Stack Overflow**: Search for error messages
- **Office Hours**: Ask your instructor or TA

## Next Steps

1. Complete `01_data_exploration.ipynb`
2. Implement activation functions
3. Test implementations in `02_model_testing.ipynb`
4. Implement the neural network class
5. Run your first training experiment
6. Conduct hyperparameter sweeps
7. Analyze results in `03_results_analysis.ipynb`

Good luck! Remember: everyone struggles with implementing neural networks from scratch. Be patient, test frequently, and don't hesitate to ask for help.

---

**Note:** This is a learning project. The goal is to understand the mathematics and algorithms behind neural networks. Take your time and make sure you understand each component before moving to the next.

