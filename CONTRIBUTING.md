# Contributing Guidelines

Thank you for contributing to the Neural Network from Scratch project! This guide will help you collaborate effectively with your team.

## Table of Contents
1. [Code Style](#code-style)
2. [Git Workflow](#git-workflow)
3. [Commit Messages](#commit-messages)
4. [Code Review](#code-review)
5. [Testing](#testing)
6. [Documentation](#documentation)

## Code Style

### Python Style Guide

Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines:

- Use 4 spaces for indentation (not tabs)
- Maximum line length: 100 characters
- Use descriptive variable names
- Add docstrings to all functions

**Example:**

```python
def compute_accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Compute classification accuracy.
    
    Args:
        y_pred: Predicted labels of shape (n_samples,)
        y_true: True labels of shape (n_samples,)
        
    Returns:
        Accuracy score between 0 and 1
    """
    return np.mean(y_pred == y_true)
```

### Naming Conventions

- **Functions**: `lowercase_with_underscores`
- **Classes**: `CapitalizedWords`
- **Constants**: `UPPERCASE_WITH_UNDERSCORES`
- **Variables**: `lowercase_with_underscores`

### Type Hints

Use type hints for function arguments and return values:

```python
def forward(self, X: np.ndarray) -> np.ndarray:
    pass
```

## Git Workflow

### Branching Strategy

1. **main**: Production-ready code
2. **feature branches**: For new features
3. **bugfix branches**: For bug fixes

### Creating a New Feature

```bash
# Update main branch
git checkout main
git pull origin main

# Create feature branch
git checkout -b feature/implement-adam-optimizer

# Make your changes...

# Commit changes
git add src/optimizers.py
git commit -m "Implement Adam optimizer"

# Push to GitHub
git push origin feature/implement-adam-optimizer
```

### Creating a Pull Request

1. Push your feature branch to GitHub
2. Go to the repository on GitHub
3. Click "New Pull Request"
4. Select your feature branch
5. Add description of changes
6. Request review from teammates
7. Address review comments
8. Merge when approved

## Commit Messages

Write clear, descriptive commit messages:

### Format

```
<type>: <subject>

<body (optional)>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

### Examples

**Good:**
```
feat: implement ReLU activation function

Added ReLU activation and its derivative with proper
handling of edge cases.
```

**Bad:**
```
fixed stuff
```

### More Examples

```
feat: add Adam optimizer
fix: correct gradient calculation in backward pass
docs: add implementation guide for activation functions
test: add unit tests for sigmoid function
refactor: simplify forward propagation logic
```

## Code Review

### As a Reviewer

- Be constructive and respectful
- Explain *why* changes are needed
- Suggest alternatives when possible
- Approve if code looks good!

**Example comments:**

Good ✅:
> "This gradient calculation looks incorrect. According to the chain rule, we should multiply by the derivative of the activation function here. See line 45 in `activations.py` for reference."

Bad ❌:
> "This is wrong."

### As an Author

- Don't take feedback personally
- Ask questions if unclear
- Make requested changes promptly
- Thank reviewers!

## Testing

### Writing Tests

Write tests for all new functionality:

```python
def test_relu_with_negative_values():
    """Test that ReLU zeros out negative values."""
    x = np.array([-1, -2, -3])
    result = relu(x)
    expected = np.array([0, 0, 0])
    assert np.allclose(result, expected)
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_activations.py

# Run with verbose output
pytest tests/ -v
```

### Test Coverage

Aim for:
- All activation functions tested
- All loss functions tested
- Forward and backward pass tested
- Edge cases covered (empty input, NaN, etc.)

## Documentation

### Docstrings

Every function should have a docstring:

```python
def cross_entropy_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Compute cross-entropy loss.
    
    Args:
        y_pred: Predicted probabilities of shape (batch_size, num_classes)
        y_true: True labels (one-hot) of shape (batch_size, num_classes)
        
    Returns:
        Cross-entropy loss value (scalar)
        
    Example:
        >>> y_pred = np.array([[0.7, 0.3], [0.4, 0.6]])
        >>> y_true = np.array([[1, 0], [0, 1]])
        >>> loss = cross_entropy_loss(y_pred, y_true)
        >>> print(f"Loss: {loss:.4f}")
    """
    pass
```

### Code Comments

Use comments to explain *why*, not *what*:

**Good:**
```python
# Subtract max for numerical stability in softmax
exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
```

**Bad:**
```python
# Compute exponential
exp_x = np.exp(x)
```

### README Updates

Update README.md when:
- Adding new features
- Changing setup instructions
- Updating dependencies

## Team Communication

### When to Communicate

- Before starting a major feature
- When stuck on a problem (>30 minutes)
- Before making breaking changes
- When you find a bug in someone else's code

### Communication Channels

- **GitHub Issues**: For bugs and feature requests
- **Pull Request Comments**: For code-specific discussions
- **Team Chat**: For quick questions
- **Meetings**: For major decisions

## Workflow Example

### Scenario: Implementing the Adam Optimizer

1. **Create issue** on GitHub:
   - Title: "Implement Adam Optimizer"
   - Description: Details about what needs to be done

2. **Create branch**:
   ```bash
   git checkout -b feature/adam-optimizer
   ```

3. **Implement**:
   - Write code in `src/optimizers.py`
   - Add docstrings
   - Follow style guide

4. **Test**:
   - Write tests in `tests/test_optimizers.py`
   - Run tests locally
   - Verify all pass

5. **Commit**:
   ```bash
   git add src/optimizers.py tests/test_optimizers.py
   git commit -m "feat: implement Adam optimizer

   Added Adam optimizer with bias correction.
   Includes unit tests and documentation."
   ```

6. **Push**:
   ```bash
   git push origin feature/adam-optimizer
   ```

7. **Create Pull Request**:
   - Add description
   - Link to issue
   - Request review

8. **Address feedback**:
   - Make requested changes
   - Push updates

9. **Merge**:
   - After approval, merge to main
   - Delete feature branch

## Resolving Conflicts

### When Conflicts Occur

```bash
# Update your branch with latest main
git checkout feature/your-feature
git fetch origin
git merge origin/main

# If conflicts, Git will mark them in files
# Edit files to resolve conflicts
# Then:
git add <resolved-files>
git commit -m "Resolve merge conflicts"
git push origin feature/your-feature
```

### Preventing Conflicts

- Pull from main frequently
- Coordinate with teammates on overlapping work
- Make small, focused changes
- Merge pull requests promptly

## Questions?

If you have questions about contributing:

1. Check existing documentation
2. Ask teammates
3. Create a GitHub issue
4. Ask in team chat

---

**Remember**: We're all learning! Don't be afraid to ask questions or make mistakes. That's how we grow as developers. 🚀

