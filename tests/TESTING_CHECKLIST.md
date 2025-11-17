# Testing Checklist

This document tracks the testing checklist items and how to run the tests.

## Checklist Items

### ✅ All activation functions return correct shapes
**Test File:** `tests/test_activations.py::TestActivationShapes`

**What it tests:**
- All activation functions (ReLU, sigmoid, tanh, softmax) preserve input shapes
- All activation derivatives preserve input shapes
- Softmax works correctly with 2D arrays (batch_size, num_classes)

**Run:** `pytest tests/test_activations.py::TestActivationShapes -v`

---

### ✅ Derivatives are numerically correct
**Test File:** `tests/test_derivatives_numerical.py`

**What it tests:**
- ReLU derivative matches numerical gradient
- Sigmoid derivative matches numerical gradient
- Tanh derivative matches numerical gradient
- MSE derivative matches numerical gradient
- Cross-entropy derivative matches numerical gradient
- Binary cross-entropy derivative matches numerical gradient
- All derivatives preserve input shapes

**Run:** `pytest tests/test_derivatives_numerical.py -v`

**Note:** Uses finite difference method to verify analytical derivatives are correct.

---

### ✅ Loss decreases with correct predictions
**Test File:** `tests/test_loss_behavior.py`

**What it tests:**
- MSE is zero for perfect predictions
- MSE increases as predictions get worse
- Cross-entropy is minimal for perfect predictions
- Cross-entropy increases as predictions get worse
- Binary cross-entropy behaves correctly
- Minimum log-likelihood behaves correctly
- All losses are non-negative
- Loss functions scale appropriately with batch size

**Run:** `pytest tests/test_loss_behavior.py -v`

---

### ⚠️ Initializations have correct variance
**Test File:** `tests/test_initializers.py`

**What it tests:**
- Random initialization returns correct shape and range
- Xavier initialization has correct variance (1/n_in)
- He initialization has correct variance (2/n_in)
- Zeros initialization returns all zeros
- Initialization variance scales correctly with input size
- Initializations are reproducible with same seed

**Run:** `pytest tests/test_initializers.py -v`

**Note:** These tests will fail until the initializer functions are implemented in `src/initializers.py`.

---

## Running All Tests

### Run all checklist tests:
```bash
python tests/run_checklist_tests.py
```

### Run individual test files:
```bash
# Activation shapes
pytest tests/test_activations.py::TestActivationShapes -v

# Numerical derivatives
pytest tests/test_derivatives_numerical.py -v

# Loss behavior
pytest tests/test_loss_behavior.py -v

# Initializers
pytest tests/test_initializers.py -v
```

### Run all tests:
```bash
pytest tests/ -v
```

---

## Test Coverage Summary

| Component | Test File | Status |
|-----------|-----------|--------|
| Activation Shapes | `test_activations.py` | ✅ Complete |
| Derivative Correctness | `test_derivatives_numerical.py` | ✅ Complete |
| Loss Behavior | `test_loss_behavior.py` | ✅ Complete |
| Initializer Variance | `test_initializers.py` | ⚠️ Ready (needs implementation) |

---

## Notes

1. **Numerical Gradient Checking**: The derivative tests use finite difference method with `epsilon=1e-7` to verify analytical derivatives. Relative error should be < 1e-5 for most functions.

2. **Initializer Tests**: The initializer tests are ready but will fail until the functions in `src/initializers.py` are implemented. Once implemented, the tests will verify:
   - Correct variance formulas
   - Proper shape handling
   - Reproducibility with seeds

3. **Loss Function Tests**: These tests verify that loss functions behave correctly:
   - Perfect predictions → minimal loss
   - Worse predictions → higher loss
   - Non-negative values
   - Proper scaling with batch size

---

## Team Tasks

- [ ] Daily stand-up meetings (15 min)
- [ ] Code reviews for each component
- [ ] Test all components together
- [ ] Document any issues encountered

---

## Next Steps

1. Implement initializer functions in `src/initializers.py`
2. Run all tests to verify everything works
3. Test components together (integration tests)
4. Document any issues found during testing

