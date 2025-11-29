#!/bin/bash
# Test script to diagnose why sweep agents aren't running
# Usage: ./test_sweep_run.sh [sweep_path]

SWEEP_PATH=${1:-""}

if [ -z "$SWEEP_PATH" ]; then
    echo "Usage: ./test_sweep_run.sh <sweep_path>"
    echo "Example: ./test_sweep_run.sh entity/project/sweep_id"
    exit 1
fi

if [ -z "$BLACKHOLE" ]; then
    echo "ERROR: BLACKHOLE environment variable is not set!"
    exit 1
fi

PROJECT_DIR="$BLACKHOLE/DeepLearningGroup71"
cd "$PROJECT_DIR" || exit 1

# Activate virtual environment
if [ -d "deeplearning/bin" ]; then
    source deeplearning/bin/activate
else
    echo "ERROR: Virtual environment not found"
    exit 1
fi

export WANDB_PROJECT="neural-network-numpy"
export WANDB_ENTITY="makssuppras1-danmarks-tekniske-universitet-dtu"

echo "=========================================="
echo "Testing Sweep Run"
echo "=========================================="
echo "Sweep Path: $SWEEP_PATH"
echo "Project: $WANDB_PROJECT"
echo "Entity: $WANDB_ENTITY"
echo ""

# Test 1: Check if we can import the training script
echo "Test 1: Checking imports..."
python -c "
import sys
sys.path.insert(0, '.')
try:
    from experiments.train_simple import main, get_default_config
    print('✓ Imports successful')
    config = get_default_config()
    print(f'✓ Default config loaded: {len(config)} parameters')
except Exception as e:
    print(f'✗ Import failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "Import test failed!"
    exit 1
fi

echo ""

# Test 2: Check if data loader works
echo "Test 2: Checking data loader..."
python -c "
import sys
sys.path.insert(0, '.')
from src.data_loader import download_fashion_mnist, load_fashion_mnist
from src.hpc_utils import get_data_dir, get_project_root
import os

project_root = get_project_root()
data_dir = get_data_dir(project_root)
print(f'Data directory: {data_dir}')

try:
    print('Checking/downloading Fashion-MNIST...')
    download_fashion_mnist(data_dir)
    print('✓ Download successful')
    
    print('Loading data...')
    X_train, y_train, X_test, y_test = load_fashion_mnist(data_dir)
    print(f'✓ Data loaded: train={X_train.shape}, test={X_test.shape}')
except Exception as e:
    print(f'✗ Data loading failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "Data loading test failed!"
    exit 1
fi

echo ""

# Test 3: Try running a single sweep run with verbose output
echo "Test 3: Running a single sweep run (this will show any errors)..."
echo "Running: wandb agent $SWEEP_PATH --count 1"
echo ""

wandb agent "$SWEEP_PATH" --count 1

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Sweep run completed successfully!"
else
    echo "✗ Sweep run failed with exit code: $EXIT_CODE"
    echo ""
    echo "Check the output above for error messages."
fi
echo "=========================================="

exit $EXIT_CODE

