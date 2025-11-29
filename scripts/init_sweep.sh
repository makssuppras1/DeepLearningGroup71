#!/bin/bash
# Initialize/create a WandB sweep without starting agents
# Usage: ./scripts/init_sweep.sh [config_file]

CONFIG_FILE=${1:-configs/fashion_mnist_sweep.yaml}

if [ -z "$BLACKHOLE" ]; then
    echo "ERROR: BLACKHOLE environment variable is not set!"
    echo "This script should be run on HPC."
    exit 1
fi

PROJECT_DIR="$BLACKHOLE/DeepLearningGroup71"
cd "$PROJECT_DIR" || exit 1

# Activate virtual environment
if [ -d "deeplearning/bin" ]; then
    source deeplearning/bin/activate
    echo "Virtual environment activated"
else
    echo "ERROR: Virtual environment not found at $PROJECT_DIR/deeplearning/bin"
    exit 1
fi

# Set WandB project
export WANDB_PROJECT="neural-network-numpy"
export WANDB_ENTITY="makssuppras1-danmarks-tekniske-universitet-dtu"

# Check WandB login
if ! wandb status &>/dev/null; then
    echo "WARNING: WandB not logged in. Run 'wandb login' first."
    echo "Continuing anyway - WandB will prompt for login if needed."
fi

echo "=========================================="
echo "Initializing WandB Sweep"
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo "Project: $WANDB_PROJECT"
echo "Entity: $WANDB_ENTITY"
echo ""

# Create sweep
echo "Creating WandB sweep..."
SWEEP_OUTPUT=$(wandb sweep "$CONFIG_FILE" 2>&1)
echo "$SWEEP_OUTPUT"

# Extract sweep ID - try multiple patterns
SWEEP_ID=""

# Pattern 1: "wandb: Creating sweep with ID: xxxxxxxx"
SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep -oP 'wandb: Creating sweep with ID: \K[^\s]+' 2>/dev/null || echo "")

# Pattern 2: "sweep/xxxxxxxx" in URL
if [ -z "$SWEEP_ID" ]; then
    SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep -oP 'sweep/\K[^\w]*[a-z0-9]+' | head -1 | tr -d '/' 2>/dev/null || echo "")
fi

# Pattern 3: Extract from "Run sweep agent with: wandb agent entity/project/xxxxxxxx"
if [ -z "$SWEEP_ID" ]; then
    SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep -oP 'wandb agent [^/]+/[^/]+/\K[a-z0-9]+' | head -1 2>/dev/null || echo "")
fi

# Pattern 4: Look for alphanumeric ID after "sweep" keyword
if [ -z "$SWEEP_ID" ]; then
    SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep -i sweep | grep -oP '[a-z0-9]{8,}' | head -1 2>/dev/null || echo "")
fi

if [ -z "$SWEEP_ID" ]; then
    echo ""
    echo "ERROR: Could not extract sweep ID from wandb output!"
    echo "Full output was:"
    echo "$SWEEP_OUTPUT"
    echo ""
    echo "Please manually extract the sweep ID from the output above."
    exit 1
fi

SWEEP_PATH="$WANDB_ENTITY/$WANDB_PROJECT/$SWEEP_ID"
echo ""
echo "=========================================="
echo "✅ Sweep created successfully!"
echo "=========================================="
echo "Sweep ID: $SWEEP_ID"
echo "Sweep Path: $SWEEP_PATH"
echo ""
echo "View sweep at:"
echo "  https://wandb.ai/$SWEEP_PATH"
echo ""
echo "To start agents, run:"
echo "  wandb agent $SWEEP_PATH --count 10"
echo ""
echo "Or use the setup script:"
echo "  ./scripts/setup_fashion_mnist_sweep.sh 10 10"
echo ""
echo "Save this sweep ID: $SWEEP_ID"

