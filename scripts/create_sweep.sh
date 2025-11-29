#!/bin/bash
# Script to create a WandB sweep on HPC
# Usage: ./create_sweep.sh <config_name> [project_name] [entity]
# Example: ./create_sweep.sh random neural-network-numpy

CONFIG_NAME=$1
PROJECT_NAME=${2:-"neural-network-numpy"}
ENTITY=${3:-""}  # Optional WandB entity (e.g., 'makssuppras1-danmarks-tekniske-universitet-dtu')

if [ -z "$CONFIG_NAME" ]; then
    echo "ERROR: Config name required!"
    echo ""
    echo "Usage: ./create_sweep.sh <config_name> [project_name] [entity]"
    echo ""
    echo "Available configs:"
    echo "  - random    : Random search (explores wide hyperparameter space)"
    echo "  - bayes     : Bayesian optimization (efficient search)"
    echo "  - activations : Grid search for activation functions"
    echo ""
    echo "Examples:"
    echo "  ./create_sweep.sh random"
    echo "  ./create_sweep.sh bayes neural-network-numpy"
    echo "  ./create_sweep.sh random my-project my-entity"
    exit 1
fi

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
else
    echo "ERROR: Virtual environment not found at $PROJECT_DIR/deeplearning/bin"
    echo "Please create it first: python3 -m venv deeplearning"
    exit 1
fi

# Check WandB login
if ! wandb status &>/dev/null; then
    echo "WARNING: WandB not logged in. Run 'wandb login' first."
    exit 1
fi

echo "Creating WandB sweep..."
echo "Config: $CONFIG_NAME"
echo "Project: $PROJECT_NAME"
if [ -n "$ENTITY" ]; then
    echo "Entity: $ENTITY"
    export WANDB_ENTITY="$ENTITY"
fi
echo ""

# Run Python script to create sweep
python experiments/sweep_config.py "$CONFIG_NAME" "$PROJECT_NAME"

echo ""
echo "To run sweep agents, use:"
echo "  sbatch scripts/submit_sweep_agent.sh <sweep-id> [count]"
echo ""
echo "To run multiple agents in parallel:"
echo "  for i in {1..5}; do sbatch scripts/submit_sweep_agent.sh <sweep-id> 10; done"

