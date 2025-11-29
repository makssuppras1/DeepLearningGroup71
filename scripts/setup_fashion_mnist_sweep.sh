#!/bin/bash
# Setup and run Fashion-MNIST sweep on HPC
# Usage: ./setup_fashion_mnist_sweep.sh [num_agents] [runs_per_agent]

NUM_AGENTS=${1:-5}      # Default: 5 agents
RUNS_PER_AGENT=${2:-10} # Default: 10 runs per agent

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
echo "Fashion-MNIST Sweep Setup"
echo "=========================================="
echo "Project: $WANDB_PROJECT"
echo "Entity: $WANDB_ENTITY"
echo "Config: configs/fashion_mnist_sweep.yaml"
echo ""

# Kill any old agents first
echo "Stopping any existing sweep agents..."
pkill -f "wandb agent" 2>/dev/null
sleep 2

# Create sweep
echo "Creating WandB sweep..."
SWEEP_OUTPUT=$(wandb sweep configs/fashion_mnist_sweep.yaml 2>&1)
SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep -oP 'wandb: Creating sweep with ID: \K[^\s]+' || echo "")

if [ -z "$SWEEP_ID" ]; then
    # Try alternative pattern
    SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep -oP 'sweep/\K[^\s]+' | head -1 || echo "")
fi

if [ -z "$SWEEP_ID" ]; then
    echo "ERROR: Could not extract sweep ID from output:"
    echo "$SWEEP_OUTPUT"
    exit 1
fi

SWEEP_PATH="$WANDB_ENTITY/$WANDB_PROJECT/$SWEEP_ID"
echo ""
echo "✅ Sweep created successfully!"
echo "Sweep ID: $SWEEP_ID"
echo "Sweep Path: $SWEEP_PATH"
echo "View at: https://wandb.ai/$SWEEP_PATH"
echo ""

# Start agents
echo "Starting $NUM_AGENTS agents (each running $RUNS_PER_AGENT runs)..."
for i in $(seq 1 $NUM_AGENTS); do
    LOG_FILE="sweep_agent_fashion_mnist_${i}.log"
    nohup wandb agent "$SWEEP_PATH" --count "$RUNS_PER_AGENT" > "$LOG_FILE" 2>&1 &
    echo "Started agent $i (PID: $!, log: $LOG_FILE)"
    sleep 1  # Small delay between starts
done

echo ""
echo "=========================================="
echo "Sweep agents started!"
echo "=========================================="
echo "Monitor logs:"
echo "  tail -f sweep_agent_fashion_mnist_*.log"
echo ""
echo "Check running agents:"
echo "  ps aux | grep 'wandb agent' | grep -v grep"
echo ""
echo "Stop all agents:"
echo "  pkill -f 'wandb agent'"
echo ""
echo "View sweep dashboard:"
echo "  https://wandb.ai/$SWEEP_PATH"
echo ""

# Wait a moment and show status
sleep 3
echo "Current agent status:"
ps aux | grep "wandb agent" | grep -v grep | wc -l | xargs echo "Running agents:"

