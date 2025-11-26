#!/bin/bash
# Run WandB sweep agent directly (without job scheduler)
# Usage: ./run_sweep_agent_direct.sh <sweep-id> [count]
# Example: ./run_sweep_agent_direct.sh n6xelg7m 10

SWEEP_ID=$1
COUNT=${2:-10}  # Default to 10 runs

if [ -z "$SWEEP_ID" ]; then
    echo "ERROR: Sweep ID required!"
    echo "Usage: ./run_sweep_agent_direct.sh <sweep-id> [count]"
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

echo "Starting WandB sweep agent"
echo "Sweep ID: $SWEEP_ID"
echo "Runs: $COUNT"
echo "Log file: sweep_agent_${SWEEP_ID}.log"
echo ""

# Run agent in background, redirect output to log file
nohup wandb agent "$SWEEP_ID" --count "$COUNT" > "sweep_agent_${SWEEP_ID}_$(date +%s).log" 2>&1 &

AGENT_PID=$!
echo "Agent started with PID: $AGENT_PID"
echo "Monitor with: tail -f sweep_agent_${SWEEP_ID}_*.log"
echo "Check if running: ps aux | grep $AGENT_PID"

