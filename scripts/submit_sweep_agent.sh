#!/bin/bash
#SBATCH --job-name=wandb_sweep
#SBATCH --time=24:00:00          # Maximum runtime: 24 hours (longer for sweeps)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8        # More CPUs for faster training
#SBATCH --mem=16G                 # More memory for larger models
#SBATCH --output=slurm-sweep-%j.out
#SBATCH --error=slurm-sweep-%j.err

# Usage: sbatch submit_sweep_agent.sh <sweep-id> [count] [entity]
# Example: sbatch submit_sweep_agent.sh abc123def456 10

SWEEP_ID=$1
COUNT=${2:-10}  # Default to 10 runs
ENTITY=${3:-""}  # Optional WandB entity (e.g., 'makssuppras1-danmarks-tekniske-universitet-dtu')

if [ -z "$SWEEP_ID" ]; then
    echo "ERROR: Sweep ID required!"
    echo "Usage: sbatch submit_sweep_agent.sh <sweep-id> [count] [entity]"
    echo "Example: sbatch submit_sweep_agent.sh abc123def456 10"
    echo "Example with entity: sbatch submit_sweep_agent.sh abc123def456 10 makssuppras1-danmarks-tekniske-universitet-dtu"
    exit 1
fi

if [ -z "$BLACKHOLE" ]; then
    echo "ERROR: BLACKHOLE environment variable is not set!"
    echo "This script should be run on HPC."
    exit 1
fi

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

PROJECT_DIR="$BLACKHOLE/DeepLearningGroup71"
cd "$PROJECT_DIR" || exit 1

# Activate virtual environment
if [ -d "deeplearning/bin" ]; then
    source deeplearning/bin/activate
    echo "Virtual environment activated"
else
    echo "ERROR: Virtual environment not found at $PROJECT_DIR/deeplearning/bin"
    echo "Please create it first: python3 -m venv deeplearning"
    exit 1
fi

# Check WandB login
if ! wandb status &>/dev/null; then
    echo "WARNING: WandB not logged in. Run 'wandb login' first."
    echo "Continuing anyway - WandB will prompt for login if needed."
fi

echo ""
echo "Running WandB sweep agent"
echo "Sweep ID: $SWEEP_ID"
echo "Runs: $COUNT"
if [ -n "$ENTITY" ]; then
    echo "Entity: $ENTITY"
    export WANDB_ENTITY="$ENTITY"
fi
echo "Project directory: $PROJECT_DIR"
echo ""

# Set WandB entity if provided
if [ -n "$ENTITY" ]; then
    export WANDB_ENTITY="$ENTITY"
    echo "Using WandB entity: $ENTITY"
fi

# Run sweep agent
# For 'program' mode sweeps, WandB will execute the program specified in sweep config
# For 'function' mode sweeps, use the wrapper script
wandb agent "$SWEEP_ID" --count "$COUNT"

echo ""
echo "=========================================="
echo "Sweep agent completed at: $(date)"
echo "=========================================="


