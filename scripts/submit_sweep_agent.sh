#!/bin/bash
#SBATCH --job-name=wandb_sweep
#SBATCH --time=12:00:00          # Maximum runtime: 12 hours
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=slurm-sweep-%j.out
#SBATCH --error=slurm-sweep-%j.err

# Usage: sbatch submit_sweep_agent.sh <sweep-id> [count]
# Example: sbatch submit_sweep_agent.sh abc123def456 10

SWEEP_ID=$1
COUNT=${2:-10}  # Default to 10 runs

if [ -z "$SWEEP_ID" ]; then
    echo "ERROR: Sweep ID required!"
    echo "Usage: sbatch submit_sweep_agent.sh <sweep-id> [count]"
    exit 1
fi

if [ -z "$BLACKHOLE" ]; then
    echo "ERROR: BLACKHOLE environment variable is not set!"
    exit 1
fi

PROJECT_DIR="$BLACKHOLE/DeepLearningGroup71"
cd "$PROJECT_DIR" || exit 1

source deeplearning/bin/activate

echo "Running WandB sweep agent"
echo "Sweep ID: $SWEEP_ID"
echo "Runs: $COUNT"
echo ""

# Run sweep agent
wandb agent "$SWEEP_ID" --count "$COUNT"

echo ""
echo "Sweep agent completed at: $(date)"

