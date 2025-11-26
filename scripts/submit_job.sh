#!/bin/bash
#SBATCH --job-name=nn_train
#SBATCH --time=24:00:00          # Maximum runtime: 24 hours
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --ntasks-per-node=1      # Number of tasks per node
#SBATCH --cpus-per-task=8        # Number of CPUs per task (adjust based on HPC limits)
#SBATCH --mem=16G                 # Memory per node (adjust based on HPC limits)
#SBATCH --output=slurm-%j.out     # Output file
#SBATCH --error=slurm-%j.err      # Error file

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Check if BLACKHOLE is set
if [ -z "$BLACKHOLE" ]; then
    echo "ERROR: BLACKHOLE environment variable is not set!"
    echo "This script should be run on HPC."
    exit 1
fi

# Set project directory
PROJECT_DIR="$BLACKHOLE/DeepLearningGroup71"

# Check if project directory exists
if [ ! -d "$PROJECT_DIR" ]; then
    echo "ERROR: Project directory not found: $PROJECT_DIR"
    echo "Please clone/upload your project to $BLACKHOLE first."
    exit 1
fi

# Change to project directory
cd "$PROJECT_DIR" || exit 1

# Activate virtual environment
if [ -d "deeplearning/bin" ]; then
    source deeplearning/bin/activate
    echo "Virtual environment activated"
else
    echo "WARNING: Virtual environment not found. Using system Python."
fi

# Print environment info
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"
echo "BLACKHOLE: $BLACKHOLE"
echo ""

# Set up HPC directories
python -c "from src.hpc_utils import setup_hpc_directories; setup_hpc_directories()"

# Default training parameters (can be overridden via command line)
DATASET=${1:-cifar10}
EPOCHS=${2:-150}
BATCH_SIZE=${3:-64}
LR=${4:-0.0003}
EXPERIMENT_NAME=${5:-hpc_baseline}

# Run training
echo "Starting training..."
echo "Dataset: $DATASET"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LR"
echo "Experiment name: $EXPERIMENT_NAME"
echo ""

python experiments/train2.py \
    --dataset "$DATASET" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --name "$EXPERIMENT_NAME"

# Print completion info
echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="

# Print reminder about syncing results
echo ""
echo "⚠️  REMINDER: Sync results back to your local machine!"
echo "   From local machine, run:"
echo "   rsync -av <username>@login9.hpc.dtu.dk:\$BLACKHOLE/DeepLearningGroup71/results/ ./results/"


