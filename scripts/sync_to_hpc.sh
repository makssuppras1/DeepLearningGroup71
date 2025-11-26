#!/bin/bash
# Script to sync project from local machine to HPC
# Usage: ./sync_to_hpc.sh [username@hpc.dtu.dk]

HPC_HOST=${1:-"<your-username>@hpc.dtu.dk"}
PROJECT_NAME="DeepLearningGroup71"

# Get project root directory (parent of scripts/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Syncing project to HPC..."
echo "Source: $PROJECT_ROOT"
echo "Destination: $HPC_HOST:\$BLACKHOLE/$PROJECT_NAME"
echo ""

# Check if rsync is available
if ! command -v rsync &> /dev/null; then
    echo "ERROR: rsync is not installed. Please install it first."
    exit 1
fi

# Sync project (exclude unnecessary files)
rsync -av --progress \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='wandb/' \
    --exclude='deeplearning/' \
    --exclude='*.ipynb_checkpoints' \
    --exclude='.DS_Store' \
    "$PROJECT_ROOT/" \
    "$HPC_HOST:\$BLACKHOLE/$PROJECT_NAME/"

echo ""
echo "✓ Project synced successfully!"
echo ""
echo "Next steps on HPC:"
echo "  1. ssh $HPC_HOST"
echo "  2. cd \$BLACKHOLE/$PROJECT_NAME"
echo "  3. source deeplearning/bin/activate  # or create venv if needed"
echo "  4. pip install -r requirements.txt"
echo "  5. python -c \"from src.hpc_utils import sync_data_to_hpc; sync_data_to_hpc('./data')\""

