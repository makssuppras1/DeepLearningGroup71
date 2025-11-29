#!/bin/bash
# Script to rerun Fashion-MNIST sweep on HPC with updated config
# Usage: ./rerun_fashion_mnist_sweep.sh [hpc_username@hpc_host]

HPC_HOST=${1:-"s204614@login9.hpc.dtu.dk"}
PROJECT_NAME="DeepLearningGroup71"
CONFIG_FILE="configs/fashion_mnist_sweep.yaml"

# Get project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "Rerunning Fashion-MNIST Sweep on HPC"
echo "=========================================="
echo ""
echo "Step 1: Syncing updated config to HPC..."
echo ""

# Sync the config file to HPC
rsync -av --progress \
    "$PROJECT_ROOT/$CONFIG_FILE" \
    "$HPC_HOST:\$BLACKHOLE/$PROJECT_NAME/$CONFIG_FILE"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Config file synced successfully!"
    echo ""
    echo "=========================================="
    echo "Next steps on HPC:"
    echo "=========================================="
    echo ""
    echo "1. SSH to HPC:"
    echo "   ssh $HPC_HOST"
    echo ""
    echo "2. Navigate to project and activate environment:"
    echo "   cd \$BLACKHOLE/$PROJECT_NAME"
    echo "   source deeplearning/bin/activate"
    echo ""
    echo "3. Create new sweep:"
    echo "   export WANDB_PROJECT=\"neural-network-numpy\""
    echo "   wandb sweep $CONFIG_FILE"
    echo ""
    echo "4. Save the sweep ID from the output, then submit agents:"
    echo "   SWEEP_PATH=\"<entity>/neural-network-numpy/<sweep-id>\""
    echo "   sbatch scripts/submit_sweep_agent.sh \$SWEEP_PATH 10"
    echo ""
    echo "   Or run multiple agents in parallel:"
    echo "   for i in {1..5}; do sbatch scripts/submit_sweep_agent.sh \$SWEEP_PATH 10; done"
    echo ""
else
    echo ""
    echo "✗ Error syncing config file!"
    exit 1
fi

