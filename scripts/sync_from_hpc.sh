#!/bin/bash
# Script to sync results from HPC to local machine
# Usage: ./sync_from_hpc.sh [username@hpc.dtu.dk]

HPC_HOST=${1:-"<your-username>@hpc.dtu.dk"}
PROJECT_NAME="DeepLearningGroup71"

# Get project root directory (parent of scripts/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Syncing results from HPC..."
echo "Source: $HPC_HOST:\$BLACKHOLE/$PROJECT_NAME/results/"
echo "Destination: $PROJECT_ROOT/results/"
echo ""

# Check if rsync is available
if ! command -v rsync &> /dev/null; then
    echo "ERROR: rsync is not installed. Please install it first."
    exit 1
fi

# Create local results directory if it doesn't exist
mkdir -p "$PROJECT_ROOT/results"

# Sync results
rsync -av --progress \
    "$HPC_HOST:\$BLACKHOLE/$PROJECT_NAME/results/" \
    "$PROJECT_ROOT/results/"

echo ""
echo "✓ Results synced successfully!"
echo ""
echo "⚠️  Remember: Data in \$BLACKHOLE will be deleted at service windows!"
echo "   Make sure to sync important results regularly."


