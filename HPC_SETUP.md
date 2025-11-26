# HPC Setup Guide

This guide explains how to use DTU's HPC system for training your neural networks.

## Overview

The HPC (High Performance Computing) system provides:
- **Faster training**: More powerful CPUs/GPUs than your local machine
- **Parallel experiments**: Run multiple hyperparameter sweeps simultaneously
- **Long-running jobs**: Better suited for 150+ epoch training runs
- **Temporary storage**: Scratch directory (`$BLACKHOLE`) for experiments

## Important Warnings

⚠️ **Data in `$BLACKHOLE` is temporary:**
- Deleted at each service window
- **All data will be permanently deleted at the end of January 2026**
- **Always sync important results back to your local machine!**

## Prerequisites

1. **SSH access to DTU HPC**
   - You should have received credentials from your course
   - Test connection: `ssh <your-username>@hpc.dtu.dk`

2. **BLACKHOLE environment variable**
   - Set automatically when you log in to HPC
   - Check: `echo $BLACKHOLE`
   - Path: `/dtu/blackhole/<your-username>/...`

## Quick Start

### 1. Connect to HPC

```bash
ssh <your-username>@hpc.dtu.dk
cd $BLACKHOLE
```

### 2. Clone/Upload Your Project

**Option A: Clone from GitHub (recommended)**
```bash
cd $BLACKHOLE
git clone <your-repo-url> DeepLearningGroup71
cd DeepLearningGroup71
```

**Option B: Upload via rsync**
```bash
# From your local machine:
rsync -av --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' \
  /path/to/DeepLearningGroup71/ \
  <your-username>@hpc.dtu.dk:$BLACKHOLE/DeepLearningGroup71/
```

### 3. Set Up Environment

```bash
# On HPC:
cd $BLACKHOLE/DeepLearningGroup71

# Create virtual environment (if not already created)
python3 -m venv deeplearning
source deeplearning/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Sync Data to HPC

**From your local machine:**
```bash
# Sync data directory to HPC
rsync -av --progress \
  /path/to/DeepLearningGroup71/data/ \
  <your-username>@hpc.dtu.dk:$BLACKHOLE/DeepLearningGroup71/data/
```

**Or use the utility script:**
```bash
# On HPC, after uploading project:
python -c "from src.hpc_utils import sync_data_to_hpc; sync_data_to_hpc('./data')"
```

### 5. Run Training

The training scripts automatically detect HPC and use `$BLACKHOLE` for data and results.

**Interactive session:**
```bash
# On HPC:
cd $BLACKHOLE/DeepLearningGroup71
source deeplearning/bin/activate
python experiments/train2.py --dataset cifar10 --epochs 150
```

**Submit as job (recommended for long runs):**
```bash
# Submit job using SLURM
sbatch scripts/submit_job.sh
```

## Job Submission (SLURM)

For long-running experiments, use SLURM job scheduler.

### Example Job Script

See `scripts/submit_job.sh` for a complete example. Basic usage:

```bash
#!/bin/bash
#SBATCH --job-name=nn_train
#SBATCH --time=24:00:00          # Max runtime: 24 hours
#SBATCH --nodes=1                # Number of nodes
#SBATCH --ntasks-per-node=1      # Tasks per node
#SBATCH --cpus-per-task=8        # CPUs per task
#SBATCH --mem=16G                # Memory per node

# Load modules (if needed)
# module load python/3.12

# Activate virtual environment
source $BLACKHOLE/DeepLearningGroup71/deeplearning/bin/activate

# Change to project directory
cd $BLACKHOLE/DeepLearningGroup71

# Run training
python experiments/train2.py \
  --dataset cifar10 \
  --epochs 150 \
  --batch-size 64 \
  --lr 0.0003 \
  --name cifar10_baseline
```

### Submit and Monitor Jobs

```bash
# Submit job
sbatch scripts/submit_job.sh

# Check job status
squeue -u $USER

# View job output (after completion)
cat slurm-<job-id>.out

# Cancel a job
scancel <job-id>
```

## Running Hyperparameter Sweeps

WandB sweeps are perfect for HPC - run multiple agents in parallel:

```bash
# On HPC, start multiple sweep agents
# Terminal 1:
wandb agent <sweep-id>

# Terminal 2:
wandb agent <sweep-id>

# Terminal 3:
wandb agent <sweep-id>
# ... etc
```

Or submit as separate jobs:
```bash
# Submit 5 parallel sweep agents
for i in {1..5}; do
  sbatch scripts/submit_sweep_agent.sh <sweep-id>
done
```

## Syncing Results Back

**⚠️ CRITICAL: Always sync results before they're deleted!**

```bash
# From your local machine:
rsync -av --progress \
  <your-username>@hpc.dtu.dk:$BLACKHOLE/DeepLearningGroup71/results/ \
  /path/to/DeepLearningGroup71/results/
```

**Or use the utility script:**
```bash
# On HPC:
python -c "from src.hpc_utils import sync_results_from_hpc; sync_results_from_hpc()"
# Then download from HPC to local
```

## Workflow Summary

### Daily Workflow

1. **Morning:**
   ```bash
   # Connect to HPC
   ssh <username>@hpc.dtu.dk
   cd $BLACKHOLE/DeepLearningGroup71
   
   # Pull latest code
   git pull origin main
   
   # Check running jobs
   squeue -u $USER
   ```

2. **Submit experiments:**
   ```bash
   # Submit training job
   sbatch scripts/submit_job.sh
   
   # Or run interactively for quick tests
   python experiments/train2.py --epochs 10
   ```

3. **Evening:**
   ```bash
   # Sync results back to local
   # (from local machine)
   rsync -av <username>@hpc.dtu.dk:$BLACKHOLE/DeepLearningGroup71/results/ ./results/
   ```

### Weekly Workflow

- **Monday**: Sync all code and data to HPC
- **Weekdays**: Submit jobs, monitor progress
- **Friday**: Sync all results back to local

## Troubleshooting

### BLACKHOLE not set
```bash
# Check if you're on HPC
hostname  # Should show hpc.dtu.dk

# If BLACKHOLE is not set, check:
ls /dtu/blackhole/
# Contact course staff if directory doesn't exist
```

### Out of disk space
```bash
# Check disk usage
du -sh $BLACKHOLE/*

# Clean up old results/models
rm -rf $BLACKHOLE/DeepLearningGroup71/results/models/old_*

# Clean up Python cache
find $BLACKHOLE -type d -name __pycache__ -exec rm -r {} +
```

### Job fails immediately
```bash
# Check job output
cat slurm-<job-id>.out

# Common issues:
# - Virtual environment not activated
# - Missing dependencies
# - Wrong Python path
```

### WandB not logging
```bash
# Make sure WandB is logged in
wandb login

# Check WandB status
wandb status
```

## Best Practices

1. **Always sync results regularly** - Don't wait until the end!
2. **Use job scheduler for long runs** - Don't run 150 epochs interactively
3. **Monitor disk usage** - Scratch space is limited
4. **Test locally first** - Verify code works before submitting long jobs
5. **Use WandB** - Results are automatically synced to cloud
6. **Keep code in Git** - Easy to sync between local and HPC

## Additional Resources

- DTU HPC Documentation: Check `/dtu/blackhole/readme.txt` on HPC
- SLURM Documentation: `man sbatch`, `man squeue`
- WandB Documentation: https://docs.wandb.ai/

## Quick Reference

```bash
# Connect to HPC
ssh <username>@hpc.dtu.dk

# Navigate to scratch
cd $BLACKHOLE

# Check HPC status
python -c "from src.hpc_utils import print_hpc_info; print_hpc_info()"

# Submit job
sbatch scripts/submit_job.sh

# Check jobs
squeue -u $USER

# Cancel job
scancel <job-id>

# Sync results (from local)
rsync -av <username>@hpc.dtu.dk:$BLACKHOLE/DeepLearningGroup71/results/ ./results/
```

