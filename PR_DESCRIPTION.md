# HPC Enabler + Sweep Config Update

## Summary
This PR adds HPC (High Performance Computing) support for running training jobs on DTU's HPC cluster, and updates the WandB sweep configuration to be compatible with the simplified training script.

## Key Changes

### HPC Support
- **New `src/hpc_utils.py`**: Utility functions for HPC environment detection and directory management
  - Automatically detects HPC environment via `BLACKHOLE` variable
  - Provides unified data/results directory paths (HPC scratch or local)
  - Creates necessary directory structure on HPC
  
- **New `HPC_SETUP.md`**: Comprehensive guide for HPC usage
  - Setup instructions
  - Job submission scripts
  - Sweep agent configuration
  - Data syncing workflows

- **New scripts**:
  - `scripts/submit_job.sh`: SLURM job submission script
  - `scripts/submit_sweep_agent.sh`: Submit WandB sweep agents to HPC
  - `scripts/sync_to_hpc.sh`: Sync code/data to HPC
  - `scripts/sync_from_hpc.sh`: Sync results back from HPC

- **Updated training scripts**: `train.py`, `train2.py`, `train_hpc.py` now use HPC utilities

### Sweep Configuration Updates
- **Updated `experiments/sweep_config.py`**:
  - Added all required fixed parameters (dataset, epochs, etc.)
  - Updated to use CIFAR-10 as default dataset
  - Larger hidden layer architectures suitable for CIFAR-10 (3072 input features)
  - Increased epochs to 150 for CIFAR-10 complexity
  - Added CLI interface for creating sweeps

- **Simplified `experiments/train.py`**:
  - Cleaner, more readable code structure
  - Fully compatible with WandB sweeps (works with `wandb.agent()`)
  - Separated concerns: data loading, training, evaluation
  - Default configuration updated for CIFAR-10

## Benefits
- ✅ Run training jobs on HPC for faster computation
- ✅ Parallel sweep agents for efficient hyperparameter search
- ✅ Automatic directory management (works locally and on HPC)
- ✅ Simplified training code that's easier to maintain
- ✅ Ready-to-use sweep configurations for CIFAR-10

## Testing
- [x] HPC utilities tested locally (fallback to local directories)
- [x] Training script works standalone and with sweeps
- [x] Sweep configs include all required parameters

## Usage

### Standalone Training
```bash
python experiments/train.py --dataset cifar10 --epochs 150
```

### WandB Sweeps
```bash
# Create sweep
python experiments/sweep_config.py random

# Run agent (local or HPC)
wandb agent <sweep_id>
```

### HPC Job Submission
```bash
# Submit training job
sbatch scripts/submit_job.sh

# Submit sweep agent
sbatch scripts/submit_sweep_agent.sh <sweep_id>
```


