# HPC WandB Sweep Guide

Quick reference guide for running WandB sweeps on DTU HPC.

## Prerequisites

- VPN connection to DTU (if connecting from outside)
- SSH access to HPC: `s204614@login9.hpc.dtu.dk`
- WandB account configured on HPC

## 1. Connect to HPC

```bash
ssh s204614@login9.hpc.dtu.dk
```

## 2. Navigate to Project

```bash
cd $BLACKHOLE/DeepLearningGroup71
```

## 3. Update Code

### Option A: Using Git (if repo is cloned)
```bash
cd $BLACKHOLE/DeepLearningGroup71
git pull origin main  # or your branch name
```

### Option B: Manual Sync from Local Machine
On your local machine (with VPN connected):
```bash
cd /path/to/DeepLearningGroup71
rsync -av --progress \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='wandb/' \
    --exclude='deeplearning/' \
    --exclude='*.ipynb_checkpoints' \
    --exclude='.DS_Store' \
    --exclude='data/' \
    ./ s204614@login9.hpc.dtu.dk:/dtu/blackhole/0d/156141/DeepLearningGroup71/
```

## 4. Activate Environment

```bash
cd $BLACKHOLE/DeepLearningGroup71
source deeplearning/bin/activate
```

## 5. Create a New Sweep (if needed)

```bash
cd $BLACKHOLE/DeepLearningGroup71
source deeplearning/bin/activate

# Set WandB project
export WANDB_PROJECT="neural-network-numpy"

# Create sweep from config
wandb sweep configs/hpc_sweep.yaml
```

**Output will look like:**
```
wandb: Creating sweep with ID: xxxxxxxx
wandb: View sweep at: https://wandb.ai/...
wandb: Run sweep agent with: wandb agent entity/project/xxxxxxxx
```

**Save the sweep ID** (e.g., `xxxxxxxx`) and full path (e.g., `entity/project/xxxxxxxx`).

## 6. Run Sweep Agents

### Single Agent (for testing)
```bash
cd $BLACKHOLE/DeepLearningGroup71
source deeplearning/bin/activate

export WANDB_PROJECT="neural-network-numpy"
export WANDB_ENTITY="makssuppras1-danmarks-tekniske-universitet-dtu"  # Optional, uses default if not set

# Replace with your actual sweep path
SWEEP_PATH="makssuppras1-danmarks-tekniske-universitet-dtu/neural-network-numpy/xxxxxxxx"

# Run one agent
wandb agent $SWEEP_PATH --count 10
```

### Multiple Agents (Parallel Sweeps)

```bash
cd $BLACKHOLE/DeepLearningGroup71
source deeplearning/bin/activate

export WANDB_PROJECT="neural-network-numpy"
export WANDB_ENTITY="makssuppras1-danmarks-tekniske-universitet-dtu"
SWEEP_PATH="makssuppras1-danmarks-tekniske-universitet-dtu/neural-network-numpy/xxxxxxxx"

# Kill any old agents first
pkill -f "wandb agent"

# Start 5 agents in parallel (each runs 10 runs)
for i in {1..5}; do
  nohup wandb agent $SWEEP_PATH --count 10 > sweep_agent_${i}.log 2>&1 &
  echo "Started agent $i"
done

# Check if they're running
sleep 2
ps aux | grep "wandb agent" | grep -v grep
```

## 7. Monitor Agents

### Check Logs
```bash
# View latest logs from agent 1
tail -30 sweep_agent_1.log

# Follow logs in real-time
tail -f sweep_agent_1.log

# Check all agent logs
tail -20 sweep_agent_*.log

# Search for errors
grep -i "error\|exception\|traceback" sweep_agent_*.log
```

### Check Agent Status
```bash
# See running agents
ps aux | grep "wandb agent" | grep -v grep

# Count running agents
ps aux | grep "wandb agent" | grep -v grep | wc -l
```

### View WandB Dashboard
Open in browser:
```
https://wandb.ai/makssuppras1-danmarks-tekniske-universitet-dtu/neural-network-numpy/sweeps/xxxxxxxx
```

## 8. Stop Agents

```bash
# Stop all agents
pkill -f "wandb agent"

# Verify they're stopped
ps aux | grep "wandb agent" | grep -v grep
```

## 9. Diagnosing Crashes

### Quick Diagnosis Commands

```bash
# Check which agents are still running
ps aux | grep "wandb agent" | grep -v grep

# Check exit status of crashed agents
tail -50 sweep_agent_*.log | grep -A 10 "Traceback\|Error\|Exception"

# Find common errors across all logs
grep -i "error\|exception\|traceback\|fatal" sweep_agent_*.log | head -20

# Check the most recent errors
for i in {1..5}; do
  echo "=== Agent $i ==="
  tail -30 sweep_agent_${i}.log | tail -10
done
```

### Common Issues & Solutions

### Issue: "unrecognized arguments" error
**Solution:** Make sure you've synced the latest `train_simple.py` (should not have argparse).

### Issue: "FileNotFoundError: data not found"
**Solution:** The script will auto-download data on first run. Wait for download to complete (~170MB for CIFAR-10).

### Issue: "wandb: ERROR Error while calling W&B API"
**Solution:** 
```bash
wandb login --relogin
# Enter your API key when prompted
```

### Issue: "Sweep not found"
**Solution:** Check that:
- `WANDB_PROJECT` is set correctly
- `WANDB_ENTITY` matches your WandB username (or unset it to use default)
- Sweep ID is correct

### Issue: Agents exit immediately or crash
**Solution:** Check logs:
```bash
# View full error from a crashed agent
cat sweep_agent_1.log

# Find the error pattern
grep -B 5 -A 10 "Traceback" sweep_agent_1.log
```

Common causes:
- Missing data (will auto-download, but may fail)
- Import errors (check Python path: `python -c "import sys; print(sys.path)"`)
- Config errors (check sweep YAML syntax)
- Memory issues (check available memory: `free -h`)
- Network issues (check internet: `ping -c 3 8.8.8.8`)

### Issue: Training is very slow (20+ minutes per epoch)
**Possible causes:**
1. **Running on CPU instead of GPU** - Check if GPU is available:
   ```bash
   nvidia-smi  # Check GPU availability
   # Or check CPU usage
   top -u $USER
   ```

2. **Too many agents competing for resources**
   - **Solution:** Reduce number of parallel agents (try 2-3 instead of 5)
   - **Solution:** Check system load: `uptime` or `htop`

3. **Large batch size or model size**
   - **Solution:** Check sweep config - reduce `batch_size` or `hidden_layers` if too large

4. **Network issues slowing WandB logging**
   - **Solution:** Check network: `ping -c 3 8.8.8.8`

### Issue: Multiple agents crashing simultaneously
**Possible causes:**
1. **Data download conflict** - Multiple agents trying to download/extract same file
   - **Solution:** Let one agent finish downloading first, or pre-download data:
   ```bash
   cd $BLACKHOLE/DeepLearningGroup71
   source deeplearning/bin/activate
   python -c "from src.data_loader import download_cifar10; from src.hpc_utils import get_data_dir, get_project_root; download_cifar10(get_data_dir(get_project_root()))"
   ```

2. **WandB API rate limiting**
   - **Solution:** Reduce number of parallel agents or add delays between starts

3. **Resource exhaustion** (memory/CPU)
   - **Solution:** Check system resources: `free -h`, `top`, reduce concurrent agents

### Issue: KeyboardInterrupt in logs
**Cause:** Agent was manually stopped (Ctrl+C) or killed
**Solution:** This is normal if you stopped an agent. Restart it if needed:
```bash
# Restart a specific agent
nohup wandb agent $SWEEP_PATH --count 10 > sweep_agent_5.log 2>&1 &
```

### Issue: Old log files with errors
**Solution:** Clean up old log files:
```bash
# Remove old log files
rm sweep_agent_n6xelg7m_*.log  # Replace with your old sweep ID
# Or remove all old logs
rm sweep_agent_*.log
# Then restart agents (they'll create new logs)
```

### Issue: "sbatch: command not found"
**Solution:** DTU HPC doesn't use SLURM on login nodes. Use `nohup` instead (as shown above).

### Issue: Import errors
**Solution:** Verify Python path and imports:
```bash
cd $BLACKHOLE/DeepLearningGroup71
source deeplearning/bin/activate
python -c "import sys; sys.path.insert(0, '.'); from experiments.train_simple import main; print('Import OK')"
```

## 10. Quick Reference

### Full Workflow (Copy-Paste Ready)

```bash
# 1. Connect
ssh s204614@login9.hpc.dtu.dk

# 2. Navigate and update
cd $BLACKHOLE/DeepLearningGroup71
git pull origin main  # or your branch

# 3. Activate environment
source deeplearning/bin/activate

# 4. Set environment variables
export WANDB_PROJECT="neural-network-numpy"
export WANDB_ENTITY="makssuppras1-danmarks-tekniske-universitet-dtu"

# 5. Kill old agents
pkill -f "wandb agent"

# 6. Start agents (replace xxxxxxxx with your sweep ID)
SWEEP_PATH="makssuppras1-danmarks-tekniske-universitet-dtu/neural-network-numpy/xxxxxxxx"
for i in {1..5}; do
  nohup wandb agent $SWEEP_PATH --count 10 > sweep_agent_${i}.log 2>&1 &
  echo "Started agent $i"
done

# 7. Monitor
sleep 5
tail -30 sweep_agent_1.log
```

### Environment Variables

```bash
export WANDB_PROJECT="neural-network-numpy"
export WANDB_ENTITY="makssuppras1-danmarks-tekniske-universitet-dtu"  # Optional
```

### Useful Commands

```bash
# Check WandB login status
wandb login

# View current runs
wandb status

# List sweeps
wandb sweep --list

# Check Python environment
which python
python --version

# Check installed packages
pip list | grep wandb
```

## Notes

- **Data Location:** Data is auto-downloaded to `$BLACKHOLE/data/` on first run
- **Logs:** Agent logs are saved to `sweep_agent_*.log` in project root
- **Results:** Model checkpoints and plots saved to `$BLACKHOLE/results/`
- **WandB:** All metrics logged to WandB dashboard in real-time
- **Parallel Runs:** Each agent runs independently - you can start/stop them individually

## Getting Help

1. Check agent logs: `tail -50 sweep_agent_1.log`
2. Check WandB dashboard for run status
3. Verify environment: `wandb login`, `python --version`
4. Check data exists: `ls -lh $BLACKHOLE/data/cifar-10-batches-py/`

