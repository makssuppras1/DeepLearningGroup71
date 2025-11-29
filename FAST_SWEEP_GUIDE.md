# Fast Fashion-MNIST Sweep on HPC - Quick Guide

This guide walks you through running a fast hyperparameter sweep on HPC with multiple parallel agents.

## ⚡ Quick Start (Copy-Paste Ready)

```bash
# 1. Connect to HPC
ssh s204614@login9.hpc.dtu.dk

# 2. Navigate to project
cd $BLACKHOLE/DeepLearningGroup71

# 3. Update code (if needed)
git pull origin main

# 4. Activate environment
source deeplearning/bin/activate

# 5. Login to WandB (if not already logged in)
wandb login

# 6. Run fast sweep with 10 parallel agents (each doing 10 runs = 100 total runs)
./scripts/setup_fashion_mnist_sweep.sh 10 10
```

## 📋 Detailed Step-by-Step Process

### Step 1: Connect to HPC

```bash
ssh s204614@login9.hpc.dtu.dk
```

**Note:** If you're off-campus, connect to DTU VPN first (`vpn.dtu.dk`).

### Step 2: Navigate to Project Directory

```bash
cd $BLACKHOLE/DeepLearningGroup71
```

Verify you're in the right place:
```bash
ls -la configs/fashion_mnist_sweep.yaml  # Should exist
```

### Step 3: Update Code (Optional but Recommended)

```bash
# Pull latest changes from git
git pull origin main

# Or if you made local changes, sync them first
# (from your local machine)
rsync -av --progress \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='wandb/' \
    --exclude='deeplearning/' \
    ./ s204614@login9.hpc.dtu.dk:$BLACKHOLE/DeepLearningGroup71/
```

### Step 4: Activate Virtual Environment

```bash
source deeplearning/bin/activate
```

Verify Python environment:
```bash
which python  # Should show path to deeplearning/bin/python
python --version
```

### Step 5: Verify WandB Login

```bash
wandb login
# Enter your API key if prompted

# Check status
wandb status
```

### Step 6: Review Sweep Configuration

The sweep config is at `configs/fashion_mnist_sweep.yaml`. It's optimized for speed:
- **Method**: `random` (fast exploration, doesn't try all combinations)
- **Epochs**: 20-50 (reduced for faster runs)
- **Early termination**: Enabled (stops poor runs early)

View the config:
```bash
cat configs/fashion_mnist_sweep.yaml
```

### Step 7: Run the Sweep

You have two options:

#### Option A: Use the Automated Script (Recommended)

```bash
# Syntax: ./scripts/setup_fashion_mnist_sweep.sh [num_agents] [runs_per_agent]
# Default: 5 agents, 10 runs each

# Fast sweep: 10 agents, 10 runs each = 100 total runs
./scripts/setup_fashion_mnist_sweep.sh 10 10

# Very fast: 20 agents, 5 runs each = 100 total runs (more parallel)
./scripts/setup_fashion_mnist_sweep.sh 20 5

# Conservative: 5 agents, 20 runs each = 100 total runs
./scripts/setup_fashion_mnist_sweep.sh 5 20
```

The script will:
1. Create the sweep from the YAML config
2. Start multiple agents in parallel
3. Show you the sweep ID and monitoring commands

#### Option B: Manual Setup (More Control)

```bash
# 1. Set environment variables
export WANDB_PROJECT="neural-network-numpy"
export WANDB_ENTITY="makssuppras1-danmarks-tekniske-universitet-dtu"

# 2. Create sweep
wandb sweep configs/fashion_mnist_sweep.yaml
# Save the sweep ID from output (e.g., "abc123def456")
# Save the full path (e.g., "makssuppras1-danmarks-tekniske-universitet-dtu/neural-network-numpy/abc123def456")

# 3. Start multiple agents
SWEEP_PATH="makssuppras1-danmarks-tekniske-universitet-dtu/neural-network-numpy/YOUR_SWEEP_ID"
NUM_AGENTS=10
RUNS_PER_AGENT=10

# Kill any old agents
pkill -f "wandb agent" 2>/dev/null

# Start agents
for i in $(seq 1 $NUM_AGENTS); do
    LOG_FILE="sweep_agent_fashion_mnist_${i}.log"
    nohup wandb agent "$SWEEP_PATH" --count "$RUNS_PER_AGENT" > "$LOG_FILE" 2>&1 &
    echo "Started agent $i (PID: $!)"
    sleep 1  # Small delay between starts
done
```

### Step 8: Monitor the Sweep

#### Check Running Agents

```bash
# Count running agents
ps aux | grep "wandb agent" | grep -v grep | wc -l

# See all running agents
ps aux | grep "wandb agent" | grep -v grep
```

#### View Logs

```bash
# View latest logs from agent 1
tail -30 sweep_agent_fashion_mnist_1.log

# Follow logs in real-time
tail -f sweep_agent_fashion_mnist_1.log

# Check all agent logs
tail -20 sweep_agent_fashion_mnist_*.log

# Search for errors
grep -i "error\|exception\|traceback" sweep_agent_fashion_mnist_*.log
```

#### View WandB Dashboard

The script will output a URL like:
```
https://wandb.ai/makssuppras1-danmarks-tekniske-universitet-dtu/neural-network-numpy/sweeps/YOUR_SWEEP_ID
```

Open this in your browser to see:
- Real-time metrics
- Best runs
- Hyperparameter comparisons
- Parallel coordinate plots

### Step 9: Stop Agents (When Done or to Pause)

```bash
# Stop all agents
pkill -f "wandb agent"

# Verify they're stopped
ps aux | grep "wandb agent" | grep -v grep
```

**Note:** Stopping agents doesn't delete completed runs. You can restart agents later to continue the sweep.

## 🚀 Optimizing for Maximum Speed

### Recommended Settings for Fast Sweeps

1. **Many Parallel Agents**: 10-20 agents running simultaneously
2. **Fewer Runs Per Agent**: 5-10 runs per agent (agents can restart)
3. **Random Method**: Already set in config (faster than grid)
4. **Reduced Epochs**: 20-50 epochs (already optimized)
5. **Early Termination**: Enabled (stops bad runs early)

### Example: Ultra-Fast Sweep

```bash
# 20 agents, 5 runs each = 100 runs total, maximum parallelism
./scripts/setup_fashion_mnist_sweep.sh 20 5
```

### Example: Balanced Speed/Quality

```bash
# 10 agents, 10 runs each = 100 runs total
./scripts/setup_fashion_mnist_sweep.sh 10 10
```

## 📊 Understanding the Results

### In WandB Dashboard

1. **Overview Tab**: Best run, summary statistics
2. **Runs Tab**: All individual runs with metrics
3. **Parallel Coordinates**: See which hyperparameters correlate with good performance
4. **Hyperparameter Importance**: Which parameters matter most

### Key Metrics to Watch

- **val_loss**: Lower is better (this is the optimization target)
- **val_acc**: Higher is better (validation accuracy)
- **train_loss**: Should decrease (monitor for overfitting)

## 🔧 Troubleshooting

### Agents Not Starting

```bash
# Check WandB login
wandb login

# Check environment
echo $WANDB_PROJECT
echo $WANDB_ENTITY

# Check if sweep exists
wandb sweep --list
```

### Agents Crashing

```bash
# Check logs for errors
tail -50 sweep_agent_fashion_mnist_1.log

# Common issues:
# - Missing data (will auto-download)
# - Import errors (check Python path)
# - Memory issues (reduce number of agents)
```

### Too Slow

```bash
# Reduce number of agents (less competition for resources)
pkill -f "wandb agent"
./scripts/setup_fashion_mnist_sweep.sh 5 10  # Fewer agents

# Or check system load
uptime
top -u $USER
```

### Want to Change Config

```bash
# Edit the config
nano configs/fashion_mnist_sweep.yaml

# Create a new sweep (old one keeps running)
wandb sweep configs/fashion_mnist_sweep.yaml
```

## 📝 Quick Reference Commands

```bash
# Start fast sweep (10 agents, 10 runs each)
./scripts/setup_fashion_mnist_sweep.sh 10 10

# Check running agents
ps aux | grep "wandb agent" | grep -v grep

# View logs
tail -f sweep_agent_fashion_mnist_1.log

# Stop all agents
pkill -f "wandb agent"

# Check WandB status
wandb status

# List sweeps
wandb sweep --list
```

## 🎯 Next Steps After Sweep Completes

1. **Review Best Runs**: Check WandB dashboard for top performers
2. **Analyze Results**: Use parallel coordinates to understand what works
3. **Run Best Config**: Train the best configuration for longer (more epochs)
4. **Export Results**: Sync results back to local machine

```bash
# From local machine, sync results
rsync -av --progress \
    s204614@login9.hpc.dtu.dk:$BLACKHOLE/DeepLearningGroup71/results/ \
    ./results/
```

## 💡 Tips

- **Start Small**: Test with 2-3 agents first to verify everything works
- **Monitor Early**: Check logs and WandB dashboard in first 10 minutes
- **Use Early Termination**: Bad runs will stop early, saving time
- **Parallel is Key**: More agents = faster exploration (up to resource limits)
- **Check Resources**: If system is slow, reduce number of agents

---

**Ready to start?** Just run:
```bash
./scripts/setup_fashion_mnist_sweep.sh 10 10
```

