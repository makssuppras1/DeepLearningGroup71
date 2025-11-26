# Iterative Hyperparameter Tuning Guide

This guide helps you iteratively improve your hyperparameters by analyzing WandB sweep results.

## Overview

1. **Run initial sweep** with broad hyperparameter ranges
2. **Export results** to CSV
3. **Analyze patterns** (I can help with this)
4. **Refine ranges** based on what works
5. **Repeat** until you find optimal hyperparameters

## Step 1: Export Sweep Data

### Option A: Using the Export Script (Recommended)

```bash
# On HPC or local machine
cd $BLACKHOLE/DeepLearningGroup71  # or your project directory
source deeplearning/bin/activate

# Export sweep data
python scripts/export_sweep_data.py makssuppras1-danmarks-tekniske-universitet-dtu/neural-network-numpy/w8eqt3av --output sweep_results.csv

# Or if sweep ID only:
python scripts/export_sweep_data.py w8eqt3av --output sweep_results.csv
```

### Option B: Manual Export from WandB UI

1. Go to your sweep page: `https://wandb.ai/your-entity/your-project/sweeps/sweep-id`
2. Click "Export" or "Download CSV"
3. Save as `sweep_results.csv`

### Option C: Using WandB API (Python)

```python
import wandb
import pandas as pd

api = wandb.Api()
sweep = api.sweep("entity/project/sweep-id")

runs_data = []
for run in sweep.runs:
    run_data = {
        'run_id': run.id,
        'state': run.state,
        **run.config,  # Hyperparameters
        **{k: v for k, v in run.summary.items() if isinstance(v, (int, float))}  # Metrics
    }
    runs_data.append(run_data)

df = pd.DataFrame(runs_data)
df.to_csv('sweep_results.csv', index=False)
```

## Step 2: Share Data for Analysis

Once you have the CSV file, you can:

1. **Share the CSV file** with me (paste contents or describe key columns)
2. **Share key statistics**:
   - Number of completed runs
   - Best performing runs (top 5-10)
   - Range of hyperparameters tried
   - Best validation loss/accuracy

3. **Or describe what you see**:
   - Which hyperparameters seem to work best?
   - Any clear patterns?
   - What's surprising?

## Step 3: Analysis & Recommendations

I can help you:

- **Identify patterns**: Which hyperparameter combinations work best?
- **Find optimal ranges**: Narrow down search space based on results
- **Suggest improvements**: What to try next
- **Detect issues**: Are there any problematic configurations?

### Example Analysis Questions:

1. **Learning Rate**: What range gives best results?
2. **Architecture**: Which hidden layer sizes perform best?
3. **Regularization**: What dropout/L2 values work?
4. **Optimizer**: Which optimizer performs best?
5. **Batch Size**: What batch sizes are optimal?

## Step 4: Update Sweep Config

Based on analysis, update `configs/hpc_sweep.yaml`:

```yaml
parameters:
  learning_rate:
    distribution: uniform
    min: 0.001  # Narrowed from 0.0001-0.1
    max: 0.01   # Based on best runs
    
  hidden_layers:
    values:
      - [256, 128]  # Keep only sizes that worked well
      - [512, 256]
      # Remove sizes that performed poorly
```

## Step 5: Run Refined Sweep

```bash
# Create new sweep with refined config
wandb sweep configs/hpc_sweep.yaml

# Run agents with new sweep ID
# ... (follow HPC_SWEEP_GUIDE.md)
```

## Iterative Process

```
Initial Sweep (Broad Ranges)
    ↓
Export Results
    ↓
Analyze Patterns
    ↓
Refine Config (Narrow Ranges)
    ↓
New Sweep (Focused Search)
    ↓
Export & Analyze Again
    ↓
Final Fine-tuning
```

## Tips

1. **Start broad**: Initial sweep should explore wide ranges
2. **Focus on top performers**: Look at top 10-20% of runs
3. **Consider trade-offs**: Accuracy vs training time
4. **Watch for overfitting**: Check train/val gap
5. **Document learnings**: Keep notes on what works

## Quick Reference

### Export Current Sweep
```bash
python scripts/export_sweep_data.py <sweep_id> --output results.csv
```

### View Best Runs in WandB
- Go to sweep page
- Sort by `val_loss` (ascending) or `val_acc` (descending)
- Review top 10-20 runs

### Key Metrics to Track
- `val_loss` - Primary optimization target
- `val_acc` - Model performance
- `train_loss` vs `val_loss` - Overfitting indicator
- Training time - Efficiency metric

## Example Workflow

```bash
# 1. Export current sweep
python scripts/export_sweep_data.py w8eqt3av -o sweep1.csv

# 2. Share CSV or key findings with me

# 3. I analyze and suggest improvements:
#    - Learning rate: 0.001-0.01 (was 0.0001-0.1)
#    - Hidden layers: [256,128] and [512,256] work best
#    - Dropout: 0.2-0.3 optimal (was 0.1-0.5)

# 4. Update configs/hpc_sweep.yaml

# 5. Create new sweep
wandb sweep configs/hpc_sweep.yaml

# 6. Run refined sweep
# ... (repeat)
```

## Getting Help

When sharing results, include:
- CSV file or key statistics
- Current sweep config
- What you've observed
- Specific questions or goals

I can then provide:
- Pattern analysis
- Hyperparameter recommendations
- Updated config suggestions
- Next steps

