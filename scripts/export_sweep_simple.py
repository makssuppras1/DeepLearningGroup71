#!/usr/bin/env python3
"""
Simple script to export WandB sweep data to CSV.
Run this directly on HPC - no dependencies beyond wandb.
"""

import wandb
import csv
import sys

def export_sweep(sweep_path, output_file="sweep_results.csv"):
    """Export sweep runs to CSV."""
    print(f"Connecting to WandB...")
    api = wandb.Api()
    
    # Parse sweep path
    parts = sweep_path.split("/")
    if len(parts) == 3:
        entity, project, sweep_id = parts
        sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    elif len(parts) == 2:
        project, sweep_id = parts
        sweep = api.sweep(f"{project}/{sweep_id}")
    else:
        # Try as just sweep ID (will use default entity/project)
        sweep_id = sweep_path
        sweep = api.sweep(sweep_path)
    
    print(f"Fetching runs from sweep: {sweep.name}")
    runs = sweep.runs
    print(f"Found {len(runs)} runs")
    
    if len(runs) == 0:
        print("No runs found!")
        return
    
    # Collect all keys
    all_config_keys = set()
    all_metric_keys = set()
    for run in runs:
        if run.config:
            all_config_keys.update([k for k in run.config.keys() if not k.startswith('_')])
        if run.summary:
            all_metric_keys.update([k for k in run.summary.keys() if isinstance(run.summary[k], (int, float))])
    
    config_keys = sorted(all_config_keys)
    metric_keys = sorted(all_metric_keys)
    
    # Write CSV
    rows = []
    for run in runs:
        row = {
            'run_id': run.id,
            'run_name': run.name,
            'state': run.state,
        }
        
        # Add config
        if run.config:
            for key in config_keys:
                val = run.config.get(key, '')
                row[key] = str(val) if isinstance(val, list) else val
        
        # Add metrics
        if run.summary:
            for key in metric_keys:
                row[key] = run.summary.get(key, '')
        
        rows.append(row)
    
    # Write to file
    all_cols = ['run_id', 'run_name', 'state'] + config_keys + metric_keys
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_cols)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\n✅ Exported {len(rows)} runs to {output_file}")
    
    # Summary
    completed = [r for r in rows if r['state'] == 'finished']
    print(f"  - Completed: {len(completed)}")
    print(f"  - Failed/Running: {len(rows) - len(completed)}")
    
    if completed and 'val_loss' in metric_keys:
        try:
            best = min([r for r in completed if r.get('val_loss')], 
                      key=lambda x: float(x['val_loss']) if x.get('val_loss') else float('inf'))
            print(f"\n🏆 Best run: {best['run_name']}")
            print(f"   val_loss: {best.get('val_loss', 'N/A')}")
            print(f"   val_acc: {best.get('val_acc', 'N/A')}")
        except:
            pass

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python export_sweep_simple.py <sweep_path> [output.csv]")
        print("Example: python export_sweep_simple.py makssuppras1-danmarks-tekniske-universitet-dtu/neural-network-numpy/w8eqt3av")
        sys.exit(1)
    
    sweep_path = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "sweep_results.csv"
    
    export_sweep(sweep_path, output_file)

