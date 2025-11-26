#!/usr/bin/env python3
"""
Export ALL runs from a WandB sweep to CSV (including running/failed/crashed runs).
Fixed version that handles crashed runs gracefully.
"""

import wandb
import csv
import sys

def export_all_runs(sweep_path, output_file="all_sweep_runs.csv"):
    """Export all runs (completed, running, failed, crashed) to CSV."""
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
        sweep = api.sweep(sweep_path)
    
    print(f"Fetching ALL runs from sweep: {sweep.name}")
    runs = sweep.runs
    print(f"Found {len(runs)} total runs")
    
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
            all_metric_keys.update([k for k in run.summary.keys() if isinstance(run.summary.get(k), (int, float))])
    
    config_keys = sorted(all_config_keys)
    metric_keys = sorted(all_metric_keys)
    
    print(f"Found {len(config_keys)} hyperparameters")
    print(f"Found {len(metric_keys)} metrics")
    
    # Write CSV
    rows = []
    for run in runs:
        # Safely get attributes
        try:
            created_at = str(run.created_at) if hasattr(run, 'created_at') and run.created_at else ''
        except:
            created_at = ''
        
        try:
            finished_at = str(run.finished_at) if hasattr(run, 'finished_at') and run.finished_at else ''
        except:
            finished_at = ''
        
        row = {
            'run_id': run.id,
            'run_name': run.name,
            'state': run.state,
            'created_at': created_at,
            'finished_at': finished_at,
        }
        
        # Add config (hyperparameters)
        if run.config:
            for key in config_keys:
                val = run.config.get(key, '')
                row[key] = str(val) if isinstance(val, list) else val
        else:
            for key in config_keys:
                row[key] = ''
        
        # Add metrics (summary)
        if run.summary:
            for key in metric_keys:
                try:
                    val = run.summary.get(key, '')
                    row[key] = val if isinstance(val, (int, float)) else str(val)
                except:
                    row[key] = ''
        else:
            for key in metric_keys:
                row[key] = ''
        
        rows.append(row)
    
    # Write to file
    all_cols = ['run_id', 'run_name', 'state', 'created_at', 'finished_at'] + config_keys + metric_keys
    
    print(f"\nWriting {len(rows)} runs to {output_file}...")
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_cols)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"✅ Exported {len(rows)} runs to {output_file}")
    
    # Summary statistics
    states = {}
    for row in rows:
        state = row['state']
        states[state] = states.get(state, 0) + 1
    
    print(f"\n📊 Run Status Summary:")
    for state, count in sorted(states.items()):
        print(f"  - {state}: {count}")
    
    completed = [r for r in rows if r['state'] == 'finished']
    if completed:
        print(f"\n✅ Completed runs: {len(completed)}")
        if 'val_loss' in metric_keys:
            try:
                best = min([r for r in completed if r.get('val_loss') and str(r.get('val_loss')) != ''], 
                          key=lambda x: float(x['val_loss']) if x.get('val_loss') else float('inf'))
                print(f"\n🏆 Best completed run:")
                print(f"   Name: {best['run_name']}")
                print(f"   ID: {best['run_id']}")
                print(f"   val_loss: {best.get('val_loss', 'N/A')}")
                print(f"   val_acc: {best.get('val_acc', 'N/A')}")
                print(f"   Key hyperparameters:")
                for key in ['learning_rate', 'batch_size', 'hidden_layers', 'optimizer', 'activation']:
                    if best.get(key):
                        print(f"     {key}: {best[key]}")
            except Exception as e:
                print(f"   (Could not determine best run: {e})")
    
    print(f"\n💡 File saved to: {output_file}")
    print(f"   File size: ", end="")
    import os
    if os.path.exists(output_file):
        size = os.path.getsize(output_file)
        print(f"{size / 1024:.1f} KB")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python export_all_runs_fixed.py <sweep_path> [output.csv]")
        print("Example: python export_all_runs_fixed.py makssuppras1-danmarks-tekniske-universitet-dtu/neural-network-numpy/w8eqt3av")
        sys.exit(1)
    
    sweep_path = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "all_sweep_runs.csv"
    
    export_all_runs(sweep_path, output_file)

