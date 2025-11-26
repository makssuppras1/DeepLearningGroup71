#!/usr/bin/env python3
"""
Export WandB sweep data to CSV for analysis.

Usage:
    python scripts/export_sweep_data.py <sweep_id> [--output output.csv]
    
Example:
    python scripts/export_sweep_data.py w8eqt3av --output sweep_results.csv
"""

import argparse
import csv
import sys
import os
import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def export_sweep_to_csv(sweep_path, output_file="sweep_results.csv"):
    """
    Export all runs from a WandB sweep to CSV.
    
    Args:
        sweep_path: Full sweep path (e.g., "entity/project/sweep_id")
        output_file: Output CSV file path
    """
    print(f"Connecting to WandB...")
    
    # Initialize WandB API
    api = wandb.Api()
    
    # Parse sweep path
    parts = sweep_path.split("/")
    if len(parts) == 3:
        entity, project, sweep_id = parts
    elif len(parts) == 2:
        # Assume default entity
        project, sweep_id = parts
        entity = None
    else:
        raise ValueError(f"Invalid sweep path format: {sweep_path}. Use 'entity/project/sweep_id' or 'project/sweep_id'")
    
    print(f"Fetching sweep: {sweep_id} from project: {project}")
    if entity:
        print(f"Entity: {entity}")
    
    try:
        if entity:
            sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
        else:
            sweep = api.sweep(f"{project}/{sweep_id}")
    except Exception as e:
        print(f"Error fetching sweep: {e}")
        print(f"\nTrying alternative format...")
        # Try with default entity
        try:
            sweep = api.sweep(f"{project}/{sweep_id}")
        except Exception as e2:
            print(f"Error: {e2}")
            return False
    
    print(f"Sweep name: {sweep.name}")
    print(f"Fetching runs...")
    
    # Get all runs from the sweep
    runs = sweep.runs
    
    if len(runs) == 0:
        print("No runs found in this sweep.")
        return False
    
    print(f"Found {len(runs)} runs")
    
    # Collect all unique hyperparameter keys
    all_keys = set()
    for run in runs:
        if run.config:
            all_keys.update(run.config.keys())
    
    # Sort keys for consistent column order
    config_keys = sorted([k for k in all_keys if not k.startswith('_')])
    
    # Metrics to extract (common training metrics)
    metric_keys = [
        'val_loss', 'val_acc', 'train_loss', 'train_acc',
        'test_loss', 'test_acc', 'epoch', 'best_val_loss', 'best_val_acc'
    ]
    
    # Collect all metric keys from runs
    all_metrics = set()
    for run in runs:
        if run.summary:
            all_metrics.update(run.summary.keys())
    
    # Filter to numeric metrics and sort
    metric_keys = sorted([k for k in all_metrics if any(m in k.lower() for m in ['loss', 'acc', 'epoch'])])
    
    # Prepare CSV data
    rows = []
    for run in runs:
        row = {
            'run_id': run.id,
            'run_name': run.name,
            'state': run.state,
            'created_at': run.created_at.isoformat() if run.created_at else '',
            'finished_at': run.finished_at.isoformat() if run.finished_at else '',
            'duration': run.summary.get('_runtime', '') if run.summary else '',
        }
        
        # Add hyperparameters
        if run.config:
            for key in config_keys:
                value = run.config.get(key, '')
                # Convert lists to strings
                if isinstance(value, list):
                    value = str(value)
                row[key] = value
        
        # Add metrics (best/final values)
        if run.summary:
            for key in metric_keys:
                value = run.summary.get(key, '')
                row[key] = value
        
        rows.append(row)
    
    # Write to CSV
    all_columns = ['run_id', 'run_name', 'state', 'created_at', 'finished_at', 'duration'] + config_keys + metric_keys
    
    print(f"\nWriting {len(rows)} runs to {output_file}...")
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_columns)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"✅ Exported {len(rows)} runs to {output_file}")
    print(f"\nColumns exported:")
    print(f"  - Metadata: run_id, run_name, state, created_at, finished_at, duration")
    print(f"  - Hyperparameters: {len(config_keys)} parameters")
    print(f"  - Metrics: {len(metric_keys)} metrics")
    
    # Print summary statistics
    completed_runs = [r for r in rows if r['state'] == 'finished']
    if completed_runs:
        print(f"\n📊 Summary:")
        print(f"  - Completed runs: {len(completed_runs)}")
        print(f"  - Failed/running: {len(rows) - len(completed_runs)}")
        
        # Find best run by val_loss
        if 'val_loss' in metric_keys:
            try:
                best_runs = sorted(
                    [r for r in completed_runs if r.get('val_loss') and isinstance(r.get('val_loss'), (int, float))],
                    key=lambda x: float(x['val_loss'])
                )
                if best_runs:
                    best = best_runs[0]
                    print(f"\n🏆 Best run (lowest val_loss):")
                    print(f"  - Run: {best['run_name']} ({best['run_id']})")
                    print(f"  - val_loss: {best.get('val_loss', 'N/A')}")
                    print(f"  - val_acc: {best.get('val_acc', 'N/A')}")
                    print(f"  - Key hyperparameters:")
                    for key in config_keys[:10]:  # Show first 10
                        if best.get(key):
                            print(f"    {key}: {best[key]}")
            except Exception as e:
                print(f"  (Could not determine best run: {e})")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Export WandB sweep data to CSV')
    parser.add_argument('sweep_path', help='Sweep path (entity/project/sweep_id or project/sweep_id)')
    parser.add_argument('--output', '-o', default='sweep_results.csv', help='Output CSV file (default: sweep_results.csv)')
    parser.add_argument('--entity', help='WandB entity (optional, can be in sweep_path)')
    parser.add_argument('--project', help='WandB project (optional, can be in sweep_path)')
    
    args = parser.parse_args()
    
    # If entity/project provided separately, construct sweep path
    if args.entity and args.project:
        sweep_path = f"{args.entity}/{args.project}/{args.sweep_path}"
    else:
        sweep_path = args.sweep_path
    
    success = export_sweep_to_csv(sweep_path, args.output)
    
    if success:
        print(f"\n💡 Next steps:")
        print(f"  1. Review the CSV file: {args.output}")
        print(f"  2. Share it with me for analysis")
        print(f"  3. I can help identify patterns and suggest better hyperparameter ranges")
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()

