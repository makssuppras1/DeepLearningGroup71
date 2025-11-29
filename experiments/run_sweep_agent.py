# Wrapper to run WandB sweep agent
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
import wandb
from experiments.train import train

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python run_sweep_agent.py <sweep-id> [count]")
        sys.exit(1)
    sweep_id = sys.argv[1]
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    print(f"Starting WandB sweep agent")
    print(f"Sweep ID: {sweep_id}")
    print(f"Runs: {count}")
    print(f"Project directory: {project_root}")
    print("")
    # Run agent - train functon called for each sweep run
    wandb.agent(sweep_id, function=train, count=count)
    print("")
    print("Sweep agent completed successfully")
