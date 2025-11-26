# HPC utilities for managing paths and data synchronization
# Supports DTU HPC with BLACKHOLE scratch directory

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional


def get_hpc_scratch_dir() -> Optional[str]:
    """
    Get the HPC scratch directory path from BLACKHOLE environment variable.
    
    Returns:
        Path to scratch directory if BLACKHOLE is set, None otherwise
    """
    return os.environ.get('BLACKHOLE', None)


def is_on_hpc() -> bool:
    """
    Check if running on HPC by checking for BLACKHOLE environment variable.
    
    Returns:
        True if BLACKHOLE is set, False otherwise
    """
    return get_hpc_scratch_dir() is not None


def get_data_dir(base_dir: Optional[str] = None) -> str:
    """
    Get the appropriate data directory path.
    Uses HPC scratch if available, otherwise uses local data directory.
    
    Args:
        base_dir: Base directory for project (defaults to project root)
        
    Returns:
        Path to data directory
    """
    if base_dir is None:
        # Get project root (assuming this file is in src/)
        base_dir = Path(__file__).parent.parent
    
    if is_on_hpc():
        scratch_dir = get_hpc_scratch_dir()
        hpc_data_dir = os.path.join(scratch_dir, 'data')
        return hpc_data_dir
    else:
        return os.path.join(base_dir, 'data')


def get_results_dir(base_dir: Optional[str] = None) -> str:
    """
    Get the appropriate results directory path.
    Uses HPC scratch if available, otherwise uses local results directory.
    
    Args:
        base_dir: Base directory for project (defaults to project root)
        
    Returns:
        Path to results directory
    """
    if base_dir is None:
        base_dir = Path(__file__).parent.parent
    
    if is_on_hpc():
        scratch_dir = get_hpc_scratch_dir()
        hpc_results_dir = os.path.join(scratch_dir, 'results')
        return hpc_results_dir
    else:
        return os.path.join(base_dir, 'results')


def setup_hpc_directories():
    """
    Create necessary directories on HPC scratch if running on HPC.
    Creates: data/, results/models/, results/plots/, results/logs/
    """
    if not is_on_hpc():
        print("Not running on HPC. Skipping HPC directory setup.")
        return
    
    scratch_dir = get_hpc_scratch_dir()
    print(f"Setting up directories on HPC scratch: {scratch_dir}")
    
    # Create directories
    directories = [
        os.path.join(scratch_dir, 'data'),
        os.path.join(scratch_dir, 'results'),
        os.path.join(scratch_dir, 'results', 'models'),
        os.path.join(scratch_dir, 'results', 'plots'),
        os.path.join(scratch_dir, 'results', 'logs'),
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  ✓ {directory}")


def sync_data_to_hpc(local_data_dir: str, hpc_data_dir: Optional[str] = None):
    """
    Sync local data directory to HPC scratch.
    Uses rsync for efficient transfer (only transfers new/changed files).
    
    Args:
        local_data_dir: Local data directory path
        hpc_data_dir: HPC data directory path (defaults to $BLACKHOLE/data)
    """
    if not is_on_hpc():
        print("Not running on HPC. Cannot sync data.")
        return
    
    if hpc_data_dir is None:
        hpc_data_dir = get_data_dir()
    
    print(f"Syncing data from {local_data_dir} to {hpc_data_dir}...")
    
    # Ensure HPC directory exists
    os.makedirs(hpc_data_dir, exist_ok=True)
    
    # Use rsync for efficient transfer
    # -a: archive mode (preserves permissions, timestamps, etc.)
    # -v: verbose
    # --progress: show progress
    # --exclude: exclude unnecessary files
    try:
        subprocess.run([
            'rsync', '-av', '--progress',
            '--exclude', '__pycache__',
            '--exclude', '*.pyc',
            f'{local_data_dir}/',
            f'{hpc_data_dir}/'
        ], check=True)
        print(f"✓ Data synced successfully to {hpc_data_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error syncing data: {e}")
        print("Falling back to manual copy...")
        # Fallback: manual copy
        if os.path.exists(local_data_dir):
            for item in os.listdir(local_data_dir):
                src = os.path.join(local_data_dir, item)
                dst = os.path.join(hpc_data_dir, item)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)
            print(f"✓ Data copied to {hpc_data_dir}")


def sync_results_from_hpc(hpc_results_dir: Optional[str] = None, 
                          local_results_dir: Optional[str] = None):
    """
    Sync results from HPC scratch back to local machine.
    
    Args:
        hpc_results_dir: HPC results directory (defaults to $BLACKHOLE/results)
        local_results_dir: Local results directory (defaults to ./results)
    """
    if not is_on_hpc():
        print("Not running on HPC. Cannot sync results.")
        return
    
    if hpc_results_dir is None:
        hpc_results_dir = get_results_dir()
    
    if local_results_dir is None:
        local_results_dir = os.path.join(Path(__file__).parent.parent, 'results')
    
    print(f"Syncing results from {hpc_results_dir} to {local_results_dir}...")
    
    # Ensure local directory exists
    os.makedirs(local_results_dir, exist_ok=True)
    
    try:
        subprocess.run([
            'rsync', '-av', '--progress',
            f'{hpc_results_dir}/',
            f'{local_results_dir}/'
        ], check=True)
        print(f"✓ Results synced successfully to {local_results_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error syncing results: {e}")
        print("Please manually copy results from HPC before they are deleted!")


def print_hpc_info():
    """Print information about HPC setup."""
    if is_on_hpc():
        scratch_dir = get_hpc_scratch_dir()
        print("=" * 60)
        print("HPC DETECTED")
        print("=" * 60)
        print(f"Scratch directory: {scratch_dir}")
        print(f"Data directory: {get_data_dir()}")
        print(f"Results directory: {get_results_dir()}")
        print("\n⚠️  WARNING: Data in scratch will be deleted at service windows")
        print("⚠️  WARNING: All data will be deleted at end of January 2026")
        print("=" * 60)
    else:
        print("Not running on HPC (BLACKHOLE not set)")
        print("Using local directories")


if __name__ == '__main__':
    # Test HPC utilities
    print_hpc_info()
    
    if is_on_hpc():
        setup_hpc_directories()
        print("\nTo sync data to HPC, run:")
        print("  python -c \"from src.hpc_utils import sync_data_to_hpc; sync_data_to_hpc('./data')\"")

