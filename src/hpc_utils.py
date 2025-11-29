# HPC utilties for DTU HPC with BLACKHOLE scratch
import os
import shutil
import subprocess
from pathlib import Path

def get_hpc_scratch_dir():
    # Get HPC scratch dir from BLACKHOLE env var
    return os.environ.get('BLACKHOLE', None)

def is_on_hpc():
    # Check if running on HPC
    return get_hpc_scratch_dir() is not None

def get_data_dir(base_dir=None):
    # Get data dir (HPC scratch or local)
    if base_dir is None:
        base_dir = Path(__file__).parent.parent
    if is_on_hpc():
        scratch_dir = get_hpc_scratch_dir()
        hpc_data_dir = os.path.join(scratch_dir, 'data')
        return hpc_data_dir
    else:
        return os.path.join(base_dir, 'data')

def get_results_dir(base_dir=None):
    # Get results dir (HPC scratch or local)
    if base_dir is None:
        base_dir = Path(__file__).parent.parent
    if is_on_hpc():
        scratch_dir = get_hpc_scratch_dir()
        hpc_results_dir = os.path.join(scratch_dir, 'results')
        return hpc_results_dir
    else:
        return os.path.join(base_dir, 'results')

def setup_hpc_directories():
    # Create necesary dirs on HPC scratch
    if not is_on_hpc():
        print("Not running on HPC. Skipping HPC directory setup.")
        return
    scratch_dir = get_hpc_scratch_dir()
    print(f"Setting up directories on HPC scratch: {scratch_dir}")
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

def sync_data_to_hpc(local_data_dir, hpc_data_dir=None):
    # Sync local data to HPC scratch
    if not is_on_hpc():
        print("Not running on HPC. Cannot sync data.")
        return
    if hpc_data_dir is None:
        hpc_data_dir = get_data_dir()
    print(f"Syncing data from {local_data_dir} to {hpc_data_dir}...")
    os.makedirs(hpc_data_dir, exist_ok=True)
    try:
        subprocess.run(['rsync', '-av', '--progress',
                       '--exclude', '__pycache__',
                       '--exclude', '*.pyc',
                       f'{local_data_dir}/',
                       f'{hpc_data_dir}/'], check=True)
        print(f"✓ Data synced successfully to {hpc_data_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error syncing data: {e}")
        print("Falling back to manual copy...")
        if os.path.exists(local_data_dir):
            for item in os.listdir(local_data_dir):
                src = os.path.join(local_data_dir, item)
                dst = os.path.join(hpc_data_dir, item)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)
            print(f"✓ Data copied to {hpc_data_dir}")

def sync_results_from_hpc(hpc_results_dir=None, local_results_dir=None):
    # Sync results from HPC back to local
    if not is_on_hpc():
        print("Not running on HPC. Cannot sync results.")
        return
    if hpc_results_dir is None:
        hpc_results_dir = get_results_dir()
    if local_results_dir is None:
        local_results_dir = os.path.join(Path(__file__).parent.parent, 'results')
    print(f"Syncing results from {hpc_results_dir} to {local_results_dir}...")
    os.makedirs(local_results_dir, exist_ok=True)
    try:
        subprocess.run(['rsync', '-av', '--progress',
                       f'{hpc_results_dir}/',
                       f'{local_results_dir}/'], check=True)
        print(f"✓ Results synced successfully to {local_results_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error syncing results: {e}")
        print("Please manually copy results from HPC before they are deleted!")

def print_hpc_info():
    # Print HPC setup info
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
    print_hpc_info()
    if is_on_hpc():
        setup_hpc_directories()
        print("\nTo sync data to HPC, run:")
        print("  python -c \"from src.hpc_utils import sync_data_to_hpc; sync_data_to_hpc('./data')\"")
