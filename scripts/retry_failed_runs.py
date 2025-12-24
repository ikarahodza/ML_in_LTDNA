"""
Retry Failed Multi-Seed Training Runs
Automatically extracts failed runs from log and re-runs them
"""
import os
import subprocess
import sys
from datetime import datetime

# Failed runs - 3 remaining ExtraTrees failures with tkinter/matplotlib issues
FAILED_RUNS = [
    (789, 'mix', 'ExTraTrees.py'),
    (1415, 'mix', 'ExTraTrees.py'),
    (1617, 'mix', 'ExTraTrees.py'),
]

def run_training(seed, dataset, script_name):
    """Run a single training script with specified seed and dataset"""
    script_path = os.path.join('scripts', script_name)
    
    # Prepare environment variables
    env = os.environ.copy()
    env['DATASET_TYPE'] = dataset
    env['MPLBACKEND'] = 'Agg'  # Use non-interactive backend to avoid tkinter issues
    
    # Construct command - dataset must be first positional argument
    cmd = [sys.executable, script_path, dataset, '--seed', str(seed)]
    
    print(f"\nRunning: {script_name} | Dataset: {dataset.upper()} | Seed: {seed}")
    print("="*80)
    
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=False,
            text=True,
            timeout=3600  # 1 hour timeout per run
        )
        
        if result.returncode == 0:
            print(f"✓ SUCCESS: {script_name} | {dataset} | Seed={seed}")
            return True
        else:
            print(f"✗ FAILED: {script_name} | {dataset} | Seed={seed} (exit code: {result.returncode})")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"✗ TIMEOUT: {script_name} | {dataset} | Seed={seed}")
        return False
    except Exception as e:
        print(f"✗ ERROR: {script_name} | {dataset} | Seed={seed}")
        print(f"   Error: {str(e)}")
        return False

def main():
    print("="*80)
    print("RETRY FAILED MULTI-SEED TRAINING RUNS")
    print("="*80)
    print(f"Total failed runs to retry: {len(FAILED_RUNS)}")
    print("="*80)
    
    # Confirm
    response = input("\nRetry all failed runs? (yes/no): ").strip().lower()
    if response != 'yes':
        print("Aborted.")
        return
    
    print(f"\nStarting retries at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Track results
    results = {
        'success': [],
        'failed': []
    }
    
    # Run each failed training
    for idx, (seed, dataset, script_name) in enumerate(FAILED_RUNS, 1):
        print(f"\n[{idx}/{len(FAILED_RUNS)}] Retrying: Seed={seed} | Dataset={dataset} | Script={script_name}")
        
        success = run_training(seed, dataset, script_name)
        
        if success:
            results['success'].append((seed, dataset, script_name))
        else:
            results['failed'].append((seed, dataset, script_name))
        
        print("="*80)
    
    # Summary
    print("\n" + "="*80)
    print("RETRY SUMMARY")
    print("="*80)
    print(f"Total retried: {len(FAILED_RUNS)}")
    print(f"Successful: {len(results['success'])}")
    print(f"Still failing: {len(results['failed'])}")
    
    if results['failed']:
        print("\nStill failing:")
        for seed, dataset, script_name in results['failed']:
            print(f"  ✗ Seed={seed:4d} | {dataset:6s} | {script_name}")
    
    # Save log
    log_filename = f"retry_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(log_filename, 'w') as f:
        f.write(f"RETRY LOG - {datetime.now()}\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total retried: {len(FAILED_RUNS)}\n")
        f.write(f"Successful: {len(results['success'])}\n")
        f.write(f"Still failing: {len(results['failed'])}\n\n")
        
        f.write("Successful runs:\n")
        for seed, dataset, script_name in results['success']:
            f.write(f"  ✓ Seed={seed:4d} | {dataset:6s} | {script_name}\n")
        
        f.write("\nStill failing runs:\n")
        for seed, dataset, script_name in results['failed']:
            f.write(f"  ✗ Seed={seed:4d} | {dataset:6s} | {script_name}\n")
    
    print(f"\nLog saved to: {log_filename}")
    print("="*80)

if __name__ == "__main__":
    main()
