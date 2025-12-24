"""
Multi-Seed Training Orchestrator
Runs all models with multiple random seeds and stores results separately
Usage: python run_multi_seed_training.py
"""
import subprocess
import os
import sys
import time
from datetime import datetime

# Configuration
SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]
DATASETS = ['mix', 'single']
MODEL_SCRIPTS = [
    'ExTraTrees.py',
    'gnb.py', 
    'logre.py',
    'mlp.py',
    'svm.py',
    'TabPFN.py',
    'XGBoost.py',
    'lightGBM.py',
    'randomForest.py'
]

# Disable telemetry
os.environ['POSTHOG_DISABLED'] = '1'
os.environ['TABPFN_TELEMETRY'] = '0'

def run_training_with_seed(model_script, dataset, seed):
    """Run a single model with a specific seed"""
    print(f"\n{'='*80}")
    print(f"Running: {model_script} | Dataset: {dataset.upper()} | Seed: {seed}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            ['python', os.path.join('scripts', model_script), dataset, '--seed', str(seed)],
            check=True,
            capture_output=False,
            text=True
        )
        
        elapsed = time.time() - start_time
        print(f"✓ Completed in {elapsed:.2f} seconds")
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"✗ Failed after {elapsed:.2f} seconds: {e}")
        return False
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"✗ Error after {elapsed:.2f} seconds: {e}")
        return False

def main():
    print("="*80)
    print("MULTI-SEED TRAINING ORCHESTRATOR")
    print("="*80)
    print(f"Seeds: {SEEDS}")
    print(f"Datasets: {DATASETS}")
    print(f"Models: {len(MODEL_SCRIPTS)}")
    print(f"Total runs: {len(SEEDS)} seeds × {len(DATASETS)} datasets × {len(MODEL_SCRIPTS)} models = {len(SEEDS) * len(DATASETS) * len(MODEL_SCRIPTS)}")
    print("="*80)
    
    # Confirm with user
    response = input("\nThis will take a long time. Continue? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Aborted.")
        sys.exit(0)
    
    overall_start = time.time()
    total_runs = len(SEEDS) * len(DATASETS) * len(MODEL_SCRIPTS)
    completed = 0
    failed = 0
    
    # Track progress
    results_log = []
    
    for seed in SEEDS:
        for dataset in DATASETS:
            for model_script in MODEL_SCRIPTS:
                success = run_training_with_seed(model_script, dataset, seed)
                
                completed += 1 if success else 0
                failed += 0 if success else 1
                
                results_log.append({
                    'seed': seed,
                    'dataset': dataset,
                    'model': model_script,
                    'success': success
                })
                
                progress = (completed + failed) / total_runs * 100
                print(f"\n📊 Progress: {completed + failed}/{total_runs} ({progress:.1f}%) | ✓ {completed} | ✗ {failed}")
    
    overall_elapsed = time.time() - overall_start
    
    print("\n" + "="*80)
    print("MULTI-SEED TRAINING COMPLETE")
    print("="*80)
    print(f"Total time: {overall_elapsed:.2f} seconds ({overall_elapsed/60:.2f} minutes)")
    print(f"Successful: {completed}/{total_runs}")
    print(f"Failed: {failed}/{total_runs}")
    
    # Save results log
    log_file = f"multi_seed_training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("MULTI-SEED TRAINING LOG\n")
        f.write("="*80 + "\n\n")
        f.write(f"Completed: {datetime.now()}\n")
        f.write(f"Total runs: {total_runs}\n")
        f.write(f"Successful: {completed}\n")
        f.write(f"Failed: {failed}\n\n")
        f.write("Detailed Results:\n")
        f.write("-"*80 + "\n")
        for entry in results_log:
            status = "✓" if entry['success'] else "✗"
            f.write(f"{status} Seed={entry['seed']:4d} | {entry['dataset']:6s} | {entry['model']}\n")
    
    print(f"\nLog saved to: {log_file}")
    print("="*80)

if __name__ == "__main__":
    main()
