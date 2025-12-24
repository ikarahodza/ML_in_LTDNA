"""
Run all 9 models for both Mix and Single datasets in sequence.

This script executes all model training scripts for both dataset types,
providing comprehensive results for the entire pipeline.

Usage: python run_all_models.py
"""

import subprocess
import sys
import os
from datetime import datetime

# Disable telemetry to prevent PostHog errors
os.environ['POSTHOG_DISABLED'] = '1'
os.environ['TABPFN_TELEMETRY'] = '0'

# Get script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define all model scripts
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

# Dataset types
DATASETS = ['mix', 'single']

def run_model(script_name, dataset_type):
    """Run a single model script with the specified dataset type."""
    script_path = os.path.join(SCRIPT_DIR, script_name)
    
    print(f"\n{'='*80}")
    print(f"Running {script_name} with {dataset_type.upper()} dataset")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    try:
        # Run the script with dataset argument
        result = subprocess.run(
            [sys.executable, script_path, dataset_type],
            cwd=SCRIPT_DIR,
            capture_output=False,
            text=True,
            check=True
        )
        
        print(f"\n{'='*80}")
        print(f"✓ SUCCESS: {script_name} ({dataset_type.upper()}) completed")
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n{'='*80}")
        print(f"✗ ERROR: {script_name} ({dataset_type.upper()}) failed")
        print(f"Error code: {e.returncode}")
        print(f"{'='*80}\n")
        return False
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"✗ EXCEPTION: {script_name} ({dataset_type.upper()}) encountered an error")
        print(f"Error: {str(e)}")
        print(f"{'='*80}\n")
        return False

def main():
    """Main execution function."""
    start_time = datetime.now()
    
    print("\n" + "="*80)
    print("RUNNING ALL MODELS FOR BOTH DATASETS")
    print("="*80)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total models: {len(MODEL_SCRIPTS)}")
    print(f"Total datasets: {len(DATASETS)}")
    print(f"Total runs: {len(MODEL_SCRIPTS) * len(DATASETS)}")
    print("="*80 + "\n")
    
    results = {
        'success': [],
        'failed': []
    }
    
    total_runs = len(MODEL_SCRIPTS) * len(DATASETS)
    current_run = 0
    
    # Run each model for each dataset
    for dataset in DATASETS:
        print(f"\n{'#'*80}")
        print(f"# DATASET: {dataset.upper()}")
        print(f"{'#'*80}\n")
        
        for model_script in MODEL_SCRIPTS:
            current_run += 1
            print(f"\nProgress: {current_run}/{total_runs}")
            
            success = run_model(model_script, dataset)
            
            result_key = f"{model_script} ({dataset})"
            if success:
                results['success'].append(result_key)
            else:
                results['failed'].append(result_key)
    
    # Print final summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Start time:    {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time:      {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration:      {duration}")
    print(f"\nTotal runs:    {total_runs}")
    print(f"Successful:    {len(results['success'])}")
    print(f"Failed:        {len(results['failed'])}")
    print("="*80)
    
    if results['success']:
        print("\n✓ SUCCESSFUL RUNS:")
        for item in results['success']:
            print(f"  - {item}")
    
    if results['failed']:
        print("\n✗ FAILED RUNS:")
        for item in results['failed']:
            print(f"  - {item}")
    
    print("\n" + "="*80)
    
    # Return exit code based on failures
    if results['failed']:
        print(f"\nWARNING: {len(results['failed'])} run(s) failed.")
        return 1
    else:
        print("\nALL RUNS COMPLETED SUCCESSFULLY!")
        return 0

if __name__ == "__main__":
    sys.exit(main())
