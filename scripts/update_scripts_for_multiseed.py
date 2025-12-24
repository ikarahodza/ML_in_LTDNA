"""
Script to update all model scripts to support --seed parameter
This script automatically modifies all 9 model scripts to accept random seed as parameter
"""
import os
import re

MODELS_TO_UPDATE = [
    'gnb.py',
    'logre.py', 
    'mlp.py',
    'svm.py',
    'TabPFN.py',
    'XGBoost.py',
    'lightGBM.py',
    'randomForest.py'
]

def update_script(filepath):
    """Update a model script to support --seed parameter"""
    print(f"\nUpdating: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Step 1: Add argparse import if not present
    if 'import argparse' not in content:
        content = content.replace('import os\n', 'import os\nimport argparse\n')
    
    # Step 2: Replace old argument parsing with new argparse-based version
    # Find the section that starts with "# Auto-detect dataset"
    old_arg_parse_pattern = r'# Auto-detect dataset.*?print\(f"No dataset specified.*?\n'
    if re.search(old_arg_parse_pattern, content, re.DOTALL):
        new_arg_parse = '''# Parse command line arguments
parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('dataset', nargs='?', default='single', choices=['mix', 'single'],
                    help='Dataset type: mix or single (default: single)')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for reproducibility (default: 42)')
args = parser.parse_args()

dataset_type = args.dataset.lower()
RANDOM_SEED = args.seed

'''
        content = re.sub(old_arg_parse_pattern, new_arg_parse, content, flags=re.DOTALL)
    
    # Step 3: Update path configuration to include seed-specific folders
    old_path_config = r"# Set paths based on dataset type\nRESULTS_FOLDER = f\"\{dataset_type\.capitalize\(\)\}_Results\"\n" + \
                     r"DATA_DIR = os\.path\.join\(PROJECT_ROOT, RESULTS_FOLDER, 'data'\)\n" + \
                     r"FIG_DIR = os\.path\.join\(PROJECT_ROOT, RESULTS_FOLDER, 'figures'\)\n" + \
                     r"MODEL_DIR = os\.path\.join\(PROJECT_ROOT, RESULTS_FOLDER, 'models'\)\n" + \
                     r"REPORTS_DIR = os\.path\.join\(PROJECT_ROOT, RESULTS_FOLDER, 'reports'\)"
    
    new_path_config = '''# Set paths based on dataset type and seed
RESULTS_FOLDER = f"{dataset_type.capitalize()}_Results"
if RANDOM_SEED != 42:
    # Store in seed-specific subfolder
    SEED_FOLDER = f"seed_{RANDOM_SEED}"
    DATA_DIR = os.path.join(PROJECT_ROOT, RESULTS_FOLDER, SEED_FOLDER, 'data')
    FIG_DIR = os.path.join(PROJECT_ROOT, RESULTS_FOLDER, SEED_FOLDER, 'figures')
    MODEL_DIR = os.path.join(PROJECT_ROOT, RESULTS_FOLDER, SEED_FOLDER, 'models')
    REPORTS_DIR = os.path.join(PROJECT_ROOT, RESULTS_FOLDER, SEED_FOLDER, 'reports')
else:
    # Use default folders for seed 42
    DATA_DIR = os.path.join(PROJECT_ROOT, RESULTS_FOLDER, 'data')
    FIG_DIR = os.path.join(PROJECT_ROOT, RESULTS_FOLDER, 'figures')
    MODEL_DIR = os.path.join(PROJECT_ROOT, RESULTS_FOLDER, 'models')
    REPORTS_DIR = os.path.join(PROJECT_ROOT, RESULTS_FOLDER, 'reports')'''
    
    content = re.sub(old_path_config, new_path_config, content)
    
    # Step 4: Update print statement to show seed
    content = re.sub(
        r'print\(f"\\n\{\'=\'\*60\}"\)\nprint\(f"Running .* with \{dataset_type\.upper\(\)\} dataset"\)\nprint\(f"Results folder: \{RESULTS_FOLDER\}"\)',
        '''print(f"\\n{'='*60}")
print(f"Running with {dataset_type.upper()} dataset")
print(f"Random Seed: {RANDOM_SEED}")
print(f"Results folder: {RESULTS_FOLDER}")''',
        content
    )
    
    # Step 5: Replace all random_state=42 with random_state=RANDOM_SEED
    content = re.sub(r'random_state=42\b', 'random_state=RANDOM_SEED', content)
    
    # Step 6: Handle RANDOM_STATE constant in randomForest.py
    if 'randomForest.py' in filepath:
        content = re.sub(r'^RANDOM_STATE = 42$', 'RANDOM_STATE = RANDOM_SEED', content, flags=re.MULTILINE)
    
    # Only write if changes were made
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✓ Updated successfully")
        return True
    else:
        print(f"  - No changes needed")
        return False

def main():
    print("="*80)
    print("UPDATING MODEL SCRIPTS FOR MULTI-SEED SUPPORT")
    print("="*80)
    
    scripts_dir = 'scripts'
    updated = 0
    
    for model_script in MODELS_TO_UPDATE:
        filepath = os.path.join(scripts_dir, model_script)
        if os.path.exists(filepath):
            if update_script(filepath):
                updated += 1
        else:
            print(f"\n✗ File not found: {filepath}")
    
    print(f"\n{'='*80}")
    print(f"✓ Update complete: {updated}/{len(MODELS_TO_UPDATE)} scripts updated")
    print(f"{'='*80}\n")
    print("ExTraTrees.py was already manually updated.")
    print("\nAll model scripts now support:")
    print("  python <script>.py [mix|single] --seed <value>")
    print("\nExamples:")
    print("  python ExTraTrees.py mix --seed 123")
    print("  python gnb.py single --seed 789")

if __name__ == "__main__":
    main()
