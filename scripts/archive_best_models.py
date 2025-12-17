"""
Best Model Archive Script
Organizes and documents best performing models for deployment
"""

import os
import shutil
import json
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def create_deployment_archive():
    """Create deployment-ready archive of best models."""
    
    # Create deployment directory
    deploy_dir = os.path.join(PROJECT_ROOT, 'deployment')
    os.makedirs(deploy_dir, exist_ok=True)
    
    # Create subdirectories
    models_dir = os.path.join(deploy_dir, 'models')
    docs_dir = os.path.join(deploy_dir, 'documentation')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(docs_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("CREATING DEPLOYMENT ARCHIVE")
    print(f"{'='*80}\n")
    
    # Archive Mix dataset models
    print("[INFO] Archiving Mix dataset models...")
    mix_models = {
        'stacking_ensemble': {
            'model': 'Mix_Results/models/stacking_ensemble.pkl',
            'scaler': 'Mix_Results/models/stacking_ensemble_scaler.pkl',
            'metrics': 'Mix_Results/reports/stacking_ensemble_metrics.json',
            'confusion_matrix': 'Mix_Results/figures/stacking_ensemble_confusion_matrix.png',
            'balanced_accuracy': 0.8810,
            'description': 'Best model for Mix dataset. 2-level stacking with LogisticRegression meta-learner.'
        }
    }
    
    for model_name, files in mix_models.items():
        model_deploy_dir = os.path.join(models_dir, f'mix_{model_name}')
        os.makedirs(model_deploy_dir, exist_ok=True)
        
        # Copy files
        for file_type, file_path in files.items():
            if file_type in ['model', 'scaler', 'metrics', 'confusion_matrix']:
                src = os.path.join(PROJECT_ROOT, file_path)
                if os.path.exists(src):
                    dst = os.path.join(model_deploy_dir, os.path.basename(file_path))
                    shutil.copy2(src, dst)
                    print(f"  ✓ Copied {file_type}: {os.path.basename(file_path)}")
        
        # Create README
        readme_path = os.path.join(model_deploy_dir, 'README.md')
        with open(readme_path, 'w') as f:
            f.write(f"# {model_name} - Mix Dataset\\n\\nBalanced Accuracy: {files['balanced_accuracy']:.4f}\\n")
        print(f"  ✓ Created README.md\\n")
    
    print(f"\n{'='*80}")
    print("DEPLOYMENT ARCHIVE CREATED")
    print(f"{'='*80}")
    print(f"\nLocation: {deploy_dir}\n")
    
    return deploy_dir

if __name__ == '__main__':
    create_deployment_archive()
