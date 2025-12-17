"""
Run complete analysis pipeline for ensemble models
Executes CV, external validation, error analysis, and SHAP for both voting and stacking ensembles
"""

import os
import sys
import subprocess

# Get paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

def run_analysis_for_dataset(dataset_name):
    """Run all analyses for a specific dataset."""
    print(f"\n{'='*80}")
    print(f"Running Analysis Pipeline for {dataset_name}_Results")
    print(f"{'='*80}\n")
    
    analysis_dir = os.path.join(PROJECT_ROOT, f"{dataset_name}_Results", "analysis_scripts")
    
    if not os.path.exists(analysis_dir):
        print(f"[WARNING] Analysis directory not found: {analysis_dir}")
        return
    
    # List of analysis scripts to run
    analyses = [
        ("Cross-Validation", "cross_validation.py"),
        ("External Validation", "external_validation.py"),
        ("Error Analysis", "error_analysis.py"),
        ("SHAP Analysis", "shap_analysis.py"),
        ("Comprehensive Report", "generate_comprehensive_report.py")
    ]
    
    python_exe = sys.executable
    
    for analysis_name, script_name in analyses:
        script_path = os.path.join(analysis_dir, script_name)
        
        if os.path.exists(script_path):
            print(f"\n{'─'*80}")
            print(f"Running {analysis_name}...")
            print(f"{'─'*80}")
            
            try:
                result = subprocess.run(
                    [python_exe, script_path],
                    cwd=os.path.join(PROJECT_ROOT, f"{dataset_name}_Results"),
                    capture_output=False,
                    text=True
                )
                
                if result.returncode == 0:
                    print(f"✓ {analysis_name} completed successfully")
                else:
                    print(f"✗ {analysis_name} failed with return code {result.returncode}")
            
            except Exception as e:
                print(f"✗ Error running {analysis_name}: {e}")
        else:
            print(f"[SKIP] {script_name} not found")
    
    print(f"\n{'='*80}")
    print(f"Completed analysis pipeline for {dataset_name}_Results")
    print(f"{'='*80}\n")

def main():
    """Run analysis pipeline for both datasets."""
    print(f"\n{'#'*80}")
    print("ENSEMBLE MODEL ANALYSIS PIPELINE")
    print(f"{'#'*80}\n")
    
    print("This will run the complete analysis pipeline for ensemble models:")
    print("  1. Cross-Validation Analysis")
    print("  2. External Validation")
    print("  3. Error Analysis")
    print("  4. SHAP Analysis")
    print("  5. Comprehensive Report Generation")
    print()
    print("Note: This may take 10-30 minutes depending on your hardware.")
    print()
    
    # Run for both datasets
    for dataset in ["Mix", "Single"]:
        run_analysis_for_dataset(dataset)
    
    print(f"\n{'#'*80}")
    print("ALL ANALYSES COMPLETE")
    print(f"{'#'*80}\n")
    
    print("Results saved in:")
    print("  - Mix_Results/cross_validation_results/")
    print("  - Mix_Results/external_validation_results/")
    print("  - Mix_Results/data/ (error analysis)")
    print("  - Mix_Results/figures/ (SHAP plots)")
    print("  - Mix_Results/reports/comprehensive_analysis_report.html")
    print()
    print("  - Single_Results/cross_validation_results/")
    print("  - Single_Results/external_validation_results/")
    print("  - Single_Results/data/ (error analysis)")
    print("  - Single_Results/figures/ (SHAP plots)")
    print("  - Single_Results/reports/comprehensive_analysis_report.html")
    print()
    print("Next: Generate final PDF reports with optimized ensemble results!")

if __name__ == '__main__':
    main()
