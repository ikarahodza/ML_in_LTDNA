"""
Run this the morning after overnight TabPFN training to:
  1. Re-run all TabPFN explainability analyses (SHAP, LIME, prototypes, PDP, IML)
  2. Re-aggregate multi-seed results (TabPFN v2.5 replaces v2)
  3. Regenerate violin plot and SHAP consensus heatmap
  4. Generate scientific summary report
"""
import subprocess
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.dirname(SCRIPT_DIR)

env = os.environ.copy()
env['PYTHONIOENCODING'] = 'utf-8'

steps = [
    ("All TabPFN explainability analyses",    [sys.executable, os.path.join(SCRIPT_DIR, "run_tabpfn_explainability.py"), "mix"]),
    ("Re-aggregating multi-seed results",     [sys.executable, os.path.join(SCRIPT_DIR, "aggregate_multi_seed_results.py")]),
    ("Regenerating violin plot",              [sys.executable, os.path.join(SCRIPT_DIR, "generate_multiseed_violin_plot.py")]),
    ("Generating scientific summary report",  [sys.executable, os.path.join(SCRIPT_DIR, "generate_scientific_summary.py")]),
]

for label, cmd in steps:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=ROOT, env=env)
    if result.returncode != 0:
        print(f"  WARNING: step failed (exit {result.returncode}) — continuing.")

print("\nAll post-processing steps complete.")
