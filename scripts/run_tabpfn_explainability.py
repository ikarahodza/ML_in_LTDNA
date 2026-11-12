"""
Run all TabPFN explainability analyses for the Mix dataset.
Run this after TabPFN-3.5 training (scripts/TabPFN.py) is complete.

Steps:
  1. tabpfn_prototypes_and_permutation.py — prototypes + permutation importance
  2. tabpfn_explainability.py             — PDP / ICE plots
  3. shap_comprehensive_analysis.py       — full SHAP for all models incl. TabPFN-3.5
  4. generate_shap_consensus_figure.py    — cross-model SHAP heatmap (needs step 3 first)

Note: this previously also listed tabpfn_lime_explanations.py and
tabpfn_iml_analysis.py as steps 3-4, but neither script exists in this repo —
they were removed here rather than silently failing on every run.

Usage:
    python scripts/run_tabpfn_explainability.py
    python scripts/run_tabpfn_explainability.py mix      # explicit dataset
"""
import subprocess
import sys
import os

DATASET = sys.argv[1] if len(sys.argv) > 1 else 'mix'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)

env = os.environ.copy()
env['PYTHONIOENCODING'] = 'utf-8'

steps = [
    (
        "1/4  Prototypes & permutation importance",
        [sys.executable, os.path.join(SCRIPT_DIR, 'tabpfn_prototypes_and_permutation.py'), DATASET],
    ),
    (
        "2/4  PDP / ICE plots (TabPFN explainability)",
        [sys.executable, os.path.join(SCRIPT_DIR, 'tabpfn_explainability.py'), DATASET],
    ),
    (
        "3/4  SHAP for all models (incl. TabPFN-3.5)",
        [sys.executable, os.path.join(SCRIPT_DIR, 'shap_comprehensive_analysis.py'), DATASET],
    ),
    (
        "4/4  Cross-model SHAP consensus heatmap",
        [sys.executable, os.path.join(SCRIPT_DIR, 'generate_shap_consensus_figure.py')],
    ),
]

print(f"\nTabPFN Explainability Runner — dataset: {DATASET.upper()}")
print("=" * 60)

failed = []
for label, cmd in steps:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=ROOT, env=env)
    if result.returncode != 0:
        print(f"  WARNING: step failed (exit {result.returncode}) — continuing.")
        failed.append(label)

print(f"\n{'='*60}")
print("All steps complete.")
if failed:
    print(f"\nFailed steps ({len(failed)}):")
    for f in failed:
        print(f"  - {f}")
else:
    print("All steps succeeded.")
