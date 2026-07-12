"""
Run all TabPFN explainability analyses for the Mix dataset.
Run this after overnight TabPFN v2.5 training is complete.

Steps:
  1. tabpfn_prototypes_and_permutation.py — prototypes + permutation importance
  2. tabpfn_explainability.py             — PDP / ICE plots
  3. tabpfn_lime_explanations.py          — LIME (TabPFN-specific, one sample)
  4. tabpfn_iml_analysis.py              — IML / ALE / kernel SHAP
  5. shap_comprehensive_analysis.py       — full SHAP for all models incl. TabPFN v2.5
  6. generate_shap_consensus_figure.py    — cross-model SHAP heatmap (needs step 5 first)

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
        "1/6  Prototypes & permutation importance",
        [sys.executable, os.path.join(SCRIPT_DIR, 'tabpfn_prototypes_and_permutation.py'), DATASET],
    ),
    (
        "2/6  PDP / ICE plots (TabPFN explainability)",
        [sys.executable, os.path.join(SCRIPT_DIR, 'tabpfn_explainability.py'), DATASET],
    ),
    (
        "3/6  LIME explanations (TabPFN)",
        [sys.executable, os.path.join(SCRIPT_DIR, 'tabpfn_lime_explanations.py'), DATASET],
    ),
    (
        "4/6  IML / ALE / kernel SHAP analysis",
        [sys.executable, os.path.join(SCRIPT_DIR, 'tabpfn_iml_analysis.py'), DATASET],
    ),
    (
        "5/6  SHAP for all models (incl. TabPFN v2.5)",
        [sys.executable, os.path.join(SCRIPT_DIR, 'shap_comprehensive_analysis.py'), DATASET],
    ),
    (
        "6/6  Cross-model SHAP consensus heatmap",
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
