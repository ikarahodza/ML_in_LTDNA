"""
Compute SHAP values for TabPFN-3.5 (Original strategy).
Fits TabPFN fresh from training data — avoids PyTorch pickle issues with joblib.

Output:
  Mix_Results_PoI/shap_explainability/model_tabpfn_original_v35/shap_importance_model_tabpfn_original_v35.csv
  Mix_Results_PoI/shap_explainability/model_tabpfn_original_v35/shap_bar_model_tabpfn_original_v35.png

Then run: python scripts/generate_shap_consensus_figure.py
"""
import os
import sys

_key_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tabpfn.key')
if os.path.exists(_key_file):
    with open(_key_file) as _f:
        os.environ.setdefault('TABPFN_TOKEN', _f.read().strip())
os.environ['TABPFN_NO_BROWSER'] = '1'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tabpfn import TabPFNClassifier
import warnings
warnings.filterwarnings('ignore')

DATASET_TYPE   = sys.argv[1].lower() if len(sys.argv) > 1 else 'mix'
TABPFN_MODEL   = 'tabpfn-v3.5-20260909.safetensors'
BACKGROUND_N   = 3    # KernelExplainer background samples (TabPFN is slow)
TEST_N         = 3    # test samples to explain
NSAMPLES       = 50   # SHAP perturbations per sample

if DATASET_TYPE == 'mix':
    DATA_FILE = 'Input_Data-Mix-train.csv'
    OUT_BASE  = Path('Mix_Results_PoI/shap_explainability')
else:
    DATA_FILE = 'Input_Data-Single-train.csv'
    OUT_BASE  = Path('Single_Results_PoI/shap_explainability')

MODEL_NAME = 'model_tabpfn_original_v35'
OUT_DIR    = OUT_BASE / MODEL_NAME
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print(f"  TabPFN-3.5 SHAP — {DATASET_TYPE.upper()}")
print("=" * 60)

# ── 1. Load and split data (same split as training) ───────────────────────────
print(f"\n[1/4] Loading data...")
df   = pd.read_csv(DATA_FILE)
X    = df.drop(columns=['sample', 'PoI', 'Classification'])
y    = df['Classification']
le   = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
)
print(f"  train={len(X_train)}, test={len(X_test)}, features={X.shape[1]}")

# ── 2. Scale (matches training regime) ────────────────────────────────────────
scaler    = StandardScaler()
X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_s  = pd.DataFrame(scaler.transform(X_test),      columns=X_test.columns,  index=X_test.index)

# ── 3. Fit TabPFN (Original — no resampling) ──────────────────────────────────
print(f"\n[2/4] Fitting TabPFN-3.5 (Original)...")
clf = TabPFNClassifier(model_path=TABPFN_MODEL)
clf.fit(X_train_s, y_train)
ba = __import__('sklearn.metrics', fromlist=['balanced_accuracy_score']).balanced_accuracy_score
y_pred = clf.predict(X_test_s)
print(f"  ✓ Fitted  |  balanced accuracy = {ba(y_test, y_pred):.4f}")

# ── 4. Kernel SHAP ────────────────────────────────────────────────────────────
print(f"\n[3/4] Computing kernel SHAP  "
      f"(background={BACKGROUND_N}, test={TEST_N}, nsamples={NSAMPLES})...")
print(f"  Expected time: ~{TEST_N * NSAMPLES * 0.5 / 60:.0f}–"
      f"{TEST_N * NSAMPLES * 2 / 60:.0f} min on CPU")

background = shap.sample(X_train_s, BACKGROUND_N, random_state=42)

def _predict(X_arr):
    return clf.predict_proba(pd.DataFrame(X_arr, columns=X_train_s.columns))

explainer  = shap.KernelExplainer(_predict, background)
X_sample   = X_test_s.iloc[:TEST_N]
shap_vals  = explainer.shap_values(X_sample, nsamples=NSAMPLES, silent=False)
print(f"  ✓ SHAP values computed")

# ── 5. Aggregate: mean |SHAP| across classes ──────────────────────────────────
n_features = X_sample.shape[1]
if isinstance(shap_vals, list):
    # list of n_classes arrays each (n_samples, n_features)
    mean_abs = np.mean([np.abs(s).mean(axis=0) for s in shap_vals], axis=0)
else:
    sv_arr = np.array(shap_vals)
    if sv_arr.ndim == 3:
        if sv_arr.shape[1] == n_features:
            # (n_samples, n_features, n_classes)
            mean_abs = np.abs(sv_arr).mean(axis=(0, 2))
        elif sv_arr.shape[2] == n_features:
            # (n_classes, n_samples, n_features)
            mean_abs = np.abs(sv_arr).mean(axis=(0, 1))
        else:
            mean_abs = np.abs(sv_arr).reshape(-1, n_features).mean(axis=0)
    else:
        mean_abs = np.abs(sv_arr).mean(axis=0)

imp_df = pd.DataFrame({
    'feature':       X_sample.columns.tolist(),
    'mean_abs_shap': mean_abs,
}).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

# ── 6. Save outputs ───────────────────────────────────────────────────────────
print(f"\n[4/4] Saving outputs...")

csv_path = OUT_DIR / f'shap_importance_{MODEL_NAME}.csv'
imp_df.to_csv(csv_path, index=False)
print(f"  ✓ {csv_path}")

top20 = imp_df.head(20)
plt.figure(figsize=(10, 8))
plt.barh(top20['feature'][::-1], top20['mean_abs_shap'][::-1], color='steelblue', edgecolor='k', alpha=0.85)
plt.xlabel('Mean |SHAP value|', fontsize=11)
plt.tight_layout()
bar_path = OUT_DIR / f'shap_bar_{MODEL_NAME}.png'
plt.savefig(bar_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ {bar_path}")

print(f"\n  Top 10 features:")
for _, row in imp_df.head(10).iterrows():
    print(f"    {row['feature']:40s}  {row['mean_abs_shap']:.6f}")

print("\n" + "=" * 60)
print("  Done. Now run:")
print("    python scripts/generate_shap_consensus_figure.py")
print("=" * 60)
