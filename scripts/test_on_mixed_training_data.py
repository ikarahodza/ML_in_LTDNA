"""
Test all trained models on Mixed_Training_Data.csv
===================================================
Models trained on Mix and Single datasets are each tested on the
combined dataset (70% Mix + 30% Single data, 381 samples).

For all models: TabPFN v2.5 is used (files ending in _v2.pkl); old v1 pkl files are skipped.
Stacking ensembles are skipped.

Outputs (in 'mixed training data results/'):
  summary_results.csv          — all models ranked by balanced accuracy
  per_class_results.csv        — per-class precision / recall / F1
  detailed_results.json        — full metrics + confusion matrices
  balanced_accuracy_all.png    — horizontal bar chart, all models
  confusion_matrices_top4.png  — 2×2 grid for top 4 models
  per_class_f1_heatmap.png     — F1 heatmap for top 10 models
"""

import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, log_loss,
    confusion_matrix, matthews_corrcoef, cohen_kappa_score,
)

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR  = PROJECT_ROOT / "mixed training data results"
RESULTS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))
try:
    from model_colors import MODEL_COLORS
except ImportError:
    MODEL_COLORS = {}

SKIP_STEMS   = {'stacking_ensemble', 'stacking_ensemble_scaler', 'trained_rf_model'}
NEEDS_SCALER = {'logistic_regression', 'svm', 'mlp', 'tabpfn'}
CLASS_ORDER  = ['Allele', 'Back_Stutter', 'Forward_Stutter', 'Pull_Up']

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("=" * 70)
print("TESTING ALL MODELS ON MIXED_TRAINING_DATA.CSV")
print("=" * 70)
print("\n[1/5] Loading data...")

df = pd.read_csv(PROJECT_ROOT / "Mixed_Training_Data.csv")
drop_cols = ['sample', 'PoI', 'Classification']
X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
y_raw = df['Classification']

le = LabelEncoder()
le.fit(CLASS_ORDER)
y = le.transform(y_raw)

X_scaled = StandardScaler().fit_transform(X)

print(f"  {len(df)} samples, {X.shape[1]} features")
print(f"  Class distribution: {dict(zip(le.classes_, np.bincount(y)))}")

# ── 2. Discover models ────────────────────────────────────────────────────────
print("\n[2/5] Discovering models...")

def parse_stem(stem, dataset):
    """Return (model_type, strategy, is_v2) or None to skip."""
    if stem in SKIP_STEMS:
        return None

    is_v2 = stem.endswith('_v2')
    base  = stem[:-3] if is_v2 else stem

    if 'tabpfn' in base:
        # Both Mix and Single: use only v2.5 (_v2 files); skip old v1
        if not is_v2:
            return None
        return ('tabpfn', base.replace('model_tabpfn_', ''), is_v2)

    if is_v2:
        return None  # non-TabPFN _v2 files don't exist, but skip defensively

    if 'lgbm' in base:
        return ('lgbm', base.replace('lgbm_model_lgbm_', ''), False)
    if 'xgboost' in base:
        return ('xgboost', base.replace('model_xgboost_', ''), False)
    if 'extratrees' in base:
        return ('extratrees', base.replace('extratrees_model_extratrees_', ''), False)
    if 'randomforest' in base:
        return ('randomforest', base.replace('model_randomforest_', ''), False)
    if 'logistic_regression' in base:
        return ('logistic_regression', base.replace('model_logistic_regression_', ''), False)
    if 'svm' in base:
        return ('svm', base.replace('model_svm_', ''), False)
    if 'mlp' in base:
        return ('mlp', base.replace('model_mlp_', ''), False)
    if 'gnb' in base:
        return ('gnb', base.replace('model_gnb_', ''), False)
    return None

all_models = []
for dataset in ['Mix', 'Single']:
    models_dir = PROJECT_ROOT / f"{dataset}_Results" / "models"
    if not models_dir.exists():
        continue
    for pkl in sorted(models_dir.glob("*.pkl")):
        parsed = parse_stem(pkl.stem, dataset)
        if parsed is None:
            continue
        model_type, strategy, is_v2 = parsed
        label = f"v2.5" if is_v2 else "v1"
        all_models.append({
            'path':         pkl,
            'dataset':      dataset,
            'model_type':   model_type,
            'strategy':     strategy,
            'version':      label,
            'needs_scaler': model_type in NEEDS_SCALER,
            'display':      f"[{dataset}] {model_type} / {strategy}" + (" (v2.5)" if is_v2 else ""),
        })

print(f"  {len(all_models)} models found")
for ds in ['Mix', 'Single']:
    n = sum(1 for m in all_models if m['dataset'] == ds)
    print(f"  {ds}: {n}")

# ── 3. Evaluate ───────────────────────────────────────────────────────────────
print("\n[3/5] Evaluating...")

results, failed = [], []

for i, m in enumerate(all_models, 1):
    print(f"  [{i:2d}/{len(all_models)}] {m['display']} ...", end=' ', flush=True)
    X_in = X_scaled if m['needs_scaler'] else X.values
    try:
        model  = joblib.load(m['path'])
        y_pred = np.asarray(model.predict(X_in))

        # Normalise predictions to integers
        if y_pred.dtype.kind in ('U', 'S', 'O'):
            y_pred = le.transform(y_pred).astype(int)
        else:
            y_pred = y_pred.astype(int)

        try:
            y_prob = model.predict_proba(X_in)
        except Exception:
            y_prob = None

        rec = {
            'dataset':           m['dataset'],
            'model_type':        m['model_type'],
            'strategy':          m['strategy'],
            'version':           m['version'],
            'display':           m['display'],
            'balanced_accuracy': balanced_accuracy_score(y, y_pred),
            'f1_macro':          f1_score(y, y_pred, average='macro'),
            'accuracy':          accuracy_score(y, y_pred),
            'precision_macro':   precision_score(y, y_pred, average='macro', zero_division=0),
            'recall_macro':      recall_score(y, y_pred, average='macro', zero_division=0),
            'matthews_corrcoef': matthews_corrcoef(y, y_pred),
            'cohen_kappa':       cohen_kappa_score(y, y_pred),
            'confusion_matrix':  confusion_matrix(y, y_pred).tolist(),
            'per_class_precision': precision_score(y, y_pred, average=None, zero_division=0).tolist(),
            'per_class_recall':    recall_score(y, y_pred, average=None, zero_division=0).tolist(),
            'per_class_f1':        f1_score(y, y_pred, average=None, zero_division=0).tolist(),
            'log_loss': None,
        }
        if y_prob is not None:
            try:
                rec['log_loss'] = log_loss(y, y_prob)
            except Exception:
                pass

        results.append(rec)
        print(f"BA={rec['balanced_accuracy']:.3f}  F1={rec['f1_macro']:.3f}")

    except Exception as e:
        print(f"FAILED: {e}")
        failed.append({'display': m['display'], 'error': str(e)})

results.sort(key=lambda r: r['balanced_accuracy'], reverse=True)
print(f"\n  {len(results)} succeeded, {len(failed)} failed")

# ── 4. Save tables ────────────────────────────────────────────────────────────
print("\n[4/5] Saving tables...")

with open(RESULTS_DIR / "detailed_results.json", 'w') as f:
    json.dump({'results': results, 'failed': failed}, f, indent=2)

keep = ['dataset','model_type','strategy','version',
        'balanced_accuracy','f1_macro','accuracy',
        'precision_macro','recall_macro',
        'matthews_corrcoef','cohen_kappa','log_loss']
pd.DataFrame([{k: r[k] for k in keep} for r in results]).to_csv(
    RESULTS_DIR / "summary_results.csv", index=False)

rows = []
for r in results:
    for i, cls in enumerate(le.classes_):
        rows.append({
            'display':    r['display'],
            'dataset':    r['dataset'],
            'model_type': r['model_type'],
            'strategy':   r['strategy'],
            'version':    r['version'],
            'class':      cls,
            'precision':  r['per_class_precision'][i],
            'recall':     r['per_class_recall'][i],
            'f1':         r['per_class_f1'][i],
        })
pd.DataFrame(rows).to_csv(RESULTS_DIR / "per_class_results.csv", index=False)

print("  summary_results.csv")
print("  per_class_results.csv")
print("  detailed_results.json")

# ── 5. Figures ────────────────────────────────────────────────────────────────
print("\n[5/5] Generating figures...")

def model_color(model_type):
    return MODEL_COLORS.get(model_type, '#607d8b')

# Figure A — horizontal bar chart, all models
fig, ax = plt.subplots(figsize=(12, max(6, len(results) * 0.38)))
ba_vals = [r['balanced_accuracy'] for r in results]
colors  = [model_color(r['model_type']) for r in results]
ylabels = [f"[{r['dataset']}] {r['model_type']} / {r['strategy']}" +
           (" ★" if r['version'] == 'v2.5' else "")
           for r in results]

ax.barh(range(len(results)), ba_vals, color=colors,
        edgecolor='white', linewidth=0.6)
for i, v in enumerate(ba_vals):
    ax.text(v + 0.004, i, f"{v:.3f}", va='center', fontsize=7.5)

ax.set_yticks(range(len(results)))
ax.set_yticklabels(ylabels, fontsize=8)
ax.set_xlabel('Balanced Accuracy', fontsize=10)
ax.set_xlim(0, 1.10)
ax.set_title('Balanced Accuracy on Mixed_Training_Data (★ = TabPFN v2.5)', fontsize=10)
ax.grid(axis='x', linestyle='--', alpha=0.4, linewidth=0.7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "balanced_accuracy_all.png", dpi=150, bbox_inches='tight')
plt.close()
print("  balanced_accuracy_all.png")

# Figure B — confusion matrices for top 4 models
top4 = results[:4]
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
for ax, r in zip(axes.flat, top4):
    cm    = np.array(r['confusion_matrix'])
    im    = ax.imshow(cm, cmap='Blues', interpolation='nearest')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ticks = np.arange(len(le.classes_))
    ax.set_xticks(ticks); ax.set_xticklabels(le.classes_, rotation=35, ha='right', fontsize=8)
    ax.set_yticks(ticks); ax.set_yticklabels(le.classes_, fontsize=8)
    thresh = cm.max() / 2.0
    for ii, jj in np.ndindex(cm.shape):
        ax.text(jj, ii, int(cm[ii, jj]), ha='center', va='center',
                fontsize=11, fontweight='bold',
                color='white' if cm[ii, jj] > thresh else 'black')
    title = (f"[{r['dataset']}] {r['model_type']} / {r['strategy']}\n"
             f"BA={r['balanced_accuracy']:.3f}  F1={r['f1_macro']:.3f}")
    ax.set_title(title, fontsize=8.5, color=model_color(r['model_type']))
    ax.set_ylabel('True', fontsize=8)
    ax.set_xlabel('Predicted', fontsize=8)

plt.suptitle('Confusion Matrices — Top 4 Models on Mixed_Training_Data', fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "confusion_matrices_top4.png", dpi=200, bbox_inches='tight')
plt.close()
print("  confusion_matrices_top4.png")

# Figure C — per-class F1 heatmap, top 10
top10 = results[:min(10, len(results))]
f1_mat = np.array([r['per_class_f1'] for r in top10])
ylabels10 = [f"[{r['dataset']}] {r['model_type']} / {r['strategy']}"
             for r in top10]

fig, ax = plt.subplots(figsize=(8, max(4, len(top10) * 0.55)))
im = ax.imshow(f1_mat, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
plt.colorbar(im, ax=ax, label='F1 score')
ax.set_xticks(range(len(le.classes_)))
ax.set_xticklabels(le.classes_, fontsize=9)
ax.set_yticks(range(len(top10)))
ax.set_yticklabels(ylabels10, fontsize=8)
for i in range(len(top10)):
    for j in range(len(le.classes_)):
        ax.text(j, i, f"{f1_mat[i, j]:.2f}", ha='center', va='center', fontsize=8.5)
ax.set_title('Per-class F1 — Top 10 Models on Mixed_Training_Data', fontsize=10)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "per_class_f1_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print("  per_class_f1_heatmap.png")

# ── Summary printout ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TOP 10 RESULTS")
print("=" * 70)
for rank, r in enumerate(results[:10], 1):
    print(f"  {rank:2d}. {r['display']}")
    print(f"      BA={r['balanced_accuracy']:.4f}  F1={r['f1_macro']:.4f}  Acc={r['accuracy']:.4f}")

if failed:
    print(f"\nFAILED ({len(failed)}):")
    for f in failed:
        print(f"  {f['display']}: {f['error'][:80]}")

print(f"\nAll outputs saved to: {RESULTS_DIR}")
print("=" * 70)
