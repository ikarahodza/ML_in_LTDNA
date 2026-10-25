"""
Redraws publication figures from existing computed data — no retraining needed.

Figures produced (all without embedded titles):
  1. Confusion matrices for the 4 best models (seed 42)
  2. SHAP bar plots for LightGBM, ExtraTrees, XGBoost, TabPFN
  3. Permutation importance aggregated (TabPFN)
  4. DiCE aggregated feature changes

Run after editing the figure-generating scripts to remove titles.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_colors import MODEL_COLORS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.dirname(SCRIPT_DIR)
RESULTS    = 'Mix_Results_PoI'
DATA_DIR   = os.path.join(ROOT, RESULTS, 'data')
FIG_DIR    = os.path.join(ROOT, RESULTS, 'figures')
SHAP_DIR   = os.path.join(ROOT, RESULTS, 'shap_explainability')
PERM_DIR   = os.path.join(ROOT, RESULTS, 'tabpfn_prototypes_and_permutation')
DICE_DIR   = os.path.join(ROOT, RESULTS, 'dice_explainability')
os.makedirs(FIG_DIR, exist_ok=True)

CLASS_NAMES = ['Allele', 'Back_Stutter', 'Forward_Stutter', 'Pull_Up']

# ── 1. Confusion matrices ─────────────────────────────────────────────────────
CONF_MODELS = [
    ('tabpfn_original_v3',   'TabPFN\n(Original)'),
    ('extratrees_smote',     'Extra Trees\n(SMOTE)'),
    ('lgbm_smote',           'LightGBM\n(SMOTE)'),
    ('xgboost_smote',        'XGBoost\n(SMOTE)'),
]

print("-- Confusion matrices --")
for key, label in CONF_MODELS:
    pred_path = os.path.join(DATA_DIR, f'predictions_{key}.csv')
    if not os.path.exists(pred_path):
        print(f"  MISSING: {pred_path}")
        continue
    df = pd.read_csv(pred_path)

    # True_Label is encoded integer; Predicted_Label is also integer
    # But Classification column has the string. Map via sorted class order.
    # Check column names
    if 'True_Label' in df.columns and 'Predicted_Label' in df.columns:
        y_true = df['True_Label'].values
        y_pred = df['Predicted_Label'].values
        # Map integer labels to class names using Classification column
        label_map = {i: c for i, c in enumerate(CLASS_NAMES)}
        y_true_str = [label_map.get(v, str(v)) for v in y_true]
        y_pred_str = [label_map.get(v, str(v)) for v in y_pred]
    else:
        y_true_str = df['True_Label'].values
        y_pred_str = df['Predicted_Label'].values

    cm = confusion_matrix(y_true_str, y_pred_str, labels=CLASS_NAMES)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(CLASS_NAMES))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(CLASS_NAMES, rotation=35, ha='right', fontsize=9)
    ax.set_yticklabels(CLASS_NAMES, fontsize=9)
    thresh = cm.max() / 2.0 if cm.size else 0
    for i, j in np.ndindex(cm.shape):
        ax.text(j, i, format(int(cm[i, j]), 'd'),
                ha='center', va='center', fontsize=11, fontweight='bold',
                color='white' if cm[i, j] > thresh else 'black')
    ax.set_ylabel('True label', fontsize=10)
    ax.set_xlabel('Predicted label', fontsize=10)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, f'confusion_matrix_{key}.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  OK confusion_matrix_{key}.png")

# ── 2. SHAP bar plots ─────────────────────────────────────────────────────────
SHAP_MODELS = [
    ('lgbm_model_lgbm_smote',
     'shap_importance_lgbm_model_lgbm_smote.csv',
     'shap_bar_lgbm_model_lgbm_smote.png',
     MODEL_COLORS['lgbm']),
    ('extratrees_model_extratrees_smote',
     'shap_importance_extratrees_model_extratrees_smote.csv',
     'shap_bar_extratrees_model_extratrees_smote.png',
     MODEL_COLORS['extratrees']),
    ('model_xgboost_original',
     'shap_importance_model_xgboost_original.csv',
     'shap_bar_model_xgboost_original.png',
     MODEL_COLORS['xgboost']),
    ('model_tabpfn_original_v3',
     'shap_importance_model_tabpfn_original_v3.csv',
     'shap_bar_model_tabpfn_original_v3.png',
     MODEL_COLORS['tabpfn']),
]

print("\n-- SHAP bar plots --")
for folder, csv_name, png_name, bar_color in SHAP_MODELS:
    csv_path = os.path.join(SHAP_DIR, folder, csv_name)
    if not os.path.exists(csv_path):
        print(f"  MISSING: {csv_path}")
        continue
    df = pd.read_csv(csv_path)
    top20 = df.head(20)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top20['feature'][::-1], top20['mean_abs_shap'][::-1],
            color=bar_color, edgecolor='k', alpha=0.85)
    ax.set_xlabel('Mean |SHAP value|', fontsize=11)
    ax.tick_params(axis='y', labelsize=8)
    plt.tight_layout()
    out = os.path.join(SHAP_DIR, folder, png_name)
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  OK {png_name}")

# ── 3. Permutation importance aggregated ─────────────────────────────────────
print("\n-- Permutation importance aggregated --")
perm_path = os.path.join(PERM_DIR, 'permutation_importance_aggregated.csv')
if os.path.exists(perm_path):
    df = pd.read_csv(perm_path)
    df = df.sort_values('mean_drop', ascending=False)
    order = np.arange(len(df))[::-1]   # already sorted; plot top→bottom

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['steelblue' if v >= 0 else '#c0392b' for v in df['mean_drop']]
    ax.barh(range(len(df)), df['mean_drop'].values[::-1],
            xerr=df['std_drop'].values[::-1] if 'std_drop' in df.columns else None,
            color=colors[::-1], edgecolor='k', alpha=0.85)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['feature'].values[::-1], fontsize=9)
    ax.axvline(0, color='k', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Mean balanced accuracy drop when feature is permuted', fontsize=10)
    plt.tight_layout()
    out = os.path.join(PERM_DIR, 'permutation_importance_aggregated.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  OK permutation_importance_aggregated.png")
else:
    print(f"  MISSING: {perm_path}")

# ── 4. DiCE aggregated ────────────────────────────────────────────────────────
print("\n-- DiCE aggregated --")
dice_path = os.path.join(DICE_DIR, 'dice_aggregated_feature_changes.csv')
if os.path.exists(dice_path):
    import seaborn as sns
    df = pd.read_csv(dice_path)
    top = df.head(20)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(range(len(top)), top['mean_change'].values[::-1],
            color=sns.color_palette("viridis", len(top)))
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top['feature'].values[::-1], fontsize=9)
    ax.set_xlabel('Mean |Δ feature| across models and counterfactuals', fontsize=11)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    out = os.path.join(DICE_DIR, 'dice_aggregated_feature_changes.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  OK dice_aggregated_feature_changes.png")
else:
    print(f"  MISSING: {dice_path}")

print("\nDone.")
