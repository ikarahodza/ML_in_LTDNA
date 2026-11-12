"""
Cross-model SHAP consensus heatmap.
Shows normalised feature importance for top features across all tree/boosting models,
revealing which signal positions are consistently identified regardless of model family.
Output: Mix_Results_PoI/figures/shap_consensus_heatmap.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import joblib
import shap

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
SHAP_DIR = os.path.join(ROOT, 'Mix_Results_PoI', 'shap_explainability')
FIG_DIR  = os.path.join(ROOT, 'Mix_Results_PoI', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

TOP_N        = 15   # features to consider per model
TOP_SHOWN    = 20   # rows shown in the final figure

# ── 1. Load pre-computed SHAP CSVs ───────────────────────────────────────────
csv_map = {
    'LightGBM\n(SMOTE)':      os.path.join(SHAP_DIR, 'lgbm_model_lgbm_smote',
                                            'shap_importance_lgbm_model_lgbm_smote.csv'),
    'Extra Trees\n(SMOTE)':   os.path.join(SHAP_DIR, 'extratrees_model_extratrees_smote',
                                            'shap_importance_extratrees_model_extratrees_smote.csv'),
    'XGBoost\n(Original)':    os.path.join(SHAP_DIR, 'model_xgboost_original',
                                            'shap_importance_model_xgboost_original.csv'),
    'TabPFN\n(Original)':     os.path.join(SHAP_DIR, 'model_tabpfn_original_v35',
                                            'shap_importance_model_tabpfn_original_v35.csv'),
}

shap_dfs = {}
for label, path in csv_map.items():
    df = pd.read_csv(path).set_index('feature')['mean_abs_shap']
    shap_dfs[label] = df
    print(f"Loaded {label.replace(chr(10),' ')}: {len(df)} features")

# ── 2. Compute RF SHAP from saved model ──────────────────────────────────────
rf_model_path = os.path.join(ROOT, 'Mix_Results_PoI', 'models', 'model_randomforest_smote.pkl')
data_path     = os.path.join(ROOT, 'Input_Data-Mix-train.csv')

if os.path.exists(rf_model_path) and os.path.exists(data_path):
    print("Computing RF SHAP from saved model...")
    raw = pd.read_csv(data_path)
    X   = raw.drop(['sample', 'PoI', 'Classification'], axis=1)
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    y   = LabelEncoder().fit_transform(raw['Classification'])
    X_tr, X_te = train_test_split(X, test_size=0.2, stratify=y, random_state=42)[:2]

    rf  = joblib.load(rf_model_path)
    exp = shap.TreeExplainer(rf)
    sv  = exp.shap_values(X_te)
    sv_arr = np.array(sv)
    # Handle both SHAP output formats:
    # Old: list of (n_samples, n_features) → stacked (n_classes, n_samples, n_features)
    # New: single array (n_samples, n_features, n_classes)
    if sv_arr.ndim == 3 and sv_arr.shape[0] == len(X_te):
        # New format: (n_samples, n_features, n_classes)
        mean_abs = np.mean(np.abs(sv_arr), axis=(0, 2))
    else:
        # Old format: (n_classes, n_samples, n_features)
        mean_abs = np.mean(np.abs(sv_arr), axis=(0, 1))
    rf_series = pd.Series(mean_abs, index=X.columns)
    shap_dfs['Random Forest\n(SMOTE)'] = rf_series
    print(f"RF SHAP computed: {len(rf_series)} features")
else:
    print("WARNING: RF model not found; skipping RF column.")

# ── 3. Build consensus matrix ────────────────────────────────────────────────
# Normalise each model's SHAP values to 0-1
norm_dfs = {}
for label, s in shap_dfs.items():
    mn, mx = s.min(), s.max()
    norm_dfs[label] = (s - mn) / (mx - mn + 1e-12)

all_features = set()
for label, s in shap_dfs.items():
    all_features.update(s.nlargest(TOP_N).index.tolist())

# Count how many models have each feature in their top-N
feature_counts = {}
for feat in all_features:
    cnt = sum(1 for s in shap_dfs.values() if feat in s.nlargest(TOP_N).index)
    feature_counts[feat] = cnt

# Sort: first by count (most models), then by mean normalised value
feature_mean_norm = {f: np.mean([norm_dfs[m].get(f, 0) for m in norm_dfs]) for f in all_features}
sorted_features = sorted(all_features,
                          key=lambda f: (feature_counts[f], feature_mean_norm[f]),
                          reverse=True)[:TOP_SHOWN]

model_labels = list(norm_dfs.keys())
matrix = np.zeros((len(sorted_features), len(model_labels)))
for j, lbl in enumerate(model_labels):
    for i, feat in enumerate(sorted_features):
        matrix[i, j] = norm_dfs[lbl].get(feat, 0.0)

# ── 4. Plot ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(len(model_labels) * 1.8 + 1.5, len(sorted_features) * 0.42 + 1.8))

cmap = plt.cm.YlOrRd
im   = ax.imshow(matrix, aspect='auto', cmap=cmap, vmin=0, vmax=1)

# Annotate cells with the count of models that have feature in top-N
for i, feat in enumerate(sorted_features):
    for j, lbl in enumerate(model_labels):
        val = matrix[i, j]
        in_top = feat in shap_dfs[lbl].nlargest(TOP_N).index
        if in_top:
            rank = list(shap_dfs[lbl].nlargest(TOP_N).index).index(feat) + 1
            text_color = 'white' if val > 0.6 else 'black'
            ax.text(j, i, f'#{rank}', ha='center', va='center',
                    fontsize=7.5, fontweight='bold', color=text_color)

ax.set_xticks(range(len(model_labels)))
ax.set_xticklabels(model_labels, fontsize=9)
ax.set_yticks(range(len(sorted_features)))

# Colour feature labels by region
def label_colour(feat):
    if feat.startswith('UpStream'):  return '#1565C0'
    if feat.startswith('DownStream'): return '#B71C1C'
    return '#1B5E20'

yticklabels = ax.set_yticklabels(sorted_features, fontsize=8)
for lbl, feat in zip(yticklabels, sorted_features):
    lbl.set_color(label_colour(feat))

# Consensus count annotation on right
ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())
ax2.set_yticks(range(len(sorted_features)))
ax2.set_yticklabels([f'{feature_counts[f]}/{len(model_labels)} models'
                      for f in sorted_features], fontsize=7.5, color='#555555')

cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
cbar.set_label('Normalised mean |SHAP|', fontsize=8)
cbar.ax.tick_params(labelsize=7)

# Legend for feature label colours
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='#1565C0', lw=2, label='UpStream context'),
    Line2D([0], [0], color='#B71C1C', lw=2, label='DownStream context'),
    Line2D([0], [0], color='#1B5E20', lw=2, label='Cross-channel (C1–C5)'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=7.5,
          framealpha=0.9, bbox_to_anchor=(1.0, -0.13))

plt.tight_layout()
out_path = os.path.join(FIG_DIR, 'shap_consensus_heatmap.png')
plt.savefig(out_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"\nSaved: {out_path}")
