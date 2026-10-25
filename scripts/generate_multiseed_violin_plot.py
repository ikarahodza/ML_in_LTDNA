"""
Multi-seed balanced accuracy violin + strip plot for all Mix models.
Shows the full distribution across 10 seeds for each model, sorted by mean.
Output: Mix_Results_PoI/figures/multiseed_violin_plot.png
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_colors import DISPLAY_COLORS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS = 'Mix_Results_PoI'
FIG_DIR = os.path.join(ROOT, RESULTS, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]

# Best strategy per model (based on 10-seed mean balanced accuracy)
MODELS = [
    ('tabpfn',              'ClassWeights',     'TabPFN'),
    ('extratrees',          'SMOTE',            'Extra Trees'),
    ('lgbm',                'SMOTE',            'LightGBM'),
    ('randomforest',        'SMOTE',            'Random Forest'),
    ('xgboost',             'SMOTE',            'XGBoost'),
    ('logistic_regression', 'SMOTE',            'LR'),
    ('mlp',                 'RandomOverSampler','MLP'),
    ('svm',                 'SMOTE',            'SVM'),
    ('gnb',                 'SMOTE',            'GNB'),
]

# Tan et al. (2025) single-seed Mix baselines for reference lines
TAN_BASELINES = {
    'MLP': 0.735,
    'SVM': 0.732,
    'LR':  0.717,
}


# ── Collect per-seed values ───────────────────────────────────────────────────
records = []
for seed in SEEDS:
    # Standard combined_results.json for all non-TabPFN models
    path = (os.path.join(ROOT, RESULTS, 'reports', 'combined_results.json')
            if seed == 42 else
            os.path.join(ROOT, RESULTS, f'seed_{seed}', 'reports', 'combined_results.json'))
    with open(path) as f:
        data = json.load(f)
    # TabPFN v3 results live in a separate JSON / seed folder
    path_v3 = (os.path.join(ROOT, RESULTS, 'reports', 'combined_results_v3.json')
               if seed == 42 else
               os.path.join(ROOT, RESULTS, f'seed_{seed}_v3', 'reports', 'combined_results_v3.json'))
    data_v3 = {}
    if os.path.exists(path_v3):
        with open(path_v3) as f:
            data_v3 = json.load(f)

    for model_key, strategy, label in MODELS:
        src = data_v3 if model_key == 'tabpfn' else data
        if model_key in src and 'results' in src[model_key]:
            strat_data = src[model_key]['results'].get(strategy)
            if strat_data:
                records.append({
                    'model': label,
                    'seed':  seed,
                    'balanced_accuracy': strat_data['balanced_accuracy'],
                })

df = pd.DataFrame(records)

# Sort models by mean balanced accuracy
model_order = (df.groupby('model')['balanced_accuracy']
                 .mean()
                 .sort_values(ascending=False)
                 .index.tolist())

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5.5))

positions = range(len(model_order))
color_map = {m: DISPLAY_COLORS.get(m, '#607d8b') for m in model_order}

violin_parts = ax.violinplot(
    [df[df['model'] == m]['balanced_accuracy'].values for m in model_order],
    positions=list(positions),
    showmeans=False, showmedians=False, showextrema=False,
    widths=0.65,
)

for i, (pc, model) in enumerate(zip(violin_parts['bodies'], model_order)):
    pc.set_facecolor(color_map[model])
    pc.set_alpha(0.35)
    pc.set_edgecolor(color_map[model])
    pc.set_linewidth(1.2)

# Box (IQR)
for i, model in enumerate(model_order):
    vals = df[df['model'] == model]['balanced_accuracy'].values
    q1, med, q3 = np.percentile(vals, [25, 50, 75])
    ax.vlines(i, q1, q3, color=color_map[model], linewidth=5, alpha=0.7, zorder=3)
    ax.scatter(i, med, s=55, color='white', zorder=5, linewidths=1.5,
               edgecolors=color_map[model])

# Individual seed points (jittered)
rng = np.random.default_rng(0)
for i, model in enumerate(model_order):
    vals = df[df['model'] == model]['balanced_accuracy'].values
    jitter = rng.uniform(-0.12, 0.12, size=len(vals))
    ax.scatter(i + jitter, vals, s=20, color=color_map[model],
               alpha=0.75, zorder=4, linewidths=0)

# Mean marker
for i, model in enumerate(model_order):
    mean_val = df[df['model'] == model]['balanced_accuracy'].mean()
    ax.scatter(i, mean_val, s=80, marker='D', color=color_map[model],
               zorder=6, linewidths=1.2, edgecolors='white')

# Tan et al. reference lines
line_styles = ['--', '-.', ':']
ref_colors  = ['#546e7a', '#78909c', '#90a4ae']
for (name, val), ls, rc in zip(TAN_BASELINES.items(), line_styles, ref_colors):
    ax.axhline(val, color=rc, linestyle=ls, linewidth=1.2, alpha=0.85,
               label=f'Tan et al. {name} = {val:.3f}')

ax.set_xticks(list(positions))
ax.set_xticklabels(model_order, fontsize=10)
ax.set_ylabel('Balanced Accuracy', fontsize=11)
ax.set_ylim(0.38, 1.01)
ax.set_xlim(-0.6, len(model_order) - 0.4)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
ax.grid(axis='y', linestyle='--', alpha=0.4, linewidth=0.8)
ax.set_axisbelow(True)

# Tan et al. legend
tan_handles = [
    plt.Line2D([0], [0], color=rc, linestyle=ls, linewidth=1.5, label=f'Tan et al. {n} ({v:.3f})')
    for (n, v), ls, rc in zip(TAN_BASELINES.items(), line_styles, ref_colors)
]
ax.legend(handles=tan_handles, fontsize=8.5, loc='lower right',
          framealpha=0.9, title='Reference baselines (single seed, 70:30 split)',
          title_fontsize=8)

plt.tight_layout()
out_path = os.path.join(FIG_DIR, 'multiseed_violin_plot.png')
plt.savefig(out_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {out_path}")
