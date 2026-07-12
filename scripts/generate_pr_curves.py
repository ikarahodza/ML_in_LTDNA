"""
PR curves for the top 4 models (TabPFN, Extra Trees, LightGBM, XGBoost).
One panel per model, 4 one-vs-rest PR curves per panel.
Uses seed-42 prediction CSVs (consistent with confusion matrices).
Output: Mix_Results/figures/pr_curves_top4.png
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.dirname(SCRIPT_DIR)
DATA_DIR   = os.path.join(ROOT, 'Mix_Results', 'data')
FIG_DIR    = os.path.join(ROOT, 'Mix_Results', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

sys.path.insert(0, SCRIPT_DIR)
from model_colors import MODEL_COLORS

# Class index → display name, colour, linestyle
CLASSES = [
    (0, 'Allele',           '#1565C0', '-'),   # blue
    (1, 'Back Stutter',     '#2E7D32', '-'),   # green
    (2, 'Forward Stutter',  '#C62828', '--'),  # red dashed — highlights rare class
    (3, 'Pull-Up',          '#E65100', '-'),   # deep orange
]
PROB_COLS_NAMED  = ['Prob_Allele', 'Prob_Back_Stutter', 'Prob_Forward_Stutter', 'Prob_Pull_Up']
PROB_COLS_INDEXED = ['Prob_0', 'Prob_1', 'Prob_2', 'Prob_3']

MODELS = [
    ('tabpfn_original_v2', 'TabPFN (Original)',   MODEL_COLORS['tabpfn']),
    ('extratrees_smote',   'Extra Trees (SMOTE)',  MODEL_COLORS['extratrees']),
    ('lgbm_smote',         'LightGBM (SMOTE)',     MODEL_COLORS['lgbm']),
    ('xgboost_smote',      'XGBoost (SMOTE)',      MODEL_COLORS['xgboost']),
]

fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)

for ax, (key, title, model_col) in zip(axes.flat, MODELS):
    csv_path = os.path.join(DATA_DIR, f'predictions_{key}.csv')
    if not os.path.exists(csv_path):
        print(f'  MISSING: {csv_path}')
        ax.set_title(title, fontsize=11, fontweight='bold', color=model_col)
        continue

    df     = pd.read_csv(csv_path)
    y_true = df['True_Label'].values
    prob_cols = PROB_COLS_NAMED if PROB_COLS_NAMED[0] in df.columns else PROB_COLS_INDEXED
    y_prob = df[prob_cols].values

    for idx, cls_name, cls_col, ls in CLASSES:
        y_bin    = (y_true == idx).astype(int)
        scores   = y_prob[:, idx]
        baseline = y_bin.mean()

        precision, recall, _ = precision_recall_curve(y_bin, scores)
        ap = average_precision_score(y_bin, scores)

        ax.plot(recall, precision,
                color=cls_col, linestyle=ls, linewidth=1.8,
                label=f'{cls_name}  (AP={ap:.2f})')

        # No-skill baseline: dashed horizontal at class prevalence
        ax.axhline(baseline, color=cls_col, linestyle=':',
                   linewidth=0.9, alpha=0.45)

    # Colour the title with the model's canonical colour
    ax.set_title(title, fontsize=11, fontweight='bold', color=model_col)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(linestyle='--', alpha=0.35, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=8.5, loc='lower left', framealpha=0.88,
              handlelength=1.6, handletextpad=0.4)

for ax in axes[1]:
    ax.set_xlabel('Recall', fontsize=11)
for ax in axes[:, 0]:
    ax.set_ylabel('Precision', fontsize=11)

plt.tight_layout(h_pad=2.8, w_pad=2.2)
out = os.path.join(FIG_DIR, 'pr_curves_top4.png')
plt.savefig(out, dpi=200, bbox_inches='tight')
plt.close()
print(f'Saved: {out}')
