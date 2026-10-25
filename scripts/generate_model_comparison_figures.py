"""
Two publication figures:
  1. Top-3 / bottom-3 models by balanced accuracy (best strategy per model, 10-seed mean +/- std)
  2. TabPFN performance across all resampling strategies (BA and F1, 10-seed mean +/- std)

Output:
  Mix_Results_PoI/figures/top3_bottom3_models.png
  Mix_Results_PoI/figures/tabpfn_by_strategy.png
"""
import os
import sys
import json
import glob as _glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.dirname(SCRIPT_DIR)
RESULTS    = 'Mix_Results_PoI'
FIG_DIR    = os.path.join(ROOT, RESULTS, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

sys.path.insert(0, SCRIPT_DIR)
from model_colors import MODEL_COLORS, MODEL_LABELS

with open(os.path.join(ROOT, RESULTS, 'multi_seed_aggregated', 'aggregated_results.json')) as f:
    data = json.load(f)

# TabPFN isn't in aggregated_results.json (it's aggregated separately, from the
# v3 JSONs); patch it in here from the 10-seed v3 reports.
_tabpfn_v3 = {}
for _path in (
    [os.path.join(ROOT, RESULTS, 'reports', 'combined_results_v3.json')] +
    sorted(_glob.glob(os.path.join(ROOT, RESULTS, 'seed_*_v3', 'reports', 'combined_results_v3.json')))
):
    with open(_path) as _f:
        _d = json.load(_f)['tabpfn']['results']
    for _strat, _m in _d.items():
        if _strat not in _tabpfn_v3:
            _tabpfn_v3[_strat] = {'ba': [], 'f1': []}
        _tabpfn_v3[_strat]['ba'].append(_m['balanced_accuracy'])
        _tabpfn_v3[_strat]['f1'].append(_m['f1_macro'])

data['tabpfn'] = {}
for _strat, _vals in _tabpfn_v3.items():
    _ba = np.array(_vals['ba'])
    _f1 = np.array(_vals['f1'])
    data['tabpfn'][_strat] = {
        'balanced_accuracy': {'mean': float(_ba.mean()), 'std': float(_ba.std())},
        'f1_macro':          {'mean': float(_f1.mean()), 'std': float(_f1.std())},
    }

STRAT_LABELS = {
    'ClassWeights':       'Class Weights',
    'Original':           'No Resampling',
    'RandomUnderSampler': 'Random Under-Sampling',
    'RandomOverSampler':  'Random Over-Sampling',
    'SMOTE':              'SMOTE',
    'BorderlineSMOTE':    'Borderline SMOTE',
}

# ── Derive best-per-model ─────────────────────────────────────────────────────
best_per_model = {}
for model_key, strategies in data.items():
    best_strat, best_metrics = max(
        strategies.items(), key=lambda kv: kv[1]['balanced_accuracy']['mean']
    )
    best_per_model[model_key] = {
        'key':     model_key,
        'label':   MODEL_LABELS.get(model_key, model_key),
        'color':   MODEL_COLORS.get(model_key, '#607d8b'),
        'ba_mean': best_metrics['balanced_accuracy']['mean'],
        'ba_std':  best_metrics['balanced_accuracy']['std'],
        'f1_mean': best_metrics['f1_macro']['mean'],
        'f1_std':  best_metrics['f1_macro']['std'],
    }

ranked  = sorted(best_per_model.values(), key=lambda x: x['ba_mean'], reverse=True)
top3    = ranked[:3]
bottom3 = ranked[-3:]

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Top-3 / Bottom-3
# ═══════════════════════════════════════════════════════════════════════════════
# Y positions: top3 at 6,5,4 — gap — bottom3 at 2,1,0
y_top    = [6, 5, 4]
y_bottom = [2, 1, 0]

fig, ax = plt.subplots(figsize=(8, 5))

for i, entry in enumerate(top3):
    ax.barh(y_top[i], entry['ba_mean'],
            xerr=entry['ba_std'], color=entry['color'],
            edgecolor='white', linewidth=0.8, height=0.65,
            error_kw=dict(ecolor='#37474f', elinewidth=1.2, capsize=4), zorder=3)
    ax.text(entry['ba_mean'] + entry['ba_std'] + 0.004,
            y_top[i], f"{entry['ba_mean']:.3f}", va='center', fontsize=8.5,
            color='#212121')

for i, entry in enumerate(bottom3):
    ax.barh(y_bottom[i], entry['ba_mean'],
            xerr=entry['ba_std'], color=entry['color'],
            edgecolor='white', linewidth=0.8, height=0.65,
            error_kw=dict(ecolor='#37474f', elinewidth=1.2, capsize=4), zorder=3)
    ax.text(entry['ba_mean'] + entry['ba_std'] + 0.004,
            y_bottom[i], f"{entry['ba_mean']:.3f}", va='center', fontsize=8.5,
            color='#212121')

ax.set_yticks(y_top + y_bottom)
ax.set_yticklabels([e['label'] for e in top3] + [e['label'] for e in bottom3],
                   fontsize=10)

# Background shading
ax.axhspan(3.5, 6.5, color='#e3f2fd', alpha=0.40, zorder=0)
ax.axhspan(-0.5, 2.5, color='#ffebee', alpha=0.40, zorder=0)

# Separator
ax.axhline(y=3, color='#90a4ae', linestyle='--', linewidth=1.0, zorder=2)

# Group labels
ax.text(1.06, 5, 'Top 3', transform=ax.get_yaxis_transform(),
        ha='left', va='center', fontsize=8.5, color='#1565c0', fontweight='bold')
ax.text(1.06, 1, 'Bottom 3', transform=ax.get_yaxis_transform(),
        ha='left', va='center', fontsize=8.5, color='#b71c1c', fontweight='bold')

ax.set_xlabel('Mean balanced accuracy (10 seeds, ± std)', fontsize=10)
ax.set_xlim(0, 1.08)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
ax.grid(axis='x', linestyle='--', alpha=0.4, linewidth=0.8)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
out1 = os.path.join(FIG_DIR, 'top3_bottom3_models.png')
plt.savefig(out1, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {out1}")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — TabPFN by resampling strategy
# ═══════════════════════════════════════════════════════════════════════════════
tabpfn_col = MODEL_COLORS['tabpfn']   # navy — consistent with all other figures
F1_COL     = '#43A047'                # green accent for F1 bars

tabpfn_entries = []
for strat, metrics in data['tabpfn'].items():
    tabpfn_entries.append({
        'label':   STRAT_LABELS.get(strat, strat),
        'ba_mean': metrics['balanced_accuracy']['mean'],
        'ba_std':  metrics['balanced_accuracy']['std'],
        'f1_mean': metrics['f1_macro']['mean'],
        'f1_std':  metrics['f1_macro']['std'],
    })
tabpfn_entries.sort(key=lambda x: x['ba_mean'], reverse=True)

n  = len(tabpfn_entries)
x  = np.arange(n)
bw = 0.35

fig, ax = plt.subplots(figsize=(8, 4.5))

bars_ba = ax.bar(x - bw/2,
                 [e['ba_mean'] for e in tabpfn_entries],
                 width=bw,
                 yerr=[e['ba_std'] for e in tabpfn_entries],
                 color=tabpfn_col, alpha=0.85, edgecolor='white',
                 error_kw=dict(ecolor='#37474f', elinewidth=1.2, capsize=4),
                 label='Balanced accuracy', zorder=3)

bars_f1 = ax.bar(x + bw/2,
                 [e['f1_mean'] for e in tabpfn_entries],
                 width=bw,
                 yerr=[e['f1_std'] for e in tabpfn_entries],
                 color=F1_COL, alpha=0.85, edgecolor='white',
                 error_kw=dict(ecolor='#37474f', elinewidth=1.2, capsize=4),
                 label='F1 macro', zorder=3)

for bar in bars_ba:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
            f'{h:.3f}', ha='center', va='bottom', fontsize=7.5, color='#212121')
for bar in bars_f1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
            f'{h:.3f}', ha='center', va='bottom', fontsize=7.5, color='#212121')

ax.set_xticks(x)
ax.set_xticklabels([e['label'] for e in tabpfn_entries], fontsize=9.5,
                   rotation=18, ha='right')
ax.set_ylabel('Score (10-seed mean ± std)', fontsize=10)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=9, framealpha=0.9)
ax.grid(axis='y', linestyle='--', alpha=0.4, linewidth=0.8)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
out2 = os.path.join(FIG_DIR, 'tabpfn_by_strategy.png')
plt.savefig(out2, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {out2}")
