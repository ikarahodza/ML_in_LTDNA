"""
Generate a print-ready scientific HTML summary report.
Open the output file in a browser and use File > Print > Save as PDF.
Output: Mix_Results/reports/scientific_summary.html
"""
import os, json, base64, datetime
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.dirname(SCRIPT_DIR)
MIX        = os.path.join(ROOT, 'Mix_Results')

# ── Data sources ──────────────────────────────────────────────────────────────
AGG_JSON     = os.path.join(MIX, 'multi_seed_aggregated', 'aggregated_results.json')
SEED42_JSON  = os.path.join(MIX, 'reports',    'combined_results.json')
TABPFN_JSON  = os.path.join(MIX, 'reports_v2', 'combined_results_v2.json')
OUT_PATH     = os.path.join(MIX, 'reports',    'scientific_summary.html')

os.makedirs(os.path.join(MIX, 'reports'), exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def img_tag(path, width='100%'):
    if not path or not os.path.exists(path):
        return '<p style="color:#999;font-style:italic;">Figure not available.</p>'
    with open(path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()
    ext = os.path.splitext(path)[1].lstrip('.')
    return f'<img src="data:image/{ext};base64,{b64}" style="width:{width};max-width:100%;border:1px solid #ddd;border-radius:4px;">'

def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding='utf-8') as f:
        return json.load(f)

# ── Load data ─────────────────────────────────────────────────────────────────
agg      = load_json(AGG_JSON)
seed42   = load_json(SEED42_JSON)
tabpfn42 = load_json(TABPFN_JSON)

# Merge seed-42 combined JSONs
combined42 = {}
combined42.update(seed42)
combined42.update(tabpfn42)

CLASS_NAMES = ['Allele', 'Back_Stutter', 'Forward_Stutter', 'Pull_Up']

TAN_BASELINES = {
    'MLP':  {'bal_acc': 0.735, 'f1': 0.831},
    'SVM':  {'bal_acc': 0.732, 'f1': 0.791},
    'LR':   {'bal_acc': 0.717, 'f1': 0.816},
    'RF':   {'bal_acc': 0.633, 'f1': 0.758},
    'GNB':  {'bal_acc': 0.546, 'f1': 0.507},
}

# Best strategy per model (for Table 1)
BEST_STRATEGY = {
    'tabpfn':              'ClassWeights',
    'extratrees':          'SMOTE',
    'lgbm':                'SMOTE',
    'randomforest':        'SMOTE',
    'xgboost':             'SMOTE',
    'logistic_regression': 'SMOTE',
    'mlp':                 'RandomOverSampler',
    'svm':                 'SMOTE',
    'gnb':                 'SMOTE',
}

DISPLAY_NAMES = {
    'tabpfn':              'TabPFN v2.5',
    'extratrees':          'Extra Trees',
    'lgbm':                'LightGBM',
    'randomforest':        'Random Forest',
    'xgboost':             'XGBoost',
    'logistic_regression': 'LR',
    'mlp':                 'MLP',
    'svm':                 'SVM',
    'gnb':                 'GNB',
}

# ── Table 1: Multi-seed balanced accuracy ─────────────────────────────────────
def build_table1():
    rows = []
    for model_key in ['tabpfn','extratrees','lgbm','randomforest','xgboost',
                      'logistic_regression','mlp','svm','gnb']:
        strategy = BEST_STRATEGY.get(model_key, 'Original')
        name     = DISPLAY_NAMES.get(model_key, model_key)
        data     = agg.get(model_key, {}).get(strategy, {})
        ba       = data.get('balanced_accuracy', {})
        f1       = data.get('f1_macro', {})
        n        = ba.get('n_seeds', '—')
        mean_ba  = ba.get('mean')
        std_ba   = ba.get('std')
        mean_f1  = f1.get('mean')

        is_tree   = model_key in ('tabpfn','extratrees','lgbm','randomforest','xgboost')
        bold_open  = '<strong>' if is_tree else ''
        bold_close = '</strong>' if is_tree else ''

        if mean_ba is not None:
            ba_str = f'{bold_open}{mean_ba:.3f} ± {std_ba:.3f}{bold_close}'
            f1_str = f'{mean_f1:.3f}' if mean_f1 is not None else '—'
        else:
            ba_str, f1_str = '—', '—'

        rows.append(f'''
        <tr class="{"tree-row" if is_tree else "baseline-row"}">
          <td>{bold_open}{name}{bold_close}</td>
          <td>{bold_open}{strategy}{bold_close}</td>
          <td style="text-align:center;">{ba_str}</td>
          <td style="text-align:center;">{f1_str}</td>
          <td style="text-align:center;">{n}</td>
        </tr>''')

    # Stacking ensemble
    rows.append('''
        <tr class="tree-row">
          <td>Stacking Ensemble</td>
          <td>—</td>
          <td style="text-align:center;">0.881 <span style="font-size:0.8em;color:#666;">(seed 42 only)</span></td>
          <td style="text-align:center;">0.843</td>
          <td style="text-align:center;">1*</td>
        </tr>''')

    # Tan et al. baselines
    rows.append('<tr><td colspan="5" class="section-divider">Tan et al. (2025) single-seed baselines (70:30 split)</td></tr>')
    for name, vals in TAN_BASELINES.items():
        rows.append(f'''
        <tr class="tan-row">
          <td>Tan et al. {name}</td>
          <td>—</td>
          <td style="text-align:center;">{vals["bal_acc"]:.3f} ‡</td>
          <td style="text-align:center;">{vals["f1"]:.3f}</td>
          <td style="text-align:center;">—</td>
        </tr>''')

    return ''.join(rows)


# ── Table 2: Per-class performance (seed 42) ──────────────────────────────────
PERCLASS_MODELS = [
    ('tabpfn',     'Original',  'TabPFN v2.5 (Original)'),
    ('xgboost',    'SMOTE',     'XGBoost (SMOTE)'),
    ('lgbm',       'SMOTE',     'LightGBM (SMOTE)'),
]

def fmt_prf(p, r, f):
    return f'P={p:.3f} / R={r:.3f} / F1={f:.3f}'

def build_table2():
    header_cols = ''.join(f'<th>{c.replace("_"," ")}<br><span style="font-weight:normal;font-size:0.85em;">(n=?)</span></th>' for c in CLASS_NAMES)
    rows = []
    for model_key, strategy, label in PERCLASS_MODELS:
        src = combined42.get(model_key, {}).get('results', {}).get(strategy)
        if src is None:
            rows.append(f'<tr><td>{label}</td>' + '<td colspan="4" style="text-align:center;color:#999;">Data not available</td></tr>')
            continue
        pp = src.get('per_class_precision', [None]*4)
        rr = src.get('per_class_recall',    [None]*4)
        ff = src.get('per_class_f1',        [None]*4)
        cells = ''
        for i in range(4):
            try:
                cells += f'<td style="font-size:0.82em;">{fmt_prf(pp[i], rr[i], ff[i])}</td>'
            except Exception:
                cells += '<td>—</td>'
        rows.append(f'<tr><td><strong>{label}</strong></td>{cells}</tr>')
    return header_cols, ''.join(rows)


header_cols, table2_rows = build_table2()
table1_rows = build_table1()

# ── Figures ───────────────────────────────────────────────────────────────────
violin_img  = img_tag(os.path.join(MIX, 'figures', 'multiseed_violin_plot.png'))
shap_img    = img_tag(os.path.join(MIX, 'figures', 'shap_consensus_heatmap.png'))
cm_tabpfn   = img_tag(os.path.join(MIX, 'figures_v2', 'confusion_matrix_tabpfn_original_v2.png'), width='72%')

date_str = datetime.date.today().strftime('%d %B %Y')

# ── HTML ──────────────────────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Scientific Summary — DNA Artifact Classification</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,wght@0,300;0,400;0,600;1,400&family=Source+Sans+3:wght@300;400;600&display=swap');

  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    font-family: 'Source Sans 3', 'Helvetica Neue', sans-serif;
    font-size: 10.5pt;
    line-height: 1.55;
    color: #1a1a1a;
    background: #f5f5f5;
  }}

  .page {{
    width: 210mm;
    min-height: 297mm;
    margin: 12mm auto;
    background: white;
    padding: 18mm 20mm 18mm 20mm;
    box-shadow: 0 2px 12px rgba(0,0,0,0.12);
  }}

  /* Header */
  .report-header {{
    border-bottom: 2.5px solid #1a3a6b;
    padding-bottom: 10px;
    margin-bottom: 18px;
  }}
  .report-title {{
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 15pt;
    font-weight: 600;
    color: #1a3a6b;
    line-height: 1.3;
    margin-bottom: 6px;
  }}
  .report-meta {{
    font-size: 8.5pt;
    color: #555;
    display: flex;
    gap: 24px;
  }}
  .badge {{
    display: inline-block;
    background: #e8f0fb;
    color: #1a3a6b;
    border: 1px solid #c0d0ee;
    border-radius: 3px;
    padding: 1px 7px;
    font-size: 7.5pt;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-left: 4px;
  }}

  /* Key metrics strip */
  .kpi-strip {{
    display: flex;
    gap: 10px;
    margin: 16px 0;
  }}
  .kpi {{
    flex: 1;
    background: #f0f4fb;
    border: 1px solid #c8d8f0;
    border-radius: 5px;
    padding: 10px 12px;
    text-align: center;
  }}
  .kpi-value {{
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 17pt;
    font-weight: 600;
    color: #1a3a6b;
    line-height: 1;
  }}
  .kpi-label {{
    font-size: 7.5pt;
    color: #555;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }}

  /* Sections */
  h2 {{
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 11.5pt;
    font-weight: 600;
    color: #1a3a6b;
    border-bottom: 1px solid #d0d8e8;
    padding-bottom: 4px;
    margin: 22px 0 10px 0;
  }}
  h3 {{
    font-size: 9.5pt;
    font-weight: 600;
    color: #333;
    margin: 14px 0 6px 0;
  }}
  p {{ margin-bottom: 7px; font-size: 9.5pt; }}

  /* Tables */
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 8.5pt;
    margin: 10px 0 14px 0;
  }}
  th {{
    background: #1a3a6b;
    color: white;
    padding: 5px 8px;
    text-align: left;
    font-weight: 600;
  }}
  td {{ padding: 4px 8px; border-bottom: 1px solid #e8e8e8; vertical-align: top; }}
  tr:hover td {{ background: #f8f9fc; }}
  .tree-row td {{ background: #f0f4fb; }}
  .tree-row:hover td {{ background: #e4ecf8; }}
  .baseline-row td {{ color: #555; }}
  .tan-row td {{ color: #777; font-style: italic; }}
  .section-divider td {{
    background: #e8ecf5;
    color: #333;
    font-weight: 600;
    font-style: normal;
    font-size: 8pt;
    padding: 5px 8px;
    letter-spacing: 0.03em;
  }}

  /* Figures */
  .figure-block {{
    margin: 14px 0;
    text-align: center;
  }}
  .figure-caption {{
    font-size: 8pt;
    color: #555;
    margin-top: 5px;
    font-style: italic;
    text-align: left;
  }}
  .figure-label {{
    font-weight: 600;
    font-style: normal;
    color: #1a3a6b;
  }}

  /* Two-column figure layout */
  .fig-row {{ display: flex; gap: 14px; margin: 14px 0; }}
  .fig-col {{ flex: 1; }}

  /* Footer */
  .report-footer {{
    margin-top: 24px;
    border-top: 1px solid #ddd;
    padding-top: 8px;
    font-size: 7.5pt;
    color: #888;
    display: flex;
    justify-content: space-between;
  }}

  /* Notes */
  .footnote {{
    font-size: 7.5pt;
    color: #666;
    margin-top: 4px;
  }}

  @media print {{
    body {{ background: white; }}
    .page {{ box-shadow: none; margin: 0; width: 100%; padding: 15mm 18mm; }}
  }}
</style>
</head>
<body>
<div class="page">

  <!-- Header -->
  <div class="report-header">
    <div class="report-title">
      Transformer-based in-context learning outperforms traditional classifiers<br>
      for peak classification in low-template DNA mixture electropherograms
      <span class="badge">Scientific Summary</span>
    </div>
    <div class="report-meta">
      <span>Karahodža I, Sehić K &nbsp;·&nbsp; University of Sarajevo</span>
      <span>TabPFN v2.5 &nbsp;·&nbsp; 10-seed evaluation</span>
      <span>Generated: {date_str}</span>
    </div>
  </div>

  <!-- KPI strip -->
  <div class="kpi-strip">
    <div class="kpi">
      <div class="kpi-value">0.847</div>
      <div class="kpi-label">TabPFN v2.5 balanced accuracy<br>(mean ± 0.045, n=10 seeds)</div>
    </div>
    <div class="kpi">
      <div class="kpi-value">+15.3%</div>
      <div class="kpi-label">Relative improvement<br>over Tan et al. best (MLP: 0.735)</div>
    </div>
    <div class="kpi">
      <div class="kpi-value">0.816</div>
      <div class="kpi-label">Best tree ensemble<br>(Extra Trees SMOTE, n=10)</div>
    </div>
    <div class="kpi">
      <div class="kpi-value">10</div>
      <div class="kpi-label">Independent random seeds<br>(stratified 80:20 split)</div>
    </div>
  </div>

  <!-- Table 1 -->
  <h2>Table 1 — Multi-seed balanced accuracy on the Mix dataset</h2>
  <table>
    <thead>
      <tr>
        <th style="width:22%;">Model</th>
        <th style="width:22%;">Best strategy</th>
        <th style="width:26%;text-align:center;">Balanced accuracy (mean ± SD)</th>
        <th style="width:16%;text-align:center;">F1-macro</th>
        <th style="width:14%;text-align:center;">n seeds</th>
      </tr>
    </thead>
    <tbody>
      {table1_rows}
    </tbody>
  </table>
  <p class="footnote">
    † TabPFN Original and ClassWeights produce identical results (internal imbalance handling).
    * Stacking Ensemble: single-seed result (seed 42); 30-fold CV on same seed yielded 0.844 ± 0.042.
    ‡ Single-seed estimates from a 70:30 train-test split (Tan et al., 2025).
    Bold rows: tree ensemble / transformer models.
  </p>

  <!-- Table 2 -->
  <h2>Table 2 — Per-class performance, seed 42 (Precision / Recall / F1)</h2>
  <table>
    <thead>
      <tr>
        <th style="width:26%;">Model</th>
        {header_cols}
      </tr>
    </thead>
    <tbody>
      {table2_rows}
    </tbody>
  </table>
  <p class="footnote">
    Seed 42, 80:20 stratified split. Test set: Allele n=45, Back Stutter n=18, Forward Stutter n=7, Pull-Up n=16.
  </p>

  <!-- Figure 1 -->
  <h2>Figure 1 — Multi-seed balanced accuracy distribution</h2>
  <div class="figure-block">
    {violin_img}
    <p class="figure-caption">
      <span class="figure-label">Figure 1.</span>
      Violin plots show the distribution of balanced accuracy across 10 random seeds for each model
      (best strategy shown). Diamond = mean; white circle = median; vertical bar = IQR; dots = individual seeds.
      Dashed reference lines indicate single-seed Tan et al. (2025) baselines.
    </p>
  </div>

  <!-- Figure 2 -->
  <h2>Figure 2 — Cross-model SHAP feature importance consensus</h2>
  <div class="figure-block">
    {shap_img}
    <p class="figure-caption">
      <span class="figure-label">Figure 2.</span>
      Normalised mean |SHAP| value (0–1 within each model) for the top-20 features appearing in at
      least one model's top-15. Cell annotation shows rank within that model. Blue labels: UpStream
      context; red: DownStream context; green: cross-channel (C1–C5).
      UpStream_97/98 and DownStream_40/44 appear consistently across all five model families,
      corresponding to one- and two-repeat-unit stutter product positions in tetranucleotide STR loci.
    </p>
  </div>

  <!-- Figure 3 -->
  <h2>Figure 3 — Confusion matrix, TabPFN v2.5 (Original, seed 42)</h2>
  <div class="figure-block" style="text-align:center;">
    {cm_tabpfn}
    <p class="figure-caption">
      <span class="figure-label">Figure 3.</span>
      Confusion matrix for TabPFN v2.5 (Original strategy, seed 42).
      Balanced accuracy: 0.835. Pull-Up recall = 1.000; Forward Stutter recall = 0.571 (n=7 test instances).
    </p>
  </div>

  <!-- Key findings -->
  <h2>Key Findings</h2>
  <h3>1. TabPFN v2.5 achieves state-of-the-art performance without hyperparameter tuning</h3>
  <p>
    TabPFN v2.5 attained a mean balanced accuracy of 0.847 ± 0.045 across 10 seeds, representing a
    15.3% relative improvement over the best result reported by Tan et al. (2025) for mixture EPGs
    (MLP: 0.735). All tree-based ensemble models (Extra Trees, LightGBM, Random Forest, XGBoost)
    also surpassed the Tan et al. ceiling, confirming that the performance bottleneck lies in classifier
    choice rather than the signal feature representation.
  </p>

  <h3>2. TabPFN is robust to class imbalance without explicit resampling</h3>
  <p>
    Applying ClassWeights or no resampling to TabPFN produced identical results (0.847 ± 0.045 for both),
    while SMOTE degraded performance to 0.753 ± 0.038. This indicates that TabPFN's pre-trained prior
    implicitly marginalises over class imbalance configurations, rendering external resampling redundant
    or harmful.
  </p>

  <h3>3. SHAP identifies biologically meaningful stutter-position features</h3>
  <p>
    UpStream_97/98 and DownStream_40/44 — corresponding to approximately two and one repeat units
    from the signal of interest — appear in the top features for all five independently trained model families.
    These positions align with the expected locations of back stutter (n−1) and forward stutter (n+1)
    products in tetranucleotide STR loci, providing model-agnostic evidence that classifiers have learned
    genuine biological relationships.
  </p>

  <h3>4. Forward Stutter remains the principal unresolved challenge</h3>
  <p>
    Forward Stutter is the smallest class (n=33 total; n=7 test instances per seed) and its feature-space
    neighbourhood is dominated by Allele (54% of top-5 prototypes) and Back Stutter (23%), with
    Forward Stutter itself appearing in only 17% of retrieved training examples. This distributional overlap
    is biologically expected and cannot be resolved without substantially more training data.
  </p>

  <h3>5. Multi-seed evaluation reveals the limits of single-seed benchmarking</h3>
  <p>
    Under 10-seed evaluation, traditional classifiers reproduced from Tan et al. achieved means of
    0.548–0.712, below the corresponding single-seed values (0.546–0.735). The gap is attributable to
    hyperparameter tuning differences and the stochastic optimism of single-seed estimates. Crucially,
    our tree-based and TabPFN models exceed the Tan et al. best result even under this stricter protocol,
    making the advantage robust.
  </p>

  <!-- Footer -->
  <div class="report-footer">
    <span>Karahodža &amp; Sehić · University of Sarajevo · {date_str}</span>
    <span>TabPFN v2.5 (pip: tabpfn 6.4.1) · scikit-learn 1.3 · 10 seeds · Mix dataset n=426</span>
  </div>

</div>
</body>
</html>"""

with open(OUT_PATH, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"Scientific summary saved to:\n  {OUT_PATH}")
print("Open in a browser and use File > Print > Save as PDF.")
