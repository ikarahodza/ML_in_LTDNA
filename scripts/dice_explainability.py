"""
DiCE (Diverse Counterfactual Explanations) for all ML models.

For each model this script generates diverse counterfactual examples:
  "What feature values would need to change for a different classification?"

This is especially valuable in forensic DNA analysis: it shows the boundary
conditions between, e.g., an Allele and Back_Stutter classification.

Coverage:
- All models (same enumeration as shap_comprehensive_analysis.py)
- TabPFN: only v2 models
- MLP / SVM / LogReg: data is standardized (matching training); counterfactual
  feature deltas are therefore in standard-deviation units

Outputs per model (in <Results>/dice_explainability/<model_name>/):
  - counterfactuals_<model>_class<k>.csv   — raw CF table per desired class
  - feature_change_freq_<model>.png        — how often each feature changed
  - feature_change_magnitude_<model>.png   — mean |delta| per feature across CFs
  - dice_summary_<model>.json              — aggregate stats
"""

import os
import sys
import json
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import dice_ml
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_TYPE = sys.argv[1] if len(sys.argv) > 1 else 'mix'
DATASET_TYPE = DATASET_TYPE.lower()

if DATASET_TYPE == 'mix':
    DATA_FILE = 'Input_Data-Mix-train.csv'
    MODELS_DIR = 'Mix_Results/models'
    OUTPUT_DIR = 'Mix_Results/dice_explainability'
elif DATASET_TYPE == 'single':
    DATA_FILE = 'Input_Data-Single-train.csv'
    MODELS_DIR = 'Single_Results/models'
    OUTPUT_DIR = 'Single_Results/dice_explainability'
else:
    print(f"Error: Unknown dataset type '{DATASET_TYPE}'. Use 'mix' or 'single'.")
    sys.exit(1)

os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = ['Allele', 'Back_Stutter', 'Forward_Stutter', 'Pull_Up']
OUTCOME_COL = 'Classification'

# Number of diverse counterfactuals per query instance and desired class
N_CFS = 4
# Number of test samples to explain per class (balanced)
N_SAMPLES_PER_CLASS = 3
# DiCE generation method: 'random' is fastest; 'genetic' is more thorough
DICE_METHOD = 'random'

# Models that need standardized input
NEEDS_SCALING = ['svm', 'mlp', 'logistic_regression']

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data():
    """Load and split data consistently with training."""
    print(f"\n[1/4] Loading {DATASET_TYPE.upper()} dataset...")
    df = pd.read_csv(DATA_FILE)

    drop_cols = ['sample', 'PoI']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    # Keep Classification as a column (DiCE needs it in the dataframe)
    y = X[OUTCOME_COL]
    X_features = X.drop(columns=[OUTCOME_COL])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_features, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"   Training samples: {len(X_tr)}")
    print(f"   Test samples:     {len(X_te)}")
    print(f"   Features:         {X_tr.shape[1]}")
    print(f"   Classes:          {sorted(y.unique())}")
    return X_tr, X_te, y_tr, y_te


# ---------------------------------------------------------------------------
# Model enumeration
# ---------------------------------------------------------------------------
def find_all_models():
    """Find ALL model pkl files, excluding TabPFN (too slow for DiCE) and scaler-only files."""
    print("\n[2/4] Discovering models...")
    models_path = Path(MODELS_DIR)
    selected, skipped = [], []

    for pkl in sorted(models_path.glob('*.pkl')):
        stem = pkl.stem
        if stem.endswith('_scaler'):
            continue
        # Skip ALL TabPFN models — their per-prediction cost makes DiCE infeasible
        if 'tabpfn' in stem.lower():
            skipped.append(stem)
            continue
        selected.append(stem)

    print(f"   Including {len(selected)} models")
    if skipped:
        print(f"   Skipped {len(skipped)} TabPFN models (inference too slow for DiCE)")
    return selected


# ---------------------------------------------------------------------------
# Model + data helpers
# ---------------------------------------------------------------------------
def load_model(model_name):
    path = Path(MODELS_DIR) / f"{model_name}.pkl"
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"   ✗ Load failed: {e}")
        return None


def get_model_data(model_name, X_train, X_test):
    """Return (X_train_scaled, X_test_scaled, scaler_or_None)."""
    if 'stacking_ensemble' in model_name:
        sp = Path(MODELS_DIR) / 'stacking_ensemble_scaler.pkl'
        if sp.exists():
            sc = joblib.load(sp)
            Xtr = pd.DataFrame(sc.transform(X_train), columns=X_train.columns, index=X_train.index)
            Xte = pd.DataFrame(sc.transform(X_test), columns=X_test.columns, index=X_test.index)
            return Xtr, Xte, sc

    if any(n in model_name.lower() for n in NEEDS_SCALING):
        sc = StandardScaler()
        Xtr = pd.DataFrame(sc.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        Xte = pd.DataFrame(sc.transform(X_test), columns=X_test.columns, index=X_test.index)
        return Xtr, Xte, sc

    return X_train, X_test, None


def select_query_instances(X_test, y_test, model, n_per_class):
    """
    Pick n_per_class correctly-predicted test instances per class.
    Falls back to any instance if there are not enough correct predictions.
    """
    try:
        y_pred = model.predict(X_test)
    except Exception:
        y_pred = np.zeros(len(X_test), dtype=object)

    queries = {}
    for cls in CLASS_NAMES:
        correct_mask = (y_test.values == cls) & (y_pred == cls)
        idx = np.where(correct_mask)[0]
        if len(idx) < n_per_class:
            # Fall back to any instance of this class
            idx = np.where(y_test.values == cls)[0]
        idx = idx[:n_per_class]
        if len(idx) > 0:
            queries[cls] = X_test.iloc[idx].copy()
    return queries


# ---------------------------------------------------------------------------
# DiCE analysis
# ---------------------------------------------------------------------------
def run_dice_for_model(model_name, model, X_train_m, X_test_m, y_train, y_test):
    """Generate counterfactuals for all classes and save outputs."""
    out_dir = Path(OUTPUT_DIR) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_names = X_train_m.columns.tolist()

    # Build training DataFrame with outcome column (DiCE requires it)
    train_df = X_train_m.copy()
    train_df[OUTCOME_COL] = y_train.values

    # DiCE data descriptor
    d = dice_ml.Data(
        dataframe=train_df,
        continuous_features=feature_names,
        outcome_name=OUTCOME_COL
    )

    # DiCE model wrapper
    m = dice_ml.Model(model=model, backend="sklearn")

    exp = dice_ml.Dice(d, m, method=DICE_METHOD)

    # Select query instances
    queries = select_query_instances(X_test_m, y_test, model, N_SAMPLES_PER_CLASS)
    if not queries:
        print("   ✗ No query instances found — skipping")
        return None

    all_cfs = []          # list of dicts for aggregation
    all_deltas = []       # absolute feature changes per CF

    for src_class, query_df in queries.items():
        for desired_cls_idx, desired_cls_name in enumerate(CLASS_NAMES):
            if desired_cls_name == src_class:
                continue  # skip same-class

            try:
                result = exp.generate_counterfactuals(
                    query_instances=query_df,
                    total_CFs=N_CFS,
                    desired_class=desired_cls_idx,
                    verbose=False
                )
            except Exception as e:
                print(f"   ⚠  CF gen failed ({src_class}→{desired_cls_name}): {e}")
                continue

            for qi, cf_ex in enumerate(result.cf_examples_list):
                if cf_ex.final_cfs_df is None or cf_ex.final_cfs_df.empty:
                    continue

                cf_df = cf_ex.final_cfs_df.copy()
                orig_row = query_df.iloc[qi] if qi < len(query_df) else query_df.iloc[-1]

                # Compute deltas (only for feature columns, not the outcome)
                feat_cols = [c for c in cf_df.columns if c != OUTCOME_COL]
                deltas = cf_df[feat_cols].subtract(orig_row[feat_cols].values, axis=1).abs()
                all_deltas.append(deltas)

                # Tag with metadata
                cf_df['_source_class'] = src_class
                cf_df['_desired_class'] = desired_cls_name
                cf_df['_query_idx'] = qi
                all_cfs.append(cf_df)

            # Save per (src→desired) CSV
            if all_cfs:
                batch = [c for c in all_cfs
                         if c['_source_class'].iloc[0] == src_class
                         and c['_desired_class'].iloc[0] == desired_cls_name]
                if batch:
                    merged = pd.concat(batch, ignore_index=True)
                    fname = f"cf_{src_class}_to_{desired_cls_name}.csv".replace(' ', '_')
                    merged.to_csv(out_dir / fname, index=False)

    if not all_cfs:
        print("   ✗ No counterfactuals generated")
        return None

    # Full CF table
    full_cf = pd.concat(all_cfs, ignore_index=True)
    full_cf.to_csv(out_dir / f'all_counterfactuals_{model_name}.csv', index=False)
    print(f"   ✓ all_counterfactuals_{model_name}.csv ({len(full_cf)} rows)")

    # Feature change analysis
    delta_df = pd.concat(all_deltas, ignore_index=True)
    mean_delta = delta_df.mean().sort_values(ascending=False)
    change_freq = (delta_df > 1e-6).mean().sort_values(ascending=False)  # fraction of CFs where feature changed

    # -- Plot 1: mean absolute change per feature (top 20) -----------------
    top20_delta = mean_delta.head(20)
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(range(len(top20_delta)), top20_delta.values[::-1],
                   color=sns.color_palette("Blues_d", len(top20_delta)))
    ax.set_yticks(range(len(top20_delta)))
    ax.set_yticklabels(top20_delta.index[::-1], fontsize=9)
    ax.set_xlabel('Mean |Δ feature| across counterfactuals', fontsize=11)
    ax.set_title(f'DiCE — Feature Change Magnitude\n{model_name}', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f'feature_change_magnitude_{model_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ feature_change_magnitude_{model_name}.png")

    # -- Plot 2: change frequency per feature (top 20) ---------------------
    top20_freq = change_freq.head(20)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(top20_freq)), top20_freq.values[::-1],
            color=sns.color_palette("Oranges_d", len(top20_freq)))
    ax.set_yticks(range(len(top20_freq)))
    ax.set_yticklabels(top20_freq.index[::-1], fontsize=9)
    ax.set_xlabel('Fraction of counterfactuals where feature changed', fontsize=11)
    ax.set_title(f'DiCE — Feature Change Frequency\n{model_name}', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f'feature_change_freq_{model_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ feature_change_freq_{model_name}.png")

    # -- Summary JSON -------------------------------------------------------
    summary = {
        'model': model_name,
        'total_counterfactuals': len(full_cf),
        'top10_features_by_change_magnitude': mean_delta.head(10).to_dict(),
        'top10_features_by_change_frequency': change_freq.head(10).to_dict(),
    }
    with open(out_dir / f'dice_summary_{model_name}.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"   ✓ dice_summary_{model_name}.json")

    return summary


# ---------------------------------------------------------------------------
# Aggregate across models
# ---------------------------------------------------------------------------
def create_aggregated_analysis(all_summaries):
    """Combine per-model summaries into a cross-model feature importance view."""
    if not all_summaries:
        return

    print(f"\n{'='*80}")
    print("  Aggregated DiCE analysis")
    print(f"{'='*80}")

    # Aggregate mean change magnitudes across models
    agg_magnitude = {}
    for summary in all_summaries:
        for feat, val in summary.get('top10_features_by_change_magnitude', {}).items():
            agg_magnitude.setdefault(feat, []).append(val)

    if not agg_magnitude:
        return

    agg_df = pd.DataFrame({
        'feature': list(agg_magnitude.keys()),
        'mean_change': [np.mean(v) for v in agg_magnitude.values()],
        'n_models': [len(v) for v in agg_magnitude.values()],
    }).sort_values('mean_change', ascending=False)

    agg_df.to_csv(Path(OUTPUT_DIR) / 'dice_aggregated_feature_changes.csv', index=False)
    print(f"   ✓ dice_aggregated_feature_changes.csv")

    top = agg_df.head(20)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(top)), top['mean_change'].values[::-1],
            color=sns.color_palette("viridis", len(top)))
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top['feature'].values[::-1], fontsize=9)
    ax.set_xlabel('Mean |Δ feature| across models and counterfactuals', fontsize=11)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(OUTPUT_DIR) / 'dice_aggregated_feature_changes.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ dice_aggregated_feature_changes.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 80)
    print(f"  DiCE COUNTERFACTUAL EXPLAINABILITY — {DATASET_TYPE.upper()}")
    print("=" * 80)
    print(f"  Method: {DICE_METHOD} | CFs per query: {N_CFS} | Samples/class: {N_SAMPLES_PER_CLASS}")

    X_train, X_test, y_train, y_test = load_data()
    model_names = find_all_models()

    if not model_names:
        print("\n✗ No models found — exiting")
        return

    print(f"\n[3/4] Running DiCE for {len(model_names)} models...")

    all_summaries = []
    for name in model_names:
        print(f"\n{'─'*70}")
        print(f"  {name}")
        print(f"{'─'*70}")

        # Resume support: skip models already completed in a previous run
        done_marker = Path(OUTPUT_DIR) / name / f'dice_summary_{name}.json'
        if done_marker.exists():
            print("   ✓ Already completed — skipping (delete folder to re-run)")
            with open(done_marker) as f:
                all_summaries.append(json.load(f))
            continue

        model = load_model(name)
        if model is None:
            print("   ✗ Could not load — skipping")
            continue

        if not hasattr(model, 'predict_proba'):
            print("   ✗ No predict_proba — skipping")
            continue

        X_train_m, X_test_m, _ = get_model_data(name, X_train, X_test)

        try:
            summary = run_dice_for_model(
                name, model, X_train_m, X_test_m, y_train, y_test
            )
            if summary is not None:
                all_summaries.append(summary)
        except Exception as e:
            print(f"   ✗ Unexpected error: {e}")

    print(f"\n[4/4] Aggregating results...")
    create_aggregated_analysis(all_summaries)

    print("\n" + "=" * 80)
    print(f"  DONE — {len(all_summaries)}/{len(model_names)} models processed")
    print(f"  Results: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == '__main__':
    main()
