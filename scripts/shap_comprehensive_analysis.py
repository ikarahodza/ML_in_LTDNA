"""
Comprehensive SHAP Analysis Script for ALL ML Models
Generates detailed SHAP visualizations and insights.

Features:
- TreeExplainer for tree-based models (ExtraTrees, RandomForest, XGBoost, LightGBM)
- LinearExplainer for logistic regression
- KernelExplainer for other models (SVM, MLP, GNB, TabPFN-3.5)
- Summary plots (beeswarm and bar charts)
- Dependence plots for top features
- Waterfall plots
- Interactive HTML force plots
- Aggregated feature importance across all models

Model coverage:
- All models in the models directory are included
- TabPFN: only the current _v35 (TabPFN-3.5) checkpoint; superseded _v3 models are skipped
- MLP, SVM, LogReg: data is standardized before SHAP (matches training regime)
"""

import os
import sys

# Must be set before tabpfn is imported (joblib unpickling triggers the import)
_key_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tabpfn.key')
if os.path.exists(_key_file):
    with open(_key_file) as _f:
        os.environ.setdefault('TABPFN_TOKEN', _f.read().strip())
os.environ['TABPFN_NO_BROWSER'] = '1'

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATASET_TYPE = sys.argv[1] if len(sys.argv) > 1 else 'mix'
DATASET_TYPE = DATASET_TYPE.lower()

if DATASET_TYPE == 'mix':
    DATA_FILE = 'Input_Data-Mix-train.csv'
    MODELS_DIR = 'Mix_Results_PoI/models'
    OUTPUT_DIR = 'Mix_Results_PoI/shap_explainability'
elif DATASET_TYPE == 'single':
    DATA_FILE = 'Input_Data-Single-train.csv'
    MODELS_DIR = 'Single_Results_PoI/models'
    OUTPUT_DIR = 'Single_Results_PoI/shap_explainability'
else:
    print(f"Error: Invalid dataset type '{DATASET_TYPE}'. Use 'mix' or 'single'")
    sys.exit(1)

os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

# Models that need standardized input (they were trained on scaled data)
NEEDS_SCALING = ['svm', 'mlp', 'logistic_regression', 'tabpfn']
# KernelExplainer sample limits for slow models
KERNEL_MAX_BACKGROUND = 50
KERNEL_MAX_TEST = 50
# TabPFN: keep very small — KernelExplainer calls predict_proba
# nsamples × n_test times, each on a batch of n_background rows
TABPFN_MAX_BACKGROUND = 5
TABPFN_MAX_TEST = 5
TABPFN_NSAMPLES = 100   # perturbations per test point (default would be 2000+)


def load_data():
    """Load and split data consistently with training (stratified, random_state=42)."""
    print(f"\n[1/5] Loading {DATASET_TYPE.upper()} dataset...")
    df = pd.read_csv(DATA_FILE)

    drop_cols = ['sample', 'PoI', 'Classification']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df['Classification']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples:     {len(X_test)}")
    print(f"   Features:         {X_train.shape[1]}")
    return X_train, X_test, y_train, y_test


def find_all_models():
    """Find ALL model pkl files with these exclusions:
    - scaler-only files
    - superseded TabPFN checkpoints (only the current _v35 TabPFN-3.5 models are kept)
    - stacking_ensemble (not a single model worth explaining this way)
    All models (including TabPFN) now live directly under MODELS_DIR.
    """
    print("\n[2/5] Discovering models...")
    models_path = Path(MODELS_DIR)

    selected = []
    skipped  = []

    for pkl in sorted(models_path.glob('*.pkl')):
        stem = pkl.stem
        if stem.endswith('_scaler'):
            continue
        if stem == 'stacking_ensemble':
            skipped.append(f"{stem} (ensemble excluded)")
            continue
        if 'tabpfn' in stem.lower() and not stem.endswith('_v35'):
            skipped.append(f"{stem} (superseded TabPFN checkpoint; using _v35 only)")
            continue
        selected.append(stem)

    print(f"   Including {len(selected)} models:")
    for name in selected:
        print(f"     + {name}")
    if skipped:
        print(f"\n   Excluded:")
        for name in skipped:
            print(f"     - {name}")
    return selected


def get_model_data(model_name, X_train, X_test):
    """Return appropriately scaled/unscaled data for the given model."""
    if 'stacking_ensemble' in model_name:
        scaler_path = Path(MODELS_DIR) / 'stacking_ensemble_scaler.pkl'
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            Xtr = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
            Xte = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
            return Xtr, Xte

    if any(n in model_name.lower() for n in NEEDS_SCALING):
        scaler = StandardScaler()
        Xtr = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        Xte = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
        return Xtr, Xte

    return X_train, X_test


def load_model(model_name):
    """Load a trained model from disk."""
    model_path = Path(MODELS_DIR) / f"{model_name}.pkl"
    if model_path.exists():
        try:
            return joblib.load(model_path)
        except Exception as e:
            print(f"   ✗ Failed to load {model_name}: {e}")
            return None
    return None


def _wrap_predict(model):
    """Return a plain lambda for predict_proba.

    SHAP 0.49+ introspects bound methods to find the underlying estimator and
    may try to set feature_names_in_ on it.  XGBoost 2.x made that property
    read-only, causing a crash.  A plain lambda has no __self__, so SHAP
    cannot reach the model object through it.
    """
    return lambda X: model.predict_proba(X)


class XGBNativeExplainer:
    """XGBoost's own tree SHAP via pred_contribs=True.

    Avoids shap.TreeExplainer entirely — SHAP 0.49 cannot parse the
    base_score field in models saved with XGBoost 2.x (it was a scalar in
    older versions, a per-class array in newer ones).
    """
    def __init__(self, model):
        import xgboost as xgb
        self._xgb = xgb
        self.booster = model.get_booster()
        self.expected_value = None  # set after first shap_values call

    def shap_values(self, X):
        dm = self._xgb.DMatrix(X)
        # shape: (n_samples, n_classes, n_features + 1)
        contribs = self.booster.predict(dm, pred_contribs=True)
        # last column per class is the bias term
        bias = contribs[:, :, -1]        # (n_samples, n_classes)
        self.expected_value = bias[0].tolist()
        sv = contribs[:, :, :-1]         # (n_samples, n_classes, n_features)
        # Return as list-of-arrays like shap.TreeExplainer does
        return [sv[:, i, :] for i in range(sv.shape[1])]


def get_explainer(model, X_train_model, model_name):
    """Create the most appropriate SHAP explainer for the model type."""
    name = model_name.lower()

    # XGBoost: use native pred_contribs — shap.TreeExplainer cannot parse the
    # base_score format of models saved with XGBoost 2.x
    if 'xgboost' in name:
        explainer = XGBNativeExplainer(model)
        return explainer, 'XGBNative'

    # Other tree-based models: standard TreeExplainer
    if any(t in name for t in ['extratrees', 'randomforest', 'lgbm', 'lightgbm']):
        try:
            explainer = shap.TreeExplainer(model)
            return explainer, 'TreeExplainer'
        except Exception as e:
            print(f"   Warning: TreeExplainer failed ({e}), falling back to Kernel...")

    # Linear models
    if 'logistic' in name:
        try:
            explainer = shap.LinearExplainer(model, X_train_model)
            return explainer, 'LinearExplainer'
        except Exception as e:
            print(f"   Warning: LinearExplainer failed ({e}), falling back to Kernel...")

    # TabPFN-3.5: KernelExplainer with very small background and capped nsamples
    if 'tabpfn' in name:
        bg = shap.sample(X_train_model, TABPFN_MAX_BACKGROUND)
        explainer = shap.KernelExplainer(_wrap_predict(model), bg)
        return explainer, 'KernelExplainer_TabPFN'

    # SVM, MLP, GNB and anything else: KernelExplainer
    bg = shap.sample(X_train_model, KERNEL_MAX_BACKGROUND)
    explainer = shap.KernelExplainer(_wrap_predict(model), bg)
    return explainer, 'KernelExplainer'


def generate_shap_values(explainer, explainer_type, X_test_model, model_name):
    """Generate SHAP values with sample limits appropriate for the explainer."""
    if 'TabPFN' in explainer_type:
        limit = TABPFN_MAX_TEST
    elif explainer_type == 'KernelExplainer':
        limit = KERNEL_MAX_TEST
    else:
        limit = 200

    X_sample = X_test_model.iloc[:limit] if len(X_test_model) > limit else X_test_model

    if 'TabPFN' in explainer_type:
        # Cap perturbations to avoid O(n_features * n_background * n_test) explosion
        shap_values = explainer.shap_values(X_sample, nsamples=TABPFN_NSAMPLES, silent=True)
    else:
        shap_values = explainer.shap_values(X_sample)

    return shap_values, X_sample


def handle_multiclass_shap(shap_values):
    """Convert multiclass SHAP output to 2-D (samples x features) mean-abs array."""
    if isinstance(shap_values, list):
        # list of [n_samples x n_features] arrays — one per class
        return np.mean(np.abs(np.stack(shap_values, axis=0)), axis=0)
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # shape (n_samples, n_features, n_classes)
        return np.mean(np.abs(shap_values), axis=2)
    return shap_values


def create_summary_plots(shap_values, X_sample, model_name, out_dir):
    """Beeswarm and bar summary plots."""
    sv = handle_multiclass_shap(shap_values)

    for plot_type in ('dot', 'bar'):
        try:
            plt.figure(figsize=(12, 8))
            shap.summary_plot(sv, X_sample, show=False, plot_type=plot_type, max_display=20)
            plt.tight_layout()
            suffix = 'beeswarm' if plot_type == 'dot' else 'bar'
            plt.savefig(out_dir / f'shap_{suffix}_{model_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✓ shap_{suffix}_{model_name}.png")
        except Exception as e:
            print(f"   ✗ {plot_type} summary plot: {e}")


def create_dependence_plots(shap_values, X_sample, model_name, out_dir, top_n=5):
    """Dependence plots for the top-N most important features."""
    sv = handle_multiclass_shap(shap_values)
    mean_abs = np.abs(sv).mean(axis=0)
    if mean_abs.ndim > 1:
        mean_abs = mean_abs.mean(axis=1)

    top_idx = np.argsort(mean_abs)[::-1][:top_n]
    feat_names = X_sample.columns.tolist()

    for idx in top_idx:
        try:
            fname = feat_names[int(idx)]
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(int(idx), sv, X_sample, show=False)
            plt.title(f'SHAP Dependence: {fname} - {model_name}', fontsize=11, fontweight='bold')
            plt.tight_layout()
            safe = fname.replace('/', '_').replace('\\', '_')[:50]
            plt.savefig(out_dir / f'shap_dep_{model_name}_{safe}.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✓ shap_dep_{model_name}_{safe}.png")
        except Exception as e:
            print(f"   ✗ dependence plot feature {idx}: {e}")


def create_waterfall_plots(shap_values, X_sample, model_name, out_dir, explainer, num=3):
    """Waterfall plots for individual predictions."""
    sv = handle_multiclass_shap(shap_values)

    base_value = 0
    if hasattr(explainer, 'expected_value'):
        ev = explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            ev = np.array(ev)
            base_value = float(ev.mean())
        else:
            base_value = float(ev)

    for i in range(min(num, len(X_sample))):
        try:
            expl = shap.Explanation(
                values=sv[i],
                base_values=base_value,
                data=X_sample.iloc[i].values,
                feature_names=X_sample.columns.tolist()
            )
            plt.figure(figsize=(10, 8))
            shap.waterfall_plot(expl, show=False)
            plt.title(f'SHAP Waterfall - Sample {i} - {model_name}', fontsize=11, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig(out_dir / f'shap_waterfall_{model_name}_s{i}.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✓ shap_waterfall_{model_name}_s{i}.png")
        except Exception as e:
            print(f"   ✗ waterfall sample {i}: {e}")


def create_force_plots(shap_values, X_sample, model_name, out_dir, explainer, num=3):
    """Force plots saved as HTML for individual predictions."""
    sv = handle_multiclass_shap(shap_values)

    base_value = 0
    if hasattr(explainer, 'expected_value'):
        ev = explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            ev = np.array(ev)
            base_value = float(ev.mean())
        else:
            base_value = float(ev)

    for i in range(min(num, len(X_sample))):
        try:
            fp = shap.force_plot(base_value, sv[i], X_sample.iloc[i], matplotlib=False)
            html_path = out_dir / f'shap_force_{model_name}_s{i}.html'
            shap.save_html(str(html_path), fp)
            print(f"   ✓ shap_force_{model_name}_s{i}.html")
        except Exception as e:
            print(f"   ✗ force plot sample {i}: {e}")


def compute_feature_importance(shap_values, X_sample, model_name):
    """Return DataFrame of mean absolute SHAP per feature."""
    sv = handle_multiclass_shap(shap_values)
    mean_abs = np.abs(sv).mean(axis=0)
    if mean_abs.ndim > 1:
        mean_abs = mean_abs.mean(axis=1)

    df = pd.DataFrame({
        'feature': X_sample.columns.tolist(),
        'mean_abs_shap': mean_abs
    }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
    return df


def process_model(model_name, X_train, X_test):
    """Full SHAP pipeline for one model."""
    print(f"\n{'='*80}")
    print(f"  {model_name}")
    print(f"{'='*80}")

    out_dir = Path(OUTPUT_DIR) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resume: skip if importance CSV already exists from a previous run
    done_marker = out_dir / f'shap_importance_{model_name}.csv'
    if done_marker.exists():
        print("   ✓ Already completed — skipping (delete folder to re-run)")
        return pd.read_csv(done_marker)

    model = load_model(model_name)
    if model is None:
        print("   ✗ Could not load model — skipping")
        return None
    print("   ✓ Model loaded")

    X_train_m, X_test_m = get_model_data(model_name, X_train, X_test)

    print("   Creating explainer...")
    try:
        explainer, expl_type = get_explainer(model, X_train_m, model_name)
        print(f"   ✓ {expl_type}")
    except Exception as e:
        print(f"   ✗ Explainer creation failed: {e}")
        return None

    print("   Generating SHAP values...")
    try:
        shap_values, X_sample = generate_shap_values(explainer, expl_type, X_test_m, model_name)
        print(f"   ✓ SHAP values computed for {len(X_sample)} samples")
    except Exception as e:
        print(f"   ✗ SHAP value generation failed: {e}")
        return None

    print("   Saving plots...")
    create_summary_plots(shap_values, X_sample, model_name, out_dir)
    create_dependence_plots(shap_values, X_sample, model_name, out_dir, top_n=5)
    create_waterfall_plots(shap_values, X_sample, model_name, out_dir, explainer, num=3)
    create_force_plots(shap_values, X_sample, model_name, out_dir, explainer, num=3)

    importance_df = compute_feature_importance(shap_values, X_sample, model_name)
    csv_path = out_dir / f'shap_importance_{model_name}.csv'
    importance_df.to_csv(csv_path, index=False)
    print(f"   ✓ shap_importance_{model_name}.csv")

    print(f"\n   Top 10 features:")
    for _, row in importance_df.head(10).iterrows():
        print(f"     {row['feature']:40s}  {row['mean_abs_shap']:.6f}")

    return importance_df


def create_aggregated_analysis(all_importance, out_dir):
    """Aggregate SHAP importance across all successfully processed models."""
    print(f"\n{'='*80}")
    print("  Aggregated analysis")
    print(f"{'='*80}")

    if not all_importance:
        print("   No data to aggregate")
        return

    combined = pd.concat(
        [df.assign(model=name) for name, df in all_importance.items()],
        ignore_index=True
    )

    agg = (
        combined.groupby('feature')['mean_abs_shap']
        .agg(['mean', 'std', 'min', 'max', 'count'])
        .sort_values('mean', ascending=False)
    )

    agg_path = Path(out_dir) / 'shap_aggregated_feature_importance.csv'
    agg.to_csv(agg_path)
    print(f"   ✓ shap_aggregated_feature_importance.csv")

    plt.figure(figsize=(12, 10))
    top30 = agg.head(30)
    colors = plt.cm.viridis(np.linspace(0, 1, len(top30)))
    plt.barh(range(len(top30)), top30['mean'], xerr=top30['std'],
             color=colors, edgecolor='black', alpha=0.8)
    plt.yticks(range(len(top30)), top30.index, fontsize=9)
    plt.xlabel('Mean Absolute SHAP Value', fontsize=12, fontweight='bold')
    plt.title(f'Top 30 Features — Aggregated SHAP Importance ({len(all_importance)} models)',
              fontsize=13, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(out_dir) / 'shap_aggregated_top_features.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ shap_aggregated_top_features.png")

    print(f"\n   Top 20 features (aggregated over {len(all_importance)} models):")
    for i, (feat, row) in enumerate(agg.head(20).iterrows(), 1):
        print(f"   {i:2d}. {feat:40s}  {row['mean']:.6f} ± {row['std']:.6f}")


def main():
    print("=" * 80)
    print(f"  COMPREHENSIVE SHAP ANALYSIS — {DATASET_TYPE.upper()}")
    print("=" * 80)

    X_train, X_test, _, _ = load_data()
    model_names = find_all_models()

    if not model_names:
        print("\n✗ No models found — exiting")
        return

    all_importance = {}
    for name in model_names:
        try:
            imp = process_model(name, X_train, X_test)
            if imp is not None:
                all_importance[name] = imp
        except Exception as e:
            print(f"\n   ✗ Unexpected error for {name}: {e}")

    create_aggregated_analysis(all_importance, OUTPUT_DIR)

    print("\n" + "=" * 80)
    print(f"  DONE — processed {len(all_importance)}/{len(model_names)} models")
    print(f"  Results: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == '__main__':
    main()
