"""
TabPFN Prototype and Permutation Importance Analysis

1. PROTOTYPE EXPLANATIONS
   For each test prediction, finds the k nearest training examples in
   normalized feature space. Since TabPFN does in-context learning, these
   are the cases the model is effectively 'comparing against'.

   Outputs:
     prototypes_all_samples.csv        — every test sample with its top-k neighbours
     prototype_purity.png              — per predicted-class prototype label agreement
     prototype_distances.png           — nearest-neighbour distance distributions
     prototype_misclassified.png       — detailed view of wrongly-predicted samples

2. PERMUTATION IMPORTANCE
   Shuffles one feature at a time and measures balanced-accuracy drop.
   Only runs over the top-N features from the aggregated SHAP results
   (TabPFN's ~245 s/call makes full permutation over 305 features infeasible).

   Outputs:
     permutation_importance_<model>.csv
     permutation_importance_<model>.png
     permutation_importance_aggregated.png
"""

import os
import sys
import time
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import balanced_accuracy_score

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_TYPE = sys.argv[1] if len(sys.argv) > 1 else 'mix'
DATASET_TYPE = DATASET_TYPE.lower()
assert DATASET_TYPE in ('mix', 'single'), "Use 'mix' or 'single'"

if DATASET_TYPE == 'mix':
    DATA_FILE  = 'Input_Data-Mix-train.csv'
    MODELS_DIR = Path('Mix_Results_PoI/models')
    SHAP_CSV   = Path('Mix_Results_PoI/shap_explainability/shap_aggregated_feature_importance.csv')
    OUT_DIR    = Path('Mix_Results_PoI/tabpfn_prototypes_and_permutation')
else:
    DATA_FILE  = 'Input_Data-Single-train.csv'
    MODELS_DIR = Path('Single_Results_PoI/models')
    SHAP_CSV   = Path('Single_Results_PoI/shap_explainability/shap_aggregated_feature_importance.csv')
    OUT_DIR    = Path('Single_Results_PoI/tabpfn_prototypes_and_permutation')

OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES  = ['Allele', 'Back_Stutter', 'Forward_Stutter', 'Pull_Up']
K_NEIGHBOURS = 5
PERM_TOP_N   = 10   # only permute top-N features from SHAP ranking
PERM_REPEATS = 1    # 1 repeat × top-10 × ~245 s × 4 models ≈ 3 h

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150
PALETTE = dict(zip(CLASS_NAMES, sns.color_palette('Set2', 4)))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_data():
    print(f"\nLoading {DATASET_TYPE.upper()} data...")
    df = pd.read_csv(DATA_FILE)
    X  = df.drop(columns=[c for c in ['sample', 'PoI', 'Classification'] if c in df.columns])
    y  = df['Classification']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"  train={len(X_train)}  test={len(X_test)}  features={X.shape[1]}")
    return X_train, X_test, y_train, y_test


def find_tabpfn_models():
    """Return TabPFN model stems for the current checkpoint (v3.5 only; skips superseded v3)."""
    stems = []
    for pkl in sorted(MODELS_DIR.glob('model_tabpfn*.pkl')):
        stem = pkl.stem
        if not stem.endswith('_v35'):
            continue   # skip superseded TabPFN-3.0 (_v3) models
        stems.append(stem)
    print(f"  Found {len(stems)} TabPFN models: {stems}")
    return stems


# ---------------------------------------------------------------------------
# ── PART 1: PROTOTYPES ──────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def compute_prototypes(X_train, X_test, y_train, y_test):
    """
    Fit a NearestNeighbors index on scaled training data; find top-k
    neighbours for every test sample.
    """
    print("\n" + "="*70)
    print("PART 1 — PROTOTYPE EXPLANATIONS")
    print("="*70)

    scaler  = StandardScaler()
    Xtr_s   = scaler.fit_transform(X_train)
    Xte_s   = scaler.transform(X_test)

    nn = NearestNeighbors(n_neighbors=K_NEIGHBOURS, metric='euclidean', n_jobs=-1)
    nn.fit(Xtr_s)
    distances, indices = nn.kneighbors(Xte_s)

    # Build detailed records
    records = []
    y_train_arr = y_train.values
    y_test_arr  = y_test.values
    X_train_arr = X_train.values

    for i in range(len(X_test)):
        row = {
            'test_idx':        i,
            'true_class':      y_test_arr[i],
        }
        for rank in range(K_NEIGHBOURS):
            nb_idx = indices[i, rank]
            row[f'proto{rank+1}_train_idx'] = nb_idx
            row[f'proto{rank+1}_class']     = y_train_arr[nb_idx]
            row[f'proto{rank+1}_distance']  = round(distances[i, rank], 4)
        records.append(row)

    proto_df = pd.DataFrame(records)
    return proto_df, distances, indices, y_train_arr, y_test_arr, scaler


def add_model_predictions(proto_df, model_stems, X_test):
    """Add each model's predicted class and confidence to proto_df."""
    for stem in model_stems:
        model = joblib.load(MODELS_DIR / f'{stem}.pkl')
        try:
            preds   = model.predict(X_test)
            probas  = model.predict_proba(X_test)
            conf    = probas.max(axis=1)
        except Exception as e:
            print(f"  ⚠ prediction failed for {stem}: {e}")
            continue
        short = stem.replace('model_tabpfn_', '').replace('_v35', '')
        proto_df[f'pred_{short}']     = preds
        proto_df[f'conf_{short}']     = conf.round(3)
        print(f"  ✓ predictions from {stem}")
    return proto_df


def plot_prototype_purity(proto_df, y_test_arr):
    """
    Heatmap: for each true class, what fraction of prototypes at each
    rank share the same class as the query?
    """
    class_order = CLASS_NAMES

    # Purity = proto_class == true_class
    fig, axes = plt.subplots(1, K_NEIGHBOURS, figsize=(14, 4), sharey=True)
    for rank in range(K_NEIGHBOURS):
        purity = np.zeros((len(class_order), len(class_order)))
        col = f'proto{rank+1}_class'
        for ri, tc in enumerate(class_order):
            mask = proto_df['true_class'] == tc
            if mask.sum() == 0:
                continue
            pc = proto_df.loc[mask, col]
            for ci, pc_name in enumerate(class_order):
                purity[ri, ci] = (pc == pc_name).mean()
        ax = axes[rank]
        sns.heatmap(purity, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=[c[:6] for c in class_order],
                    yticklabels=[c[:6] for c in class_order] if rank == 0 else False,
                    vmin=0, vmax=1, ax=ax, cbar=(rank == K_NEIGHBOURS-1))
        ax.set_title(f'Rank {rank+1}', fontsize=10)
        if rank == 0:
            ax.set_ylabel('True class of query', fontsize=9)
        ax.set_xlabel('Prototype class', fontsize=9)

    fig.suptitle('Prototype Class Distribution by Query True Class\n'
                 '(row = true query class, col = prototype class, value = fraction)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'prototype_purity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ prototype_purity.png")


def plot_prototype_distances(proto_df):
    """Box plot of nearest-neighbour distances per true class."""
    dist_records = []
    for _, row in proto_df.iterrows():
        for rank in range(K_NEIGHBOURS):
            dist_records.append({
                'true_class': row['true_class'],
                'rank': f'Rank {rank+1}',
                'distance': row[f'proto{rank+1}_distance'],
            })
    dist_df = pd.DataFrame(dist_records)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=dist_df, x='rank', y='distance', hue='true_class',
                palette=PALETTE, ax=ax)
    ax.set_title('Nearest-Neighbour Distances by True Class',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Prototype rank')
    ax.set_ylabel('Euclidean distance (normalised feature space)')
    ax.legend(title='True class', bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'prototype_distances.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ prototype_distances.png")


def plot_misclassified_prototypes(proto_df, model_stems):
    """
    For each misclassified test sample (per model), show a horizontal bar
    chart of prototype labels and distances.
    """
    # Use first model's predictions as reference
    if not model_stems:
        return
    short0 = model_stems[0].replace('model_tabpfn_', '').replace('_v35', '')
    pred_col = f'pred_{short0}'
    if pred_col not in proto_df.columns:
        return

    wrong = proto_df[proto_df['true_class'] != proto_df[pred_col]].copy()
    if wrong.empty:
        print("  ✓ No misclassified samples for prototype plot")
        return

    n = min(6, len(wrong))
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n))
    if n == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, wrong.head(n).iterrows()):
        labels    = [row[f'proto{r+1}_class'] for r in range(K_NEIGHBOURS)]
        distances = [row[f'proto{r+1}_distance'] for r in range(K_NEIGHBOURS)]
        colors    = [PALETTE.get(lbl, 'grey') for lbl in labels]

        bars = ax.barh(range(K_NEIGHBOURS), distances, color=colors, edgecolor='k', alpha=0.85)
        ax.set_yticks(range(K_NEIGHBOURS))
        ax.set_yticklabels([f'#{r+1}: {lbl}' for r, lbl in enumerate(labels)], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Distance (normalised)')
        ax.set_title(
            f"Sample {int(row['test_idx'])} | True: {row['true_class']} | "
            f"Predicted: {row[pred_col]}",
            fontsize=10, fontweight='bold'
        )

    # Legend
    from matplotlib.patches import Patch
    handles = [Patch(color=PALETTE[c], label=c) for c in CLASS_NAMES]
    fig.legend(handles=handles, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, -0.02), fontsize=9)
    fig.suptitle('Prototypes for Misclassified Test Samples', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'prototype_misclassified.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ prototype_misclassified.png ({n} samples)")


def run_prototype_analysis(X_train, X_test, y_train, y_test, model_stems, X_test_scaled=None):
    proto_df, distances, indices, y_train_arr, y_test_arr, _ = compute_prototypes(
        X_train, X_test, y_train, y_test
    )

    # Models were trained on scaled data — use scaled X_test for predictions
    X_for_pred = X_test_scaled if X_test_scaled is not None else X_test
    print("\n  Getting model predictions for prototype table...")
    proto_df = add_model_predictions(proto_df, model_stems, X_for_pred)

    csv_path = OUT_DIR / 'prototypes_all_samples.csv'
    proto_df.to_csv(csv_path, index=False)
    print(f"  ✓ prototypes_all_samples.csv ({len(proto_df)} rows)")

    print("\n  Generating prototype plots...")
    plot_prototype_purity(proto_df, y_test_arr)
    plot_prototype_distances(proto_df)
    plot_misclassified_prototypes(proto_df, model_stems)

    # Quick summary
    print("\n  Prototype class purity (rank-1 neighbour):")
    for cls in CLASS_NAMES:
        mask = proto_df['true_class'] == cls
        if mask.sum() == 0:
            continue
        same = (proto_df.loc[mask, 'proto1_class'] == cls).mean()
        print(f"    {cls:<20s}  {same*100:.1f}% of nearest neighbours share true class")


# ---------------------------------------------------------------------------
# ── PART 2: PERMUTATION IMPORTANCE ─────────────────────────────────────────
# ---------------------------------------------------------------------------

def load_top_shap_features(n=PERM_TOP_N):
    """Read aggregated SHAP CSV and return the top-N feature names."""
    if not SHAP_CSV.exists():
        print(f"  ⚠ SHAP aggregated CSV not found at {SHAP_CSV}")
        print("    Run shap_comprehensive_analysis.py first.")
        return None
    agg = pd.read_csv(SHAP_CSV, index_col=0)
    # columns: mean, std, min, max, count — sorted descending by mean already
    top_features = agg.sort_values('mean', ascending=False).head(n).index.tolist()
    print(f"  Top {n} features from SHAP aggregation:")
    for i, f in enumerate(top_features, 1):
        print(f"    {i:2d}. {f}")
    return top_features


def permutation_importance_manual(model, X_test, y_test, features, n_repeats=1):
    """
    Manual permutation importance over a feature subset.
    Shows progress since each TabPFN call is slow.
    """
    from sklearn.preprocessing import LabelEncoder
    baseline_preds = model.predict(X_test)
    # TabPFN returns numeric labels; y_test may have string labels — align them
    if hasattr(y_test, 'iloc'):
        y_vals = y_test.values
    else:
        y_vals = np.array(y_test)
    if y_vals.dtype.kind in ('U', 'O') and np.array(baseline_preds).dtype.kind not in ('U', 'O'):
        le = LabelEncoder()
        y_vals = le.fit_transform(y_vals)
    baseline_score = balanced_accuracy_score(y_vals, baseline_preds)
    print(f"    Baseline balanced accuracy: {baseline_score:.4f}")

    importances = np.zeros((len(features), n_repeats))
    feat_idx    = {f: X_test.columns.get_loc(f) for f in features if f in X_test.columns}

    for i, feat in enumerate(features):
        if feat not in feat_idx:
            print(f"    ⚠ Feature not found: {feat}")
            continue
        col_i = feat_idx[feat]
        for r in range(n_repeats):
            X_perm = X_test.copy()
            rng    = np.random.RandomState(42 + r)
            X_perm.iloc[:, col_i] = rng.permutation(X_perm.iloc[:, col_i].values)

            t0 = time.time()
            perm_preds = model.predict(X_perm)
            elapsed    = time.time() - t0
            perm_score = balanced_accuracy_score(y_vals, perm_preds)
            importances[i, r] = baseline_score - perm_score
            print(f"    [{i+1}/{len(features)}] {feat[:35]:35s}  "
                  f"drop={importances[i,r]:+.4f}  ({elapsed:.0f}s)")

    return baseline_score, importances


def plot_permutation_importance(feature_names, importances, model_name, baseline):
    mean_imp = importances.mean(axis=1)
    std_imp  = importances.std(axis=1) if importances.shape[1] > 1 else np.zeros(len(feature_names))
    order    = np.argsort(mean_imp)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#e74c3c' if v > 0 else '#95a5a6' for v in mean_imp[order]]
    ax.barh(range(len(order)), mean_imp[order][::-1],
            xerr=std_imp[order][::-1] if importances.shape[1] > 1 else None,
            color=colors[::-1], edgecolor='k', alpha=0.85)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([feature_names[j] for j in order[::-1]], fontsize=9)
    ax.axvline(0, color='k', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Balanced accuracy drop when feature is permuted', fontsize=11)
    plt.tight_layout()
    fig.savefig(OUT_DIR / f'permutation_importance_{model_name}.png',
                dpi=150, bbox_inches='tight')
    plt.close()


def run_permutation_analysis(X_test, y_test, model_stems, top_features):
    print("\n" + "="*70)
    print("PART 2 — PERMUTATION IMPORTANCE")
    print("="*70)
    est_mins = len(top_features) * PERM_REPEATS * 245 / 60
    print(f"  Features to permute: {len(top_features)}")
    print(f"  Repeats: {PERM_REPEATS}  |  Est. time: ~{est_mins:.0f} min total")

    all_importances = {}

    for stem in model_stems:
        print(f"\n  ── {stem} ──")
        model = joblib.load(MODELS_DIR / f'{stem}.pkl')
        baseline, importances = permutation_importance_manual(
            model, X_test, y_test, top_features, n_repeats=PERM_REPEATS
        )
        all_importances[stem] = importances

        # Per-model CSV
        imp_df = pd.DataFrame({
            'feature':     top_features,
            'importance':  importances.mean(axis=1),
            'std':         importances.std(axis=1),
        }).sort_values('importance', ascending=False)
        short = stem.replace('model_tabpfn_', '').replace('_v35', '')
        imp_df.to_csv(OUT_DIR / f'permutation_importance_{short}.csv', index=False)
        print(f"  ✓ permutation_importance_{short}.csv")
        plot_permutation_importance(top_features, importances, short, baseline)
        print(f"  ✓ permutation_importance_{short}.png")

    # Aggregated plot across models
    if len(all_importances) > 1:
        agg_matrix = np.stack([v.mean(axis=1) for v in all_importances.values()])  # (n_models, n_features)
        agg_mean   = agg_matrix.mean(axis=0)
        agg_std    = agg_matrix.std(axis=0)
        order      = np.argsort(agg_mean)[::-1]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(order)), agg_mean[order][::-1],
                xerr=agg_std[order][::-1],
                color='steelblue', edgecolor='k', alpha=0.85)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([top_features[j] for j in order[::-1]], fontsize=9)
        ax.axvline(0, color='k', linewidth=0.8, linestyle='--')
        ax.set_xlabel('Mean balanced accuracy drop (aggregated across TabPFN variants)', fontsize=10)
        plt.tight_layout()
        fig.savefig(OUT_DIR / 'permutation_importance_aggregated.png',
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  ✓ permutation_importance_aggregated.png")

        # Aggregated CSV
        agg_df = pd.DataFrame({
            'feature':    top_features,
            'mean_drop':  agg_mean,
            'std_drop':   agg_std,
        }).sort_values('mean_drop', ascending=False)
        agg_df.to_csv(OUT_DIR / 'permutation_importance_aggregated.csv', index=False)
        print(f"  ✓ permutation_importance_aggregated.csv")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print(f"  TabPFN PROTOTYPE & PERMUTATION IMPORTANCE — {DATASET_TYPE.upper()}")
    print("=" * 70)

    X_train, X_test, y_train, y_test = load_data()
    model_stems = find_tabpfn_models()

    if not model_stems:
        print("No TabPFN models found — exiting")
        return

    # Scale test data to match training regime (models trained on StandardScaler output)
    _ss = StandardScaler()
    _ss.fit(X_train)
    X_test_scaled = pd.DataFrame(
        _ss.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    # ── Part 1: Prototypes ────────────────────────────────────────────────
    proto_csv = OUT_DIR / 'prototypes_all_samples.csv'
    if proto_csv.exists():
        print(f"\nPrototypes already done — skipping (delete {proto_csv.name} to re-run)")
    else:
        run_prototype_analysis(X_train, X_test, y_train, y_test, model_stems, X_test_scaled)

    # ── Part 2: Permutation importance ───────────────────────────────────
    top_features = load_top_shap_features(PERM_TOP_N)
    if top_features is not None:
        run_permutation_analysis(X_test_scaled, y_test, model_stems, top_features)
    else:
        print("\nSkipping permutation importance (no SHAP aggregated CSV found).")

    print("\n" + "=" * 70)
    print(f"  DONE — results in {OUT_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()
