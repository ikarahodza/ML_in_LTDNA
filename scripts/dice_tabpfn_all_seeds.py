"""
DiCE (Diverse Counterfactual Explanations) for TabPFN — all 5 class-imbalance
strategies, across all 10 seeds, on the Mix or Single dataset (pass as the
first positional arg; defaults to mix).

TabPFN is excluded from the general scripts/dice_explainability.py run ("too
slow for DiCE" — it skips every *.pkl with 'tabpfn' in the name). This script
instead fits TabPFN fresh per (seed, strategy), exactly as
scripts/dice_tabpfn_timing_probe.py demonstrates for a single query/target
pair, and runs DiCE against that freshly-fit model. Output schema matches
dice_explainability.py exactly, so results sit alongside the other models'
DiCE folders unchanged:
  <results_dir>/dice_explainability/<model_name>/
    all_counterfactuals_<model_name>.csv
    cf_<source_class>_to_<desired_class>.csv   (one per class pair)
    feature_change_magnitude_<model_name>.png
    feature_change_freq_<model_name>.png
    dice_summary_<model_name>.json

Strategies (matching scripts/TabPFN.py's VERSION_SUFFIX='_v35' output):
  Original            model_tabpfn_original_v35            no resampling, balance_probabilities=False
  ClassWeights         model_tabpfn_classweights_v35         no resampling, balance_probabilities=True
  SMOTE                model_tabpfn_smote_v35                imblearn SMOTE oversampling
  RandomOverSampler    model_tabpfn_randomoversampler_v35    imblearn RandomOverSampler
  RandomUnderSampler   model_tabpfn_randomundersampler_v35   imblearn RandomUnderSampler

Seeds: 42 (results land directly under <dataset>_Results_PoI/) and
       123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021
       (results land under <dataset>_Results_PoI/seed_<N>_v35/)

Uses the tabpfn-v3.5-20260909.safetensors checkpoint (TabPFN-3.5, released by
Prior Labs; requires tabpfn>=9.0.0 — the model registry entry for v3.5 does
not exist in earlier tabpfn releases such as 8.1.0). This supersedes the
earlier "_v3" (TabPFN-3.0) result tree, which is left untouched on disk.

WARNING — this is expensive. scripts/dice_tabpfn_timing_probe.py exists
specifically to estimate the per-(query,target) cost on your hardware before
you commit to a full run: with the project defaults (3 samples/class x 4
classes x 3 targets = 36 pairs per strategy), a full run is
10 seeds x 5 strategies x 36 pairs. Run the probe first. Use --seeds /
--strategies to split the work across sessions or machines; already-finished
(seed, strategy) combinations are skipped automatically (delete the model's
output folder to force a re-run).

Does NOT touch dice_aggregated_feature_changes.{csv,png} — that cross-model
rollup also covers the ~44 non-TabPFN models and must be regenerated
separately (see dice_explainability.py's create_aggregated_analysis) once
all models you care about are done.

Usage:
  python scripts/dice_tabpfn_all_seeds.py              # mix (default)
  python scripts/dice_tabpfn_all_seeds.py single
  python scripts/dice_tabpfn_all_seeds.py mix --seeds 42 123 --strategies Original ClassWeights
  python scripts/dice_tabpfn_all_seeds.py single --dry-run
"""
import os
import sys
import json
import argparse
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
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

warnings.filterwarnings('ignore')

os.environ['TABPFN_NO_BROWSER'] = '1'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
_key_file = os.path.join(PROJECT_ROOT, 'tabpfn.key')
if os.path.exists(_key_file):
    with open(_key_file) as _f:
        os.environ.setdefault('TABPFN_TOKEN', _f.read().strip())

from tabpfn import TabPFNClassifier

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TABPFN_MODEL = 'tabpfn-v3.5-20260909.safetensors'
DATA_FILE    = None   # set in main() from the dataset arg
RESULTS_ROOT = None   # set in main() from the dataset arg

OUTCOME_COL = 'Classification'
CLASS_NAMES = ['Allele', 'Back_Stutter', 'Forward_Stutter', 'Pull_Up']

SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]

STRATEGY_KEYS = {
    'Original':           'original',
    'ClassWeights':        'classweights',
    'SMOTE':                'smote',
    'RandomOverSampler':    'randomoversampler',
    'RandomUnderSampler':   'randomundersampler',
}

N_CFS = 4                    # counterfactuals per (query, target) pair
N_SAMPLES_PER_CLASS = 3       # query instances per class
DICE_METHOD = 'random'

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def seed_results_dir(seed):
    if seed == 42:
        return Path(RESULTS_ROOT)
    return Path(RESULTS_ROOT) / f'seed_{seed}_v35'


def model_name_for(strategy):
    return f'model_tabpfn_{STRATEGY_KEYS[strategy]}_v35'


# ---------------------------------------------------------------------------
# Data loading / resampling
# ---------------------------------------------------------------------------
def load_data(seed):
    df = pd.read_csv(DATA_FILE)
    drop_cols = ['sample', 'PoI']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = X[OUTCOME_COL]
    X_features = X.drop(columns=[OUTCOME_COL])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_features, y, test_size=0.2, stratify=y, random_state=seed
    )
    return X_tr, X_te, y_tr, y_te


def resample_for_strategy(strategy, X_train, y_train, seed):
    if strategy == 'SMOTE':
        return SMOTE(random_state=seed).fit_resample(X_train, y_train)
    if strategy == 'RandomOverSampler':
        return RandomOverSampler(random_state=seed).fit_resample(X_train, y_train)
    if strategy == 'RandomUnderSampler':
        return RandomUnderSampler(random_state=seed).fit_resample(X_train, y_train)
    return X_train, y_train  # Original, ClassWeights: no resampling


def fit_tabpfn(strategy, X_train, y_train, seed):
    clf = TabPFNClassifier(
        model_path=TABPFN_MODEL,
        balance_probabilities=(strategy == 'ClassWeights'),
        random_state=seed,
    )
    X_fit, y_fit = resample_for_strategy(strategy, X_train, y_train, seed)
    clf.fit(X_fit.values, y_fit.values)
    return clf


# ---------------------------------------------------------------------------
# Query selection (mirrors dice_explainability.py's select_query_instances)
# ---------------------------------------------------------------------------
def select_query_instances(X_test, y_test, model, n_per_class):
    try:
        y_pred = model.predict(X_test.values)
    except Exception:
        y_pred = np.zeros(len(X_test), dtype=object)

    queries = {}
    for cls in CLASS_NAMES:
        correct_mask = (y_test.values == cls) & (y_pred == cls)
        idx = np.where(correct_mask)[0]
        if len(idx) < n_per_class:
            idx = np.where(y_test.values == cls)[0]
        idx = idx[:n_per_class]
        if len(idx) > 0:
            queries[cls] = X_test.iloc[idx].copy()
    return queries


# ---------------------------------------------------------------------------
# DiCE run for one (seed, strategy) — mirrors dice_explainability.py's
# run_dice_for_model output schema exactly.
# ---------------------------------------------------------------------------
def run_dice_for_tabpfn(model_name, clf, X_train, X_test, y_train, y_test, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_names = X_train.columns.tolist()

    train_df = X_train.copy()
    train_df[OUTCOME_COL] = y_train.values

    d = dice_ml.Data(dataframe=train_df, continuous_features=feature_names,
                      outcome_name=OUTCOME_COL)
    m = dice_ml.Model(model=clf, backend="sklearn")
    exp = dice_ml.Dice(d, m, method=DICE_METHOD)

    classes_sorted = sorted(y_train.unique())
    queries = select_query_instances(X_test, y_test, clf, N_SAMPLES_PER_CLASS)
    if not queries:
        print("   x No query instances found — skipping")
        return None

    all_cfs = []
    all_deltas = []

    for src_class, query_df in queries.items():
        for desired_cls_name in CLASS_NAMES:
            if desired_cls_name == src_class:
                continue
            desired_cls_idx = classes_sorted.index(desired_cls_name)

            try:
                result = exp.generate_counterfactuals(
                    query_instances=query_df, total_CFs=N_CFS,
                    desired_class=desired_cls_idx, verbose=False
                )
            except Exception as e:
                print(f"   !  CF gen failed ({src_class}->{desired_cls_name}): {e}")
                continue

            batch = []
            for qi, cf_ex in enumerate(result.cf_examples_list):
                if cf_ex.final_cfs_df is None or cf_ex.final_cfs_df.empty:
                    continue

                cf_df = cf_ex.final_cfs_df.copy()
                orig_row = query_df.iloc[qi] if qi < len(query_df) else query_df.iloc[-1]

                feat_cols = [c for c in cf_df.columns if c != OUTCOME_COL]
                deltas = cf_df[feat_cols].subtract(orig_row[feat_cols].values, axis=1).abs()
                all_deltas.append(deltas)

                cf_df['_source_class'] = src_class
                cf_df['_desired_class'] = desired_cls_name
                cf_df['_query_idx'] = qi
                all_cfs.append(cf_df)
                batch.append(cf_df)

            if batch:
                merged = pd.concat(batch, ignore_index=True)
                fname = f"cf_{src_class}_to_{desired_cls_name}.csv".replace(' ', '_')
                merged.to_csv(out_dir / fname, index=False)

    if not all_cfs:
        print("   x No counterfactuals generated")
        return None

    full_cf = pd.concat(all_cfs, ignore_index=True)
    full_cf.to_csv(out_dir / f'all_counterfactuals_{model_name}.csv', index=False)
    print(f"   + all_counterfactuals_{model_name}.csv ({len(full_cf)} rows)")

    delta_df = pd.concat(all_deltas, ignore_index=True)
    mean_delta = delta_df.mean().sort_values(ascending=False)
    change_freq = (delta_df > 1e-6).mean().sort_values(ascending=False)

    top20_delta = mean_delta.head(20)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(top20_delta)), top20_delta.values[::-1],
            color=sns.color_palette("Blues_d", len(top20_delta)))
    ax.set_yticks(range(len(top20_delta)))
    ax.set_yticklabels(top20_delta.index[::-1], fontsize=9)
    ax.set_xlabel('Mean |Delta feature| across counterfactuals', fontsize=11)
    ax.set_title(f'DiCE — Feature Change Magnitude\n{model_name}', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f'feature_change_magnitude_{model_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   + feature_change_magnitude_{model_name}.png")

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
    print(f"   + feature_change_freq_{model_name}.png")

    summary = {
        'model': model_name,
        'total_counterfactuals': len(full_cf),
        'top10_features_by_change_magnitude': mean_delta.head(10).to_dict(),
        'top10_features_by_change_frequency': change_freq.head(10).to_dict(),
    }
    with open(out_dir / f'dice_summary_{model_name}.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"   + dice_summary_{model_name}.json")

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global DATA_FILE, RESULTS_ROOT

    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset', nargs='?', default='mix', choices=['mix', 'single'],
                         help="Dataset to run on (default: mix)")
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS,
                         help=f'Subset of seeds to run (default: all {len(SEEDS)})')
    parser.add_argument('--strategies', nargs='+', default=list(STRATEGY_KEYS.keys()),
                         choices=list(STRATEGY_KEYS.keys()),
                         help='Subset of strategies to run (default: all 5)')
    parser.add_argument('--dry-run', action='store_true',
                         help='List planned work and resume status without fitting anything')
    args = parser.parse_args()

    dataset_label = 'Mix' if args.dataset == 'mix' else 'Single'
    DATA_FILE = os.path.join(PROJECT_ROOT, f'Input_Data-{dataset_label}-train.csv')
    RESULTS_ROOT = os.path.join(PROJECT_ROOT, f'{dataset_label}_Results_PoI')

    print("=" * 80)
    print(f"  DiCE COUNTERFACTUAL EXPLAINABILITY — TabPFN, {dataset_label} dataset")
    print("=" * 80)
    print(f"  Seeds:      {args.seeds}")
    print(f"  Strategies: {args.strategies}")
    print(f"  Method: {DICE_METHOD} | CFs/pair: {N_CFS} | Samples/class: {N_SAMPLES_PER_CLASS}")
    print()

    all_summaries = []
    n_done, n_run, n_skip = 0, 0, 0

    for seed in args.seeds:
        out_root = seed_results_dir(seed) / 'dice_explainability'
        print(f"\n{'#'*70}\n  SEED {seed}  ->  {out_root}\n{'#'*70}")

        X_train = X_test = y_train = y_test = None  # loaded lazily, once per seed

        for strategy in args.strategies:
            model_name = model_name_for(strategy)
            out_dir = out_root / model_name
            done_marker = out_dir / f'dice_summary_{model_name}.json'

            if done_marker.exists():
                print(f"\n  [{strategy}] {model_name}: already done — skipping")
                n_skip += 1
                with open(done_marker) as f:
                    all_summaries.append(json.load(f))
                continue

            if args.dry_run:
                print(f"\n  [{strategy}] {model_name}: WOULD RUN")
                n_run += 1
                continue

            print(f"\n  {'-'*66}\n  [{strategy}] {model_name}\n  {'-'*66}")

            if X_train is None:
                X_train, X_test, y_train, y_test = load_data(seed)

            try:
                clf = fit_tabpfn(strategy, X_train, y_train, seed)
            except Exception as e:
                print(f"   x TabPFN fit failed: {e}")
                continue

            try:
                summary = run_dice_for_tabpfn(
                    model_name, clf, X_train, X_test, y_train, y_test, out_dir
                )
                if summary is not None:
                    all_summaries.append(summary)
                    n_done += 1
            except Exception as e:
                print(f"   x Unexpected error: {e}")

    print("\n" + "=" * 80)
    if args.dry_run:
        print(f"  DRY RUN — {n_run} to run, {n_skip} already done")
    else:
        print(f"  DONE — {n_done} completed this run, {n_skip} already done, "
              f"{len(all_summaries)} total summaries available")
    print("=" * 80)


if __name__ == '__main__':
    main()
