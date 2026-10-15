"""
One-shot timing probe for DiCE-on-TabPFN.

Fits a single fresh TabPFN, generates counterfactuals for exactly ONE
(query, desired_class) pair, and times it. Then extrapolates to the full
dice_explainability.py-style run so you know what you're signing up for
before kicking off the real thing.

Usage:
    python scripts/dice_tabpfn_timing_probe.py [mix|single] [--include-poi] [--n-estimators N]

Defaults match the project's "regular" DiCE settings: n_estimators=8 (TabPFN
library default), total_CFs=4, 3 query samples/class, all 4 classes, 3 target
classes per query (skip same-class) — i.e. 36 (query,target) pairs per
TabPFN strategy variant, 5 variants (Original/ClassWeights/SMOTE/ROS/RUS).
"""
import os
import sys
import time

os.environ['TABPFN_NO_BROWSER'] = '1'
_key_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tabpfn.key')
if os.path.exists(_key_file):
    with open(_key_file) as _f:
        os.environ.setdefault('TABPFN_TOKEN', _f.read().strip())

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import dice_ml
from sklearn.model_selection import train_test_split

_args = sys.argv[1:]
_pos = [a for a in _args if not a.startswith('-')]
INCLUDE_POI = '--include-poi' in _args
DATASET_TYPE = (_pos[0].lower() if _pos else 'mix')
N_EST = 8
if '--n-estimators' in _args:
    N_EST = int(_args[_args.index('--n-estimators') + 1])

DATA_FILE = 'Input_Data-Mix-train.csv' if DATASET_TYPE == 'mix' else 'Input_Data-Single-train.csv'
OUTCOME_COL = 'Classification'

# Full-run shape, for the extrapolation at the end — matches dice_explainability.py
N_CFS = 4
N_SAMPLES_PER_CLASS = 3
N_CLASSES = 4
N_TARGETS_PER_QUERY = N_CLASSES - 1          # skip same-class
PAIRS_PER_STRATEGY = N_SAMPLES_PER_CLASS * N_CLASSES * N_TARGETS_PER_QUERY
N_STRATEGIES = 5                             # Original, ClassWeights, SMOTE, ROS, RUS

print("=" * 70)
print("  DiCE-on-TabPFN timing probe")
print("=" * 70)
print(f"  dataset={DATASET_TYPE}  include_poi={INCLUDE_POI}  n_estimators={N_EST}")
print()

# --- 1. Load data (same split as the real pipeline) -------------------------
t0 = time.perf_counter()
df = pd.read_csv(DATA_FILE)
drop_cols = ['sample'] if INCLUDE_POI else ['sample', 'PoI']
X = df.drop(columns=[c for c in drop_cols if c in df.columns])
y = X[OUTCOME_COL]
X_features = X.drop(columns=[OUTCOME_COL])

X_train, X_test, y_train, y_test = train_test_split(
    X_features, y, test_size=0.2, stratify=y, random_state=42
)
print(f"[1/4] Data loaded: train={len(X_train)}, test={len(X_test)}, "
      f"features={X_train.shape[1]}  ({time.perf_counter()-t0:.1f}s)")

# --- 2. Fit a fresh TabPFN (Original strategy — no resampling) --------------
t0 = time.perf_counter()
from tabpfn import TabPFNClassifier
clf = TabPFNClassifier(n_estimators=N_EST, softmax_temperature=0.7,
                        balance_probabilities=False, random_state=42)
clf.fit(X_train.values, y_train.values)
fit_time = time.perf_counter() - t0
print(f"[2/4] TabPFN fit (Original, n_estimators={N_EST})  ({fit_time:.1f}s)")

# --- 3. Build the DiCE wrapper (identical setup to dice_explainability.py) --
train_df = X_train.copy()
train_df[OUTCOME_COL] = y_train.values
d = dice_ml.Data(dataframe=train_df, continuous_features=X_train.columns.tolist(),
                  outcome_name=OUTCOME_COL)
m = dice_ml.Model(model=clf, backend="sklearn")
exp = dice_ml.Dice(d, m, method='random')

# --- 4. Pick ONE query instance and ONE target class ------------------------
y_pred = clf.predict(X_test.values)
src_class = y_test.iloc[0]
correct_mask = (y_test.values == src_class) & (y_pred == src_class)
idx = np.where(correct_mask)[0]
if len(idx) == 0:
    idx = np.where(y_test.values == src_class)[0]
query_df = X_test.iloc[idx[:1]].copy()

classes_sorted = sorted(y.unique())
desired_cls_name = next(c for c in classes_sorted if c != src_class)
desired_cls_idx = classes_sorted.index(desired_cls_name)

print(f"[3/4] Query: 1 instance, true class={src_class!r}  ->  "
      f"desired class={desired_cls_name!r}")

# --- 5. Time exactly one (query, target) counterfactual generation ---------
t0 = time.perf_counter()
result = exp.generate_counterfactuals(
    query_instances=query_df, total_CFs=N_CFS,
    desired_class=desired_cls_idx, verbose=False
)
cf_time = time.perf_counter() - t0

n_found = 0
cf_ex = result.cf_examples_list[0]
if cf_ex.final_cfs_df is not None:
    n_found = len(cf_ex.final_cfs_df)

print(f"[4/4] One (query,target) pair done: {n_found}/{N_CFS} counterfactuals "
      f"found  ({cf_time:.1f}s)")

# --- Extrapolation -----------------------------------------------------------
print()
print("=" * 70)
print("  Extrapolation to the full run")
print("=" * 70)
print(f"  Per-(query,target) time measured: {cf_time:.1f}s")
print(f"  Pairs per TabPFN strategy: {N_SAMPLES_PER_CLASS} samples/class x "
      f"{N_CLASSES} classes x {N_TARGETS_PER_QUERY} targets = {PAIRS_PER_STRATEGY}")
per_strategy_s = PAIRS_PER_STRATEGY * cf_time
print(f"  -> ~{per_strategy_s/60:.0f} min per strategy "
      f"({per_strategy_s/3600:.1f} h)")
total_s = per_strategy_s * N_STRATEGIES
print(f"  Across all {N_STRATEGIES} strategies "
      f"(Original, ClassWeights, SMOTE, ROS, RUS):")
print(f"  -> ~{total_s/3600:.1f} h total")
print()
print("  NOTE: this one sample is a noisy estimate (TabPFN CF search time can")
print("  vary 2-3x by query/target difficulty). Run this probe 2-3x and use")
print("  the max, not the first number, if you need a safer upper bound.")
print("  Fit time (one-off per strategy, not per pair): "
      f"{fit_time:.1f}s — negligible next to the CF search cost.")
