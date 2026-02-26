import pandas as pd
import argparse
import sys
import importlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, log_loss, recall_score,
    precision_score, classification_report, confusion_matrix, precision_recall_curve,
    average_precision_score, roc_curve, roc_auc_score, mean_squared_error, mean_absolute_error,
    precision_recall_fscore_support
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import joblib
import os

# Disable TabPFN telemetry to prevent PostHog connection errors
os.environ['POSTHOG_DISABLED'] = '1'
os.environ['TABPFN_TELEMETRY'] = '0'
import helper_functions as hf

os.environ["HF_TOKEN"] = "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"

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--tabpfn-key', dest='tabpfn_key', default=None)
args, _ = parser.parse_known_args()

def _get_tabpfn_key(cli_key=None):
    if cli_key:
        return cli_key
    for name in ('TABPFN_KEY', 'TABPFN_API_KEY'):
        val = os.environ.get(name)
        if val:
            return val
    key_file = os.path.join(os.getcwd(), 'tabpfn.key')
    if os.path.exists(key_file):
        try:
            with open(key_file, 'r', encoding='utf-8') as fh:
                return fh.read().strip()
        except Exception:
            return None
    return None

tabpfn_key = _get_tabpfn_key(args.tabpfn_key)
if tabpfn_key:
    os.environ['TABPFN_KEY'] = tabpfn_key
    src = 'CLI' if args.tabpfn_key else ('env' if any(os.environ.get(n) for n in ('TABPFN_KEY','TABPFN_API_KEY')) else 'file')
    print(f"TabPFN API key found ({src}) — using it for this run.")
else:
    print("ERROR: TabPFN API key not found. Provide it with --tabpfn-key, set TABPFN_KEY/TABPFN_API_KEY in the environment, or place it in 'tabpfn.key' in the repo root.")
    sys.exit(1)

from tabpfn import TabPFNClassifier

MODEL_NAME = 'tabpfn'
VERSION_SUFFIX = '_v2'  # Add version suffix to distinguish new results from old ones
# Get the absolute path to the project root and construct paths from there
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('dataset', nargs='?', default='single', choices=['mix', 'single'],
                    help='Dataset type: mix or single (default: single)')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for reproducibility (default: 42)')
args = parser.parse_args()

dataset_type = args.dataset.lower()
RANDOM_SEED = args.seed


# Set paths based on dataset type and seed
RESULTS_FOLDER = f"{dataset_type.capitalize()}_Results"
if RANDOM_SEED != 42:
    # Store in seed-specific subfolder with version suffix
    SEED_FOLDER = f"seed_{RANDOM_SEED}{VERSION_SUFFIX}"
    DATA_DIR = os.path.join(PROJECT_ROOT, RESULTS_FOLDER, SEED_FOLDER, 'data')
    FIG_DIR = os.path.join(PROJECT_ROOT, RESULTS_FOLDER, SEED_FOLDER, 'figures')
    MODEL_DIR = os.path.join(PROJECT_ROOT, RESULTS_FOLDER, SEED_FOLDER, 'models')
    REPORTS_DIR = os.path.join(PROJECT_ROOT, RESULTS_FOLDER, SEED_FOLDER, 'reports')
else:
    # Use default folders for seed 42 with version suffix
    DATA_DIR = os.path.join(PROJECT_ROOT, RESULTS_FOLDER, f'data{VERSION_SUFFIX}')
    FIG_DIR = os.path.join(PROJECT_ROOT, RESULTS_FOLDER, f'figures{VERSION_SUFFIX}')
    MODEL_DIR = os.path.join(PROJECT_ROOT, RESULTS_FOLDER, f'models{VERSION_SUFFIX}')
    REPORTS_DIR = os.path.join(PROJECT_ROOT, RESULTS_FOLDER, f'reports{VERSION_SUFFIX}')
DATA_FILE = os.path.join(PROJECT_ROOT, f'Input_Data-{dataset_type.capitalize()}-train.csv')

print(f"{'='*60}")
print(f"Running with {dataset_type.upper()} dataset")
print(f"Random Seed: {RANDOM_SEED}")
print(f"Results folder: {RESULTS_FOLDER}")
print(f"Data file: {os.path.basename(DATA_FILE)}")
print(f"{'='*60}\n")

df_m = pd.read_csv(DATA_FILE) 
original_data = df_m.copy()

X = df_m.drop(['sample', 'PoI', 'Classification'], axis=1)
y_raw = df_m['Classification']

le = LabelEncoder()
y = le.fit_transform(y_raw)

indices = np.arange(len(df_m))
X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
    X, y, indices,
    test_size=0.2,
    stratify=y,
    random_state=RANDOM_SEED
)

# Scale features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nClass distribution in training set:")
print(Counter(y_train))
print(f"\nFeature scaling applied with StandardScaler")

# Use shared helpers from `helper_functions.py` as `hf`

def train_tabpfn_pipeline(X_train, y_train, X_test, y_test, strategy_name="Original", class_weight=None):
    # TabPFN doesn't support class_weight parameter directly, so we note it for the strategy name
    clf = TabPFNClassifier()
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    try:
        y_pred_proba = clf.predict_proba(X_test)
    except Exception:
        y_pred_proba = None
    class_names = le.classes_
    
    metrics = hf.compute_metrics(y_test, y_pred, y_pred_proba)
    print(f"\n{strategy_name} Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    print("\nPer-Class Accuracy:")
    for i, class_name in enumerate(le.classes_):
        mask = (y_test == i)
        if mask.sum() > 0:
            print(f"{class_name}: {accuracy_score(y_test[mask], y_pred[mask]):.4f}")
    
    safe_name = f"{MODEL_NAME}_{strategy_name.replace(' ', '_').lower()}{VERSION_SUFFIX}"
    cm_path = os.path.join(FIG_DIR, f'confusion_matrix_{safe_name}.png')
    cm_path = hf.plot_confusion_matrix(confusion_matrix(y_test, y_pred), class_names, save_path=cm_path)

    pred_path = os.path.join(DATA_DIR, f'predictions_{safe_name}.csv')
    try:
        pred_path = hf.save_predictions(original_data, y_test, y_pred, y_pred_proba, test_idx, class_names, save_path=pred_path)
    except Exception as e:
        print(f"WARNING: Failed to save predictions: {e}")
        pred_path = None

    error_path = os.path.join(DATA_DIR, f'error_analysis_{safe_name}.csv')
    try:
        error_path = hf.analyze_errors(original_data, y_test, y_pred, y_pred_proba, test_idx, class_names, save_path=error_path)
    except Exception as e:
        print(f"WARNING: Failed to save error analysis: {e}")
        error_path = None

    # ROC and PR plots
    if y_pred_proba is not None:
        roc_path = os.path.join(FIG_DIR, f'roc_curves_{safe_name}.png')
        roc_path = hf.plot_roc_curves(y_test, y_pred_proba, class_names, save_path=roc_path)
        pr_path = os.path.join(FIG_DIR, f'pr_curves_{safe_name}.png')
        pr_path = hf.plot_precision_recall_curves(y_test, y_pred_proba, class_names, save_path=pr_path)
    else:
        roc_path = None
        pr_path = None

    model_path = os.path.join(MODEL_DIR, f'model_{safe_name}.pkl')
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, model_path)
    print(f"TabPFN model saved to: {model_path}")

    assets = {
        'confusion_matrix': cm_path,
        'predictions': pred_path,
        'error_analysis': error_path,
        'roc_curves': roc_path,
        'pr_curves': pr_path,
        'model': model_path
    }

    return clf, assets

# Prepare sampling strategies with scaled data
# Find majority class for undersampling
class_counts = Counter(y_train)
majority_class = max(class_counts, key=class_counts.get)
print(f"\nMajority class: {le.inverse_transform([majority_class])[0]} with {class_counts[majority_class]} samples")

strategies = {
    "Original": (X_train_scaled, y_train),
    "SMOTE": SMOTE(random_state=RANDOM_SEED).fit_resample(X_train_scaled, y_train),
    "RandomUnderSampler": RandomUnderSampler(random_state=RANDOM_SEED, sampling_strategy='auto').fit_resample(X_train_scaled, y_train),
    "ClassWeights": (X_train_scaled, y_train)  # Note: TabPFN doesn't natively support class_weight, but we keep balanced data
}

print("\nStrategy sample counts:")
for name, (X_tr, y_tr) in strategies.items():
    print(f"{name}: {Counter(y_tr)}")

models = {}
assets = {}
for name, (X_tr, y_tr) in strategies.items():
    print(f"\n==== Running strategy: {name} ====")
    class_weight = 'balanced' if name == "ClassWeights" else None
    model, asset = train_tabpfn_pipeline(X_tr, y_tr, X_test_scaled, y_test, strategy_name=name, class_weight=class_weight)
    models[name] = model
    assets[name] = asset

def generate_html_report(results, class_names, output_path=None, assets=None):
    if output_path is None:
        output_path = os.path.join(REPORTS_DIR, f'comparison_report_{MODEL_NAME}{VERSION_SUFFIX}.html')
    hf.generate_model_html_report(MODEL_NAME, results, class_names, output_path=output_path, dataset=DATA_FILE, fig_dir=FIG_DIR, reports_dir=REPORTS_DIR, assets=assets)

results = {}
for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = None
    try:
        y_pred_proba = model.predict_proba(X_test_scaled)
    except Exception:
        y_pred_proba = None
    
    # Calculate per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None, zero_division=0)
    
    results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'mse': mean_squared_error(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'log_loss': (log_loss(y_test, y_pred_proba) if y_pred_proba is not None else None),
        'per_class_precision': precision.tolist(),
        'per_class_recall': recall.tolist(),
        'per_class_f1': f1.tolist(),
        'per_class_support': support.tolist(),
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    # Add enhanced loss metrics
    if y_pred_proba is not None:
        try:
            results[name]['focal_loss'] = hf.focal_loss(y_test, y_pred_proba, gamma=2.0)
            results[name]['weighted_log_loss'] = hf.class_weighted_log_loss(y_test, y_pred_proba)
        except Exception:
            results[name]['focal_loss'] = results[name]['log_loss']
            results[name]['weighted_log_loss'] = results[name]['log_loss']
    
    # Add binary classification metrics if applicable
    binary_metrics = hf.compute_binary_classification_metrics(y_test, y_pred)
    results[name].update(binary_metrics)

class_names = le.classes_.tolist()
generate_html_report(results, class_names, assets=assets)
# Append to combined results JSON and regenerate combined HTML
try:
    json_path = os.path.join(REPORTS_DIR, f'combined_results{VERSION_SUFFIX}.json')
    output_path = os.path.join(REPORTS_DIR, f'combined_report_all_models{VERSION_SUFFIX}.html')
    hf.append_model_results_to_json(MODEL_NAME, results, class_names, json_path=json_path, assets=assets)
    hf.generate_combined_html(json_path=json_path, output_path=output_path)
except Exception as e:
    print(f"Warning: failed to append/generate combined report: {e}")
