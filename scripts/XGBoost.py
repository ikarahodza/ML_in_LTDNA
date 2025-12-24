import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             log_loss, recall_score, confusion_matrix,
                             precision_recall_curve, average_precision_score,
                             roc_curve, roc_auc_score, mean_squared_error, mean_absolute_error,
                             precision_recall_fscore_support)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE, RandomOverSampler
import joblib
import os
import argparse
import sys
import shap
import helper_functions as hf

# Get the project root directory (parent of scripts/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

MODEL_NAME = 'xgboost'

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
    # Store in seed-specific subfolder
    SEED_FOLDER = f"seed_{RANDOM_SEED}"
    DATA_DIR = os.path.join(PROJECT_ROOT, RESULTS_FOLDER, SEED_FOLDER, 'data')
    FIG_DIR = os.path.join(PROJECT_ROOT, RESULTS_FOLDER, SEED_FOLDER, 'figures')
    MODEL_DIR = os.path.join(PROJECT_ROOT, RESULTS_FOLDER, SEED_FOLDER, 'models')
    REPORTS_DIR = os.path.join(PROJECT_ROOT, RESULTS_FOLDER, SEED_FOLDER, 'reports')
else:
    # Use default folders for seed 42
    DATA_DIR = os.path.join(PROJECT_ROOT, RESULTS_FOLDER, 'data')
    FIG_DIR = os.path.join(PROJECT_ROOT, RESULTS_FOLDER, 'figures')
    MODEL_DIR = os.path.join(PROJECT_ROOT, RESULTS_FOLDER, 'models')
    REPORTS_DIR = os.path.join(PROJECT_ROOT, RESULTS_FOLDER, 'reports')
DATA_FILE = os.path.join(PROJECT_ROOT, f'Input_Data-{dataset_type.capitalize()}-train.csv')

print(f"\n{'='*60}")
print(f"Running XGBoost with {dataset_type.upper()} dataset")
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

# Use shared helpers from `helper_functions.py` as `hf`

def train_xgb_pipeline(X_train, y_train, X_test, y_test, feature_names, strategy_name="Original"):
    # Split training data for early stopping validation
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.15, random_state=RANDOM_SEED, stratify=y_train
    )
    
    model = XGBClassifier(
        n_estimators=1000,  # Increased for early stopping
        max_depth=6,
        learning_rate=0.1,  # Reduced for better convergence
        objective='multi:softprob',
        eval_metric='mlogloss',
        early_stopping_rounds=50,  # Early stopping parameter in constructor
        random_state=RANDOM_SEED
    )
    model.fit(
        X_train_split, y_train_split,
        eval_set=[(X_val_split, y_val_split)],
        verbose=False
    )
    
    y_pred = model.predict(X_test)
    try:
        y_pred_proba = model.predict_proba(X_test)
    except Exception:
        y_pred_proba = None
    class_names = [str(c) for c in np.unique(y_test)]
    
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    logloss = log_loss(y_test, y_pred_proba)
    per_class_recall = recall_score(y_test, y_pred, average=None)
    
    print(f"\n{strategy_name} - Accuracy: {acc:.4f}, Balanced Acc: {bal_acc:.4f}, F1-macro: {f1_macro:.4f}")

    safe_name = f"{MODEL_NAME}_{strategy_name.replace(' ', '_').lower()}"

    cm_path = os.path.join(FIG_DIR, f'confusion_matrix_{safe_name}.png')
    try:
        cm_path = hf.plot_confusion_matrix(confusion_matrix(y_test, y_pred), class_names, save_path=cm_path)
    except Exception as e:
        print(f"WARNING: Failed to save confusion matrix: {e}")
        cm_path = None

    fi_path = os.path.join(FIG_DIR, f'feature_importance_{safe_name}.png')
    try:
        fi_path, fi_csv = hf.plot_feature_importance(model, feature_names, save_path=fi_path)
    except Exception:
        fi_path = hf.plot_feature_importance(model, feature_names, save_path=fi_path)

    pred_path = os.path.join(DATA_DIR, f'predictions_{safe_name}.csv')
    try:
        pred_path = hf.save_predictions(original_data, y_test, y_pred, y_pred_proba, test_idx, class_names,
                         save_path=pred_path)
    except Exception as e:
        print(f"WARNING: Failed to save predictions: {e}")
        pred_path = None

    error_path = os.path.join(DATA_DIR, f'error_analysis_{safe_name}.csv')
    try:
        error_path = hf.analyze_errors(original_data, y_test, y_pred, y_pred_proba, test_idx, class_names,
                       save_path=error_path)
    except Exception as e:
        print(f"WARNING: Failed to save error analysis: {e}")
        error_path = None

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
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")

    # Generate SHAP values and plots
    shap_summary_path = os.path.join(FIG_DIR, f'shap_summary_{safe_name}.png')
    shap_bar_path = os.path.join(FIG_DIR, f'shap_bar_{safe_name}.png')
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        # SHAP summary plot (beeswarm)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False, plot_type="dot")
        plt.tight_layout()
        os.makedirs(os.path.dirname(shap_summary_path) or '.', exist_ok=True)
        plt.savefig(shap_summary_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved SHAP summary plot: {shap_summary_path}")
        
        # SHAP bar plot (mean absolute values)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False, plot_type="bar")
        plt.tight_layout()
        plt.savefig(shap_bar_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved SHAP bar plot: {shap_bar_path}")
    except Exception as e:
        print(f"WARNING: Failed to generate SHAP plots: {e}")
        shap_summary_path = None
        shap_bar_path = None

    assets = {
        'confusion_matrix': cm_path,
        'feature_importance': fi_path,
        'predictions': pred_path,
        'error_analysis': error_path,
        'roc_curves': roc_path,
        'pr_curves': pr_path,
        'model': model_path,
        'shap_summary': shap_summary_path,
        'shap_bar': shap_bar_path
    }

    return model, assets

strategies = {
    "Original": (X_train, y_train),
    "SMOTE": SMOTE(random_state=RANDOM_SEED).fit_resample(X_train, y_train),
    "RandomOverSampler": RandomOverSampler(random_state=RANDOM_SEED).fit_resample(X_train, y_train)
}

models = {}
assets = {}
for name, (X_tr, y_tr) in strategies.items():
    print(f"\n==== Running strategy: {name} ====")
    model, asset = train_xgb_pipeline(X_tr, y_tr, X_test, y_test, X.columns.tolist(), strategy_name=name)
    models[name] = model
    assets[name] = asset

def generate_html_report(results, class_names, output_path=None, assets=None):
    if output_path is None:
        output_path = os.path.join(REPORTS_DIR, f'comparison_report_{MODEL_NAME}.html')
    hf.generate_model_html_report(MODEL_NAME, results, class_names, output_path=output_path, dataset=DATA_FILE, fig_dir=FIG_DIR, reports_dir=REPORTS_DIR, assets=assets)

results = {}
for name, model in models.items():
    y_pred = model.predict(X_test)
    y_pred_proba = None
    try:
        y_pred_proba = model.predict_proba(X_test)
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
    json_path = os.path.join(REPORTS_DIR, 'combined_results.json')
    output_path = os.path.join(REPORTS_DIR, 'combined_report_all_models.html')
    hf.append_model_results_to_json(MODEL_NAME, results, class_names, json_path=json_path, assets=assets)
    hf.generate_combined_html(json_path=json_path, output_path=output_path)
except Exception as e:
    print(f"Warning: failed to append/generate combined report: {e}")
