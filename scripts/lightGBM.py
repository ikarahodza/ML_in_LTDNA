import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import ( accuracy_score, classification_report, confusion_matrix, balanced_accuracy_score, f1_score, log_loss, recall_score, mean_squared_error, mean_absolute_error, precision_recall_fscore_support)
from imblearn.over_sampling import SMOTE, RandomOverSampler
from lightgbm import LGBMClassifier
import lightgbm as lgb
import joblib
import os
import sys
import shap
import helper_functions as hf

# Get the project root directory (parent of scripts/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

MODEL_NAME = 'lgbm'

# Auto-detect dataset from command line argument
# Usage: python lightGBM.py mix  OR  python lightGBM.py single
if len(sys.argv) > 1:
    dataset_type = sys.argv[1].lower()
    if dataset_type not in ['mix', 'single']:
        print(f"Error: Invalid dataset type '{dataset_type}'. Use 'mix' or 'single'.")
        print("Usage: python lightGBM.py [mix|single]")
        sys.exit(1)
else:
    # Default to single if no argument provided
    dataset_type = 'single'
    print(f"No dataset specified. Using default: {dataset_type}")

# Set paths based on dataset type
RESULTS_FOLDER = f"{dataset_type.capitalize()}_Results"
DATA_DIR = os.path.join(PROJECT_ROOT, RESULTS_FOLDER, 'data')
FIG_DIR = os.path.join(PROJECT_ROOT, RESULTS_FOLDER, 'figures')
MODEL_DIR = os.path.join(PROJECT_ROOT, RESULTS_FOLDER, 'models')
REPORTS_DIR = os.path.join(PROJECT_ROOT, RESULTS_FOLDER, 'reports')
DATA_FILE = os.path.join(PROJECT_ROOT, f'Input_Data-{dataset_type.capitalize()}-train.csv')

print(f"\n{'='*60}")
print(f"Running LightGBM with {dataset_type.upper()} dataset")
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
    random_state=42
)

# Use shared helpers from `helper_functions.py` as `hf`

def train_eval_plot(X_tr, y_tr, title, strategy_name):
    # Split training data for early stopping validation
    X_tr_split, X_val_split, y_tr_split, y_val_split = train_test_split(
        X_tr, y_tr, test_size=0.15, random_state=42, stratify=y_tr
    )
    
    model = LGBMClassifier(
        n_estimators=1000,  # Increased for early stopping
        max_depth=6,
        learning_rate=0.1,  # Reduced for better convergence
        objective='multiclass',
        class_weight='balanced',  # Optimize class weights
        random_state=42,
        verbose=-1
    )
    model.fit(
        X_tr_split, y_tr_split,
        eval_set=[(X_val_split, y_val_split)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    y_pred = model.predict(X_test)
    try:
        y_pred_proba = model.predict_proba(X_test)
    except Exception:
        y_pred_proba = None
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    print(f"\n===== {title} =====")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Confusion Matrix 
    cm_path = os.path.join(FIG_DIR, f"confusion_matrix_{strategy_name}.png")
    cm_path = hf.plot_confusion_matrix(confusion_matrix(y_test, y_pred), le.classes_, save_path=cm_path)
    
    # Feature Importance 
    fi_path = os.path.join(FIG_DIR, f"feature_importance_{strategy_name}.png")
    try:
        fi_path, fi_csv = hf.plot_feature_importance(model, X.columns.values, save_path=fi_path)
    except Exception:
        fi_path = hf.plot_feature_importance(model, X.columns.values, save_path=fi_path)
    
    # Save Predictions 
    pred_path = os.path.join(DATA_DIR, f"predictions_{strategy_name}.csv")
    pred_path = hf.save_predictions(original_data, y_test, y_pred, y_pred_proba, test_indices=test_idx, class_names=le.classes_, save_path=pred_path)
    
    # Analyze Errors 
    error_path = os.path.join(DATA_DIR, f"error_analysis_{strategy_name}.csv")
    error_path = hf.analyze_errors(original_data, y_test, y_pred, y_pred_proba, test_indices=test_idx, class_names=le.classes_, save_path=error_path)

    # ROC and PR curves
    if y_pred_proba is not None:
        roc_path = os.path.join(FIG_DIR, f'roc_curves_{strategy_name}.png')
        roc_path = hf.plot_roc_curves(y_test, y_pred_proba, le.classes_, save_path=roc_path)
        pr_path = os.path.join(FIG_DIR, f'pr_curves_{strategy_name}.png')
        pr_path = hf.plot_precision_recall_curves(y_test, y_pred_proba, le.classes_, save_path=pr_path)
    else:
        roc_path = None
        pr_path = None

    #  Save Model 
    model_path = os.path.join(MODEL_DIR, f"lgbm_model_{strategy_name}.pkl")
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")

    # Generate SHAP values and plots
    shap_summary_path = os.path.join(FIG_DIR, f'shap_summary_{strategy_name}.png')
    shap_bar_path = os.path.join(FIG_DIR, f'shap_bar_{strategy_name}.png')
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        # SHAP summary plot (beeswarm)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test, feature_names=X.columns.tolist(), show=False, plot_type="dot")
        plt.tight_layout()
        os.makedirs(os.path.dirname(shap_summary_path) or '.', exist_ok=True)
        plt.savefig(shap_summary_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved SHAP summary plot: {shap_summary_path}")
        
        # SHAP bar plot (mean absolute values)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, feature_names=X.columns.tolist(), show=False, plot_type="bar")
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
    "SMOTE": SMOTE(random_state=42).fit_resample(X_train, y_train),
    "RandomOverSampler": RandomOverSampler(random_state=42).fit_resample(X_train, y_train)
}

models = {}
assets = {}
for name, (X_tr, y_tr) in strategies.items():
    safe_name = f"{MODEL_NAME}_{name.replace(' ', '_').lower()}"
    print(f"\n==== Running strategy: {name} ====")
    model, asset = train_eval_plot(X_tr, y_tr, title=name, strategy_name=safe_name)
    models[name] = model
    assets[name] = asset

def generate_html_report(results, class_names, output_path=None, assets=None):
    if output_path is None:
        output_path = os.path.join(REPORTS_DIR, f'comparison_report_{MODEL_NAME}.html')
    hf.generate_model_html_report(MODEL_NAME, results, class_names, output_path=output_path, dataset=DATA_FILE, fig_dir=FIG_DIR, reports_dir=REPORTS_DIR, assets=assets)

# Build results by evaluating each trained model on the test set
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
