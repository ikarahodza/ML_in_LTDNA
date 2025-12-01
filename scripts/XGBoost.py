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
import helper_functions as hf

MODEL_NAME = 'xgboost'
DATA_DIR = 'data'
FIG_DIR = 'figures'
MODEL_DIR = 'models'
REPORTS_DIR = 'reports'

data_mix = 'Input_Data-Mix-train.csv'
data_single = 'Input_Data-Single-train.csv'
DATA_FILE = data_mix

df_m = pd.read_csv(data_mix)
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

def train_xgb_pipeline(X_train, y_train, X_test, y_test, feature_names, strategy_name="Original"):
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.3,
        objective='multi:softprob',
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42
    )
    model.fit(X_train, y_train)
    
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
    cm_path = hf.plot_confusion_matrix(confusion_matrix(y_test, y_pred), class_names, save_path=cm_path)

    fi_path = os.path.join(FIG_DIR, f'feature_importance_{safe_name}.png')
    try:
        fi_path, fi_csv = hf.plot_feature_importance(model, feature_names, save_path=fi_path)
    except Exception:
        fi_path = hf.plot_feature_importance(model, feature_names, save_path=fi_path)

    pred_path = os.path.join(DATA_DIR, f'predictions_{safe_name}.csv')
    pred_path = hf.save_predictions(original_data, y_test, y_pred, y_pred_proba, test_idx, class_names,
                     save_path=pred_path)

    error_path = os.path.join(DATA_DIR, f'error_analysis_{safe_name}.csv')
    error_path = hf.analyze_errors(original_data, y_test, y_pred, y_pred_proba, test_idx, class_names,
                   save_path=error_path)

    if y_pred_proba is not None:
        roc_path = os.path.join(FIG_DIR, f'roc_curves_{safe_name}.png')
        roc_path = hf.plot_roc_curves(y_test, y_pred_proba, class_names, save_path=roc_path)
        pr_path = os.path.join(FIG_DIR, f'pr_curves_{safe_name}.png')
        pr_path = hf.plot_precision_recall_curves(y_test, y_pred_proba, class_names, save_path=pr_path)
    else:
        roc_path = None
        pr_path = None

    model_path = os.path.join(MODEL_DIR, f'model_{safe_name}.pkl')
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")

    assets = {
        'confusion_matrix': cm_path,
        'feature_importance': fi_path,
        'predictions': pred_path,
        'error_analysis': error_path,
        'roc_curves': roc_path,
        'pr_curves': pr_path,
        'model': model_path
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

class_names = le.classes_.tolist()
generate_html_report(results, class_names, assets=assets)
# Append to combined results JSON and regenerate combined HTML
try:
    hf.append_model_results_to_json(MODEL_NAME, results, class_names, assets=assets)
    hf.generate_combined_html()
except Exception as e:
    print(f"Warning: failed to append/generate combined report: {e}")
