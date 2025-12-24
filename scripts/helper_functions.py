import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, log_loss, recall_score,
    precision_score, confusion_matrix, precision_recall_curve, average_precision_score,
    mean_squared_error,
    roc_curve, roc_auc_score
)
from sklearn.calibration import CalibratedClassifierCV
import joblib
import seaborn as sns
import json
import os


def focal_loss(y_true, y_pred_proba, gamma=2.0, alpha=None):
    """
    Calculate Focal Loss for multi-class classification.
    
    Focal Loss = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Args:
        y_true: True labels (1D array of class indices)
        y_pred_proba: Predicted probabilities (2D array, shape: [n_samples, n_classes])
        gamma: Focusing parameter (higher = more focus on hard examples). Default=2.0
        alpha: Class weights (1D array or None). If None, no class weighting.
    
    Returns:
        float: Average focal loss
    """
    n_samples = len(y_true)
    n_classes = y_pred_proba.shape[1]
    
    # Clip probabilities to prevent log(0)
    y_pred_proba = np.clip(y_pred_proba, 1e-7, 1 - 1e-7)
    
    # Convert y_true to one-hot encoding
    y_true_one_hot = np.zeros((n_samples, n_classes))
    y_true_one_hot[np.arange(n_samples), y_true] = 1
    
    # Calculate p_t (probability of the true class)
    p_t = np.sum(y_pred_proba * y_true_one_hot, axis=1)
    
    # Calculate focal loss
    focal_weight = (1 - p_t) ** gamma
    cross_entropy = -np.log(p_t)
    
    if alpha is not None:
        # Apply class weights
        alpha_t = alpha[y_true]
        loss = alpha_t * focal_weight * cross_entropy
    else:
        loss = focal_weight * cross_entropy
    
    return np.mean(loss)


def class_weighted_log_loss(y_true, y_pred_proba, class_weights=None):
    """
    Calculate class-weighted log loss.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        class_weights: Dict or array of class weights
    
    Returns:
        float: Weighted log loss
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    # Compute class weights if not provided
    if class_weights is None:
        classes = np.unique(y_true)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_true)
        class_weights = dict(zip(classes, class_weights))
    
    # Convert to dict if array
    if isinstance(class_weights, np.ndarray):
        classes = np.unique(y_true)
        class_weights = dict(zip(classes, class_weights))
    
    # Calculate sample weights
    sample_weights = np.array([class_weights[y] for y in y_true])
    
    # Calculate weighted log loss
    return log_loss(y_true, y_pred_proba, sample_weight=sample_weights)


def apply_label_smoothing(y_pred_proba, epsilon=0.1):
    """
    Apply label smoothing to predicted probabilities to prevent overconfidence.
    
    Args:
        y_pred_proba: Predicted probabilities (2D array)
        epsilon: Smoothing parameter (0 = no smoothing, 1 = uniform distribution)
    
    Returns:
        Smoothed probabilities
    """
    n_classes = y_pred_proba.shape[1]
    smoothed = y_pred_proba * (1 - epsilon) + epsilon / n_classes
    return smoothed


def calibrate_classifier(model, X_train, y_train, method='sigmoid', cv=5):
    """
    Calibrate a classifier to improve probability estimates.
    
    Args:
        model: Trained classifier
        X_train: Training features
        y_train: Training labels
        method: 'sigmoid' (Platt scaling) or 'isotonic'
        cv: Number of cross-validation folds
    
    Returns:
        Calibrated classifier
    """
    calibrated_model = CalibratedClassifierCV(
        model, 
        method=method, 
        cv=cv,
        ensemble=True
    )
    calibrated_model.fit(X_train, y_train)
    return calibrated_model

def plot_confusion_matrix(cm, class_names, save_path='confusion_matrix.png', cmap=plt.cm.Blues):
    """Plot and save a confusion matrix. Returns the saved file path.

    Signature: plot_confusion_matrix(cm, class_names, save_path)
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    thresh = cm.max() / 2. if cm.size else 0
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(int(cm[i, j]), 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix: {save_path}")
    return save_path

def plot_roc_curves(y_test, y_pred_proba, class_names, save_path='roc_curves.png'):
    from sklearn.preprocessing import label_binarize
    y_test_bin = label_binarize(y_test, classes=range(len(class_names)))
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        auc_score = roc_auc_score(y_test_bin[:, i], y_pred_proba[:, i])
        plt.plot(fpr, tpr, linewidth=2, label=f'{class_name} (AUC={auc_score:.3f})')
    plt.plot([0,1],[0,1],'k--', alpha=0.5)
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('ROC Curves', fontsize=16, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC curves: {save_path}")
    return save_path

def plot_precision_recall_curves(y_test, y_pred_proba, class_names, save_path='pr_curves.png'):
    from sklearn.preprocessing import label_binarize
    y_test_bin = label_binarize(y_test, classes=range(len(class_names)))
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_pred_proba[:, i])
        ap = average_precision_score(y_test_bin[:, i], y_pred_proba[:, i])
        plt.plot(recall, precision, linewidth=2, label=f'{class_name} (AP={ap:.3f})')
    plt.xlabel('Recall', fontsize=12, fontweight='bold')
    plt.ylabel('Precision', fontsize=12, fontweight='bold')
    plt.title('Precision-Recall Curves', fontsize=16, fontweight='bold')
    plt.legend(loc='lower left')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved precision-recall curves: {save_path}")
    return save_path

# ------------------ Metrics ------------------
def compute_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision (macro)": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall (macro)": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1 Score (macro)": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Log Loss": log_loss(y_true, y_prob),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "Mean Squared Error": float(mean_squared_error(y_true, y_pred))
    }
    
    # Add enhanced loss metrics
    try:
        metrics["Focal Loss"] = focal_loss(y_true, y_prob, gamma=2.0)
        metrics["Weighted Log Loss"] = class_weighted_log_loss(y_true, y_prob)
    except Exception as e:
        # If enhanced metrics fail, use standard log loss
        metrics["Focal Loss"] = metrics["Log Loss"]
        metrics["Weighted Log Loss"] = metrics["Log Loss"]

    if cm.shape == (2, 2):
        TN, FP, FN, TP = cm.ravel()
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        precision_pos = TP / (TP + FP) if (TP + FP) > 0 else 0
        npv = TN / (TN + FN) if (TN + FN) > 0 else 0
        
        # Calculate Informedness and Markedness
        informedness = sensitivity + specificity - 1  # Bookmaker Informedness (BM)
        markedness = precision_pos + npv - 1  # DeltaP

        metrics.update({
            "Sensitivity (TPR)": sensitivity,
            "Specificity (TNR)": specificity,
            "Precision (positive)": precision_pos,
            "NPV": npv,
            "Informedness": informedness,
            "Markedness": markedness,
        })

    return metrics


def compute_binary_classification_metrics(y_true, y_pred):
    """
    Compute additional classification metrics: Sensitivity, Specificity, 
    Precision, NPV, Informedness, and Markedness.
    
    For binary classification: returns standard binary metrics.
    For multi-class: returns macro-averaged metrics using one-vs-rest approach.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if cm.shape == (2, 2):
        # Binary classification
        TN, FP, FN, TP = cm.ravel()
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        precision_pos = TP / (TP + FP) if (TP + FP) > 0 else 0
        npv = TN / (TN + FN) if (TN + FN) > 0 else 0
        
        # Calculate Informedness and Markedness
        informedness = sensitivity + specificity - 1  # Bookmaker Informedness (BM)
        markedness = precision_pos + npv - 1  # DeltaP
        
        return {
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'precision_positive': float(precision_pos),
            'npv': float(npv),
            'informedness': float(informedness),
            'markedness': float(markedness),
        }
    
    elif cm.shape[0] > 2:
        # Multi-class classification: use one-vs-rest macro-averaging
        n_classes = cm.shape[0]
        informedness_scores = []
        markedness_scores = []
        
        for class_idx in range(n_classes):
            # Create binary confusion matrix for this class (one-vs-rest)
            TP = cm[class_idx, class_idx]
            FN = cm[class_idx, :].sum() - TP
            FP = cm[:, class_idx].sum() - TP
            TN = cm.sum() - TP - FN - FP
            
            # Calculate metrics for this class
            sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
            specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            npv = TN / (TN + FN) if (TN + FN) > 0 else 0
            
            # Calculate Informedness and Markedness for this class
            class_informedness = sensitivity + specificity - 1
            class_markedness = precision + npv - 1
            
            informedness_scores.append(class_informedness)
            markedness_scores.append(class_markedness)
        
        # Macro-average across all classes
        return {
            'informedness': float(np.mean(informedness_scores)),
            'markedness': float(np.mean(markedness_scores)),
        }
    
    return {}


def save_predictions(data, y_test, y_pred, y_pred_proba, test_indices, class_names, save_path='predictions.csv'):
    """Save predictions for test indices to CSV and return the saved path.

    Required signature: save_predictions(data, y_test, y_pred, y_pred_proba, test_indices, class_names, save_path)
    """
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    results = data.iloc[test_indices].copy()
    results = results.reset_index(drop=True)
    results['True_Label'] = list(y_test)
    results['Predicted_Label'] = list(y_pred)
    # handle missing probabilities gracefully
    if y_pred_proba is not None:
        for i, class_name in enumerate(class_names):
            try:
                results[f'Prob_{class_name}'] = y_pred_proba[:, i]
            except Exception:
                # fallback if y_pred_proba shape mismatch
                results[f'Prob_{class_name}'] = None
    else:
        for class_name in class_names:
            results[f'Prob_{class_name}'] = None

    results.to_csv(save_path, index=False)
    print(f"Saved predictions: {save_path}")
    return save_path

def analyze_errors(data, y_test, y_pred, y_pred_proba, test_indices, class_names, save_path='error_analysis.csv'):
    """Generate an error analysis CSV and return the saved path.

    Required signature: analyze_errors(data, y_test, y_pred, y_pred_proba, test_indices, class_names, save_path)
    """
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    errors = data.iloc[test_indices].copy()
    errors = errors.reset_index(drop=True)
    errors['True_Label'] = list(y_test)
    errors['Predicted_Label'] = list(y_pred)
    errors['Correct'] = [bool(t == p) for t, p in zip(y_test, y_pred)]

    if y_pred_proba is not None:
        try:
            for i, class_name in enumerate(class_names):
                errors[f'Prob_{class_name}'] = y_pred_proba[:, i]
            errors['Confidence'] = y_pred_proba.max(axis=1)
        except Exception:
            # If shape mismatch or failure, fill with None
            for class_name in class_names:
                errors[f'Prob_{class_name}'] = None
            errors['Confidence'] = None
    else:
        for class_name in class_names:
            errors[f'Prob_{class_name}'] = None
        errors['Confidence'] = None

    misclassified = errors[~errors['Correct']].sort_values('Confidence', ascending=False) if 'Confidence' in errors.columns and errors['Confidence'].notnull().any() else errors[~errors['Correct']]
    misclassified.to_csv(save_path, index=False)
    print(f"Saved error analysis: {save_path} ({len(misclassified)} misclassifications)")
    return save_path

def save_model(model, save_path='model.pkl'):
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    joblib.dump(model, save_path)
    print(f"Saved model: {save_path}")
    return save_path


def plot_feature_importance(model, feature_names, top_n=20, save_path='feature_importance.png', importance_type='auto', X=None, y=None, random_state=42):
    """Plot and save top feature importances from a fitted model.

    Parameters
    - model: fitted estimator or array-like importances
    - feature_names: list of feature names (same length as X columns)
    - top_n: how many top features to plot
    - save_path: path to save the PNG
    - importance_type: 'auto'|'split'|'gain'|'permutation'|'shap'|'weight'|'cover'
      - 'auto' will pick a sensible default for the model (e.g., LGBM gain/split, sklearn feature_importances_)
    - X, y: required for permutation importance and SHAP when importance_type is set to those
    - random_state: passed to permutation importance

    The function will attempt to use model-specific APIs (LightGBM, XGBoost, sklearn).
    If SHAP or permutation importance is requested but unavailable (missing packages or X/y not provided),
    the function will fall back to available importances and notify the user.
    """
    method_used = None

    # Helper to write CSV
    def _save_csv(names, imps, method, out_path):
        df = pd.DataFrame({'feature': list(names), 'importance': list(imps)})
        df['method'] = method
        df = df.sort_values('importance', ascending=False)
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Saved feature importances CSV to: {out_path}")

    importances = None

    # If model is actually an array-like of importances
    if not hasattr(model, '__class__') or isinstance(model, (list, tuple, np.ndarray)):
        importances = np.asarray(model)
        method_used = 'array'

    # Try model-specific importances
    if importances is None:
        # LightGBM
        try:
            import lightgbm as lgb
            if isinstance(model, lgb.basic.Booster) or hasattr(model, 'booster_') or isinstance(model, lgb.LGBMModel):
                # choose importance type
                if importance_type in ('auto', 'gain'):
                    try:
                        importances = np.asarray(model.feature_importance(importance_type='gain'))
                        method_used = 'gain'
                    except Exception:
                        importances = np.asarray(model.booster_.feature_importance(importance_type='gain'))
                        method_used = 'gain'
                elif importance_type == 'split':
                    try:
                        importances = np.asarray(model.feature_importance(importance_type='split'))
                        method_used = 'split'
                    except Exception:
                        importances = np.asarray(model.booster_.feature_importance(importance_type='split'))
                        method_used = 'split'
        except Exception:
            pass

    # XGBoost
    if importances is None:
        try:
            import xgboost as xgb
            # sklearn wrapper provides get_booster or get_score
            if hasattr(model, 'get_booster') or hasattr(model, 'booster'):
                try:
                    booster = model.get_booster() if hasattr(model, 'get_booster') else model.booster()
                    if importance_type in ('auto', 'gain'):
                        score = booster.get_score(importance_type='gain')
                        # map f0->importance
                        imps = np.zeros(len(feature_names), dtype=float)
                        for k, v in score.items():
                            if k.startswith('f'):
                                idx = int(k[1:])
                                if idx < len(imps):
                                    imps[idx] = v
                        importances = imps
                        method_used = 'gain'
                    elif importance_type == 'weight':
                        score = booster.get_score(importance_type='weight')
                        imps = np.zeros(len(feature_names), dtype=float)
                        for k, v in score.items():
                            if k.startswith('f'):
                                idx = int(k[1:])
                                if idx < len(imps):
                                    imps[idx] = v
                        importances = imps
                        method_used = 'weight'
                except Exception:
                    pass
        except Exception:
            pass

    # sklearn-style feature_importances_
    if importances is None:
        if hasattr(model, 'feature_importances_'):
            try:
                importances = np.asarray(model.feature_importances_)
                method_used = 'feature_importances_'
            except Exception:
                importances = None

    # Permutation importance
    if (importance_type == 'permutation' or (importance_type == 'auto' and importances is None)):
        try:
            from sklearn.inspection import permutation_importance
            if X is not None and y is not None:
                r = permutation_importance(model, X, y, n_repeats=10, random_state=random_state, n_jobs=1)
                importances = r.importances_mean
                method_used = 'permutation'
        except Exception:
            pass

    # SHAP importance (mean absolute shap values)
    if (importance_type == 'shap') or (importance_type == 'auto' and importances is None):
        try:
            import shap
            if X is not None:
                try:
                    # Tree explainer when possible
                    try:
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(X)
                    except Exception:
                        explainer = shap.Explainer(model, X)
                        shap_values = explainer(X)

                    # shap_values may be array or list (for multiclass)
                    if isinstance(shap_values, list):
                        # average across classes
                        arr = np.mean([np.abs(s).mean(axis=0) for s in shap_values], axis=0)
                    else:
                        # shap_values.values for newer APIs
                        vals = getattr(shap_values, 'values', shap_values)
                        arr = np.mean(np.abs(vals), axis=0)

                    importances = np.asarray(arr)
                    method_used = 'shap'
                except Exception:
                    importances = None
        except Exception:
            pass

    # Final fallback: if still None, attempt attribute, else raise
    if importances is None:
        # Last attempt: attribute access or error
        try:
            importances = np.asarray(getattr(model, 'feature_importances_', None))
            if importances is None or len(importances) == 0:
                raise Exception('no feature_importances_')
            method_used = 'feature_importances_'
        except Exception as e:
            print(f"Could not determine feature importances automatically: {e}")
            # Return gracefully with a clear message
            raise RuntimeError('Could not determine feature importances for the provided model. Try passing `importance_type`, or provide `X`/`y` for permutation/shap, or pass an explicit importances array.')

    # Ensure importances is numpy array and matches feature_names
    importances = np.asarray(importances, dtype=float)
    if len(importances) != len(feature_names):
        # If lengths mismatch, try to truncate/pad
        minlen = min(len(importances), len(feature_names))
        importances = importances[:minlen]
        feature_names = feature_names[:minlen]

    # Save CSV alongside plot
    csv_path = os.path.splitext(save_path)[0] + f'_importances_{method_used}.csv'
    try:
        _save_csv(feature_names, importances, method_used, csv_path)
    except Exception:
        print(f"Warning: could not save feature importances CSV: {csv_path}")

    # Plot top_n
    indices = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(10, max(6, top_n * 0.35)))
    plt.title(f'Top {min(top_n, len(indices))} Feature Importances ({method_used})', fontsize=14, fontweight='bold')
    plt.barh(range(len(indices)), importances[indices][::-1], color='steelblue')
    labels = [feature_names[i] for i in indices][::-1]
    plt.yticks(range(len(indices)), labels)
    plt.xlabel('Importance', fontsize=12, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Feature importance plot saved to: {save_path} (method: {method_used})")
    plt.close()
    return save_path, csv_path


def save_feature_importances_csv(feature_names, importances, method, output_path='feature_importances.csv'):
    """Save feature importances to a short CSV report with chosen method."""
    df = pd.DataFrame({'feature': list(feature_names), 'importance': list(importances)})
    df['method'] = method
    df = df.sort_values('importance', ascending=False)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved feature importances CSV to: {output_path}")
    return output_path


def _sanitize_results_for_json(results):
    """Convert results dict (which may contain numpy arrays) into JSON-serializable structure."""
    out = {}
    for strat, r in results.items():
        rr = {}
        for k, v in r.items():
            if isinstance(v, (np.floating, float)):
                rr[k] = float(v)
            elif isinstance(v, (np.integer, int)):
                rr[k] = int(v)
            elif hasattr(v, 'tolist'):
                try:
                    rr[k] = v.tolist()
                except Exception:
                    rr[k] = str(v)
            else:
                try:
                    json.dumps(v)
                    rr[k] = v
                except Exception:
                    rr[k] = str(v)
        out[strat] = rr
    return out


def append_model_results_to_json(model_name, results, class_names, json_path='reports/combined_results.json', assets=None):
    """Append a model's `results` dict to a JSON file keyed by model name.

    `results` is expected to be the same structure used across model scripts: mapping
    strategy -> metrics dict (including numeric and list values). `class_names` is
    used when generating the combined HTML later. `assets` is a dict mapping strategy
    names to asset paths (confusion_matrix, feature_importance, etc.).
    """
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    data = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
        except Exception:
            data = {}

    model_data = {
        'results': _sanitize_results_for_json(results),
        'class_names': [str(c) for c in class_names]
    }
    
    # Add assets if provided
    if assets:
        model_data['assets'] = assets
    
    data[model_name] = model_data

    with open(json_path, 'w', encoding='utf-8') as fh:
        json.dump(data, fh, indent=2)

    print(f"Appended results for model '{model_name}' to {json_path}")
    return json_path


def generate_combined_html(json_path='reports/combined_results.json', output_path='reports/combined_report_all_models.html'):
    """Generate a combined HTML report from the JSON file containing results for multiple models.

    This implementation uses a Jinja2 template `reports/combined_report_template.html` when
    Jinja2 is available. If not installed, it falls back to a simple HTML generator.
    """
    if not os.path.exists(json_path):
        print(f"No combined results file found at {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as fh:
        data = json.load(fh)

    template_path = os.path.join(os.path.dirname(json_path), 'combined_report_template.html')

    # Try to use Jinja2 for nicer templating
    try:
        from jinja2 import Environment, FileSystemLoader, select_autoescape
        env = Environment(
            loader=FileSystemLoader(os.path.dirname(template_path)),
            autoescape=select_autoescape(['html', 'xml'])
        )
        template = env.get_template(os.path.basename(template_path))
        rendered = template.render(data=data, timestamp=__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as fh:
            fh.write(rendered)

        print(f"Generated combined HTML report with Jinja2 template: {output_path}")
        return output_path
    except Exception:
        # Fallback to the simple generator if Jinja2 isn't available or templating fails
        pass

    # Fallback simple HTML generator (keeps previous behavior)
    timestamp = __import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    html = [
        '<html>',
        '<head><meta charset="utf-8"><title>Combined Comparison Report</title></head>',
        '<body>',
        f'<h1>Combined Comparison Report</h1>',
        f'<div>Generated: {timestamp}</div>'
    ]

    for model_name, content in data.items():
        html.append(f'<h2>Model: {model_name}</h2>')
        results = content.get('results', {})
        class_names = content.get('class_names', [])

        # Summary table per strategy
        html.append('<table border="1" cellpadding="6" cellspacing="0">')
        html.append('<tr><th>Strategy</th><th>Accuracy</th><th>Balanced Accuracy</th><th>F1 (macro)</th><th>MSE</th><th>Log Loss</th><th>Informedness</th><th>Markedness</th></tr>')
        for strat, r in results.items():
            acc = r.get('accuracy', 'N/A')
            bacc = r.get('balanced_accuracy', 'N/A')
            f1 = r.get('f1_macro', 'N/A')
            # try multiple keys for mse
            mse_val = r.get('mse', None)
            if mse_val is None:
                mse_val = r.get('Mean Squared Error', 'N/A')
            ll = r.get('log_loss', 'N/A')
            if isinstance(ll, list):
                ll = ll[0] if ll else 'N/A'
            informedness = r.get('informedness', 'N/A')
            markedness = r.get('markedness', 'N/A')
            html.append(f'<tr><td>{strat}</td><td>{acc}</td><td>{bacc}</td><td>{f1}</td><td>{mse_val}</td><td>{ll}</td><td>{informedness}</td><td>{markedness}</td></tr>')
        html.append('</table>')

        # Per-class recall
        if class_names:
            html.append('<h3>Per-class Recall</h3>')
            html.append('<table border="1" cellpadding="6" cellspacing="0">')
            html.append('<tr><th>Strategy</th>')
            for cn in class_names:
                html.append(f'<th>{cn}</th>')
            html.append('</tr>')
            for strat, r in results.items():
                html.append(f'<tr><td>{strat}</td>')
                recalls = r.get('per_class_recall', [])
                for val in recalls:
                    html.append(f'<td>{val}</td>')
                html.append('</tr>')
            html.append('</table>')

    html.append('</body></html>')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(html))

    print(f"Generated combined HTML report (fallback): {output_path}")
    return output_path


def generate_model_html_report(model_name, results, class_names, output_path=None, dataset=None, fig_dir='figures', reports_dir='reports', assets=None):
    """Generate a per-model HTML comparison report.

    - `model_name`: string identifier used in the filename and title
    - `results`: mapping strategy -> metrics dict (must include keys used by other scripts)
    - `class_names`: list of class names (strings)
    - `output_path`: optional path; default: `reports/comparison_report_{model_name}.html`
    - `dataset`: optional dataset name to display in header
    """
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if output_path is None:
        output_path = os.path.join(reports_dir, f'comparison_report_{model_name}.html')

    ds = dataset if dataset is not None else ''

    html = [
        '<!doctype html>',
        '<html lang="en">',
        '<head>',
        '  <meta charset="utf-8">',
        '  <meta name="viewport" content="width=device-width, initial-scale=1">',
        f'  <title>Comparison Report - {model_name}</title>',
        '  <style>',
        '    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial; margin: 24px; color: #222; }',
        '    h1 { margin: 0 0 8px 0; }',
        '    h2 { margin-top: 24px; margin-bottom: 12px; }',
        '    h3 { margin-top: 20px; margin-bottom: 10px; }',
        '    .meta { color: #666; margin-bottom: 16px; }',
        '    table { border-collapse: collapse; width: 100%; margin-top: 8px; margin-bottom: 12px; }',
        '    table, th, td { border: 1px solid #dcdcdc; }',
        '    th, td { padding: 8px 10px; text-align: left; }',
        '    th { background: #f6f8fa; font-weight: 600; }',
        '    .strategy-name { width: 220px; }',
        '    .small { font-size: 0.9em; color: #555; margin-bottom: 8px; }',
        '    footer { margin-top: 32px; padding-top: 16px; border-top: 1px solid #e1e4e8; color: #666; font-size: 0.9em; text-align: center; }',
        '  </style>',
        '</head>',
        '<body>',
        f'  <h1>Comparison Report - {model_name}</h1>',
        f'  <div class="meta">Generated: {timestamp}</div>',
        f'  <div class="meta">Dataset: {ds}</div>',
        f'  <div class="small">Strategies: {", ".join(results.keys())}</div>'
    ]

    html.append('  <h2>Summary by Strategy</h2>')
    html.append('  <table>')
    html.append('    <thead>')
    html.append('      <tr>')
    html.append('        <th class="strategy-name">Strategy</th>')
    html.append('        <th>Accuracy</th>')
    html.append('        <th>Balanced Accuracy</th>')
    html.append('        <th>F1 (macro)</th>')
    html.append('        <th>Log Loss</th>')
    html.append('        <th>Focal Loss</th>')
    html.append('        <th>Weighted Log Loss</th>')
    html.append('        <th>Informedness</th>')
    html.append('        <th>Markedness</th>')
    html.append('      </tr>')
    html.append('    </thead>')
    html.append('    <tbody>')

    strategy_list = list(results.keys())
    for name, r in results.items():
        # Format basic metrics
        try:
            acc = f"{r['accuracy']:.4f}"
        except Exception:
            acc = r.get('accuracy', 'N/A')
        try:
            bacc = f"{r['balanced_accuracy']:.4f}"
        except Exception:
            bacc = r.get('balanced_accuracy', 'N/A')
        try:
            f1m = f"{r['f1_macro']:.4f}"
        except Exception:
            f1m = r.get('f1_macro', 'N/A')
        
        # Format log loss
        logloss_val = r.get('log_loss')
        if logloss_val is not None:
            try:
                logloss_display = f"{float(logloss_val):.4f}" if logloss_val != 'N/A' else 'N/A'
            except Exception:
                logloss_display = str(logloss_val)
        else:
            logloss_display = 'N/A'
        
        # Format focal loss
        focal_val = r.get('focal_loss')
        if focal_val is not None:
            try:
                focal_display = f"{float(focal_val):.4f}"
            except Exception:
                focal_display = str(focal_val)
        else:
            focal_display = 'N/A'
        
        # Format weighted log loss
        weighted_ll_val = r.get('weighted_log_loss')
        if weighted_ll_val is not None:
            try:
                weighted_ll_display = f"{float(weighted_ll_val):.4f}"
            except Exception:
                weighted_ll_display = str(weighted_ll_val)
        else:
            weighted_ll_display = 'N/A'

        html.append(f'      <tr>')
        html.append(f'        <td class="strategy-name">{name}</td>')
        html.append(f'        <td>{acc}</td>')
        html.append(f'        <td>{bacc}</td>')
        html.append(f'        <td>{f1m}</td>')
        html.append(f'        <td>{logloss_display}</td>')
        html.append(f'        <td>{focal_display}</td>')
        html.append(f'        <td>{weighted_ll_display}</td>')
        
        # Format informedness and markedness
        informedness_val = r.get('informedness')
        if informedness_val is not None:
            try:
                informedness_display = f"{float(informedness_val):.4f}"
            except Exception:
                informedness_display = str(informedness_val)
        else:
            informedness_display = 'N/A'
        
        markedness_val = r.get('markedness')
        if markedness_val is not None:
            try:
                markedness_display = f"{float(markedness_val):.4f}"
            except Exception:
                markedness_display = str(markedness_val)
        else:
            markedness_display = 'N/A'
        
        html.append(f'        <td>{informedness_display}</td>')
        html.append(f'        <td>{markedness_display}</td>')
        html.append(f'      </tr>')

    html.append('    </tbody>')
    html.append('  </table>')

    # Per-class metrics
    if class_names:
        for name, r in results.items():
            html.append(f'  <h3>Per-class Metrics - {name}</h3>')
            html.append('  <table>')
            html.append('    <thead>')
            html.append('      <tr>')
            html.append('        <th class="strategy-name">Class</th>')
            html.append('        <th>Precision</th>')
            html.append('        <th>Recall</th>')
            html.append('        <th>F1-Score</th>')
            html.append('        <th>Support</th>')
            html.append('      </tr>')
            html.append('    </thead>')
            html.append('    <tbody>')
            
            precisions = r.get('per_class_precision', [])
            recalls = r.get('per_class_recall', [])
            f1s = r.get('per_class_f1', [])
            supports = r.get('per_class_support', [])
            
            for i, cn in enumerate(class_names):
                try:
                    prec = f"{float(precisions[i]):.4f}" if i < len(precisions) else 'N/A'
                except Exception:
                    prec = 'N/A'
                try:
                    rec = f"{float(recalls[i]):.4f}" if i < len(recalls) else 'N/A'
                except Exception:
                    rec = 'N/A'
                try:
                    f1 = f"{float(f1s[i]):.4f}" if i < len(f1s) else 'N/A'
                except Exception:
                    f1 = 'N/A'
                try:
                    sup = int(supports[i]) if i < len(supports) else 'N/A'
                except Exception:
                    sup = 'N/A'
                
                html.append(f'      <tr>')
                html.append(f'        <td class="strategy-name">{cn}</td>')
                html.append(f'        <td>{prec}</td>')
                html.append(f'        <td>{rec}</td>')
                html.append(f'        <td>{f1}</td>')
                html.append(f'        <td>{sup}</td>')
                html.append(f'      </tr>')
            
            html.append('    </tbody>')
            html.append('  </table>')

    # Add visualizations section
    if assets:
        html.append('  <h2>Visualizations</h2>')
        for name in strategy_list:
            if name in assets:
                asset = assets[name]
                html.append(f'  <h3>{name}</h3>')
                
                # Confusion Matrix
                cm_path = asset.get('confusion_matrix')
                if cm_path and os.path.exists(cm_path):
                    # Convert absolute path to relative path from reports directory
                    rel_path = os.path.relpath(cm_path, reports_dir)
                    html.append(f'  <div style="margin: 20px 0;">')
                    html.append(f'    <h4>Confusion Matrix</h4>')
                    html.append(f'    <img src="{rel_path}" alt="Confusion Matrix - {name}" style="max-width: 600px; border: 1px solid #ddd; padding: 10px; background: white;">')
                    html.append(f'  </div>')
                
                # Feature Importance
                fi_path = asset.get('feature_importance')
                if fi_path and os.path.exists(fi_path):
                    rel_path = os.path.relpath(fi_path, reports_dir)
                    html.append(f'  <div style="margin: 20px 0;">')
                    html.append(f'    <h4>Feature Importance</h4>')
                    html.append(f'    <img src="{rel_path}" alt="Feature Importance - {name}" style="max-width: 800px; border: 1px solid #ddd; padding: 10px; background: white;">')
                    html.append(f'  </div>')

    html.append('  <footer>Report generated by ML_in_LTDNA pipeline</footer>')
    html.append('</body>')
    html.append('</html>')

    output_dir = os.path.dirname(output_path)
    if output_dir:  # Only create directory if output_path includes a directory
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as fh:
        fh.write('\n')
        fh.write('\n    '.join(html))
        fh.write('\n')

    print(f"Generated model HTML report: {output_path}")
    return output_path
