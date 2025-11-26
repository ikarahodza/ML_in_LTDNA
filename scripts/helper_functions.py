import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, log_loss, recall_score,
    precision_score, confusion_matrix, precision_recall_curve, average_precision_score,
    roc_curve, roc_auc_score
)
import joblib
import seaborn as sns

# ------------------ Plot Functions ------------------
def plot_confusion_matrix(cm, class_names, save_path='confusion_matrix.png', cmap=plt.cm.Blues):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

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
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

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
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

# ------------------ Metrics ------------------
def compute_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision (macro)": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall (macro)": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1 Score (macro)": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Log Loss": log_loss(y_true, y_prob)
    }

# ------------------ Save Results ------------------
def save_predictions(data, y_test, y_pred, y_pred_proba, test_indices, class_names, save_path='predictions.csv'):
    results = data.iloc[test_indices].copy()
    results['True_Label'] = y_test
    results['Predicted_Label'] = y_pred
    for i, class_name in enumerate(class_names):
        results[f'Prob_{class_name}'] = y_pred_proba[:, i]
    results.to_csv(save_path, index=False)
    print(f"Saved: {save_path}")

def analyze_errors(data, y_test, y_pred, y_pred_proba, test_indices, class_names, save_path='error_analysis.csv'):
    errors = data.iloc[test_indices].copy()
    errors['True_Label'] = y_test
    errors['Predicted_Label'] = y_pred
    errors['Correct'] = (y_test == y_pred)
    for i, class_name in enumerate(class_names):
        errors[f'Prob_{class_name}'] = y_pred_proba[:, i]
    errors['Confidence'] = y_pred_proba.max(axis=1)
    misclassified = errors[~errors['Correct']].sort_values('Confidence', ascending=False)
    misclassified.to_csv(save_path, index=False)
    print(f"Saved: {save_path} ({len(misclassified)} misclassifications)")
    return misclassified

def save_model(model, save_path='model.pkl'):
    joblib.dump(model, save_path)
    print(f"Saved model: {save_path}")
