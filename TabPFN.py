from tabpfn import TabPFNClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, log_loss, recall_score,
    precision_score, classification_report, confusion_matrix, precision_recall_curve,
    average_precision_score, roc_curve, roc_auc_score
)
from imblearn.over_sampling import SMOTE, RandomOverSampler
import joblib

# ================= Load Data =================
CSV_FILE_PATH = "Input_Data-Mix-train.csv"
df_m = pd.read_csv(CSV_FILE_PATH)
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

# ================= Helper Functions =================
def plot_confusion_matrix(cm, class_names, save_path='confusion_matrix_tabpfn.png'):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Reds)
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
    print(f"Confusion matrix saved to: {save_path}")
    plt.close()

def plot_roc_curves(y_test, y_pred_proba, class_names, save_path='roc_curves_tabpfn.png'):
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
    print(f"ROC curves saved to: {save_path}")
    plt.close()

def plot_precision_recall_curves(y_test, y_pred_proba, class_names, save_path='pr_curves_tabpfn.png'):
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
    print(f"Precision-Recall curves saved to: {save_path}")
    plt.close()

def save_predictions(data, y_test, y_pred, y_pred_proba, test_indices, class_names, save_path='predictions_tabpfn.csv'):
    results = data.iloc[test_indices].copy()
    results['True_Label'] = y_test
    results['Predicted_Label'] = y_pred
    for i, class_name in enumerate(class_names):
        results[f'Prob_{class_name}'] = y_pred_proba[:, i]
    results.to_csv(save_path, index=False)
    print(f"Predictions saved to: {save_path}")

def analyze_errors(data, y_test, y_pred, y_pred_proba, test_indices, class_names, save_path='error_analysis_tabpfn.csv'):
    errors = data.iloc[test_indices].copy()
    errors['True_Label'] = y_test
    errors['Predicted_Label'] = y_pred
    errors['Correct'] = (y_test == y_pred)
    for i, class_name in enumerate(class_names):
        errors[f'Prob_{class_name}'] = y_pred_proba[:, i]
    errors['Confidence'] = y_pred_proba.max(axis=1)
    misclassified = errors[~errors['Correct']].sort_values('Confidence', ascending=False)
    misclassified.to_csv(save_path, index=False)
    print(f"Error analysis saved to: {save_path} ({len(misclassified)} misclassifications)")
    return misclassified

def compute_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision (macro)": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall (macro)": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1 Score (macro)": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Log Loss": log_loss(y_true, y_prob)
    }

# ================= Train and Save TabPFN Models =================
def train_tabpfn_pipeline(X_train, y_train, X_test, y_test, strategy_name="Original"):
    clf = TabPFNClassifier()
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)
    class_names = [str(c) for c in np.unique(y_test)]
    
    metrics = compute_metrics(y_test, y_pred, y_pred_proba)
    print(f"\n{strategy_name} Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Per-class accuracy
    print("\nPer-Class Accuracy:")
    for i, class_name in enumerate(le.classes_):
        mask = (y_test == i)
        if mask.sum() > 0:
            print(f"{class_name}: {accuracy_score(y_test[mask], y_pred[mask]):.4f}")
    
    # Save outputs
    safe_name = strategy_name.replace(" ", "_").lower()
    plot_confusion_matrix(confusion_matrix(y_test, y_pred), class_names, save_path=f'confusion_matrix_{safe_name}.png')
    plot_roc_curves(y_test, y_pred_proba, class_names, save_path=f'roc_curves_{safe_name}.png')
    plot_precision_recall_curves(y_test, y_pred_proba, class_names, save_path=f'pr_curves_{safe_name}.png')
    save_predictions(original_data, y_test, y_pred, y_pred_proba, test_idx, class_names, save_path=f'predictions_{safe_name}.csv')
    analyze_errors(original_data, y_test, y_pred, y_pred_proba, test_idx, class_names, save_path=f'error_analysis_{safe_name}.csv')
    
    # Save model
    model_path = f'model_{safe_name}.pkl'
    joblib.dump(clf, model_path)
    print(f"TabPFN model saved to: {model_path}")
    
    return clf

# ================= Apply strategies =================
strategies = {
    "Original": (X_train, y_train),
    "SMOTE": SMOTE(random_state=42).fit_resample(X_train, y_train),
    "RandomOverSampler": RandomOverSampler(random_state=42).fit_resample(X_train, y_train)
}

models = {}
for name, (X_tr, y_tr) in strategies.items():
    print(f"\n==== Running strategy: {name} ====")
    models[name] = train_tabpfn_pipeline(X_tr, y_tr, X_test, y_test, strategy_name=name)
