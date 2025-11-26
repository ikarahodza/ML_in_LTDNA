import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             log_loss, recall_score, confusion_matrix,
                             precision_recall_curve, average_precision_score,
                             roc_curve, roc_auc_score)
from xgboost import XGBClassifier
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

# Get train/test split and also keep test indices
indices = np.arange(len(df_m))
X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
    X, y, indices,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ================= Helper Functions =================
def plot_confusion_matrix(cm, class_names, save_path='confusion_matrix_xgboost.png'):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
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
    print(f"\nConfusion matrix saved to: {save_path}")
    plt.close()

def plot_feature_importance(model, feature_names, top_n=20, save_path='feature_importance_xgboost.png'):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    plt.figure(figsize=(10, 8))
    plt.title(f'Top {top_n} Feature Importances', fontsize=16, fontweight='bold')
    plt.barh(range(top_n), importances[indices], color='steelblue')
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Importance', fontsize=12, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Feature importance plot saved to: {save_path}")
    plt.close()

def plot_roc_curves(y_test, y_pred_proba, class_names, save_path='roc_curves_xgboost.png'):
    from sklearn.preprocessing import label_binarize
    y_test_bin = label_binarize(y_test, classes=range(len(class_names)))
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        auc_score = roc_auc_score(y_test_bin[:, i], y_pred_proba[:, i])
        plt.plot(fpr, tpr, linewidth=2, label=f'{class_name} (AUC={auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('ROC Curves', fontsize=16, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"ROC curves saved to: {save_path}")
    plt.close()

def plot_precision_recall_curves(y_test, y_pred_proba, class_names, save_path='pr_curves_xgboost.png'):
    from sklearn.preprocessing import label_binarize
    y_test_bin = label_binarize(y_test, classes=range(len(class_names)))
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_pred_proba[:, i])
        ap = average_precision_score(y_test_bin[:, i], y_pred_proba[:, i])
        plt.plot(recall, precision, linewidth=2, label=f'{class_name} (AP={ap:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12, fontweight='bold')
    plt.ylabel('Precision', fontsize=12, fontweight='bold')
    plt.title('Precision-Recall Curves', fontsize=16, fontweight='bold')
    plt.legend(loc='lower left')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Precision-Recall curves saved to: {save_path}")
    plt.close()

def save_predictions(data, y_test, y_pred, y_pred_proba, test_indices, class_names, save_path='predictions_xgboost.csv'):
    results = data.iloc[test_indices].copy()
    results['True_Label'] = y_test
    results['Predicted_Label'] = y_pred
    for i, class_name in enumerate(class_names):
        results[f'Prob_{class_name}'] = y_pred_proba[:, i]
    results.to_csv(save_path, index=False)
    print(f"Predictions saved to: {save_path}")

def analyze_errors(data, y_test, y_pred, y_pred_proba, test_indices, class_names, save_path='error_analysis_xgboost.csv'):
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

# ================= Train and Save XGB Models =================
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
    y_pred_proba = model.predict_proba(X_test)
    class_names = [str(c) for c in np.unique(y_test)]
    
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    logloss = log_loss(y_test, y_pred_proba)
    per_class_recall = recall_score(y_test, y_pred, average=None)
    
    print(f"\n{strategy_name} - Accuracy: {acc:.4f}, Balanced Acc: {bal_acc:.4f}, F1-macro: {f1_macro:.4f}")

    safe_name = strategy_name.replace(" ", "_").lower()
    
    # Save outputs
    plot_confusion_matrix(confusion_matrix(y_test, y_pred), class_names, save_path=f'confusion_matrix_{safe_name}.png')
    plot_feature_importance(model, feature_names, save_path=f'feature_importance_{safe_name}.png')
    plot_roc_curves(y_test, y_pred_proba, class_names, save_path=f'roc_curves_{safe_name}.png')
    plot_precision_recall_curves(y_test, y_pred_proba, class_names, save_path=f'pr_curves_{safe_name}.png')
    save_predictions(original_data, y_test, y_pred, y_pred_proba, test_idx, class_names,
                     save_path=f'predictions_{safe_name}.csv')
    analyze_errors(original_data, y_test, y_pred, y_pred_proba, test_idx, class_names,
                   save_path=f'error_analysis_{safe_name}.csv')
    
    # Save model
    model_path = f'model_{safe_name}.pkl'
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    return model

# ================= Apply strategies =================
strategies = {
    "Original": (X_train, y_train),
    "SMOTE": SMOTE(random_state=42).fit_resample(X_train, y_train),
    "RandomOverSampler": RandomOverSampler(random_state=42).fit_resample(X_train, y_train)
}

models = {}
for name, (X_tr, y_tr) in strategies.items():
    print(f"\n==== Running strategy: {name} ====")
    models[name] = train_xgb_pipeline(X_tr, y_tr, X_test, y_test, X.columns.tolist(), strategy_name=name)
