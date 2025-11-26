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
import os

MODEL_NAME = 'xgboost'
DATA_DIR = 'data'
FIG_DIR = 'figures'
MODEL_DIR = 'models'
REPORTS_DIR = 'reports'

df_m = pd.read_csv("Input_Data-Mix-train.csv")
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

#  Helper Functions 
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

    safe_name = f"{MODEL_NAME}_{strategy_name.replace(' ', '_').lower()}"

    plot_confusion_matrix(confusion_matrix(y_test, y_pred), class_names, save_path=os.path.join(FIG_DIR, f'confusion_matrix_{safe_name}.png'))
    plot_feature_importance(model, feature_names, save_path=os.path.join(FIG_DIR, f'feature_importance_{safe_name}.png'))
    save_predictions(original_data, y_test, y_pred, y_pred_proba, test_idx, class_names,
                     save_path=os.path.join(DATA_DIR, f'predictions_{safe_name}.csv'))
    analyze_errors(original_data, y_test, y_pred, y_pred_proba, test_idx, class_names,
                   save_path=os.path.join(DATA_DIR, f'error_analysis_{safe_name}.csv'))
    
    model_path = os.path.join(MODEL_DIR, f'model_{safe_name}.pkl')
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    return model

strategies = {
    "Original": (X_train, y_train),
    "SMOTE": SMOTE(random_state=42).fit_resample(X_train, y_train),
    "RandomOverSampler": RandomOverSampler(random_state=42).fit_resample(X_train, y_train)
}

models = {}
for name, (X_tr, y_tr) in strategies.items():
    print(f"\n==== Running strategy: {name} ====")
    models[name] = train_xgb_pipeline(X_tr, y_tr, X_test, y_test, X.columns.tolist(), strategy_name=name)

def generate_html_report(results, class_names, output_path=f'comparison_report_{MODEL_NAME}.html'):
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""
    <html>
    <head><meta charset='utf-8'><title>Comparison Report - {MODEL_NAME}</title></head>
    <body>
    <h1>Comparison Report - {MODEL_NAME}</h1>
    <div>Generated: {timestamp}</div>
    <div>Dataset: Input_Data-Mix-train.csv</div>
    <table border='1' cellpadding='6' cellspacing='0'>
    <tr><th>Strategy</th><th>Accuracy</th><th>Balanced Accuracy</th><th>F1 (macro)</th><th>Log Loss</th></tr>
    """

    for name, r in results.items():
        logloss_display = r['log_loss'] if (r.get('log_loss') is not None) else 'N/A'
        html += f"<tr><td>{name}</td><td>{r['accuracy']:.4f}</td><td>{r['balanced_accuracy']:.4f}</td><td>{r['f1_macro']:.4f}</td><td>{logloss_display}</td></tr>"

    html += "</table>\n"

    # Per-class recall table
    html += "<h2>Per-class Recall</h2>\n"
    html += "<table border='1' cellpadding='6' cellspacing='0'><tr><th>Strategy</th>"
    for cn in class_names:
        html += f"<th>{cn}</th>"
    html += "</tr>\n"
    for name, r in results.items():
        html += f"<tr><td>{name}</td>"
        for recall in r.get('per_class_recall', []):
            html += f"<td>{recall:.4f}</td>"
        html += "</tr>\n"
    html += "</table>\n"

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"\nComparison report saved to: {output_path}")

results = {}
for name, model in models.items():
    y_pred = model.predict(X_test)
    y_pred_proba = None
    try:
        y_pred_proba = model.predict_proba(X_test)
    except Exception:
        y_pred_proba = None
    results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'log_loss': (log_loss(y_test, y_pred_proba) if y_pred_proba is not None else None),
        'per_class_recall': recall_score(y_test, y_pred, average=None),
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }

class_names = [str(c) for c in np.unique(y)]
generate_html_report(results, class_names)
