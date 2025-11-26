import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, balanced_accuracy_score, f1_score, log_loss, recall_score
from sklearn.ensemble import ExtraTreesClassifier
from imblearn.over_sampling import SMOTE, RandomOverSampler
import joblib

MODEL_NAME = 'extratrees'
DATA_DIR = os.path.join('..') if False else 'data'
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
def plot_confusion_matrix(cm, class_names, save_path='confusion_matrix_extratrees.png'):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Confusion matrix saved to: {save_path}")
    plt.close()

def plot_feature_importance(model, feature_names, top_n=15, save_path='feature_importance_extratrees.png'):
    importance = model.feature_importances_
    sorted_idx = importance.argsort()[::-1][:top_n]
    plt.figure(figsize=(8, 6))
    plt.barh(feature_names[sorted_idx][::-1], importance[sorted_idx][::-1])
    plt.title(f"Top {top_n} Feature Importances", fontsize=14, fontweight='bold')
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Feature importance saved to: {save_path}")
    plt.close()

def save_predictions(data, y_test, y_pred, test_indices, class_names, save_path='predictions_extratrees.csv'):
    results = data.iloc[test_indices].copy()
    results['True_Label'] = y_test
    results['Predicted_Label'] = y_pred
    results.to_csv(save_path, index=False)
    print(f"Predictions saved to: {save_path}")

def analyze_errors(data, y_test, y_pred, test_indices, save_path='error_analysis_extratrees.csv'):
    errors = data.iloc[test_indices].copy()
    errors['True_Label'] = y_test
    errors['Predicted_Label'] = y_pred
    errors['Correct'] = (y_test == y_pred)
    misclassified = errors[~errors['Correct']].copy()
    misclassified.to_csv(save_path, index=False)
    print(f"Error analysis saved to: {save_path} ({len(misclassified)} misclassifications)")
    return misclassified

def train_eval_plot(X_tr, y_tr, title, strategy_name):
    model = ExtraTreesClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_test)
    
    #  Metrics 
    acc = accuracy_score(y_test, y_pred)
    print(f"\n===== {title} =====")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    #  Confusion Matrix 
    cm_path = os.path.join(FIG_DIR, f"confusion_matrix_{strategy_name}.png")
    plot_confusion_matrix(confusion_matrix(y_test, y_pred), le.classes_, save_path=cm_path)
    
    #  Feature Importance 
    fi_path = os.path.join(FIG_DIR, f"feature_importance_{strategy_name}.png")
    plot_feature_importance(model, X.columns.values, save_path=fi_path)
    
    #  Save Predictions 
    pred_path = os.path.join(DATA_DIR, f"predictions_{strategy_name}.csv")
    save_predictions(original_data, y_test, y_pred, test_idx, le.classes_, save_path=pred_path)
    
    #  Analyze Errors 
    error_path = os.path.join(DATA_DIR, f"error_analysis_{strategy_name}.csv")
    analyze_errors(original_data, y_test, y_pred, test_idx, save_path=error_path)
    
    #  Save Model 
    model_path = os.path.join(MODEL_DIR, f"extratrees_model_{strategy_name}.pkl")
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
    safe_name = f"{MODEL_NAME}_{name.replace(' ', '_').lower()}"
    print(f"\n==== Running strategy: {name} ====")
    models[name] = train_eval_plot(X_tr, y_tr, title=name, strategy_name=safe_name)

def generate_html_report(results, class_names, output_path=None):
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if output_path is None:
        output_path = os.path.join(REPORTS_DIR, f'comparison_report_{MODEL_NAME}.html')

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

    html += "</body></html>"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
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
