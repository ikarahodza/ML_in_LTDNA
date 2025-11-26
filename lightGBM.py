import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
from imblearn.over_sampling import SMOTE, RandomOverSampler
from lightgbm import LGBMClassifier
import joblib

# ================= Load Data =================
CSV_FILE_PATH = "Input_Data-Mix-train.csv"
df_m = pd.read_csv(CSV_FILE_PATH)
original_data = df_m.copy()

X = df_m.drop(['sample', 'PoI', 'Classification'], axis=1)
y_raw = df_m['Classification']

le = LabelEncoder()
y = le.fit_transform(y_raw)

# Train/test split with indices
indices = np.arange(len(df_m))
X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
    X, y, indices,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ================= Helper Functions =================
def plot_confusion_matrix(cm, class_names, save_path='confusion_matrix_lgbm.png'):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Confusion matrix saved to: {save_path}")
    plt.close()

def plot_feature_importance(model, feature_names, top_n=15, save_path='feature_importance_lgbm.png'):
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

def save_predictions(data, y_test, y_pred, test_indices, class_names, save_path='predictions_lgbm.csv'):
    results = data.iloc[test_indices].copy()
    results['True_Label'] = y_test
    results['Predicted_Label'] = y_pred
    results.to_csv(save_path, index=False)
    print(f"Predictions saved to: {save_path}")

def analyze_errors(data, y_test, y_pred, test_indices, save_path='error_analysis_lgbm.csv'):
    errors = data.iloc[test_indices].copy()
    errors['True_Label'] = y_test
    errors['Predicted_Label'] = y_pred
    errors['Correct'] = (y_test == y_pred)
    misclassified = errors[~errors['Correct']].copy()
    misclassified.to_csv(save_path, index=False)
    print(f"Error analysis saved to: {save_path} ({len(misclassified)} misclassifications)")
    return misclassified

# ================= Train and Evaluate Function =================
def train_eval_plot(X_tr, y_tr, title, strategy_name):
    model = LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.3,
        objective='multiclass',
        random_state=42
    )
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_test)
    
    # ---- Metrics ----
    acc = accuracy_score(y_test, y_pred)
    print(f"\n===== {title} =====")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # ---- Confusion Matrix ----
    cm_path = f"confusion_matrix_{strategy_name}.png"
    plot_confusion_matrix(confusion_matrix(y_test, y_pred), le.classes_, save_path=cm_path)
    
    # ---- Feature Importance ----
    fi_path = f"feature_importance_{strategy_name}.png"
    plot_feature_importance(model, X.columns.values, save_path=fi_path)
    
    # ---- Save Predictions ----
    pred_path = f"predictions_{strategy_name}.csv"
    save_predictions(original_data, y_test, y_pred, test_idx, le.classes_, save_path=pred_path)
    
    # ---- Analyze Errors ----
    error_path = f"error_analysis_{strategy_name}.csv"
    analyze_errors(original_data, y_test, y_pred, test_idx, save_path=error_path)
    
    # ---- Save Model ----
    model_path = f"lgbm_model_{strategy_name}.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    return model

# ================= Apply Strategies =================
strategies = {
    "Original": (X_train, y_train),
    "SMOTE": SMOTE(random_state=42).fit_resample(X_train, y_train),
    "RandomOverSampler": RandomOverSampler(random_state=42).fit_resample(X_train, y_train)
}

models = {}
for name, (X_tr, y_tr) in strategies.items():
    safe_name = name.replace(" ", "_").lower()
    print(f"\n==== Running strategy: {name} ====")
    models[name] = train_eval_plot(X_tr, y_tr, title=name, strategy_name=safe_name)
