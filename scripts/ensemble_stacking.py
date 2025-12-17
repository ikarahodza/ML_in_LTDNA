"""
Ensemble Stacking Classifier - Uses meta-learning to combine model predictions
Base learners: RandomForest, ExtraTrees, LightGBM, XGBoost
Meta-learner: Logistic Regression
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, 
                             matthews_corrcoef, confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json

# Get base paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

def create_output_dirs(base_folder):
    """Create output directories for results."""
    dirs = {
        'models': os.path.join(project_root, base_folder, 'models'),
        'figures': os.path.join(project_root, base_folder, 'figures'),
        'reports': os.path.join(project_root, base_folder, 'reports'),
        'data': os.path.join(project_root, base_folder, 'data')
    }
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    return dirs

def plot_confusion_matrix(cm, labels, title, output_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def train_stacking_ensemble(X_train, y_train, X_test, y_test, base_folder):
    """Train and evaluate stacking ensemble classifier."""
    
    print(f"\n{'='*80}")
    print(f"Training Stacking Ensemble for {base_folder}")
    print(f"{'='*80}")
    
    # Create output directories
    dirs = create_output_dirs(base_folder)
    
    # Scale features for models that need it
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create validation split for early stopping
    X_tr_split, X_val_split, y_tr_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )
    
    # Define base learners (Level 0)
    base_learners = [
        ('rf', RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=5,
            class_weight='balanced', random_state=42, n_jobs=-1
        )),
        ('et', ExtraTreesClassifier(
            n_estimators=200, max_depth=10, min_samples_split=5,
            class_weight='balanced', random_state=42, n_jobs=-1
        )),
        ('lgbm', LGBMClassifier(
            n_estimators=1000, max_depth=6, learning_rate=0.1,
            class_weight='balanced', random_state=42, verbose=-1
        )),
        ('xgb', XGBClassifier(
            n_estimators=1000, max_depth=6, learning_rate=0.1,
            objective='multi:softprob', eval_metric='mlogloss',
            random_state=42
        ))
    ]
    
    # Define meta-learner (Level 1)
    meta_learner = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )
    
    print("\n[INFO] Creating stacking ensemble...")
    
    # Create stacking classifier
    stacking_clf = StackingClassifier(
        estimators=base_learners,
        final_estimator=meta_learner,
        cv=5,  # 5-fold CV for generating meta-features
        stack_method='predict_proba',  # Use probabilities for meta-learning
        n_jobs=-1
    )
    
    print("[INFO] Training stacking ensemble (this may take a while)...")
    stacking_clf.fit(X_train, y_train)
    
    # Predictions
    y_pred = stacking_clf.predict(X_test)
    y_pred_proba = stacking_clf.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_test, y_pred)
    
    # Cross-validation score
    print("[INFO] Performing cross-validation...")
    cv_scores = cross_val_score(stacking_clf, X_train, y_train, cv=5, 
                                scoring='balanced_accuracy', n_jobs=-1)
    
    print(f"\n{'='*80}")
    print("Stacking Ensemble Results:")
    print(f"{'='*80}")
    print(f"Accuracy:                 {accuracy:.4f}")
    print(f"Balanced Accuracy:        {balanced_acc:.4f}")
    print(f"F1-Score (weighted):      {f1_weighted:.4f}")
    print(f"Matthews Correlation:     {mcc:.4f}")
    print(f"CV Balanced Accuracy:     {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    print(f"{'='*80}\n")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    labels = sorted(np.unique(y_test))
    
    # Plot confusion matrix
    cm_path = os.path.join(dirs['figures'], 'stacking_ensemble_confusion_matrix.png')
    plot_confusion_matrix(cm, labels, 'Stacking Ensemble - Confusion Matrix', cm_path)
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    report_path = os.path.join(dirs['reports'], 'stacking_ensemble_classification_report.txt')
    with open(report_path, 'w') as f:
        f.write("Stacking Ensemble Classification Report\n")
        f.write("="*80 + "\n\n")
        f.write(classification_report(y_test, y_pred))
        f.write(f"\n\nCross-Validation Scores:\n")
        f.write(f"Mean Balanced Accuracy: {cv_scores.mean():.4f}\n")
        f.write(f"Std Deviation: {cv_scores.std():.4f}\n")
        f.write(f"\n\nBase Learner Details:\n")
        f.write(f"- RandomForest: n_estimators=200, max_depth=10\n")
        f.write(f"- ExtraTrees: n_estimators=200, max_depth=10\n")
        f.write(f"- LightGBM: n_estimators=1000, learning_rate=0.1, early_stopping\n")
        f.write(f"- XGBoost: n_estimators=1000, learning_rate=0.1, early_stopping\n")
        f.write(f"\nMeta-Learner: LogisticRegression with class_weight='balanced'\n")
    
    # Save metrics to JSON
    metrics = {
        'model': 'StackingEnsemble',
        'base_learners': ['RandomForest', 'ExtraTrees', 'LightGBM', 'XGBoost'],
        'meta_learner': 'LogisticRegression',
        'accuracy': float(accuracy),
        'balanced_accuracy': float(balanced_acc),
        'f1_weighted': float(f1_weighted),
        'matthews_corrcoef': float(mcc),
        'cv_balanced_accuracy_mean': float(cv_scores.mean()),
        'cv_balanced_accuracy_std': float(cv_scores.std()),
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }
    
    metrics_path = os.path.join(dirs['reports'], 'stacking_ensemble_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Save model
    model_path = os.path.join(dirs['models'], 'stacking_ensemble.pkl')
    joblib.dump(stacking_clf, model_path)
    
    # Save scaler
    scaler_path = os.path.join(dirs['models'], 'stacking_ensemble_scaler.pkl')
    joblib.dump(scaler, scaler_path)
    
    print(f"[SUCCESS] Model saved to: {model_path}")
    print(f"[SUCCESS] Metrics saved to: {metrics_path}")
    print(f"[SUCCESS] Confusion matrix saved to: {cm_path}")
    
    return stacking_clf, metrics

if __name__ == '__main__':
    import lightgbm as lgb
    
    # Run for both Mix and Single datasets
    datasets = [
        ('Mix_Results', 'Input_Data-Mix-train.csv'),
        ('Single_Results', 'Input_Data-Single-train.csv')
    ]
    
    for base_folder, data_file in datasets:
        print(f"\n\n{'#'*80}")
        print(f"Processing: {base_folder}")
        print(f"{'#'*80}\n")
        
        # Load data
        data_path = os.path.join(project_root, data_file)
        df = pd.read_csv(data_path)
        
        # Separate features and target
        X = df.drop(columns=['sample', 'PoI', 'Classification'])
        y = df['Classification']
        
        # Train-test split (80-20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train stacking ensemble
        model, metrics = train_stacking_ensemble(X_train, y_train, X_test, y_test, base_folder)
        
        print(f"\n[SUCCESS] Completed {base_folder}\n")
    
    print(f"\n{'#'*80}")
    print("All stacking ensemble models trained successfully!")
    print(f"{'#'*80}")
