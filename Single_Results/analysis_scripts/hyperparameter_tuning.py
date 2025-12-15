"""
Hyperparameter Tuning Framework for Top Performing Models
Uses RandomizedSearchCV for efficient hyperparameter search with cross-validation.
Based on model_performance_analysis.txt findings.
"""

import numpy as np
import pandas as pd
import json
import joblib
import os
from datetime import datetime
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score, f1_score, make_scorer
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from imblearn.over_sampling import SMOTE, RandomOverSampler
import warnings
warnings.filterwarnings('ignore')

# Get project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_DIR = os.path.dirname(SCRIPT_DIR)  # Single_Results
PROJECT_ROOT = os.path.dirname(ANALYSIS_DIR)

# Configuration
DATA_FILE = os.path.join(PROJECT_ROOT, 'Input_Data-Single-train.csv')
RESULTS_DIR = os.path.join(ANALYSIS_DIR, 'hyperparameter_tuning_results')
N_ITER = 50  # Number of parameter settings sampled
CV_FOLDS = 5  # Cross-validation folds
N_JOBS = -1  # Use all available cores
RANDOM_STATE = 42

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load and prepare data
print("="*80)
print("HYPERPARAMETER TUNING FRAMEWORK")
print("="*80)
print("\n[1/6] Loading data...")

df = pd.read_csv(DATA_FILE)
X = df.drop(['sample', 'PoI', 'Classification'], axis=1)
y_raw = df['Classification']

le = LabelEncoder()
y = le.fit_transform(y_raw)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

print(f"   Training samples: {len(X_train)}")
print(f"   Test samples: {len(X_test)}")
print(f"   Features: {X_train.shape[1]}")
print(f"   Classes: {len(np.unique(y))}")

# Define parameter grids for each model
print("\n[2/6] Defining hyperparameter search spaces...")

param_grids = {
    'lgbm_randomoversampler': {
        'model': LGBMClassifier(random_state=RANDOM_STATE, verbose=-1),
        'resampling': 'randomoversampler',
        'params': {
            'n_estimators': [50, 100, 200, 300, 500],
            'max_depth': [3, 5, 7, 10, 15, -1],
            'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
            'num_leaves': [15, 31, 50, 70, 100],
            'min_child_samples': [5, 10, 20, 30, 50],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.01, 0.1, 0.5, 1.0],
            'reg_lambda': [0, 0.01, 0.1, 0.5, 1.0],
            'objective': ['multiclass']
        }
    },
    
    'xgboost_smote': {
        'model': XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='mlogloss'),
        'resampling': 'smote',
        'params': {
            'n_estimators': [50, 100, 200, 300, 500],
            'max_depth': [3, 5, 7, 10, 15],
            'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
            'min_child_weight': [1, 3, 5, 7, 10],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2, 0.3, 0.5],
            'reg_alpha': [0, 0.01, 0.1, 0.5, 1.0],
            'reg_lambda': [0.5, 1.0, 1.5, 2.0, 3.0],
            'objective': ['multi:softprob']
        }
    },
    
    'randomforest_smote': {
        'model': RandomForestClassifier(random_state=RANDOM_STATE),
        'resampling': 'smote',
        'params': {
            'n_estimators': [50, 100, 200, 300, 500],
            'max_depth': [5, 10, 15, 20, 30, None],
            'min_samples_split': [2, 5, 10, 20, 30],
            'min_samples_leaf': [1, 2, 4, 8, 10],
            'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7],
            'bootstrap': [True, False],
            'max_samples': [0.5, 0.7, 0.9, None],
            'criterion': ['gini', 'entropy']
        }
    },
    
    'extratrees_smote': {
        'model': ExtraTreesClassifier(random_state=RANDOM_STATE),
        'resampling': 'smote',
        'params': {
            'n_estimators': [50, 100, 200, 300, 500],
            'max_depth': [5, 10, 15, 20, 30, None],
            'min_samples_split': [2, 5, 10, 20, 30],
            'min_samples_leaf': [1, 2, 4, 8, 10],
            'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7],
            'bootstrap': [True, False],
            'criterion': ['gini', 'entropy']
        }
    }
}

print(f"   Models configured: {len(param_grids)}")
for model_name, config in param_grids.items():
    n_combinations = np.prod([len(v) for v in config['params'].values()])
    print(f"   - {model_name}: {n_combinations:,} possible combinations")

# Define scoring metrics
print("\n[3/6] Setting up scoring metrics...")
scoring = {
    'balanced_accuracy': make_scorer(balanced_accuracy_score),
    'f1_macro': make_scorer(f1_score, average='macro')
}
print("   Primary metric: balanced_accuracy")
print("   Secondary metric: f1_macro")

# Function to apply resampling
def apply_resampling(X_train, y_train, strategy):
    """Apply the specified resampling strategy."""
    if strategy == 'smote':
        sampler = SMOTE(random_state=RANDOM_STATE)
        return sampler.fit_resample(X_train, y_train)
    elif strategy == 'randomoversampler':
        sampler = RandomOverSampler(random_state=RANDOM_STATE)
        return sampler.fit_resample(X_train, y_train)
    else:
        return X_train, y_train

# Function to run hyperparameter tuning
def tune_model(model_name, config):
    """Run RandomizedSearchCV for a specific model configuration."""
    print(f"\n{'='*80}")
    print(f"TUNING: {model_name.upper()}")
    print(f"{'='*80}")
    
    # Apply resampling
    print(f"\n   Applying resampling: {config['resampling']}...")
    X_train_resampled, y_train_resampled = apply_resampling(
        X_train, y_train, config['resampling']
    )
    print(f"   Resampled training samples: {len(X_train_resampled)}")
    
    # Setup cross-validation
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    # Create RandomizedSearchCV
    print(f"\n   Starting RandomizedSearchCV...")
    print(f"   - Iterations: {N_ITER}")
    print(f"   - CV Folds: {CV_FOLDS}")
    print(f"   - Jobs: {N_JOBS}")
    
    random_search = RandomizedSearchCV(
        estimator=config['model'],
        param_distributions=config['params'],
        n_iter=N_ITER,
        scoring=scoring,
        refit='balanced_accuracy',
        cv=cv,
        verbose=1,
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
        return_train_score=True
    )
    
    # Fit the search
    start_time = datetime.now()
    random_search.fit(X_train_resampled, y_train_resampled)
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"\n   ✓ Tuning completed in {duration:.2f} seconds ({duration/60:.2f} minutes)")
    
    # Evaluate on test set
    print(f"\n   Evaluating best model on test set...")
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_bal_acc = balanced_accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"   - Test Balanced Accuracy: {test_bal_acc:.4f}")
    print(f"   - Test F1-Macro: {test_f1:.4f}")
    
    # Prepare results
    results = {
        'model_name': model_name,
        'resampling_strategy': config['resampling'],
        'best_params': random_search.best_params_,
        'best_cv_balanced_accuracy': random_search.best_score_,
        'cv_results': {
            'mean_test_balanced_accuracy': random_search.cv_results_['mean_test_balanced_accuracy'].tolist(),
            'mean_test_f1_macro': random_search.cv_results_['mean_test_f1_macro'].tolist(),
            'std_test_balanced_accuracy': random_search.cv_results_['std_test_balanced_accuracy'].tolist(),
            'std_test_f1_macro': random_search.cv_results_['std_test_f1_macro'].tolist(),
            'params': [str(p) for p in random_search.cv_results_['params']]
        },
        'test_performance': {
            'balanced_accuracy': test_bal_acc,
            'f1_macro': test_f1
        },
        'tuning_duration_seconds': duration,
        'n_iterations': N_ITER,
        'cv_folds': CV_FOLDS,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save results
    results_file = os.path.join(RESULTS_DIR, f'{model_name}_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n   ✓ Results saved to: {results_file}")
    
    # Save best model
    model_file = os.path.join(RESULTS_DIR, f'{model_name}_best_model.pkl')
    joblib.dump(best_model, model_file)
    print(f"   ✓ Best model saved to: {model_file}")
    
    return results

# Main execution
def main():
    print("\n[4/6] Starting hyperparameter tuning...")
    print(f"   This will take considerable time (~{len(param_grids) * 10}-{len(param_grids) * 30} minutes)")
    print("   Consider running overnight or on a powerful machine.\n")
    
    all_results = {}
    
    for idx, (model_name, config) in enumerate(param_grids.items(), 1):
        print(f"\n[Progress: {idx}/{len(param_grids)}]")
        try:
            results = tune_model(model_name, config)
            all_results[model_name] = results
        except Exception as e:
            print(f"\n   ✗ ERROR tuning {model_name}: {e}")
            all_results[model_name] = {'error': str(e)}
    
    # Save combined results
    print("\n[5/6] Saving combined results...")
    combined_file = os.path.join(RESULTS_DIR, 'all_tuning_results.json')
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"   ✓ Combined results saved to: {combined_file}")
    
    # Print summary
    print("\n[6/6] TUNING SUMMARY")
    print("="*80)
    print(f"\n{'Model':<30} {'Best CV Bal Acc':<20} {'Test Bal Acc':<15} {'Test F1-Macro':<15}")
    print("-"*80)
    
    for model_name, results in all_results.items():
        if 'error' not in results:
            cv_score = results['best_cv_balanced_accuracy']
            test_bal = results['test_performance']['balanced_accuracy']
            test_f1 = results['test_performance']['f1_macro']
            print(f"{model_name:<30} {cv_score:<20.4f} {test_bal:<15.4f} {test_f1:<15.4f}")
        else:
            print(f"{model_name:<30} {'ERROR':<20} {'-':<15} {'-':<15}")
    
    print("\n" + "="*80)
    print("✓ HYPERPARAMETER TUNING COMPLETE")
    print(f"   Results directory: {RESULTS_DIR}/")
    print("="*80)

if __name__ == '__main__':
    main()
