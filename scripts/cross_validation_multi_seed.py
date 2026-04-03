"""
Multi-Seed Cross-Validation Analysis (30-Fold)
Performs 30-fold stratified cross-validation across all 10 random seeds.
Aggregates results to assess model stability across different random initializations.
"""

import numpy as np
import pandas as pd
import json
import joblib
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, accuracy_score,
    precision_score, recall_score
)
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE, RandomOverSampler
import warnings
warnings.filterwarnings('ignore')

# Configuration
SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]
DATA_FILE = 'Input_Data-Mix-train.csv'
OUTPUT_DIR = 'Mix_Results/cross_validation_multi_seed'
CV_FOLDS = 30  # 30-fold cross-validation

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("MULTI-SEED CROSS-VALIDATION ANALYSIS (30-Fold)")
print("="*80)
print(f"Seeds: {SEEDS}")
print(f"CV Folds: {CV_FOLDS}")

# Load data
print("\n[1/5] Loading data...")
df = pd.read_csv(DATA_FILE)
X = df.drop(['sample', 'PoI', 'Classification'], axis=1)
y_raw = df['Classification']

le = LabelEncoder()
y = le.fit_transform(y_raw)
class_names = le.classes_

print(f"   Total samples: {len(X)}")
print(f"   Features: {X.shape[1]}")
print(f"   Classes: {class_names.tolist()}")

# Define models to evaluate (faster models for cross-validation)
print("\n[2/5] Defining models...")
models_config = {
    'lgbm_original': {
        'model_class': LGBMClassifier,
        'params': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 
                   'objective': 'multiclass', 'verbose': -1, 'n_jobs': -1},
        'resampling': None
    },
    'lgbm_smote': {
        'model_class': LGBMClassifier,
        'params': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 
                   'objective': 'multiclass', 'verbose': -1, 'n_jobs': -1},
        'resampling': 'smote'
    },
    'xgboost_original': {
        'model_class': XGBClassifier,
        'params': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1,
                   'objective': 'multi:softprob', 'eval_metric': 'mlogloss',
                   'use_label_encoder': False, 'n_jobs': -1},
        'resampling': None
    },
    'xgboost_smote': {
        'model_class': XGBClassifier,
        'params': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1,
                   'objective': 'multi:softprob', 'eval_metric': 'mlogloss',
                   'use_label_encoder': False, 'n_jobs': -1},
        'resampling': 'smote'
    },
    'extratrees_original': {
        'model_class': ExtraTreesClassifier,
        'params': {'n_estimators': 100, 'n_jobs': -1},
        'resampling': None
    },
    'randomforest_original': {
        'model_class': RandomForestClassifier,
        'params': {'n_estimators': 100, 'n_jobs': -1},
        'resampling': None
    },
    'logistic_original': {
        'model_class': LogisticRegression,
        'params': {'max_iter': 1000, 'class_weight': 'balanced', 'n_jobs': -1},
        'resampling': None
    }
}

print(f"   Models to evaluate: {len(models_config)}")

# Apply resampling
def apply_resampling(X, y, strategy, random_state):
    """Apply resampling strategy."""
    if strategy == 'smote':
        sampler = SMOTE(random_state=random_state)
        return sampler.fit_resample(X, y)
    elif strategy == 'randomoversampler':
        sampler = RandomOverSampler(random_state=random_state)
        return sampler.fit_resample(X, y)
    else:
        return X, y

# Perform cross-validation for one seed
def cross_validate_single_seed(model_config, X, y, seed, model_name):
    """Perform 30-fold CV for a single seed."""
    
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    
    # Store fold results
    fold_results = {
        'accuracy': [],
        'balanced_accuracy': [],
        'f1_macro': [],
        'precision_macro': [],
        'recall_macro': []
    }
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Apply resampling on training fold only
        if model_config['resampling']:
            X_train_fold, y_train_fold = apply_resampling(
                X_train_fold, y_train_fold, model_config['resampling'], seed
            )
        
        # Create and train model with seed
        params = model_config['params'].copy()
        if 'random_state' in model_config['model_class'].__init__.__code__.co_varnames:
            params['random_state'] = seed
        
        model = model_config['model_class'](**params)
        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_val_fold)
        
        # Calculate metrics
        fold_results['accuracy'].append(accuracy_score(y_val_fold, y_pred))
        fold_results['balanced_accuracy'].append(balanced_accuracy_score(y_val_fold, y_pred))
        fold_results['f1_macro'].append(f1_score(y_val_fold, y_pred, average='macro'))
        fold_results['precision_macro'].append(precision_score(y_val_fold, y_pred, average='macro', zero_division=0))
        fold_results['recall_macro'].append(recall_score(y_val_fold, y_pred, average='macro', zero_division=0))
    
    # Calculate statistics for this seed
    results = {}
    for metric, values in fold_results.items():
        results[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'values': values
        }
    
    return results

# Run cross-validation across all seeds
print("\n[3/5] Running cross-validation across all seeds...")
all_results = {}

for model_name, model_config in models_config.items():
    print(f"\n{'─'*60}")
    print(f"Model: {model_name}")
    print(f"{'─'*60}")
    
    seed_results = {}
    
    for seed in SEEDS:
        print(f"   Seed {seed}...", end=' ', flush=True)
        
        try:
            results = cross_validate_single_seed(model_config, X, y, seed, model_name)
            seed_results[f'seed_{seed}'] = results
            print(f"✓ Bal Acc: {results['balanced_accuracy']['mean']:.4f} ± {results['balanced_accuracy']['std']:.4f}")
        except Exception as e:
            print(f"✗ Error: {e}")
            seed_results[f'seed_{seed}'] = {'error': str(e)}
    
    all_results[model_name] = {
        'config': {
            'resampling': model_config['resampling'],
            'cv_folds': CV_FOLDS
        },
        'seed_results': seed_results
    }

# Aggregate results across seeds
print("\n[4/5] Aggregating results across seeds...")
aggregated_results = {}

for model_name, data in all_results.items():
    seed_results = data['seed_results']
    
    # Collect metrics across all seeds
    metrics_across_seeds = {
        'accuracy': [],
        'balanced_accuracy': [],
        'f1_macro': [],
        'precision_macro': [],
        'recall_macro': []
    }
    
    valid_seeds = []
    for seed_key, seed_data in seed_results.items():
        if 'error' not in seed_data:
            valid_seeds.append(seed_key)
            for metric in metrics_across_seeds.keys():
                metrics_across_seeds[metric].append(seed_data[metric]['mean'])
    
    # Calculate aggregated statistics
    aggregated = {}
    for metric, values in metrics_across_seeds.items():
        if values:
            aggregated[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'ci_95': 1.96 * np.std(values) / np.sqrt(len(values)),  # 95% confidence interval
                'values': values
            }
    
    aggregated_results[model_name] = {
        'aggregated_metrics': aggregated,
        'num_seeds': len(valid_seeds),
        'cv_folds_per_seed': CV_FOLDS,
        'total_evaluations': len(valid_seeds) * CV_FOLDS,
        'resampling': data['config']['resampling']
    }
    
    print(f"   {model_name}: {len(valid_seeds)} seeds, {len(valid_seeds) * CV_FOLDS} total evaluations")

# Save results
print("\n[5/5] Saving results and creating visualizations...")

# Save JSON results
results_file = os.path.join(OUTPUT_DIR, 'multi_seed_cv_results.json')
json_results = {}
for model_name, data in aggregated_results.items():
    json_results[model_name] = {
        'num_seeds': data['num_seeds'],
        'cv_folds_per_seed': data['cv_folds_per_seed'],
        'total_evaluations': data['total_evaluations'],
        'resampling': data['resampling'],
        'metrics': {}
    }
    for metric, stats in data['aggregated_metrics'].items():
        json_results[model_name]['metrics'][metric] = {
            'mean': float(stats['mean']),
            'std': float(stats['std']),
            'min': float(stats['min']),
            'max': float(stats['max']),
            'ci_95': float(stats['ci_95']),
            'values': [float(v) for v in stats['values']]
        }

with open(results_file, 'w') as f:
    json.dump(json_results, f, indent=2)
print(f"✓ Results saved to: {results_file}")

# Create comprehensive visualizations
print("\nCreating visualizations...")

# Plot 1: Comparison with confidence intervals
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

model_names = list(aggregated_results.keys())
bal_acc_means = [aggregated_results[m]['aggregated_metrics']['balanced_accuracy']['mean'] for m in model_names]
bal_acc_ci = [aggregated_results[m]['aggregated_metrics']['balanced_accuracy']['ci_95'] for m in model_names]
f1_means = [aggregated_results[m]['aggregated_metrics']['f1_macro']['mean'] for m in model_names]
f1_ci = [aggregated_results[m]['aggregated_metrics']['f1_macro']['ci_95'] for m in model_names]

# Sort by balanced accuracy
sorted_indices = np.argsort(bal_acc_means)[::-1]
model_names = [model_names[i] for i in sorted_indices]
bal_acc_means = [bal_acc_means[i] for i in sorted_indices]
bal_acc_ci = [bal_acc_ci[i] for i in sorted_indices]
f1_means = [f1_means[i] for i in sorted_indices]
f1_ci = [f1_ci[i] for i in sorted_indices]

x = np.arange(len(model_names))
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(model_names)))

# Balanced Accuracy
ax1.barh(x, bal_acc_means, xerr=bal_acc_ci, color=colors, alpha=0.8, capsize=5)
ax1.set_yticks(x)
ax1.set_yticklabels(model_names)
ax1.set_xlabel('Balanced Accuracy', fontsize=12, fontweight='bold')
ax1.set_title(f'Balanced Accuracy - Multi-Seed ({len(SEEDS)} seeds × {CV_FOLDS} folds = {len(SEEDS)*CV_FOLDS} evaluations)',
             fontsize=13, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)
ax1.invert_yaxis()
for i, (mean, ci) in enumerate(zip(bal_acc_means, bal_acc_ci)):
    ax1.text(mean + ci + 0.01, i, f'{mean:.4f}', va='center', fontsize=9)

# F1-Macro
ax2.barh(x, f1_means, xerr=f1_ci, color=colors, alpha=0.8, capsize=5)
ax2.set_yticks(x)
ax2.set_yticklabels(model_names)
ax2.set_xlabel('F1-Score (Macro)', fontsize=12, fontweight='bold')
ax2.set_title(f'F1-Score - Multi-Seed ({len(SEEDS)} seeds × {CV_FOLDS} folds = {len(SEEDS)*CV_FOLDS} evaluations)',
             fontsize=13, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)
ax2.invert_yaxis()
for i, (mean, ci) in enumerate(zip(f1_means, f1_ci)):
    ax2.text(mean + ci + 0.01, i, f'{mean:.4f}', va='center', fontsize=9)

plt.tight_layout()
plot_file = os.path.join(OUTPUT_DIR, 'multi_seed_cv_comparison.png')
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Comparison plot saved: {plot_file}")

# Plot 2: Box plot showing distribution across seeds
fig, ax = plt.subplots(figsize=(14, 8))

data_for_boxplot = []
labels_for_boxplot = []
for model_name in model_names:
    values = aggregated_results[model_name]['aggregated_metrics']['balanced_accuracy']['values']
    data_for_boxplot.append(values)
    labels_for_boxplot.append(model_name)

bp = ax.boxplot(data_for_boxplot, labels=labels_for_boxplot, patch_artist=True, vert=False)

# Color boxes
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(data_for_boxplot)))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_xlabel('Balanced Accuracy (Mean of 30-Fold CV)', fontsize=12, fontweight='bold')
ax.set_title(f'Distribution of Performance Across {len(SEEDS)} Seeds\nEach point = Mean of {CV_FOLDS}-Fold CV',
            fontsize=13, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
ax.invert_yaxis()

plt.tight_layout()
boxplot_file = os.path.join(OUTPUT_DIR, 'multi_seed_distribution.png')
plt.savefig(boxplot_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Distribution plot saved: {boxplot_file}")

# Generate summary report
print("\nGenerating summary report...")

summary = []
summary.append("="*80)
summary.append("MULTI-SEED CROSS-VALIDATION ANALYSIS SUMMARY")
summary.append("="*80)
summary.append(f"\nConfiguration:")
summary.append(f"  - Seeds: {SEEDS}")
summary.append(f"  - CV Folds per Seed: {CV_FOLDS}")
summary.append(f"  - Total Evaluations per Model: {len(SEEDS)} × {CV_FOLDS} = {len(SEEDS) * CV_FOLDS}")
summary.append(f"  - Total Samples: {len(X)}")
summary.append(f"  - Features: {X.shape[1]}")
summary.append(f"  - Classes: {len(class_names)}")
summary.append(f"\nEvaluated Models: {len(aggregated_results)}")
summary.append("")
summary.append("-"*100)
summary.append(f"{'Model':<30} {'Resampling':<15} {'Bal Acc':<20} {'F1-Macro':<20} {'CI 95%':<10}")
summary.append("-"*100)

# Sort by balanced accuracy
sorted_models = sorted(
    aggregated_results.items(),
    key=lambda x: x[1]['aggregated_metrics']['balanced_accuracy']['mean'],
    reverse=True
)

for model_name, data in sorted_models:
    resampling = data['resampling'] if data['resampling'] else 'None'
    bal_acc = data['aggregated_metrics']['balanced_accuracy']['mean']
    bal_std = data['aggregated_metrics']['balanced_accuracy']['std']
    bal_ci = data['aggregated_metrics']['balanced_accuracy']['ci_95']
    f1 = data['aggregated_metrics']['f1_macro']['mean']
    f1_std = data['aggregated_metrics']['f1_macro']['std']
    summary.append(
        f"{model_name:<30} {resampling:<15} "
        f"{bal_acc:.4f} ± {bal_std:.4f}   "
        f"{f1:.4f} ± {f1_std:.4f}   "
        f"±{bal_ci:.4f}"
    )

summary.append("-"*100)
summary.append("\nKEY OBSERVATIONS:")
summary.append(f"  - Each model evaluated {len(SEEDS) * CV_FOLDS} times ({len(SEEDS)} seeds × {CV_FOLDS} folds)")
summary.append("  - Lower std dev indicates consistent performance across seeds")
summary.append("  - CI 95% shows the margin of error at 95% confidence level")
summary.append("  - Narrower confidence intervals indicate more reliable estimates")
summary.append("\n" + "="*80)

summary_text = "\n".join(summary)
print("\n" + summary_text)

# Save summary
summary_file = os.path.join(OUTPUT_DIR, 'multi_seed_cv_summary.txt')
with open(summary_file, 'w') as f:
    f.write(summary_text)
print(f"\n✓ Summary saved to: {summary_file}")

print("\n" + "="*80)
print("✓ MULTI-SEED CROSS-VALIDATION ANALYSIS COMPLETE")
print(f"   Results directory: {OUTPUT_DIR}/")
print(f"   Total evaluations per model: {len(SEEDS) * CV_FOLDS}")
print("="*80)
