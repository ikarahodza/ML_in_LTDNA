"""
Cross-Validation Analysis for Model Stability Assessment
Evaluates both baseline and tuned models using stratified k-fold cross-validation.
Generates detailed performance reports and visualization plots.
"""

import numpy as np
import pandas as pd
import json
import joblib
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, accuracy_score,
    make_scorer, precision_score, recall_score
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

# Get project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_DIR = os.path.dirname(SCRIPT_DIR)  # Single_Results
PROJECT_ROOT = os.path.dirname(ANALYSIS_DIR)

# Configuration
DATA_FILE = os.path.join(PROJECT_ROOT, 'Input_Data-Single-train.csv')
RESULTS_DIR = os.path.join(ANALYSIS_DIR, 'cross_validation_results')
TUNING_DIR = os.path.join(ANALYSIS_DIR, 'hyperparameter_tuning_results')
CV_FOLDS = 30  # 30-fold cross-validation for robust estimates
RANDOM_STATE = 42

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)

print("="*80)
print("CROSS-VALIDATION ANALYSIS")
print("="*80)

# Load data
print("\n[1/7] Loading data...")
df = pd.read_csv(DATA_FILE)
X = df.drop(['sample', 'PoI', 'Classification'], axis=1)
y_raw = df['Classification']

le = LabelEncoder()
y = le.fit_transform(y_raw)
class_names = le.classes_

print(f"   Total samples: {len(X)}")
print(f"   Features: {X.shape[1]}")
print(f"   Classes: {class_names.tolist()}")

# Define scoring metrics
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'balanced_accuracy': make_scorer(balanced_accuracy_score),
    'f1_macro': make_scorer(f1_score, average='macro'),
    'precision_macro': make_scorer(precision_score, average='macro', zero_division=0),
    'recall_macro': make_scorer(recall_score, average='macro', zero_division=0)
}

# Load tuned models if available
print("\n[2/7] Loading tuned models...")
tuned_models = {}
if os.path.exists(TUNING_DIR):
    for filename in os.listdir(TUNING_DIR):
        if filename.endswith('_best_model.pkl'):
            model_name = filename.replace('_best_model.pkl', '')
            model_path = os.path.join(TUNING_DIR, filename)
            try:
                tuned_models[model_name] = joblib.load(model_path)
                print(f"   ✓ Loaded: {model_name}")
            except Exception as e:
                print(f"   ✗ Failed to load {model_name}: {e}")
else:
    print("   No tuned models found. Will use baseline models only.")

# Define baseline models for comparison
print("\n[3/7] Defining baseline models...")
baseline_models = {
    'lgbm_baseline': {
        'model': LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.3, 
                               objective='multiclass', random_state=RANDOM_STATE, verbose=-1),
        'resampling': 'randomoversampler'
    },
    'xgboost_baseline': {
        'model': XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.3,
                              objective='multi:softprob', eval_metric='mlogloss',
                              use_label_encoder=False, random_state=RANDOM_STATE),
        'resampling': 'smote'
    },
    'randomforest_baseline': {
        'model': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        'resampling': 'smote'
    },
    'extratrees_baseline': {
        'model': ExtraTreesClassifier(n_estimators=100, random_state=RANDOM_STATE),
        'resampling': 'smote'
    },
    'logre_baseline': {
        'model': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced'),
        'resampling': 'smote'
    },
    'svm_baseline': {
        'model': SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE, class_weight='balanced'),
        'resampling': 'smote'
    },
    'gnb_baseline': {
        'model': GaussianNB(),
        'resampling': 'smote'
    },
    'mlp_baseline': {
        'model': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=RANDOM_STATE, early_stopping=True),
        'resampling': 'smote'
    }
}

print(f"   Baseline models: {len(baseline_models)}")

# Apply resampling within CV
def apply_resampling(X, y, strategy):
    """Apply resampling strategy."""
    if strategy == 'smote':
        sampler = SMOTE(random_state=RANDOM_STATE)
        return sampler.fit_resample(X, y)
    elif strategy == 'randomoversampler':
        sampler = RandomOverSampler(random_state=RANDOM_STATE)
        return sampler.fit_resample(X, y)
    else:
        return X, y

# Perform cross-validation
def cross_validate_model(model, X, y, model_name, resampling_strategy=None):
    """Perform stratified k-fold cross-validation."""
    print(f"\n   Evaluating: {model_name}")
    print(f"   Resampling: {resampling_strategy if resampling_strategy else 'None'}")
    
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    # Store fold results
    fold_results = {metric: [] for metric in scoring.keys()}
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Apply resampling on training fold only
        if resampling_strategy:
            X_train_fold, y_train_fold = apply_resampling(
                X_train_fold, y_train_fold, resampling_strategy
            )
        
        # Train and evaluate
        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_val_fold)
        
        # Calculate metrics
        fold_results['accuracy'].append(accuracy_score(y_val_fold, y_pred))
        fold_results['balanced_accuracy'].append(balanced_accuracy_score(y_val_fold, y_pred))
        fold_results['f1_macro'].append(f1_score(y_val_fold, y_pred, average='macro'))
        fold_results['precision_macro'].append(precision_score(y_val_fold, y_pred, average='macro', zero_division=0))
        fold_results['recall_macro'].append(recall_score(y_val_fold, y_pred, average='macro', zero_division=0))
    
    # Calculate statistics
    results = {}
    for metric, values in fold_results.items():
        results[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'values': values
        }
    
    print(f"   Balanced Accuracy: {results['balanced_accuracy']['mean']:.4f} ± {results['balanced_accuracy']['std']:.4f}")
    print(f"   F1-Macro: {results['f1_macro']['mean']:.4f} ± {results['f1_macro']['std']:.4f}")
    
    return results

# Run cross-validation for all models
print("\n[4/7] Running cross-validation...")
all_results = {}

# Evaluate baseline models
print("\n   BASELINE MODELS:")
for model_name, config in baseline_models.items():
    try:
        results = cross_validate_model(
            config['model'], X, y, model_name, config['resampling']
        )
        all_results[model_name] = {
            'results': results,
            'type': 'baseline',
            'resampling': config['resampling']
        }
    except Exception as e:
        print(f"   ✗ Error with {model_name}: {e}")
        all_results[model_name] = {'error': str(e)}

# Evaluate tuned models
if tuned_models:
    print("\n   TUNED MODELS:")
    for model_name, model in tuned_models.items():
        try:
            # Get resampling strategy from model name
            resampling = 'randomoversampler' if 'randomoversampler' in model_name else 'smote'
            results = cross_validate_model(model, X, y, model_name, resampling)
            all_results[model_name] = {
                'results': results,
                'type': 'tuned',
                'resampling': resampling
            }
        except Exception as e:
            print(f"   ✗ Error with {model_name}: {e}")
            all_results[model_name] = {'error': str(e)}

# Save results
print("\n[5/7] Saving results...")
results_file = os.path.join(RESULTS_DIR, 'cv_results.json')
# Convert numpy types for JSON serialization
json_results = {}
for model_name, data in all_results.items():
    if 'error' not in data:
        json_results[model_name] = {
            'type': data['type'],
            'resampling': data['resampling'],
            'metrics': {}
        }
        for metric, stats in data['results'].items():
            json_results[model_name]['metrics'][metric] = {
                'mean': float(stats['mean']),
                'std': float(stats['std']),
                'min': float(stats['min']),
                'max': float(stats['max']),
                'values': [float(v) for v in stats['values']]
            }
    else:
        json_results[model_name] = data

with open(results_file, 'w') as f:
    json.dump(json_results, f, indent=2)
print(f"   ✓ Results saved to: {results_file}")

# Create visualizations
print("\n[6/7] Creating visualizations...")

# Plot 1: Box plots of balanced accuracy
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Prepare data for plotting
models = []
values = []
types = []
for model_name, data in all_results.items():
    if 'error' not in data:
        models.append(model_name)
        values.append(data['results']['balanced_accuracy']['values'])
        types.append(data['type'])

# Balanced Accuracy boxplot
ax1 = axes[0]
positions = np.arange(len(models))
bp1 = ax1.boxplot(values, positions=positions, labels=models, patch_artist=True)

# Color by type
colors = ['lightblue' if t == 'baseline' else 'lightcoral' for t in types]
for patch, color in zip(bp1['boxes'], colors):
    patch.set_facecolor(color)

ax1.set_ylabel('Balanced Accuracy', fontsize=12)
ax1.set_title(f'Cross-Validation Results ({CV_FOLDS}-Fold) - Balanced Accuracy', 
             fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
ax1.tick_params(axis='x', rotation=45)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='lightblue', label='Baseline'),
    Patch(facecolor='lightcoral', label='Tuned')
]
ax1.legend(handles=legend_elements, loc='lower right')

# F1-Macro boxplot
ax2 = axes[1]
f1_values = []
for model_name, data in all_results.items():
    if 'error' not in data:
        f1_values.append(data['results']['f1_macro']['values'])

bp2 = ax2.boxplot(f1_values, positions=positions, labels=models, patch_artist=True)
for patch, color in zip(bp2['boxes'], colors):
    patch.set_facecolor(color)

ax2.set_ylabel('F1-Score (Macro)', fontsize=12)
ax2.set_title(f'Cross-Validation Results ({CV_FOLDS}-Fold) - F1-Score', 
             fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
ax2.tick_params(axis='x', rotation=45)
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plot_file = os.path.join(RESULTS_DIR, 'cv_boxplots.png')
plt.savefig(plot_file, dpi=150, bbox_inches='tight')
plt.close()
print(f"   ✓ Boxplot saved to: {plot_file}")

# Plot 2: Comparison bar chart with error bars
fig, ax = plt.subplots(figsize=(12, 6))

model_names = []
bal_acc_means = []
bal_acc_stds = []
f1_means = []
f1_stds = []

for model_name, data in all_results.items():
    if 'error' not in data:
        model_names.append(model_name)
        bal_acc_means.append(data['results']['balanced_accuracy']['mean'])
        bal_acc_stds.append(data['results']['balanced_accuracy']['std'])
        f1_means.append(data['results']['f1_macro']['mean'])
        f1_stds.append(data['results']['f1_macro']['std'])

x = np.arange(len(model_names))
width = 0.35

bars1 = ax.bar(x - width/2, bal_acc_means, width, yerr=bal_acc_stds, 
               label='Balanced Accuracy', capsize=5, alpha=0.7, color='steelblue')
bars2 = ax.bar(x + width/2, f1_means, width, yerr=f1_stds,
               label='F1-Macro', capsize=5, alpha=0.7, color='coral')

ax.set_ylabel('Score', fontsize=12)
ax.set_title(f'Mean Performance with Std Dev ({CV_FOLDS}-Fold CV)', 
            fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 1.0)

plt.tight_layout()
barplot_file = os.path.join(RESULTS_DIR, 'cv_comparison.png')
plt.savefig(barplot_file, dpi=150, bbox_inches='tight')
plt.close()
print(f"   ✓ Bar chart saved to: {barplot_file}")

# Generate summary report
print("\n[7/7] Generating summary report...")

summary = []
summary.append("="*80)
summary.append("CROSS-VALIDATION ANALYSIS SUMMARY")
summary.append("="*80)
summary.append(f"\nConfiguration:")
summary.append(f"  - CV Folds: {CV_FOLDS}")
summary.append(f"  - Total Samples: {len(X)}")
summary.append(f"  - Features: {X.shape[1]}")
summary.append(f"  - Classes: {len(class_names)}")
summary.append(f"\nEvaluated Models: {len([k for k, v in all_results.items() if 'error' not in v])}")
summary.append("")
summary.append("-"*80)
summary.append(f"{'Model':<30} {'Type':<10} {'Bal Acc':<15} {'F1-Macro':<15} {'Std Dev':<10}")
summary.append("-"*80)

# Sort by balanced accuracy
sorted_results = sorted(
    [(k, v) for k, v in all_results.items() if 'error' not in v],
    key=lambda x: x[1]['results']['balanced_accuracy']['mean'],
    reverse=True
)

for model_name, data in sorted_results:
    model_type = data['type']
    bal_acc = data['results']['balanced_accuracy']['mean']
    bal_std = data['results']['balanced_accuracy']['std']
    f1 = data['results']['f1_macro']['mean']
    summary.append(f"{model_name:<30} {model_type:<10} {bal_acc:.4f} ± {bal_std:.4f}  {f1:.4f}")

summary.append("-"*80)
summary.append("\nKEY OBSERVATIONS:")
summary.append("  - Lower std dev indicates more stable performance across folds")
summary.append("  - Compare baseline vs tuned to assess hyperparameter tuning impact")
summary.append("  - Models with high variance may overfit or be sensitive to data splits")
summary.append("\n" + "="*80)

summary_text = "\n".join(summary)
print(summary_text)

# Save summary
summary_file = os.path.join(RESULTS_DIR, 'cv_summary.txt')
with open(summary_file, 'w') as f:
    f.write(summary_text)
print(f"\n✓ Summary saved to: {summary_file}")

print("\n" + "="*80)
print("✓ CROSS-VALIDATION ANALYSIS COMPLETE")
print(f"   Results directory: {RESULTS_DIR}/")
print("="*80)
