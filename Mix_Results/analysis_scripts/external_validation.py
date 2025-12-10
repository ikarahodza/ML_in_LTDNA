"""
External Validation Dataset Analysis and Testing
Tests trained models on independent dataset (Input_Data-Single-train.csv)
to assess generalization and detect overfitting.
"""

import numpy as np
import pandas as pd
import json
import joblib
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_recall_fscore_support, confusion_matrix,
    log_loss, classification_report, mean_squared_error, mean_absolute_error
)
import warnings
warnings.filterwarnings('ignore')

# Configuration
TRAIN_DATA = 'Input_Data-Mix-train.csv'  # Original training data
EXTERNAL_DATA = 'Input_Data-Single-train.csv'  # External validation data
MODEL_DIRS = ['models', 'hyperparameter_tuning_results']
RESULTS_DIR = 'external_validation_results'
REPORTS_DIR = 'reports'

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)

print("="*80)
print("EXTERNAL VALIDATION ANALYSIS")
print("="*80)

# Load and analyze datasets
print("\n[1/6] Loading and analyzing datasets...")

# Training dataset
df_train = pd.read_csv(TRAIN_DATA)
X_train_full = df_train.drop(['sample', 'PoI', 'Classification'], axis=1)
y_train_raw = df_train['Classification']

# External validation dataset
df_external = pd.read_csv(EXTERNAL_DATA)
X_external = df_external.drop(['sample', 'PoI', 'Classification'], axis=1)
y_external_raw = df_external['Classification']

# Encode labels
le = LabelEncoder()
le.fit(y_train_raw)  # Fit on training data
y_external = le.transform(y_external_raw)
class_names = le.classes_

print(f"\n   Training Dataset:")
print(f"   - Samples: {len(df_train)}")
print(f"   - Features: {X_train_full.shape[1]}")
print(f"   - Classes: {class_names.tolist()}")
print(f"   - Class distribution: {dict(zip(*np.unique(y_train_raw, return_counts=True)))}")

print(f"\n   External Validation Dataset:")
print(f"   - Samples: {len(df_external)}")
print(f"   - Features: {X_external.shape[1]}")
print(f"   - Classes: {class_names.tolist()}")
print(f"   - Class distribution: {dict(zip(*np.unique(y_external_raw, return_counts=True)))}")

# Check feature consistency
print(f"\n   Feature consistency check:")
if list(X_train_full.columns) == list(X_external.columns):
    print(f"   ✓ Features match ({len(X_train_full.columns)} features)")
else:
    print(f"   ✗ WARNING: Features do not match!")
    missing_in_external = set(X_train_full.columns) - set(X_external.columns)
    missing_in_train = set(X_external.columns) - set(X_train_full.columns)
    if missing_in_external:
        print(f"     Missing in external: {missing_in_external}")
    if missing_in_train:
        print(f"     Missing in train: {missing_in_train}")

# Load trained models
print("\n[2/6] Loading trained models...")
models_to_test = {}

for model_dir in MODEL_DIRS:
    if not os.path.exists(model_dir):
        continue
    
    for filename in os.listdir(model_dir):
        if filename.endswith('.pkl') and 'model' in filename.lower():
            model_name = filename.replace('.pkl', '')
            model_path = os.path.join(model_dir, filename)
            try:
                model = joblib.load(model_path)
                models_to_test[model_name] = {
                    'model': model,
                    'source': model_dir
                }
                print(f"   ✓ Loaded: {model_name} (from {model_dir})")
            except Exception as e:
                print(f"   ✗ Failed to load {model_name}: {e}")

print(f"\n   Total models loaded: {len(models_to_test)}")

# Load original training results for comparison
print("\n[3/6] Loading original training results for comparison...")
original_results = {}
results_file = os.path.join(REPORTS_DIR, 'combined_results.json')
if os.path.exists(results_file):
    with open(results_file, 'r') as f:
        combined_results = json.load(f)
    
    for model_key, model_data in combined_results.items():
        if 'results' in model_data:
            for strategy, metrics in model_data['results'].items():
                key = f"{model_key}_{strategy.lower().replace(' ', '_')}"
                original_results[key] = {
                    'balanced_accuracy': metrics.get('balanced_accuracy', 0),
                    'f1_macro': metrics.get('f1_macro', 0),
                    'accuracy': metrics.get('accuracy', 0)
                }
    print(f"   ✓ Loaded original results for {len(original_results)} model-strategy combinations")
else:
    print("   ⚠ Original results not found. Comparison will be limited.")

# Evaluate models on external dataset
print("\n[4/6] Evaluating models on external validation set...")

external_results = {}

for model_name, model_info in models_to_test.items():
    print(f"\n   Testing: {model_name}")
    
    try:
        model = model_info['model']
        
        # Predict
        y_pred = model.predict(X_external)
        
        # Try to get probabilities
        try:
            y_pred_proba = model.predict_proba(X_external)
            has_proba = True
        except:
            y_pred_proba = None
            has_proba = False
        
        # Calculate metrics
        accuracy = accuracy_score(y_external, y_pred)
        bal_acc = balanced_accuracy_score(y_external, y_pred)
        f1_macro = f1_score(y_external, y_pred, average='macro')
        mse = mean_squared_error(y_external, y_pred)
        mae = mean_absolute_error(y_external, y_pred)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_external, y_pred, average=None, zero_division=0
        )
        
        # Log loss if probabilities available
        if has_proba:
            logloss = log_loss(y_external, y_pred_proba)
        else:
            logloss = None
        
        # Confusion matrix
        cm = confusion_matrix(y_external, y_pred)
        
        external_results[model_name] = {
            'accuracy': accuracy,
            'balanced_accuracy': bal_acc,
            'f1_macro': f1_macro,
            'mse': mse,
            'mae': mae,
            'log_loss': logloss,
            'per_class_precision': precision.tolist(),
            'per_class_recall': recall.tolist(),
            'per_class_f1': f1.tolist(),
            'per_class_support': support.tolist(),
            'confusion_matrix': cm.tolist(),
            'source': model_info['source']
        }
        
        print(f"     Balanced Accuracy: {bal_acc:.4f}")
        print(f"     F1-Macro: {f1_macro:.4f}")
        print(f"     Accuracy: {accuracy:.4f}")
        
    except Exception as e:
        print(f"     ✗ Error: {e}")
        external_results[model_name] = {'error': str(e)}

# Save results
print("\n[5/6] Saving results...")
results_file = os.path.join(RESULTS_DIR, 'external_validation_results.json')
with open(results_file, 'w') as f:
    json.dump(external_results, f, indent=2)
print(f"   ✓ Results saved to: {results_file}")

# Create comparison visualizations
print("\n[6/6] Creating visualizations...")

# Prepare comparison data
comparison_data = []
for model_name, ext_results in external_results.items():
    if 'error' not in ext_results:
        # Try to match with original results
        orig_key = model_name
        if orig_key not in original_results:
            # Try variations
            for key in original_results.keys():
                if key in model_name or model_name in key:
                    orig_key = key
                    break
        
        row = {
            'model': model_name,
            'external_bal_acc': ext_results['balanced_accuracy'],
            'external_f1': ext_results['f1_macro'],
            'external_accuracy': ext_results['accuracy'],
        }
        
        if orig_key in original_results:
            row['train_bal_acc'] = original_results[orig_key]['balanced_accuracy']
            row['train_f1'] = original_results[orig_key]['f1_macro']
            row['train_accuracy'] = original_results[orig_key]['accuracy']
            row['bal_acc_diff'] = ext_results['balanced_accuracy'] - original_results[orig_key]['balanced_accuracy']
            row['f1_diff'] = ext_results['f1_macro'] - original_results[orig_key]['f1_macro']
        else:
            row['train_bal_acc'] = None
            row['train_f1'] = None
            row['train_accuracy'] = None
            row['bal_acc_diff'] = None
            row['f1_diff'] = None
        
        comparison_data.append(row)

df_comparison = pd.DataFrame(comparison_data)

# Plot 1: External validation performance
fig, ax = plt.subplots(figsize=(12, 6))

models = df_comparison['model'].values
x = np.arange(len(models))
width = 0.35

bars1 = ax.bar(x - width/2, df_comparison['external_bal_acc'], width, 
              label='Balanced Accuracy', alpha=0.7, color='steelblue')
bars2 = ax.bar(x + width/2, df_comparison['external_f1'], width,
              label='F1-Macro', alpha=0.7, color='coral')

ax.set_ylabel('Score', fontsize=12)
ax.set_title('External Validation Performance', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 1.0)

plt.tight_layout()
plot_file = os.path.join(RESULTS_DIR, 'external_validation_performance.png')
plt.savefig(plot_file, dpi=150, bbox_inches='tight')
plt.close()
print(f"   ✓ Performance plot saved to: {plot_file}")

# Plot 2: Train vs External comparison (if available)
if df_comparison['train_bal_acc'].notna().any():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Filter rows with training data
    df_comp = df_comparison[df_comparison['train_bal_acc'].notna()]
    
    # Balanced Accuracy comparison
    ax1 = axes[0]
    x = np.arange(len(df_comp))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, df_comp['train_bal_acc'], width,
                   label='Training', alpha=0.7, color='lightgreen')
    bars2 = ax1.bar(x + width/2, df_comp['external_bal_acc'], width,
                   label='External', alpha=0.7, color='lightcoral')
    
    ax1.set_ylabel('Balanced Accuracy', fontsize=12)
    ax1.set_title('Training vs External Validation\nBalanced Accuracy', 
                 fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_comp['model'], rotation=45, ha='right', fontsize=8)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 1.0)
    
    # F1 comparison
    ax2 = axes[1]
    bars1 = ax2.bar(x - width/2, df_comp['train_f1'], width,
                   label='Training', alpha=0.7, color='lightgreen')
    bars2 = ax2.bar(x + width/2, df_comp['external_f1'], width,
                   label='External', alpha=0.7, color='lightcoral')
    
    ax2.set_ylabel('F1-Macro', fontsize=12)
    ax2.set_title('Training vs External Validation\nF1-Score', 
                 fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(df_comp['model'], rotation=45, ha='right', fontsize=8)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, 1.0)
    
    plt.tight_layout()
    comp_file = os.path.join(RESULTS_DIR, 'train_vs_external_comparison.png')
    plt.savefig(comp_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Comparison plot saved to: {comp_file}")

# Generate summary report
print("\nGenerating summary report...")

summary = []
summary.append("="*80)
summary.append("EXTERNAL VALIDATION SUMMARY")
summary.append("="*80)
summary.append(f"\nValidation Dataset: {EXTERNAL_DATA}")
summary.append(f"  - Samples: {len(df_external)}")
summary.append(f"  - Features: {X_external.shape[1]}")
summary.append(f"  - Classes: {len(class_names)}")
summary.append(f"\nModels Tested: {len([k for k, v in external_results.items() if 'error' not in v])}")
summary.append("")
summary.append("-"*80)
summary.append(f"{'Model':<40} {'Bal Acc':<12} {'F1-Macro':<12} {'Accuracy':<12}")
summary.append("-"*80)

# Sort by balanced accuracy
sorted_results = sorted(
    [(k, v) for k, v in external_results.items() if 'error' not in v],
    key=lambda x: x[1]['balanced_accuracy'],
    reverse=True
)

for model_name, results in sorted_results:
    bal_acc = results['balanced_accuracy']
    f1 = results['f1_macro']
    acc = results['accuracy']
    summary.append(f"{model_name:<40} {bal_acc:<12.4f} {f1:<12.4f} {acc:<12.4f}")

summary.append("-"*80)

# Add generalization analysis
if df_comparison['train_bal_acc'].notna().any():
    summary.append("\nGENERALIZATION ANALYSIS:")
    summary.append(f"{'Model':<40} {'Train→Ext':<15} {'Performance':<15}")
    summary.append("-"*80)
    
    for _, row in df_comp.iterrows():
        model = row['model']
        diff = row['bal_acc_diff']
        
        if diff is not None:
            if abs(diff) < 0.05:
                status = "Stable"
            elif diff < 0:
                status = f"Drop ({diff:.4f})"
            else:
                status = f"Improve ({diff:+.4f})"
            
            summary.append(f"{model:<40} {status:<15} {row['external_bal_acc']:.4f}")
    
    summary.append("")
    summary.append("INTERPRETATION:")
    summary.append("  - Stable: Performance similar on external data (good generalization)")
    summary.append("  - Drop: Lower performance on external data (possible overfitting)")
    summary.append("  - Improve: Better performance on external data (rare, but possible)")

summary.append("\n" + "="*80)

summary_text = "\n".join(summary)
print(summary_text)

# Save summary
summary_file = os.path.join(RESULTS_DIR, 'external_validation_summary.txt')
with open(summary_file, 'w') as f:
    f.write(summary_text)
print(f"\n✓ Summary saved to: {summary_file}")

print("\n" + "="*80)
print("✓ EXTERNAL VALIDATION COMPLETE")
print(f"   Results directory: {RESULTS_DIR}/")
print("="*80)
