"""
Test All Models on Mixed_Training_Data.xlsx
============================================
Tests all trained models (Mix and Single) on the Mixed_Training_Data.xlsx file.
Prioritizes TabPFN v2 models and provides comprehensive evaluation metrics.
"""

import pandas as pd
import numpy as np
import os
import sys
import pickle
import joblib
import json
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, log_loss,
    confusion_matrix, classification_report,
    matthews_corrcoef, cohen_kappa_score
)

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "mixed training data results"
RESULTS_DIR.mkdir(exist_ok=True)

DATA_FILE = PROJECT_ROOT / "Mixed_Training_Data.csv"

print("=" * 80)
print("TESTING ALL MODELS ON MIXED_TRAINING_DATA.CSV")
print("=" * 80)

# =============================================================================
# Step 1: Load Test Data
# =============================================================================
print("\n[1/5] Loading Mixed_Training_Data.csv...")
try:
    df = pd.read_csv(DATA_FILE)
    print(f"   âś“ Loaded {len(df)} samples")
    print(f"   âś“ Columns: {list(df.columns)}")
except Exception as e:
    print(f"   âś— Error loading file: {e}")
    sys.exit(1)

# Prepare features and target
drop_cols = ['sample', 'PoI', 'Classification']
X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
y_raw = df['Classification'] if 'Classification' in df.columns else None

if y_raw is None:
    print("   âś— No 'Classification' column found!")
    sys.exit(1)

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y_raw)

print(f"   âś“ Features shape: {X.shape}")
print(f"   âś“ Classes: {le.classes_}")
print(f"   âś“ Class distribution: {dict(zip(le.classes_, np.bincount(y)))}")

# Scale features (needed for some models)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)


def normalize_predicted_labels(y_pred_raw, label_encoder):
    """Convert model predictions to integer-encoded labels for metric compatibility."""
    normalized = []
    for value in np.asarray(y_pred_raw).tolist():
        if isinstance(value, (int, np.integer)):
            normalized.append(int(value))
            continue
        if isinstance(value, (float, np.floating)) and float(value).is_integer():
            normalized.append(int(value))
            continue
        normalized.append(int(label_encoder.transform([str(value)])[0]))
    return np.asarray(normalized, dtype=int)


def normalize_predicted_proba(y_pred_proba_raw, model_classes, label_encoder):
    """Align probability columns to encoded label order expected by sklearn metrics."""
    y_pred_proba = np.asarray(y_pred_proba_raw)
    n_classes = len(label_encoder.classes_)

    if y_pred_proba.ndim != 2 or y_pred_proba.shape[1] != n_classes:
        return y_pred_proba

    if model_classes is None:
        return y_pred_proba

    encoded_order = []
    for cls in np.asarray(model_classes).tolist():
        if isinstance(cls, (int, np.integer)):
            encoded_order.append(int(cls))
        else:
            encoded_order.append(int(label_encoder.transform([str(cls)])[0]))

    aligned = np.zeros_like(y_pred_proba)
    for source_idx, target_idx in enumerate(encoded_order):
        aligned[:, target_idx] = y_pred_proba[:, source_idx]

    return aligned

# =============================================================================
# Step 2: Discover All Models
# =============================================================================
print("\n[2/5] Discovering models...")

def find_models(base_dir, dataset_type):
    """Find all model files in a directory."""
    models_found = []
    
    # Check models_v2 first (TabPFN v2 - prioritized)
    v2_dir = base_dir / f"{dataset_type}_Results" / "models_v2"
    if v2_dir.exists():
        for model_file in v2_dir.glob("*.pkl"):
            model_name = model_file.stem
            # Parse model name
            if 'tabpfn' in model_name:
                parts = model_name.replace('model_', '').split('_')
                model_type = 'tabpfn'
                strategy = '_'.join(parts[1:]).replace('_v2', '')
                models_found.append({
                    'path': model_file,
                    'dataset': dataset_type,
                    'model_type': model_type,
                    'strategy': strategy,
                    'version': 'v2',
                    'display_name': f"TabPFN_{strategy}_v2"
                })
    
    # Check main models directory
    models_dir = base_dir / f"{dataset_type}_Results" / "models"
    if models_dir.exists():
        for model_file in models_dir.glob("*.pkl"):
            model_name = model_file.stem
            
            # Skip v2 models (already processed)
            if '_v2' in model_name or model_name in ['stacking_ensemble_scaler', 'trained_rf_model']:
                continue
            
            # Skip TabPFN v1 if v2 exists
            if 'tabpfn' in model_name and v2_dir.exists():
                continue
            
            # Parse model type and strategy
            if 'lgbm' in model_name:
                model_type = 'lgbm'
                strategy = model_name.replace('lgbm_model_lgbm_', '')
            elif 'xgboost' in model_name:
                model_type = 'xgboost'
                strategy = model_name.replace('model_xgboost_', '')
            elif 'extratrees' in model_name:
                model_type = 'extratrees'
                strategy = model_name.replace('extratrees_model_extratrees_', '')
            elif 'randomforest' in model_name:
                model_type = 'randomforest'
                strategy = model_name.replace('model_randomforest_', '')
            elif 'logistic_regression' in model_name:
                model_type = 'logistic_regression'
                strategy = model_name.replace('model_logistic_regression_', '')
            elif 'svm' in model_name:
                model_type = 'svm'
                strategy = model_name.replace('model_svm_', '')
            elif 'gnb' in model_name:
                model_type = 'gaussiannb'
                strategy = model_name.replace('model_gnb_', '')
            elif 'mlp' in model_name:
                model_type = 'mlp'
                strategy = model_name.replace('model_mlp_', '')
            elif 'stacking_ensemble' in model_name and 'scaler' not in model_name:
                model_type = 'stacking_ensemble'
                strategy = 'ensemble'
            else:
                continue
            
            models_found.append({
                'path': model_file,
                'dataset': dataset_type,
                'model_type': model_type,
                'strategy': strategy,
                'version': 'v1',
                'display_name': f"{model_type}_{strategy}"
            })
    
    return models_found

# Find all models
all_models = []
all_models.extend(find_models(PROJECT_ROOT, 'Mix'))
all_models.extend(find_models(PROJECT_ROOT, 'Single'))

print(f"   âś“ Found {len(all_models)} models total")
print(f"   âś“ Mix dataset: {sum(1 for m in all_models if m['dataset'] == 'Mix')}")
print(f"   âś“ Single dataset: {sum(1 for m in all_models if m['dataset'] == 'Single')}")
print(f"   âś“ TabPFN v2 models: {sum(1 for m in all_models if m['version'] == 'v2')}")

# =============================================================================
# Step 3: Test All Models
# =============================================================================
print("\n[3/5] Testing models...")

results = []
failed_models = []

for idx, model_info in enumerate(all_models, 1):
    model_path = model_info['path']
    display_name = f"{model_info['dataset']}_{model_info['display_name']}"
    
    print(f"\n   [{idx}/{len(all_models)}] Testing {display_name}...")
    
    try:
        # Load model - try joblib first, then pickle with different protocols
        model = None
        load_errors = []
        
        # Try joblib first (better for sklearn models)
        try:
            model = joblib.load(model_path)
        except Exception as e1:
            load_errors.append(f"joblib: {str(e1)[:50]}")
            
            # Try pickle with different protocols
            for encoding in [None, 'latin1', 'bytes']:
                try:
                    with open(model_path, 'rb') as f:
                        if encoding:
                            model = pickle.load(f, encoding=encoding)
                        else:
                            model = pickle.load(f)
                    break
                except Exception as e2:
                    load_errors.append(f"pickle({encoding}): {str(e2)[:30]}")
        
        if model is None:
            raise Exception(f"All load attempts failed: {'; '.join(load_errors[:3])}")
        
        # Determine if scaling is needed
        needs_scaling = model_info['model_type'] in ['logistic_regression', 'svm', 'mlp', 'tabpfn']
        X_test = X_scaled if needs_scaling else X.values
        
        # Make predictions with timeout protection
        try:
            y_pred = model.predict(X_test)
        except KeyboardInterrupt:
            raise  # Re-raise keyboard interrupt
        except Exception as pred_error:
            raise Exception(f"Prediction failed: {str(pred_error)[:100]}")

        y_pred = normalize_predicted_labels(y_pred, le)
        
        # Get probabilities if available
        try:
            y_pred_proba = model.predict_proba(X_test)
        except Exception:
            y_pred_proba = None

        model_classes = getattr(model, 'classes_', None)
        if y_pred_proba is not None:
            y_pred_proba = normalize_predicted_proba(y_pred_proba, model_classes, le)
        
        # Calculate metrics
        metrics = {
            'dataset': model_info['dataset'],
            'model_type': model_info['model_type'],
            'strategy': model_info['strategy'],
            'version': model_info['version'],
            'display_name': display_name,
            
            # Core metrics
            'accuracy': accuracy_score(y, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y, y_pred),
            'f1_macro': f1_score(y, y_pred, average='macro'),
            'f1_weighted': f1_score(y, y_pred, average='weighted'),
            
            # Per-class metrics
            'precision_macro': precision_score(y, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y, y_pred, average='macro', zero_division=0),
            
            # Additional metrics
            'matthews_corrcoef': matthews_corrcoef(y, y_pred),
            'cohen_kappa': cohen_kappa_score(y, y_pred),
        }
        
        # Add log loss if probabilities available
        if y_pred_proba is not None:
            try:
                metrics['log_loss'] = log_loss(y, y_pred_proba, labels=np.arange(len(le.classes_)))
            except Exception:
                metrics['log_loss'] = None
        else:
            metrics['log_loss'] = None
        
        # Per-class metrics
        per_class_precision = precision_score(y, y_pred, average=None, zero_division=0)
        per_class_recall = recall_score(y, y_pred, average=None, zero_division=0)
        per_class_f1 = f1_score(y, y_pred, average=None, zero_division=0)
        
        metrics['per_class_precision'] = per_class_precision.tolist()
        metrics['per_class_recall'] = per_class_recall.tolist()
        metrics['per_class_f1'] = per_class_f1.tolist()
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        results.append(metrics)
        
        print(f"      [OK] Accuracy: {metrics['accuracy']:.4f}")
        print(f"      [OK] Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"      [OK] F1 (macro): {metrics['f1_macro']:.4f}")
        if metrics['log_loss'] is not None:
            print(f"      [OK] Log Loss: {metrics['log_loss']:.4f}")
        
    except Exception as e:
        print(f"      [X] Failed: {e}")
        failed_models.append({
            'display_name': display_name,
            'error': str(e)
        })

print(f"\n   âś“ Successfully tested: {len(results)}/{len(all_models)} models")
if failed_models:
    print(f"   âś— Failed: {len(failed_models)} models")

# =============================================================================
# Step 4: Generate Reports
# =============================================================================
print("\n[4/5] Generating reports...")

# Save detailed JSON results
json_output = RESULTS_DIR / "detailed_results.json"
with open(json_output, 'w') as f:
    json.dump({
        'test_data': {
            'file': str(DATA_FILE),
            'samples': len(df),
            'features': X.shape[1],
            'classes': le.classes_.tolist(),
            'class_distribution': {str(k): int(v) for k, v in zip(le.classes_, np.bincount(y))}
        },
        'results': results,
        'failed_models': failed_models
    }, f, indent=2)
print(f"   âś“ Saved: {json_output}")

# Create summary CSV
summary_df = pd.DataFrame(results)
summary_cols = ['dataset', 'model_type', 'strategy', 'version', 'accuracy', 
                'balanced_accuracy', 'f1_macro', 'f1_weighted', 
                'precision_macro', 'recall_macro', 'log_loss',
                'matthews_corrcoef', 'cohen_kappa']
summary_df = summary_df[summary_cols]
summary_df = summary_df.sort_values('balanced_accuracy', ascending=False)

csv_output = RESULTS_DIR / "summary_results.csv"
summary_df.to_csv(csv_output, index=False)
print(f"   âś“ Saved: {csv_output}")

# Create detailed report with per-class metrics
detailed_rows = []
for result in results:
    base_info = {
        'dataset': result['dataset'],
        'model_type': result['model_type'],
        'strategy': result['strategy'],
        'version': result['version'],
        'accuracy': result['accuracy'],
        'balanced_accuracy': result['balanced_accuracy'],
        'f1_macro': result['f1_macro'],
        'log_loss': result['log_loss']
    }
    
    # Add per-class metrics
    for i, class_name in enumerate(le.classes_):
        row = base_info.copy()
        row['class'] = class_name
        row['precision'] = result['per_class_precision'][i]
        row['recall'] = result['per_class_recall'][i]
        row['f1'] = result['per_class_f1'][i]
        detailed_rows.append(row)

detailed_df = pd.DataFrame(detailed_rows)
detailed_output = RESULTS_DIR / "detailed_per_class_results.csv"
detailed_df.to_csv(detailed_output, index=False)
print(f"   âś“ Saved: {detailed_output}")

# =============================================================================
# Step 5: Print Top Models Summary
# =============================================================================
print("\n[5/5] Top Models Summary")
print("=" * 80)

# Overall top 10
print("\nTop 10 Models (by Balanced Accuracy):")
print("-" * 80)
top_10 = summary_df.head(10)
for idx, row in top_10.iterrows():
    print(f"\n{row['dataset']:6s} | {row['model_type']:20s} | {row['strategy']:20s} | v{row['version']}")
    print(f"   Accuracy:          {row['accuracy']:.4f}")
    print(f"   Balanced Accuracy: {row['balanced_accuracy']:.4f}")
    print(f"   F1 (macro):        {row['f1_macro']:.4f}")
    if pd.notna(row['log_loss']):
        print(f"   Log Loss:          {row['log_loss']:.4f}")

# Top TabPFN v2 models
print("\n" + "=" * 80)
print("Top TabPFN v2 Models:")
print("-" * 80)
tabpfn_v2 = summary_df[summary_df['version'] == 'v2'].head(5)
for idx, row in tabpfn_v2.iterrows():
    print(f"\n{row['dataset']:6s} | TabPFN v2 | {row['strategy']:20s}")
    print(f"   Accuracy:          {row['accuracy']:.4f}")
    print(f"   Balanced Accuracy: {row['balanced_accuracy']:.4f}")
    print(f"   F1 (macro):        {row['f1_macro']:.4f}")
    if pd.notna(row['log_loss']):
        print(f"   Log Loss:          {row['log_loss']:.4f}")

# Dataset comparison
print("\n" + "=" * 80)
print("Best Model Per Dataset:")
print("-" * 80)
for dataset in ['Mix', 'Single']:
    dataset_best = summary_df[summary_df['dataset'] == dataset].iloc[0]
    print(f"\n{dataset} Dataset Best:")
    print(f"   Model: {dataset_best['model_type']} ({dataset_best['strategy']})")
    print(f"   Accuracy:          {dataset_best['accuracy']:.4f}")
    print(f"   Balanced Accuracy: {dataset_best['balanced_accuracy']:.4f}")
    print(f"   F1 (macro):        {dataset_best['f1_macro']:.4f}")

print("\n" + "=" * 80)

# =============================================================================
# Step 6: Write RESULTS_SUMMARY.md
# =============================================================================
print("\n[6/6] Writing RESULTS_SUMMARY.md...")

class_dist = {str(k): int(v) for k, v in zip(le.classes_, np.bincount(y))}

md_lines = []
md_lines.append("# Mixed Training Data Testing Results")
md_lines.append("## Summary\n")
md_lines.append(f"Tested all trained models on Mixed_Training_Data.csv ({len(df)} samples, {X.shape[1]} features)\n")
md_lines.append("### Test Data Distribution:")
for cls, cnt in class_dist.items():
    md_lines.append(f"- {cls}: {cnt} samples")

md_lines.append("\n---\n")
md_lines.append(f"## Results: {len(results)} Models Successfully Tested\n")

# Top 10 overall
md_lines.append("## Top 10 Models by Balanced Accuracy\n")
top10 = summary_df.head(10).reset_index(drop=True)
for rank, (_, row) in enumerate(top10.iterrows(), 1):
    ll_str = f"{row['log_loss']:.4f}" if pd.notna(row.get('log_loss')) else "N/A"
    md_lines.append(f"### {rank}. {row['dataset']} {row['model_type'].replace('_', ' ').title()} + {row['strategy'].replace('_', ' ').title()} (v{row['version']})")
    md_lines.append(f"- **Accuracy**: {row['accuracy']*100:.2f}%")
    md_lines.append(f"- **Balanced Accuracy**: {row['balanced_accuracy']*100:.2f}%")
    md_lines.append(f"- **F1 Macro**: {row['f1_macro']*100:.2f}%")
    md_lines.append(f"- **Log Loss**: {ll_str}")
    md_lines.append(f"- **Matthews Correlation**: {row['matthews_corrcoef']:.4f}")
    md_lines.append(f"- **Cohen Kappa**: {row['cohen_kappa']:.4f}")
    md_lines.append("")

md_lines.append("---\n")

# TabPFN v2 section
tabpfn_v2 = summary_df[summary_df['version'] == 'v2']
if not tabpfn_v2.empty:
    md_lines.append("## TabPFN v2 Results\n")
    md_lines.append("| Dataset | Strategy | Accuracy | Balanced Acc | F1 Macro | Log Loss | MCC |")
    md_lines.append("|---------|----------|----------|--------------|----------|----------|-----|")
    for _, row in tabpfn_v2.iterrows():
        ll_str = f"{row['log_loss']:.4f}" if pd.notna(row.get('log_loss')) else "N/A"
        md_lines.append(f"| {row['dataset']} | {row['strategy']} | {row['accuracy']*100:.2f}% | {row['balanced_accuracy']*100:.2f}% | {row['f1_macro']*100:.2f}% | {ll_str} | {row['matthews_corrcoef']:.4f} |")
    md_lines.append("")
    md_lines.append("---\n")

# All results table
md_lines.append("## All Model Results\n")
md_lines.append("| Dataset | Model | Strategy | Ver | Accuracy | Balanced Acc | F1 Macro | Log Loss | MCC | Kappa |")
md_lines.append("|---------|-------|----------|-----|----------|--------------|----------|----------|-----|-------|")
for _, row in summary_df.iterrows():
    ll_str = f"{row['log_loss']:.4f}" if pd.notna(row.get('log_loss')) else "N/A"
    md_lines.append(f"| {row['dataset']} | {row['model_type']} | {row['strategy']} | v{row['version']} | {row['accuracy']*100:.2f}% | {row['balanced_accuracy']*100:.2f}% | {row['f1_macro']*100:.2f}% | {ll_str} | {row['matthews_corrcoef']:.4f} | {row['cohen_kappa']:.4f} |")
md_lines.append("")
md_lines.append("---\n")

# Best per dataset
md_lines.append("## Best Model Per Dataset\n")
for dataset in ['Mix', 'Single']:
    ds_results = summary_df[summary_df['dataset'] == dataset]
    if not ds_results.empty:
        best = ds_results.iloc[0]
        ll_str = f"{best['log_loss']:.4f}" if pd.notna(best.get('log_loss')) else "N/A"
        md_lines.append(f"### {dataset} Dataset")
        md_lines.append(f"- **Best Model**: {best['model_type']} + {best['strategy']} (v{best['version']})")
        md_lines.append(f"- **Accuracy**: {best['accuracy']*100:.2f}%")
        md_lines.append(f"- **Balanced Accuracy**: {best['balanced_accuracy']*100:.2f}%")
        md_lines.append(f"- **F1 Macro**: {best['f1_macro']*100:.2f}%")
        md_lines.append(f"- **Log Loss**: {ll_str}")
        md_lines.append("")

md_lines.append("---\n")

# Failed models
if failed_models:
    md_lines.append(f"## Failed Models ({len(failed_models)} total)\n")
    for fm in failed_models:
        md_lines.append(f"- **{fm['display_name']}**: {fm['error'][:120]}")
    md_lines.append("")
    md_lines.append("---\n")

md_lines.append("## File Index\n")
md_lines.append("- `summary_results.csv` — All model metrics (sorted by balanced accuracy)")
md_lines.append("- `detailed_results.json` — Full results including per-class metrics and confusion matrices")
md_lines.append("- `detailed_per_class_results.csv` — Per-class precision/recall/F1 breakdown")

md_output = RESULTS_DIR / "RESULTS_SUMMARY.md"
with open(md_output, 'w', encoding='utf-8') as f:
    f.write('\n'.join(md_lines))
print(f"   [OK] Saved: {md_output}")

print("\n" + "=" * 80)
print("[OK] TESTING COMPLETE")
print(f"[OK] Results saved to: {RESULTS_DIR}")
print("=" * 80)

