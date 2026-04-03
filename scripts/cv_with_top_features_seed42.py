"""
Cross-Validation with Top Features from Seed 42
Uses feature importance from seed 42 to select top features and runs 30-fold CV.
Compares performance with full feature set vs top features only.
Usage: python cv_with_top_features_seed42.py [--top-n 20]
"""

import numpy as np
import pandas as pd
import json
import os
import argparse
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

# Disable TabPFN telemetry
os.environ['POSTHOG_DISABLED'] = '1'
os.environ['TABPFN_TELEMETRY'] = '0'
try:
    from tabpfn import TabPFNClassifier
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False
    print("⚠ Warning: TabPFN not available")

# Configuration
DATA_FILE = 'Input_Data-Mix-train.csv'
FEATURE_IMPORTANCE_DIR = 'Mix_Results/figures'
CV_FOLDS = 30
RANDOM_STATE = 42

def load_feature_importance(algorithm='lgbm', strategy='original'):
    """Load feature importance from seed 42 results."""
    # Find the CSV file
    pattern = f"feature_importance_{algorithm}_{strategy}_importances_"
    csv_file = None
    
    for file in os.listdir(FEATURE_IMPORTANCE_DIR):
        if file.startswith(pattern) and file.endswith('.csv'):
            csv_file = file
            break
    
    if not csv_file:
        raise FileNotFoundError(f"Feature importance file not found for {algorithm} {strategy}")
    
    csv_path = os.path.join(FEATURE_IMPORTANCE_DIR, csv_file)
    df = pd.read_csv(csv_path)
    
    print(f"✓ Loaded feature importance from: {csv_file}")
    print(f"  Total features: {len(df)}")
    
    return df

def select_top_features(feature_importance_df, top_n=20):
    """Select top N features by importance."""
    top_features = feature_importance_df.head(top_n)
    feature_names = top_features['feature'].tolist()
    
    print(f"\n✓ Selected top {top_n} features:")
    for i, row in top_features.iterrows():
        print(f"  {i+1:2d}. {row['feature']:40s} (importance: {row['importance']:.4f})")
    
    return feature_names

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

def cross_validate_model(model, X, y, model_name, resampling_strategy=None):
    """Perform stratified k-fold cross-validation."""
    print(f"\n  Evaluating: {model_name}")
    
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    # Store fold results
    fold_results = {
        'accuracy': [],
        'balanced_accuracy': [],
        'f1_macro': [],
        'precision_macro': [],
        'recall_macro': []
    }
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
        if fold_idx % 10 == 0:
            print(f"    Fold {fold_idx}/{CV_FOLDS}...")
        
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
        fold_results['f1_macro'].append(f1_score(y_val_fold, y_pred, average='macro', zero_division=0))
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
    
    print(f"    ✓ Balanced Accuracy: {results['balanced_accuracy']['mean']:.4f} ± {results['balanced_accuracy']['std']:.4f}")
    print(f"    ✓ F1-Macro: {results['f1_macro']['mean']:.4f} ± {results['f1_macro']['std']:.4f}")
    
    return results

def create_comparison_visualization(comparison_results, output_dir):
    """Create visualizations comparing full vs reduced feature sets."""
    
    metrics = ['balanced_accuracy', 'f1_macro', 'accuracy', 'precision_macro', 'recall_macro']
    metric_names = ['Balanced Accuracy', 'F1-Macro', 'Accuracy', 'Precision', 'Recall']
    
    # Extract data for plotting
    models = list(comparison_results.keys())
    
    # Adjust figure size for more models
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    axes = axes.flatten()
    
    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx]
        
        full_means = []
        full_stds = []
        reduced_means = []
        reduced_stds = []
        
        for model in models:
            if 'error' not in comparison_results[model]:
                full_means.append(comparison_results[model]['full_features']['metrics'][metric]['mean'])
                full_stds.append(comparison_results[model]['full_features']['metrics'][metric]['std'])
                reduced_means.append(comparison_results[model]['top_features']['metrics'][metric]['mean'])
                reduced_stds.append(comparison_results[model]['top_features']['metrics'][metric]['std'])
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, full_means, width, yerr=full_stds, 
                       label='Full Features', alpha=0.8, capsize=5)
        bars2 = ax.bar(x + width/2, reduced_means, width, yerr=reduced_stds,
                       label='Top Features', alpha=0.8, capsize=5)
        
        ax.set_ylabel(metric_name, fontweight='bold')
        ax.set_title(f'{metric_name} Comparison', fontweight='bold', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    # Remove extra subplot
    fig.delaxes(axes[5])
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'feature_reduction_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Comparison visualization saved: {output_path}")
    return output_path

def create_detailed_metrics_table(comparison_results, output_dir):
    """Create detailed comparison table."""
    
    rows = []
    for model_name, data in comparison_results.items():
        if 'error' not in data:
            for feature_type in ['full_features', 'top_features']:
                metrics = data[feature_type]['metrics']
                row = {
                    'Model': model_name,
                    'Feature Set': 'Full' if feature_type == 'full_features' else f"Top {data['top_n']}",
                    'Num Features': data[feature_type]['num_features'],
                    'Balanced Acc (Mean)': f"{metrics['balanced_accuracy']['mean']:.4f}",
                    'Balanced Acc (Std)': f"{metrics['balanced_accuracy']['std']:.4f}",
                    'F1-Macro (Mean)': f"{metrics['f1_macro']['mean']:.4f}",
                    'F1-Macro (Std)': f"{metrics['f1_macro']['std']:.4f}",
                    'Accuracy (Mean)': f"{metrics['accuracy']['mean']:.4f}",
                    'Accuracy (Std)': f"{metrics['accuracy']['std']:.4f}",
                    'Precision (Mean)': f"{metrics['precision_macro']['mean']:.4f}",
                    'Recall (Mean)': f"{metrics['recall_macro']['mean']:.4f}"
                }
                rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, 'feature_reduction_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"✓ Comparison table saved: {csv_path}")
    
    return csv_path

def generate_summary_report(comparison_results, top_n, top_features, output_dir):
    """Generate text summary report."""
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("CROSS-VALIDATION WITH TOP FEATURES FROM SEED 42")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append(f"Configuration:")
    report_lines.append(f"  - Random Seed: {RANDOM_STATE}")
    report_lines.append(f"  - CV Folds: {CV_FOLDS}")
    report_lines.append(f"  - Top Features Selected: {top_n}")
    report_lines.append("")
    report_lines.append("-" * 80)
    report_lines.append(f"TOP {top_n} FEATURES (from LGBM Original - Seed 42)")
    report_lines.append("-" * 80)
    for i, feat in enumerate(top_features, 1):
        report_lines.append(f"  {i:2d}. {feat}")
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("RESULTS SUMMARY")
    report_lines.append("=" * 80)
    
    for model_name, data in comparison_results.items():
        if 'error' in data:
            report_lines.append(f"\n{model_name}: ERROR - {data['error']}")
            continue
        
        report_lines.append(f"\n{model_name}")
        report_lines.append("-" * 40)
        
        full_ba = data['full_features']['metrics']['balanced_accuracy']
        top_ba = data['top_features']['metrics']['balanced_accuracy']
        full_f1 = data['full_features']['metrics']['f1_macro']
        top_f1 = data['top_features']['metrics']['f1_macro']
        
        report_lines.append(f"  Full Features ({data['full_features']['num_features']} features):")
        report_lines.append(f"    Balanced Accuracy: {full_ba['mean']:.4f} ± {full_ba['std']:.4f}")
        report_lines.append(f"    F1-Macro:         {full_f1['mean']:.4f} ± {full_f1['std']:.4f}")
        
        report_lines.append(f"  Top {top_n} Features:")
        report_lines.append(f"    Balanced Accuracy: {top_ba['mean']:.4f} ± {top_ba['std']:.4f}")
        report_lines.append(f"    F1-Macro:         {top_f1['mean']:.4f} ± {top_f1['std']:.4f}")
        
        # Calculate performance difference
        ba_diff = top_ba['mean'] - full_ba['mean']
        f1_diff = top_f1['mean'] - full_f1['mean']
        ba_pct = (ba_diff / full_ba['mean']) * 100
        f1_pct = (f1_diff / full_f1['mean']) * 100
        
        report_lines.append(f"  Performance Change:")
        report_lines.append(f"    Balanced Accuracy: {ba_diff:+.4f} ({ba_pct:+.2f}%)")
        report_lines.append(f"    F1-Macro:         {f1_diff:+.4f} ({f1_pct:+.2f}%)")
    
    report_lines.append("\n" + "=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)
    
    # Save report
    report_path = os.path.join(output_dir, 'cv_top_features_summary.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✓ Summary report saved: {report_path}")
    return report_path

def main(top_n=20):
    """Main analysis workflow."""
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION WITH TOP FEATURES FROM SEED 42")
    print("=" * 80)
    
    # Create output directory
    output_dir = f'Mix_Results/cv_top_{top_n}_features_seed42'
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Load data
    print("\n[1/6] Loading data...")
    df = pd.read_csv(DATA_FILE)
    X = df.drop(['sample', 'PoI', 'Classification'], axis=1)
    y_raw = df['Classification']
    
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = le.classes_
    
    print(f"  Total samples: {len(X)}")
    print(f"  Full feature set: {X.shape[1]} features")
    print(f"  Classes: {class_names.tolist()}")
    
    # Load feature importance from seed 42
    print("\n[2/6] Loading feature importance from seed 42...")
    feature_importance_df = load_feature_importance(algorithm='lgbm', strategy='original')
    
    # Select top features
    print("\n[3/6] Selecting top features...")
    top_features = select_top_features(feature_importance_df, top_n)
    
    # Check if all features exist in data
    missing_features = [f for f in top_features if f not in X.columns]
    if missing_features:
        print(f"\n⚠ Warning: {len(missing_features)} features not found in data:")
        for f in missing_features:
            print(f"  - {f}")
        top_features = [f for f in top_features if f in X.columns]
        print(f"\nUsing {len(top_features)} available features")
    
    X_reduced = X[top_features]
    
    # Define models to test
    print("\n[4/6] Defining models...")
    models = {
        'LGBM': {
            'model': LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.3,
                                    objective='multiclass', random_state=RANDOM_STATE, verbose=-1),
            'resampling': 'randomoversampler'
        },
        'XGBoost': {
            'model': XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.3,
                                   objective='multi:softprob', eval_metric='mlogloss',
                                   random_state=RANDOM_STATE),
            'resampling': 'smote'
        },
        'RandomForest': {
            'model': RandomForestClassifier(n_estimators=100, max_depth=None, random_state=RANDOM_STATE),
            'resampling': 'smote'
        },
        'ExtraTrees': {
            'model': ExtraTreesClassifier(n_estimators=100, random_state=RANDOM_STATE),
            'resampling': 'smote'
        },
        'LogisticReg': {
            'model': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced'),
            'resampling': 'smote'
        },
        'SVM': {
            'model': SVC(kernel='rbf', random_state=RANDOM_STATE, probability=True),
            'resampling': 'smote'
        },
        'GaussianNB': {
            'model': GaussianNB(),
            'resampling': 'smote'
        },
        'MLP': {
            'model': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=RANDOM_STATE),
            'resampling': 'smote'
        }
    }
    
    # Add TabPFN if available
    if TABPFN_AVAILABLE:
        models['TabPFN'] = {
            'model': TabPFNClassifier(),
            'resampling': None
        }
    
    print(f"  Models to evaluate: {len(models)}")
    
    # Run cross-validation
    print("\n[5/6] Running 30-fold cross-validation...")
    comparison_results = {}
    
    for model_name, config in models.items():
        print(f"\n{'─' * 60}")
        print(f"Model: {model_name}")
        print(f"{'─' * 60}")
        
        try:
            # Test with full features
            print("\n► Testing with FULL feature set...")
            full_results = cross_validate_model(
                config['model'], X, y, 
                f"{model_name}_full", 
                config['resampling']
            )
            
            # Test with reduced features (need to create a new model instance)
            print(f"\n► Testing with TOP {top_n} features...")
            if model_name == 'LGBM':
                reduced_model = LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.3,
                                               objective='multiclass', random_state=RANDOM_STATE, verbose=-1)
            elif model_name == 'XGBoost':
                reduced_model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.3,
                                              objective='multi:softprob', eval_metric='mlogloss',
                                              random_state=RANDOM_STATE)
            elif model_name == 'RandomForest':
                reduced_model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=RANDOM_STATE)
            elif model_name == 'ExtraTrees':
                reduced_model = ExtraTreesClassifier(n_estimators=100, random_state=RANDOM_STATE)
            elif model_name == 'LogisticReg':
                reduced_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced')
            elif model_name == 'SVM':
                reduced_model = SVC(kernel='rbf', random_state=RANDOM_STATE, probability=True)
            elif model_name == 'GaussianNB':
                reduced_model = GaussianNB()
            elif model_name == 'MLP':
                reduced_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=RANDOM_STATE)
            elif model_name == 'TabPFN':
                reduced_model = TabPFNClassifier()
            else:
                raise ValueError(f"Unknown model: {model_name}")
            
            reduced_results = cross_validate_model(
                reduced_model, X_reduced, y,
                f"{model_name}_top{top_n}",
                config['resampling']
            )
            
            # Store comparison
            comparison_results[model_name] = {
                'full_features': {
                    'num_features': X.shape[1],
                    'metrics': full_results
                },
                'top_features': {
                    'num_features': len(top_features),
                    'metrics': reduced_results
                },
                'top_n': top_n,
                'resampling': config['resampling']
            }
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            comparison_results[model_name] = {'error': str(e)}
    
    # Save results
    print("\n[6/6] Generating outputs...")
    
    # Save JSON results
    json_path = os.path.join(output_dir, 'cv_comparison_results.json')
    # Convert numpy types for JSON serialization
    json_results = {}
    for model_name, data in comparison_results.items():
        if 'error' not in data:
            json_results[model_name] = {
                'top_n': data['top_n'],
                'resampling': data['resampling'],
                'full_features': {
                    'num_features': data['full_features']['num_features'],
                    'metrics': {}
                },
                'top_features': {
                    'num_features': data['top_features']['num_features'],
                    'metrics': {}
                }
            }
            for feature_type in ['full_features', 'top_features']:
                for metric, stats in data[feature_type]['metrics'].items():
                    json_results[model_name][feature_type]['metrics'][metric] = {
                        'mean': float(stats['mean']),
                        'std': float(stats['std']),
                        'min': float(stats['min']),
                        'max': float(stats['max']),
                        'values': [float(v) for v in stats['values']]
                    }
        else:
            json_results[model_name] = data
    
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"✓ JSON results saved: {json_path}")
    
    # Save top features list
    features_path = os.path.join(output_dir, f'top_{top_n}_features.txt')
    with open(features_path, 'w') as f:
        f.write(f"Top {top_n} Features from Seed 42 (LGBM Original)\n")
        f.write("=" * 60 + "\n\n")
        for i, feat in enumerate(top_features, 1):
            f.write(f"{i:2d}. {feat}\n")
    print(f"✓ Feature list saved: {features_path}")
    
    # Create visualizations
    create_comparison_visualization(comparison_results, output_dir)
    
    # Create comparison table
    create_detailed_metrics_table(comparison_results, output_dir)
    
    # Generate summary report
    generate_summary_report(comparison_results, top_n, top_features, output_dir)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nAll results saved to: {output_dir}")
    print("\nKey findings:")
    
    for model_name, data in comparison_results.items():
        if 'error' not in data:
            full_ba = data['full_features']['metrics']['balanced_accuracy']['mean']
            top_ba = data['top_features']['metrics']['balanced_accuracy']['mean']
            diff = top_ba - full_ba
            
            print(f"  {model_name}: {diff:+.4f} change with top {top_n} features")
    
    return output_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run 30-fold CV with top features from seed 42')
    parser.add_argument('--top-n', type=int, default=20,
                        help='Number of top features to select (default: 20)')
    
    args = parser.parse_args()
    
    main(top_n=args.top_n)
