"""
Comprehensive SHAP Analysis Script for ML Models
Generates detailed SHAP visualizations and insights using Python 3.12 compatible libraries.

Features:
- TreeExplainer for tree-based models (ExtraTrees, RandomForest, XGBoost, LightGBM)
- KernelExplainer for other models (SVM, MLP, LogReg, GNB)
- Summary plots (beeswarm and bar charts)
- Dependence plots for top features
- Force plots for individual predictions
- Waterfall plots
- Interactive HTML visualizations
- Feature interaction analysis
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATASET_TYPE = sys.argv[1] if len(sys.argv) > 1 else 'mix'
DATASET_TYPE = DATASET_TYPE.lower()

if DATASET_TYPE == 'mix':
    DATA_FILE = 'Input_Data-Mix-train.csv'
    MODELS_DIR = 'Mix_Results/models'
    OUTPUT_DIR = 'Mix_Results/shap_explainability'
    FIGURES_DIR = 'Mix_Results/figures'
elif DATASET_TYPE == 'single':
    DATA_FILE = 'Input_Data-Single-train.csv'
    MODELS_DIR = 'Single_Results/models'
    OUTPUT_DIR = 'Single_Results/shap_explainability'
    FIGURES_DIR = 'Single_Results/figures'
else:
    print(f"Error: Invalid dataset type '{DATASET_TYPE}'. Use 'mix' or 'single'")
    sys.exit(1)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300


def load_data():
    """Load and prepare the dataset."""
    print(f"\n[1/6] Loading {DATASET_TYPE.upper()} dataset...")
    df = pd.read_csv(DATA_FILE)
    
    # Drop non-feature columns
    drop_cols = ['sample', 'PoI', 'Classification']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df['Classification']
    
    # 80-20 split (consistent with training)
    split_idx = int(0.8 * len(X))
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Features: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test


def find_best_models():
    """Find the best performing models based on multi-seed aggregated results."""
    print("\n[2/6] Finding best models...")
    
    multi_seed_dir = Path(MODELS_DIR).parent / 'multi_seed_aggregated'
    
    if multi_seed_dir.exists():
        summary_file = multi_seed_dir / 'multi_seed_summary.json'
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            
            # Sort by balanced accuracy
            sorted_models = sorted(
                summary.items(),
                key=lambda x: x[1].get('balanced_accuracy', {}).get('mean', 0),
                reverse=True
            )
            
            print(f"   Found {len(sorted_models)} models in multi-seed summary")
            print("\n   Top 5 models:")
            for i, (name, metrics) in enumerate(sorted_models[:5], 1):
                ba = metrics.get('balanced_accuracy', {}).get('mean', 0)
                print(f"   {i}. {name}: {ba:.4f}")
            
            return [name for name, _ in sorted_models[:10]]  # Top 10
    
    # Fallback: scan models directory
    print("   Using directory scan fallback...")
    model_files = list(Path(MODELS_DIR).glob('*.pkl'))
    priority_models = ['extratrees', 'tabpfn', 'xgboost', 'lightgbm', 'randomforest']
    
    selected = []
    for priority in priority_models:
        for mf in model_files:
            if priority in mf.stem.lower() and mf.stem not in selected:
                selected.append(mf.stem)
                break
    
    # Add remaining models
    for mf in model_files:
        if mf.stem not in selected:
            selected.append(mf.stem)
    
    print(f"   Found {len(selected)} models")
    return selected[:10]  # Limit to 10


def load_model(model_name):
    """Load a trained model."""
    model_path = Path(MODELS_DIR) / f"{model_name}.pkl"
    if not model_path.exists():
        return None
    
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        print(f"   ✗ Failed to load {model_name}: {e}")
        return None


def get_explainer(model, X_train, model_name):
    """Create appropriate SHAP explainer for the model."""
    # Tree-based models: use TreeExplainer (fast and exact)
    tree_models = ['extratrees', 'randomforest', 'xgboost', 'lightgbm', 'lgbm']
    if any(tm in model_name.lower() for tm in tree_models):
        try:
            explainer = shap.TreeExplainer(model)
            explainer_type = 'TreeExplainer'
            return explainer, explainer_type
        except Exception as e:
            print(f"   Warning: TreeExplainer failed ({e}), skipping model...")
            return None, None
    
    # For linear models, use LinearExplainer
    if 'logistic' in model_name.lower() or 'logre' in model_name.lower():
        try:
            explainer = shap.LinearExplainer(model, X_train)
            explainer_type = 'LinearExplainer'
            return explainer, explainer_type
        except Exception as e:
            print(f"   Warning: LinearExplainer failed ({e}), skipping model...")
            return None, None
    
    # Skip models that don't support fast explainers (TabPFN, MLP, SVM, GNB)
    slow_models = ['tabpfn', 'mlp', 'svm', 'gnb']
    if any(sm in model_name.lower() for sm in slow_models):
        print(f"   Skipping {model_name}: KernelExplainer too slow for this model type")
        return None, None
    
    # Fallback: KernelExplainer (model-agnostic but slower) - only for smaller models
    print(f"   Using KernelExplainer (may be slow)...")
    background_size = min(50, len(X_train))
    background = shap.sample(X_train, background_size)
    explainer = shap.KernelExplainer(model.predict_proba, background)
    explainer_type = 'KernelExplainer'
    
    return explainer, explainer_type


def generate_shap_values(explainer, explainer_type, X_test, max_samples=100):
    """Generate SHAP values with appropriate method."""
    # Limit test samples for computational efficiency
    X_test_sample = X_test.iloc[:max_samples] if len(X_test) > max_samples else X_test
    
    if explainer_type == 'TreeExplainer':
        shap_values = explainer.shap_values(X_test_sample)
    elif explainer_type == 'LinearExplainer':
        shap_values = explainer.shap_values(X_test_sample)
    else:  # KernelExplainer
        # KernelExplainer is slow, use fewer samples
        X_kernel_sample = X_test_sample.iloc[:min(50, len(X_test_sample))]
        shap_values = explainer.shap_values(X_kernel_sample)
        X_test_sample = X_kernel_sample
    
    return shap_values, X_test_sample


def handle_multiclass_shap(shap_values):
    """Handle multiclass SHAP values (convert to single array for summary)."""
    if isinstance(shap_values, list):
        # List of arrays (one per class) - use first class or average
        return np.array(shap_values[0]) if len(shap_values) > 0 else shap_values
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # 3D array (samples x features x classes) - average across classes
        return np.mean(np.abs(shap_values), axis=2)
    return shap_values


def create_summary_plots(shap_values, X_test_sample, model_name, output_dir):
    """Create SHAP summary plots (beeswarm and bar)."""
    # Handle multiclass
    shap_values_plot = handle_multiclass_shap(shap_values)
    
    # 1. Beeswarm plot (summary_plot with dots)
    try:
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values_plot,
            X_test_sample,
            show=False,
            plot_type="dot",
            max_display=20
        )
        plt.title(f'SHAP Summary (Beeswarm) - {model_name}', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(output_dir / f'shap_summary_beeswarm_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved: shap_summary_beeswarm_{model_name}.png")
    except Exception as e:
        print(f"   ✗ Failed beeswarm plot: {e}")
    
    # 2. Bar plot (mean absolute SHAP values)
    try:
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values_plot,
            X_test_sample,
            show=False,
            plot_type="bar",
            max_display=20
        )
        plt.title(f'SHAP Feature Importance - {model_name}', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(output_dir / f'shap_bar_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved: shap_bar_{model_name}.png")
    except Exception as e:
        print(f"   ✗ Failed bar plot: {e}")


def create_dependence_plots(shap_values, X_test_sample, model_name, output_dir, top_n=5):
    """Create SHAP dependence plots for top features."""
    shap_values_plot = handle_multiclass_shap(shap_values)
    
    # Calculate mean absolute SHAP values for each feature
    mean_abs_shap = np.abs(shap_values_plot).mean(axis=0)
    
    # Handle 2D mean_abs_shap (should be 1D)
    if mean_abs_shap.ndim > 1:
        mean_abs_shap = mean_abs_shap.mean(axis=1) if mean_abs_shap.shape[1] > 1 else mean_abs_shap.flatten()
    
    top_indices = np.argsort(mean_abs_shap)[::-1][:top_n]
    feature_names = X_test_sample.columns.tolist()
    
    for idx in top_indices:
        try:
            feat_name = feature_names[int(idx)]
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(
                int(idx),
                shap_values_plot,
                X_test_sample,
                show=False
            )
            plt.title(f'SHAP Dependence: {feat_name} - {model_name}', fontsize=12, fontweight='bold')
            plt.tight_layout()
            
            # Sanitize filename
            safe_feat_name = feat_name.replace('/', '_').replace('\\', '_')[:50]
            plt.savefig(
                output_dir / f'shap_dependence_{model_name}_{safe_feat_name}.png',
                dpi=300,
                bbox_inches='tight'
            )
            plt.close()
            print(f"   ✓ Saved: shap_dependence_{model_name}_{safe_feat_name}.png")
        except Exception as e:
            print(f"   ✗ Failed dependence plot for feature {idx}: {e}")


def create_waterfall_plots(shap_values, X_test_sample, model_name, output_dir, explainer, num_samples=3):
    """Create waterfall plots for individual predictions."""
    shap_values_plot = handle_multiclass_shap(shap_values)
    
    # Use SHAP Explanation objects for waterfall (newer SHAP API)
    try:
        # Try new API with Explanation objects
        if hasattr(explainer, 'expected_value'):
            base_value = explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                if isinstance(base_value, list):
                    base_value = base_value[0] if len(base_value) > 0 else 0
                else:
                    base_value = float(base_value.mean()) if base_value.size > 1 else float(base_value)
        else:
            base_value = 0
        
        for i in range(min(num_samples, len(X_test_sample))):
            try:
                # Create Explanation object
                explanation = shap.Explanation(
                    values=shap_values_plot[i],
                    base_values=base_value,
                    data=X_test_sample.iloc[i].values,
                    feature_names=X_test_sample.columns.tolist()
                )
                
                plt.figure(figsize=(10, 8))
                shap.waterfall_plot(explanation, show=False)
                plt.title(f'SHAP Waterfall - Sample {i} - {model_name}', fontsize=12, fontweight='bold', pad=20)
                plt.tight_layout()
                plt.savefig(
                    output_dir / f'shap_waterfall_{model_name}_sample_{i}.png',
                    dpi=300,
                    bbox_inches='tight'
                )
                plt.close()
                print(f"   ✓ Saved: shap_waterfall_{model_name}_sample_{i}.png")
            except Exception as e:
                print(f"   ✗ Failed waterfall plot for sample {i}: {e}")
                
    except Exception as e:
        print(f"   ✗ Failed waterfall plots: {e}")


def create_force_plots(shap_values, X_test_sample, model_name, output_dir, explainer, num_samples=3):
    """Create force plots for individual predictions and save as HTML."""
    shap_values_plot = handle_multiclass_shap(shap_values)
    
    try:
        if hasattr(explainer, 'expected_value'):
            base_value = explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                if isinstance(base_value, list):
                    base_value = base_value[0] if len(base_value) > 0 else 0
                else:
                    base_value = float(base_value.mean()) if base_value.size > 1 else float(base_value)
        else:
            base_value = 0
        
        for i in range(min(num_samples, len(X_test_sample))):
            try:
                force_plot = shap.force_plot(
                    base_value,
                    shap_values_plot[i],
                    X_test_sample.iloc[i],
                    matplotlib=False
                )
                
                html_path = output_dir / f'shap_force_{model_name}_sample_{i}.html'
                shap.save_html(str(html_path), force_plot)
                print(f"   ✓ Saved: shap_force_{model_name}_sample_{i}.html")
            except Exception as e:
                print(f"   ✗ Failed force plot for sample {i}: {e}")
                
    except Exception as e:
        print(f"   ✗ Failed force plots: {e}")


def analyze_feature_importance(shap_values, X_test_sample, model_name):
    """Analyze and return feature importance from SHAP values."""
    shap_values_plot = handle_multiclass_shap(shap_values)
    
    # Calculate mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values_plot).mean(axis=0)
    
    # Handle 2D mean_abs_shap (should be 1D)
    if mean_abs_shap.ndim > 1:
        mean_abs_shap = mean_abs_shap.mean(axis=1) if mean_abs_shap.shape[1] > 1 else mean_abs_shap.flatten()
    
    # Create DataFrame
    feature_names = X_test_sample.columns.tolist()
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=False)
    
    return importance_df


def create_interaction_analysis(shap_values, X_test_sample, model_name, output_dir, top_n=10):
    """Analyze feature interactions using SHAP interaction values (TreeExplainer only)."""
    # Note: Interaction values are computationally expensive and only work with TreeExplainer
    print(f"   Note: Interaction analysis is computationally intensive, skipping for now")
    # Could be implemented with shap_interaction_values = explainer.shap_interaction_values(X_test_sample)


def process_model(model_name, X_train, X_test):
    """Process a single model: load, explain, visualize."""
    print(f"\n{'='*80}")
    print(f"Processing: {model_name}")
    print(f"{'='*80}")
    
    model_output_dir = Path(OUTPUT_DIR) / model_name
    model_output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load model
    model = load_model(model_name)
    if model is None:
        print(f"   ✗ Skipping {model_name} (failed to load)")
        return None
    
    print("   ✓ Model loaded")
    
    # Create explainer
    print(f"\n[3/6] Creating SHAP explainer...")
    try:
        explainer, explainer_type = get_explainer(model, X_train, model_name)
        if explainer is None:
            print(f"   ✗ Skipping {model_name} (no suitable explainer)")
            return None
        print(f"   ✓ Using {explainer_type}")
    except Exception as e:
        print(f"   ✗ Failed to create explainer: {e}")
        return None
    
    # Generate SHAP values
    print(f"\n[4/6] Generating SHAP values...")
    try:
        shap_values, X_test_sample = generate_shap_values(explainer, explainer_type, X_test)
        print(f"   ✓ Generated SHAP values for {len(X_test_sample)} test samples")
    except Exception as e:
        print(f"   ✗ Failed to generate SHAP values: {e}")
        return None
    
    # Create visualizations
    print(f"\n[5/6] Creating visualizations...")
    
    # Summary plots
    create_summary_plots(shap_values, X_test_sample, model_name, model_output_dir)
    
    # Dependence plots
    create_dependence_plots(shap_values, X_test_sample, model_name, model_output_dir, top_n=5)
    
    # Waterfall plots
    create_waterfall_plots(shap_values, X_test_sample, model_name, model_output_dir, explainer, num_samples=3)
    
    # Force plots (HTML)
    create_force_plots(shap_values, X_test_sample, model_name, model_output_dir, explainer, num_samples=3)
    
    # Analyze feature importance
    print(f"\n[6/6] Analyzing feature importance...")
    importance_df = analyze_feature_importance(shap_values, X_test_sample, model_name)
    
    # Save feature importance CSV
    csv_path = model_output_dir / f'shap_feature_importance_{model_name}.csv'
    importance_df.to_csv(csv_path, index=False)
    print(f"   ✓ Saved: shap_feature_importance_{model_name}.csv")
    
    # Print top features
    print(f"\n   Top 10 features by SHAP importance:")
    for i, row in importance_df.head(10).iterrows():
        print(f"   {i+1:2d}. {row['feature']:40s} {row['mean_abs_shap']:.6f}")
    
    return importance_df


def create_aggregated_analysis(all_importance_dfs, output_dir):
    """Create aggregated analysis across all models."""
    print(f"\n{'='*80}")
    print("Creating aggregated analysis across all models")
    print(f"{'='*80}")
    
    if not all_importance_dfs:
        print("   No importance data to aggregate")
        return
    
    # Combine all importance DataFrames
    combined_df = pd.concat([
        df.assign(model=name) for name, df in all_importance_dfs.items()
    ], ignore_index=True)
    
    # Calculate aggregate statistics
    aggregate_importance = combined_df.groupby('feature')['mean_abs_shap'].agg([
        'mean', 'std', 'min', 'max', 'count'
    ]).sort_values('mean', ascending=False)
    
    # Save aggregated results
    agg_csv_path = output_dir / 'shap_aggregated_feature_importance.csv'
    aggregate_importance.to_csv(agg_csv_path)
    print(f"   ✓ Saved: shap_aggregated_feature_importance.csv")
    
    # Create visualization
    plt.figure(figsize=(12, 10))
    top_30 = aggregate_importance.head(30)
    
    plt.barh(range(len(top_30)), top_30['mean'], xerr=top_30['std'],
             color=plt.cm.viridis(np.linspace(0, 1, len(top_30))),
             edgecolor='black', alpha=0.8)
    plt.yticks(range(len(top_30)), top_30.index, fontsize=9)
    plt.xlabel('Mean Absolute SHAP Value', fontsize=12, fontweight='bold')
    plt.ylabel('Feature', fontsize=12, fontweight='bold')
    plt.title('Top 30 Features by Aggregated SHAP Importance', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'shap_aggregated_top_features.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: shap_aggregated_top_features.png")
    
    # Print summary
    print(f"\n   Top 20 features by aggregated SHAP importance:")
    for i, (feat, row) in enumerate(aggregate_importance.head(20).iterrows(), 1):
        print(f"   {i:2d}. {feat:40s} Mean: {row['mean']:.6f} ± {row['std']:.6f}")


def main():
    """Main analysis pipeline."""
    print("="*80)
    print(f"COMPREHENSIVE SHAP ANALYSIS - {DATASET_TYPE.upper()} DATASET")
    print("="*80)
    
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Find best models
    model_names = find_best_models()
    
    if not model_names:
        print("\n✗ No models found!")
        return
    
    print(f"\n   Will analyze {len(model_names)} models")
    
    # Process each model
    all_importance_dfs = {}
    
    for model_name in model_names:
        try:
            importance_df = process_model(model_name, X_train, X_test)
            if importance_df is not None:
                all_importance_dfs[model_name] = importance_df
        except Exception as e:
            print(f"\n   ✗ Failed to process {model_name}: {e}")
            continue
    
    # Create aggregated analysis
    if all_importance_dfs:
        create_aggregated_analysis(all_importance_dfs, Path(OUTPUT_DIR))
    
    print("\n" + "="*80)
    print(f"✓ SHAP ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"\nGenerated visualizations:")
    print(f"  * SHAP summary (beeswarm) plots")
    print(f"  * SHAP feature importance (bar) plots")
    print(f"  * SHAP dependence plots (top 5 features per model)")
    print(f"  * SHAP waterfall plots (3 samples per model)")
    print(f"  * SHAP force plots HTML (3 samples per model)")
    print(f"  * Feature importance CSVs")
    print(f"  * Aggregated analysis across all models")


if __name__ == '__main__':
    main()
