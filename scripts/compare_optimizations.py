"""
Compare performance of original vs optimized models
Shows improvements from Phase 1 optimizations across all datasets
"""

import os
import json
import pandas as pd
import numpy as np

# Get project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

def load_model_metrics(base_folder, model_name, strategy='original'):
    """Load metrics for a specific model and strategy."""
    # Try different possible file patterns
    patterns = [
        os.path.join(PROJECT_ROOT, base_folder, 'data', f'error_analysis_{model_name}_{strategy}.csv'),
        os.path.join(PROJECT_ROOT, base_folder, 'reports', f'{model_name}_metrics.json'),
        os.path.join(PROJECT_ROOT, base_folder, 'reports', 'combined_results.json')
    ]
    
    for pattern in patterns:
        if os.path.exists(pattern):
            if pattern.endswith('.csv'):
                df = pd.read_csv(pattern)
                if len(df) > 0:
                    return {
                        'accuracy': df['accuracy'].iloc[0] if 'accuracy' in df.columns else None,
                        'balanced_accuracy': df['balanced_accuracy'].iloc[0] if 'balanced_accuracy' in df.columns else None,
                        'f1_macro': df['f1_macro'].iloc[0] if 'f1_macro' in df.columns else None
                    }
            elif pattern.endswith('.json'):
                with open(pattern, 'r') as f:
                    data = json.load(f)
                    if model_name in data:
                        return data[model_name]
    return None

def load_ensemble_metrics(base_folder, ensemble_type):
    """Load metrics for ensemble models."""
    metrics_file = os.path.join(PROJECT_ROOT, base_folder, 'reports', f'{ensemble_type}_ensemble_metrics.json')
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            return json.load(f)
    return None

def compare_models(dataset_name):
    """Compare original vs optimized models for a dataset."""
    print(f"\n{'='*80}")
    print(f"OPTIMIZATION COMPARISON: {dataset_name}")
    print(f"{'='*80}\n")
    
    base_folder = f"{dataset_name}_Results"
    
    # Models to compare - showing current best results from combined_results.json
    models = {
        'SVM': 'svm',
        'MLP': 'mlp', 
        'LightGBM': 'lgbm',
        'XGBoost': 'xgboost',
        'RandomForest': 'randomforest',
        'ExtraTrees': 'extratrees',
        'GaussianNB': 'gnb',
        'LogisticRegression': 'logistic_regression'
    }
    
    results = []
    
    # Load combined_results.json for metrics
    combined_file = os.path.join(PROJECT_ROOT, base_folder, 'reports', 'combined_results.json')
    if os.path.exists(combined_file):
        with open(combined_file, 'r') as f:
            combined_data = json.load(f)
    else:
        combined_data = {}
    
    # Get best results for each model
    for display_name, model_key in models.items():
        if model_key in combined_data:
            model_data = combined_data[model_key]
            
            # Extract results - handle different structures
            if 'results' in model_data:
                strategies = model_data['results']
            else:
                strategies = model_data
            
            # Find the best strategy
            best_metrics = None
            best_acc = 0
            best_strategy = ""
            
            for strategy, metrics in strategies.items():
                if isinstance(metrics, dict) and 'balanced_accuracy' in metrics:
                    if metrics['balanced_accuracy'] > best_acc:
                        best_acc = metrics['balanced_accuracy']
                        best_metrics = metrics
                        best_strategy = strategy
            
            if best_metrics:
                improvement = {
                    'model': f"{display_name} ({best_strategy})",
                    'accuracy': best_metrics.get('accuracy', 0),
                    'balanced_acc': best_metrics.get('balanced_accuracy', 0),
                    'f1': best_metrics.get('f1_macro', 0)
                }
                results.append(improvement)
    
    # Add ensemble results
    voting_metrics = load_ensemble_metrics(base_folder, 'voting')
    if voting_metrics:
        results.append({
            'model': '✨ Voting Ensemble',
            'accuracy': voting_metrics.get('accuracy', 0),
            'balanced_acc': voting_metrics.get('balanced_accuracy', 0),
            'f1': voting_metrics.get('f1_weighted', 0)
        })
    
    stacking_metrics = load_ensemble_metrics(base_folder, 'stacking')
    if stacking_metrics:
        results.append({
            'model': '⭐ Stacking Ensemble',
            'accuracy': stacking_metrics.get('accuracy', 0),
            'balanced_acc': stacking_metrics.get('balanced_accuracy', 0),
            'f1': stacking_metrics.get('f1_weighted', 0)
        })
    
    # Display results
    if results:
        df = pd.DataFrame(results)
        
        # Sort by balanced accuracy
        df = df.sort_values('balanced_acc', ascending=False)
        
        print("Model Performance Ranking:")
        print("-" * 80)
        print(f"{'Rank':<6} {'Model':<40} {'Accuracy':<12} {'Bal. Acc':<12} {'F1':<12}")
        print("-" * 80)
        
        for idx, row in enumerate(df.itertuples(), 1):
            print(f"{idx:<6} {row.model:<40} {row.accuracy:<12.4f} {row.balanced_acc:<12.4f} {row.f1:<12.4f}")
        
        # Summary statistics
        print(f"\n{'='*80}")
        print("PERFORMANCE SUMMARY")
        print(f"{'='*80}")
        
        # Best models
        best_overall = df.iloc[0]
        print(f"\n🏆 BEST OVERALL MODEL: {best_overall['model']}")
        print(f"   Balanced Accuracy: {best_overall['balanced_acc']:.4f}")
        print(f"   Accuracy:          {best_overall['accuracy']:.4f}")
        print(f"   F1 Score:          {best_overall['f1']:.4f}")
        
        # Top 3
        print(f"\n📊 TOP 3 MODELS:")
        for idx, row in enumerate(df.head(3).itertuples(), 1):
            print(f"   {idx}. {row.model}")
            print(f"      Balanced Accuracy: {row.balanced_acc:.4f}")
        
        # Ensemble comparison
        ensemble_df = df[df['model'].str.contains('Ensemble', na=False)]
        non_ensemble_df = df[~df['model'].str.contains('Ensemble', na=False)]
        
        if len(ensemble_df) > 0 and len(non_ensemble_df) > 0:
            best_individual = non_ensemble_df.iloc[0]
            best_ensemble = ensemble_df.iloc[0]
            
            improvement = (best_ensemble['balanced_acc'] - best_individual['balanced_acc']) * 100
            
            print(f"\n🎯 ENSEMBLE IMPROVEMENT:")
            print(f"   Best Individual:  {best_individual['model']}")
            print(f"      Bal. Accuracy: {best_individual['balanced_acc']:.4f}")
            print(f"   Best Ensemble:    {best_ensemble['model']}")
            print(f"      Bal. Accuracy: {best_ensemble['balanced_acc']:.4f}")
            print(f"   Improvement:      {improvement:+.2f}%")
        
        # Save to CSV
        output_file = os.path.join(PROJECT_ROOT, base_folder, 'reports', 'optimization_comparison.csv')
        df.to_csv(output_file, index=False)
        print(f"\n[INFO] Comparison saved to: {output_file}")
        
    else:
        print("[WARNING] No results found for comparison")

def main():
    """Run comparison for both datasets."""
    print(f"\n{'#'*80}")
    print("PHASE 1 OPTIMIZATION RESULTS")
    print(f"{'#'*80}")
    
    # Compare both datasets
    compare_models("Mix")
    compare_models("Single")
    
    print(f"\n{'#'*80}")
    print("OPTIMIZATION COMPARISON COMPLETE")
    print(f"{'#'*80}\n")
    
    print("\nOptimizations Applied:")
    print("  ✓ Feature Scaling (SVM/MLP) - StandardScaler for distance-based models")
    print("  ✓ Class Weights (SVM/LGBM/XGB) - Balanced class weights for imbalanced data")
    print("  ✓ Early Stopping (LGBM/XGB) - Validation-based early stopping")
    print("  ✓ Voting Ensemble - Weighted soft voting of top 5 models")
    print("  ✓ Stacking Ensemble - Meta-learning with LogisticRegression")
    print("\nNext Steps:")
    print("  - Review optimization_comparison.csv in each Results folder")
    print("  - Check ensemble confusion matrices in figures/ folders")
    print("  - Consider Phase 2 optimizations if needed (see PIPELINE_OPTIMIZATION_REPORT.md)")

if __name__ == '__main__':
    main()
