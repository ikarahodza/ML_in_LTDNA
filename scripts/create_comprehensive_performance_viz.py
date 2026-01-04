"""
Comprehensive Performance Visualizations
Creates two types of visualizations for seed 42, aggregated seeds, and CV results:
1. Model + Strategy (top 10) with color-coded models and shaded strategies
2. Best Model per Strategy with misclassification rates
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
OUTPUT_DIR = 'reports/performance_visualizations'
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Color scheme for models
MODEL_COLORS = {
    'RandomForest': '#2E7D32',      # Green
    'ExtraTrees': '#1976D2',        # Blue
    'LightGBM': '#F57C00',          # Orange
    'XGBoost': '#C62828',           # Red
    'GaussianNB': '#7B1FA2',        # Purple
    'LogisticRegression': '#C2185B',# Pink
    'MLP': '#00838F',               # Cyan
    'TabPFN': '#FFB300',            # Amber
}

def normalize_model_name(name):
    """Normalize model names to consistent format."""
    name_lower = name.lower()
    if 'randomforest' in name_lower or 'random_forest' in name_lower:
        return 'RandomForest'
    elif 'extratrees' in name_lower or 'extra_trees' in name_lower:
        return 'ExtraTrees'
    elif 'lightgbm' in name_lower or 'lgbm' in name_lower:
        return 'LightGBM'
    elif 'xgboost' in name_lower or 'xgb' in name_lower:
        return 'XGBoost'
    elif 'gaussiannb' in name_lower or 'gnb' in name_lower:
        return 'GaussianNB'
    elif 'logistic' in name_lower or 'logreg' in name_lower:
        return 'LogisticRegression'
    elif 'mlp' in name_lower:
        return 'MLP'
    elif 'tabpfn' in name_lower:
        return 'TabPFN'
    return name

def normalize_strategy(strategy):
    """Normalize strategy names."""
    if not strategy or strategy.lower() == 'none':
        return 'Original'
    strategy_lower = strategy.lower()
    if 'original' in strategy_lower or 'baseline' in strategy_lower:
        return 'Original'
    elif 'classweight' in strategy_lower or 'class_weight' in strategy_lower:
        return 'ClassWeights'
    elif 'smote' in strategy_lower:
        return 'SMOTE'
    elif 'randomoversampler' in strategy_lower or 'random_over' in strategy_lower or 'ros' in strategy_lower:
        return 'RandomOverSampler'
    elif 'adasyn' in strategy_lower:
        return 'ADASYN'
    return strategy

def get_color_shade(base_color, strategy, n_strategies=4):
    """Get a shade of the base color for a strategy."""
    strategy_order = ['Original', 'ClassWeights', 'SMOTE', 'RandomOverSampler', 'ADASYN']
    try:
        idx = strategy_order.index(strategy)
    except:
        idx = 0
    
    # Convert hex to RGB
    base_color = base_color.lstrip('#')
    r, g, b = int(base_color[0:2], 16), int(base_color[2:4], 16), int(base_color[4:6], 16)
    
    # Create shade (darker to lighter)
    factor = 0.6 + (idx * 0.4 / max(n_strategies - 1, 1))
    r, g, b = int(r * factor), int(g * factor), int(b * factor)
    
    return f'#{r:02x}{g:02x}{b:02x}'

def load_seed_results(dataset='Mix'):
    """Load seed 123 (seed 42 equivalent) results."""
    results = []
    
    # Try Mix_Results/seed_123 and Single_Results/seed_123
    for ds_type in [dataset]:
        json_path = Path(f'{ds_type}_Results/seed_123/reports/combined_results.json')
        if json_path.exists():
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            for model_name, model_data in data.items():
                if 'results' not in model_data:
                    continue
                    
                for strategy, metrics in model_data['results'].items():
                    if not isinstance(metrics, dict):
                        continue
                        
                    results.append({
                        'Dataset': ds_type,
                        'Model': normalize_model_name(model_name),
                        'Strategy': normalize_strategy(strategy),
                        'Accuracy': metrics.get('accuracy', 0),
                        'Balanced_Accuracy': metrics.get('balanced_accuracy', 0),
                        'F1_Macro': metrics.get('f1_macro', 0),
                        'Log_Loss': metrics.get('log_loss', 0),
                        'Misclassification_Rate': 1 - metrics.get('accuracy', 0)
                    })
    
    return pd.DataFrame(results)

def load_aggregated_results(dataset='Mix'):
    """Load multi-seed aggregated results."""
    results = []
    
    json_path = Path(f'{dataset}_Results/multi_seed_aggregated/aggregated_results.json')
    if json_path.exists():
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        for model_name, strategies in data.items():
            for strategy, metrics in strategies.items():
                if not isinstance(metrics, dict):
                    continue
                
                results.append({
                    'Dataset': dataset,
                    'Model': normalize_model_name(model_name),
                    'Strategy': normalize_strategy(strategy),
                    'Accuracy': metrics.get('accuracy', {}).get('mean', 0),
                    'Balanced_Accuracy': metrics.get('balanced_accuracy', {}).get('mean', 0),
                    'F1_Macro': metrics.get('f1_macro', {}).get('mean', 0),
                    'Log_Loss': metrics.get('log_loss', {}).get('mean', 0),
                    'Misclassification_Rate': 1 - metrics.get('accuracy', {}).get('mean', 0)
                })
    
    return pd.DataFrame(results)

def load_cv_results():
    """Load cross-validation results."""
    results = []
    
    json_path = Path('cross_validation_results/cv_results.json')
    if json_path.exists():
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        for model_key, model_data in data.items():
            # Parse model name and strategy from key
            model_type = model_data.get('type', 'baseline')
            resampling = model_data.get('resampling', 'Original')
            
            # Extract model name from key
            if 'lgbm' in model_key.lower():
                model = 'LightGBM'
            elif 'xgb' in model_key.lower():
                model = 'XGBoost'
            elif 'rf' in model_key.lower():
                model = 'RandomForest'
            elif 'et' in model_key.lower():
                model = 'ExtraTrees'
            else:
                model = normalize_model_name(model_key.split('_')[0])
            
            metrics = model_data.get('metrics', {})
            
            results.append({
                'Dataset': 'CV',
                'Model': model,
                'Strategy': normalize_strategy(resampling),
                'Accuracy': metrics.get('accuracy', {}).get('mean', 0),
                'Balanced_Accuracy': metrics.get('balanced_accuracy', {}).get('mean', 0),
                'F1_Macro': metrics.get('f1_macro', {}).get('mean', 0),
                'Log_Loss': metrics.get('log_loss', {}).get('mean', 0),
                'Misclassification_Rate': 1 - metrics.get('accuracy', {}).get('mean', 0)
            })
    
    return pd.DataFrame(results)

def create_model_strategy_viz(df, metric, title_suffix, dataset_name, output_prefix):
    """Create visualization 1: Model + Strategy (top 10)."""
    
    # Get top 10 model-strategy combinations
    df_sorted = df.sort_values(metric, ascending=(metric == 'Log_Loss'))
    top_10 = df_sorted.head(10).copy()
    
    # Create combined label
    top_10['Label'] = top_10['Model'] + ' - ' + top_10['Strategy']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Get colors for each bar
    colors = []
    for idx, row in top_10.iterrows():
        base_color = MODEL_COLORS.get(row['Model'], '#757575')
        shade = get_color_shade(base_color, row['Strategy'])
        colors.append(shade)
    
    # Create bars
    y_pos = np.arange(len(top_10))
    bars = ax.barh(y_pos, top_10[metric], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add misclassification rate as text on bars
    for i, (bar, row) in enumerate(zip(bars, top_10.itertuples())):
        missclass_pct = row.Misclassification_Rate * 100
        x_pos = bar.get_width() * 0.5
        
        # Add metric value
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, 
                f'{row._asdict()[metric]:.3f}',
                ha='center', va='center', fontweight='bold', fontsize=9, color='white')
        
        # Add misclassification rate
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{missclass_pct:.1f}% miss',
                ha='left', va='center', fontsize=8, style='italic', color='red')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_10['Label'], fontsize=10)
    ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax.set_title(f'{metric.replace("_", " ").title()} - Top 10 Model + Strategy\n{title_suffix}',
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add legend for models
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=model, edgecolor='black') 
                      for model, color in MODEL_COLORS.items() if model in top_10['Model'].values]
    ax.legend(handles=legend_elements, loc='lower right', title='Models', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/{output_prefix}_model_strategy_{metric.lower()}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f'   ✓ Saved: {output_prefix}_model_strategy_{metric.lower()}.png')

def create_best_model_viz(df, metric, title_suffix, dataset_name, output_prefix):
    """Create visualization 2: Best strategy per model."""
    
    # For each model, select best strategy
    if metric == 'Log_Loss':
        best_per_model = df.loc[df.groupby('Model')[metric].idxmin()]
    else:
        best_per_model = df.loc[df.groupby('Model')[metric].idxmax()]
    
    # Sort by metric
    best_per_model = best_per_model.sort_values(metric, ascending=(metric == 'Log_Loss'))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Get colors
    colors = [MODEL_COLORS.get(model, '#757575') for model in best_per_model['Model']]
    
    # Create bars
    y_pos = np.arange(len(best_per_model))
    bars = ax.barh(y_pos, best_per_model[metric], color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add labels on bars
    for i, (bar, row) in enumerate(zip(bars, best_per_model.itertuples())):
        missclass_pct = row.Misclassification_Rate * 100
        x_pos = bar.get_width() * 0.5
        
        # Add metric value
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, 
                f'{row._asdict()[metric]:.3f}',
                ha='center', va='center', fontweight='bold', fontsize=10, color='white')
        
        # Add strategy and misclassification
        strategy_text = f'{row.Strategy} | {missclass_pct:.1f}% miss'
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                strategy_text,
                ha='left', va='center', fontsize=9, style='italic')
    
    # Create labels with model name
    labels = [f'{model}' for model in best_per_model['Model']]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11, fontweight='bold')
    ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax.set_title(f'{metric.replace("_", " ").title()} - Best Strategy per Model\n{title_suffix}',
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/{output_prefix}_best_model_{metric.lower()}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f'   ✓ Saved: {output_prefix}_best_model_{metric.lower()}.png')

def main():
    """Main visualization pipeline."""
    print("="*80)
    print("COMPREHENSIVE PERFORMANCE VISUALIZATIONS")
    print("="*80)
    
    metrics = ['Balanced_Accuracy', 'F1_Macro', 'Log_Loss']
    
    # Process each dataset source
    datasets = [
        ('Mix', 'seed', 'Seed 42 (Mix Dataset)'),
        ('Single', 'seed', 'Seed 42 (Single Dataset)'),
        ('Mix', 'agg', 'Aggregated Multi-Seed (Mix Dataset)'),
        ('Single', 'agg', 'Aggregated Multi-Seed (Single Dataset)'),
        ('CV', 'cv', '30-Fold Cross-Validation'),
    ]
    
    for dataset_type, source_type, title in datasets:
        print(f"\n{title}")
        print("-" * 80)
        
        # Load data
        if source_type == 'seed':
            df = load_seed_results(dataset_type)
            prefix = f'{dataset_type.lower()}_seed42'
        elif source_type == 'agg':
            df = load_aggregated_results(dataset_type)
            prefix = f'{dataset_type.lower()}_aggregated'
        else:  # cv
            df = load_cv_results()
            prefix = 'cv_30fold'
        
        if df.empty:
            print(f"   ⚠ No data found for {title}")
            continue
        
        print(f"   Loaded {len(df)} model-strategy combinations")
        
        # Create visualizations for each metric
        for metric in metrics:
            print(f"\n   Creating {metric} visualizations...")
            
            # Visualization 1: Model + Strategy (top 10)
            create_model_strategy_viz(df, metric, title, dataset_type, prefix)
            
            # Visualization 2: Best model per strategy
            create_best_model_viz(df, metric, title, dataset_type, prefix)
    
    print("\n" + "="*80)
    print("✓ ALL VISUALIZATIONS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print("\nGenerated 2 types of visualizations for each metric:")
    print("  1. model_strategy_* - Top 10 combinations with color-coded models")
    print("  2. best_model_* - Best strategy per model")
    print("\nMetrics:")
    print("  - Balanced Accuracy")
    print("  - F1 Macro Score")
    print("  - Log Loss")
    print("\nDatasets:")
    print("  - Seed 42 (Mix and Single)")
    print("  - Aggregated Multi-Seed (Mix and Single)")
    print("  - 30-Fold Cross-Validation")

if __name__ == '__main__':
    main()
