"""
Visualize Top 3 Models from Cross-Validation Results
Creates bar charts showing:
1. Top 3 model-strategy combinations
2. Top 3 models overall (averaged across strategies)
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Professional styling
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#fafafa'

CHART_COLORS = ['#5e81ac', '#a3be8c', '#ebcb8b', '#d08770', '#b48ead', '#88c0d0']

def load_cv_results(dataset_type='Mix'):
    """Load cross-validation results."""
    cv_path1 = f'{dataset_type}_Results/cross_validation_results/cv_results.json'
    cv_path2 = 'cross_validation_results/cv_results.json'
    
    cv_path = cv_path1 if os.path.exists(cv_path1) else cv_path2
    
    if not os.path.exists(cv_path):
        print(f"✗ CV results not found")
        return None
    
    with open(cv_path, 'r') as f:
        return json.load(f)

def get_top_strategies(cv_data):
    """Get top 3 model-strategy combinations."""
    results = []
    for model_name, model_data in cv_data.items():
        bal_acc = model_data['metrics']['balanced_accuracy']['mean']
        std = model_data['metrics']['balanced_accuracy']['std']
        results.append({
            'name': model_name,
            'display': model_name.replace('_', ' ').title(),
            'bal_acc': bal_acc,
            'std': std
        })
    
    # Sort by balanced accuracy
    results.sort(key=lambda x: x['bal_acc'], reverse=True)
    return results[:3]

def get_top_models_overall(cv_data):
    """Get top 3 models overall """
    # Extract base model name (remove resampling strategy suffix)
    model_groups = defaultdict(list)
    
    for model_name, model_data in cv_data.items():
        # Extract base model name
        base_name = model_name.replace('_baseline', '').replace('_tuned', '')
        for strategy in ['_smote', '_randomoversampler', '_classweights', '_randomundersampler', '_original']:
            base_name = base_name.replace(strategy, '')
        
        bal_acc = model_data['metrics']['balanced_accuracy']['mean']
        std = model_data['metrics']['balanced_accuracy']['std']
        
        model_groups[base_name].append({
            'bal_acc': bal_acc,
            'std': std
        })
    
    # Average across strategies
    averaged_results = []
    for base_name, results in model_groups.items():
        avg_bal_acc = np.mean([r['bal_acc'] for r in results])
        # Propagate uncertainty (combined std)
        combined_std = np.sqrt(np.mean([r['std']**2 for r in results]))
        
        averaged_results.append({
            'name': base_name,
            'display': base_name.replace('_', ' ').title(),
            'bal_acc': avg_bal_acc,
            'std': combined_std
        })
    
    # Sort by balanced accuracy
    averaged_results.sort(key=lambda x: x['bal_acc'], reverse=True)
    return averaged_results[:3]

def create_visualization(dataset_type='Mix'):
    """Create top 3 visualizations."""
    print("="*80)
    print(f"TOP 3 MODELS VISUALIZATION - {dataset_type} DATASET")
    print("="*80)
    
    # Load data
    print("\n[1/4] Loading CV results...")
    cv_data = load_cv_results(dataset_type)
    if not cv_data:
        return
    
    print(f"✓ Loaded {len(cv_data)} models")
    
    # Get top models
    print("\n[2/4] Identifying top performers...")
    top_strategies = get_top_strategies(cv_data)
    top_models_overall = get_top_models_overall(cv_data)
    
    print("\nTop 3 Model-Strategy Combinations:")
    for i, model in enumerate(top_strategies, 1):
        print(f"  {i}. {model['display']}: {model['bal_acc']:.4f} ± {model['std']:.4f}")
    
    print("\nTop 3 Models Overall (Averaged Across Strategies):")
    for i, model in enumerate(top_models_overall, 1):
        print(f"  {i}. {model['display']}: {model['bal_acc']:.4f} ± {model['std']:.4f}")
    
    # Create visualization
    print("\n[3/4] Creating visualization...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Chart 1: Top 3 Model-Strategy Combinations
    names1 = [m['display'] for m in top_strategies]
    means1 = [m['bal_acc'] for m in top_strategies]
    stds1 = [m['std'] for m in top_strategies]
    
    bars1 = ax1.barh(names1, means1, color=CHART_COLORS[:3], 
                     alpha=0.8, edgecolor='black', linewidth=2)
    ax1.errorbar(means1, range(len(names1)), xerr=stds1,
                fmt='none', ecolor='black', capsize=8, capthick=2)
    
    ax1.set_xlabel('Balanced Accuracy', fontweight='bold', fontsize=12)
    ax1.set_title(f'{dataset_type} Dataset\nTop 3 Model-Strategy Combinations (30-Fold CV)',
                 fontweight='bold', fontsize=13, pad=15)
    ax1.set_xlim(0, 1.0)
    ax1.grid(axis='x', alpha=0.3)
    ax1.axvline(x=0.8, color='green', linestyle='--', alpha=0.5, linewidth=1.5, label='Target: 0.80')
    ax1.legend(fontsize=10)
    
    # Add value labels
    for i, (bar, mean, std) in enumerate(zip(bars1, means1, stds1)):
        ax1.text(mean + 0.02, bar.get_y() + bar.get_height()/2,
                f'{mean:.4f}\n±{std:.4f}',
                va='center', fontsize=10, fontweight='bold')
    
    # Add rank badges
    colors_rank = ['#FFD700', '#C0C0C0', '#CD7F32']  # Gold, Silver, Bronze
    for i, bar in enumerate(bars1):
        ax1.text(0.02, bar.get_y() + bar.get_height()/2,
                f'#{i+1}', va='center', ha='left',
                fontsize=12, fontweight='bold', color=colors_rank[i],
                bbox=dict(boxstyle='circle', facecolor='white', edgecolor=colors_rank[i], linewidth=2))
    
    # Chart 2: Top 3 Models Overall
    names2 = [m['display'] for m in top_models_overall]
    means2 = [m['bal_acc'] for m in top_models_overall]
    stds2 = [m['std'] for m in top_models_overall]
    
    bars2 = ax2.barh(names2, means2, color=CHART_COLORS[3:6],
                     alpha=0.8, edgecolor='black', linewidth=2)
    ax2.errorbar(means2, range(len(names2)), xerr=stds2,
                fmt='none', ecolor='black', capsize=8, capthick=2)
    
    ax2.set_xlabel('Balanced Accuracy', fontweight='bold', fontsize=12)
    ax2.set_title(f'{dataset_type} Dataset\nTop 3 Models Overall ',
                 fontweight='bold', fontsize=13, pad=15)
    ax2.set_xlim(0, 1.0)
    ax2.grid(axis='x', alpha=0.3)
    ax2.axvline(x=0.8, color='green', linestyle='--', alpha=0.5, linewidth=1.5, label='Target: 0.80')
    ax2.legend(fontsize=10)
    
    # Add value labels
    for i, (bar, mean, std) in enumerate(zip(bars2, means2, stds2)):
        ax2.text(mean + 0.02, bar.get_y() + bar.get_height()/2,
                f'{mean:.4f}\n±{std:.4f}',
                va='center', fontsize=10, fontweight='bold')
    
    # Add rank badges
    for i, bar in enumerate(bars2):
        ax2.text(0.02, bar.get_y() + bar.get_height()/2,
                f'#{i+1}', va='center', ha='left',
                fontsize=12, fontweight='bold', color=colors_rank[i],
                bbox=dict(boxstyle='circle', facecolor='white', edgecolor=colors_rank[i], linewidth=2))
    
    plt.tight_layout()
    
    # Save figure
    output_dir = f'{dataset_type}_Results/figures'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'top3_cv_models.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    
    plt.close()
    
    print("\n[4/4] ✓ Visualization complete")
    print("="*80)
    
    return output_path

if __name__ == "__main__":
    import sys
    
    dataset_type = sys.argv[1] if len(sys.argv) > 1 else 'Mix'
    create_visualization(dataset_type)
