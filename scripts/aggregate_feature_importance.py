"""
Multi-Seed Feature Importance Aggregator
Aggregates feature importance across multiple random seeds and algorithms
Usage: python aggregate_feature_importance.py [mix|single]
"""
import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Configuration
SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]

# Algorithms with feature importance capability
ALGORITHMS = [
    'xgboost',
    'lgbm', 
    'extratrees',
    'logistic_regression'
]

SAMPLING_STRATEGIES = [
    'original',
    'smote',
    'randomoversampler'
]

def get_seed_folder_path(dataset_type, seed):
    """Get the path to a seed's folder"""
    results_folder = f"{dataset_type.capitalize()}_Results"
    
    # Seed 42 is in the main results folder (no seed subfolder)
    if seed == 42:
        return results_folder
    else:
        return os.path.join(results_folder, f"seed_{seed}")

def load_feature_importance(dataset_type, seed, algorithm, strategy):
    """Load feature importance CSV for a specific combination"""
    seed_folder = get_seed_folder_path(dataset_type, seed)
    figures_dir = os.path.join(seed_folder, 'figures')
    
    # Find matching feature importance file
    pattern = f"feature_importance_{algorithm}_{strategy}_importances_"
    csv_files = []
    
    if os.path.exists(figures_dir):
        for file in os.listdir(figures_dir):
            if file.startswith(pattern) and file.endswith('.csv'):
                csv_files.append(file)
    
    if not csv_files:
        return None
    
    # Load the CSV file
    csv_path = os.path.join(figures_dir, csv_files[0])
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        print(f"⚠ Warning: Could not load {csv_path}: {e}")
        return None

def aggregate_feature_importance_for_model(dataset_type, algorithm, strategy):
    """Aggregate feature importance across all seeds for a specific algorithm and strategy"""
    all_importances = []
    available_seeds = []
    
    for seed in SEEDS:
        df = load_feature_importance(dataset_type, seed, algorithm, strategy)
        if df is not None:
            all_importances.append(df)
            available_seeds.append(seed)
    
    if not all_importances:
        return None, []
    
    # Merge all dataframes
    # Each df should have columns: feature, importance
    merged = all_importances[0].copy()
    merged.columns = ['feature', f'seed_{available_seeds[0]}']
    
    for i, df in enumerate(all_importances[1:], 1):
        df_temp = df.copy()
        df_temp.columns = ['feature', f'seed_{available_seeds[i]}']
        merged = merged.merge(df_temp, on='feature', how='outer')
    
    # Fill NaN with 0 (feature not present in some seeds)
    seed_cols = [col for col in merged.columns if col.startswith('seed_')]
    merged[seed_cols] = merged[seed_cols].fillna(0)
    
    # Calculate statistics
    merged['mean_importance'] = merged[seed_cols].mean(axis=1)
    merged['std_importance'] = merged[seed_cols].std(axis=1)
    merged['cv_importance'] = merged['std_importance'] / (merged['mean_importance'] + 1e-10)
    
    # Sort by mean importance
    merged = merged.sort_values('mean_importance', ascending=False)
    
    return merged, available_seeds

def create_importance_heatmap(df, algorithm, strategy, output_dir, top_n=30):
    """Create heatmap showing feature importance across seeds"""
    if df is None or len(df) == 0:
        return None
    
    # Get top N features by mean importance
    top_features = df.head(top_n)
    
    # Get seed columns
    seed_cols = [col for col in df.columns if col.startswith('seed_')]
    
    # Prepare data for heatmap
    heatmap_data = top_features[['feature'] + seed_cols].set_index('feature')
    
    # Create figure
    plt.figure(figsize=(14, max(8, top_n * 0.3)))
    sns.heatmap(heatmap_data, annot=False, cmap='YlOrRd', cbar_kws={'label': 'Importance'})
    plt.title(f'Feature Importance Across Seeds\n{algorithm.upper()} - {strategy}', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Seed', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'heatmap_{algorithm}_{strategy}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def create_top_features_barplot(df, algorithm, strategy, output_dir, top_n=20):
    """Create bar plot of top N features with error bars"""
    if df is None or len(df) == 0:
        return None
    
    top_features = df.head(top_n)
    
    plt.figure(figsize=(12, max(6, top_n * 0.3)))
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
    
    plt.barh(range(len(top_features)), top_features['mean_importance'], 
             xerr=top_features['std_importance'], color=colors, alpha=0.8)
    
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Mean Importance', fontsize=12, fontweight='bold')
    plt.ylabel('Feature', fontsize=12, fontweight='bold')
    plt.title(f'Top {top_n} Features by Mean Importance\n{algorithm.upper()} - {strategy}',
              fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'top_features_{algorithm}_{strategy}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def create_algorithm_comparison(all_results, dataset_type, output_dir, top_n=20):
    """Compare feature importance across algorithms (using 'original' strategy)"""
    
    comparison_data = {}
    for algo in ALGORITHMS:
        if (algo, 'original') in all_results:
            df, seeds = all_results[(algo, 'original')]
            if df is not None and len(df) > 0:
                comparison_data[algo] = df[['feature', 'mean_importance']].set_index('feature')
    
    if not comparison_data:
        return None
    
    # Merge all algorithms
    merged = pd.concat(comparison_data, axis=1)
    merged.columns = merged.columns.get_level_values(0)
    merged = merged.fillna(0)
    
    # Get features that appear in at least one algorithm
    merged['max_importance'] = merged.max(axis=1)
    top_features = merged.nlargest(top_n, 'max_importance').drop('max_importance', axis=1)
    
    # Create grouped bar plot
    fig, ax = plt.subplots(figsize=(14, max(8, top_n * 0.3)))
    
    x = np.arange(len(top_features))
    width = 0.8 / len(comparison_data)
    
    for i, algo in enumerate(comparison_data.keys()):
        offset = (i - len(comparison_data) / 2) * width + width / 2
        ax.barh(x + offset, top_features[algo], width, label=algo.upper(), alpha=0.8)
    
    ax.set_yticks(x)
    ax.set_yticklabels(top_features.index)
    ax.set_xlabel('Mean Importance', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
    ax.set_title(f'Feature Importance Comparison Across Algorithms\n{dataset_type.upper()} Dataset (Original Strategy)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'algorithm_comparison_original.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def create_consistency_analysis(all_results, dataset_type, output_dir):
    """Analyze which features are consistently important across algorithms and strategies"""
    
    all_features = defaultdict(lambda: {'count': 0, 'mean_ranks': [], 'algorithms': set()})
    
    for (algo, strategy), (df, seeds) in all_results.items():
        if df is not None and len(df) > 0:
            # Get top 20 features for this algo-strategy combination
            top_features = df.head(20)
            for idx, row in top_features.iterrows():
                feature = row['feature']
                rank = idx + 1
                all_features[feature]['count'] += 1
                all_features[feature]['mean_ranks'].append(rank)
                all_features[feature]['algorithms'].add(algo)
    
    # Create consistency dataframe
    consistency_data = []
    for feature, data in all_features.items():
        consistency_data.append({
            'feature': feature,
            'appearances': data['count'],
            'mean_rank': np.mean(data['mean_ranks']),
            'num_algorithms': len(data['algorithms']),
            'algorithms': ', '.join(sorted(data['algorithms']))
        })
    
    consistency_df = pd.DataFrame(consistency_data)
    consistency_df = consistency_df.sort_values('appearances', ascending=False)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, 'feature_consistency_analysis.csv')
    consistency_df.to_csv(csv_path, index=False)
    
    # Create visualization
    top_consistent = consistency_df.head(25)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
    
    # Plot 1: Number of appearances
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_consistent)))
    ax1.barh(range(len(top_consistent)), top_consistent['appearances'], color=colors, alpha=0.8)
    ax1.set_yticks(range(len(top_consistent)))
    ax1.set_yticklabels(top_consistent['feature'])
    ax1.set_xlabel('Number of Appearances in Top 20', fontsize=11, fontweight='bold')
    ax1.set_title('Most Consistently Important Features\n(Across Algorithms & Strategies)', 
                  fontsize=12, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)
    
    # Plot 2: Mean rank (lower is better)
    colors2 = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(top_consistent)))
    ax2.barh(range(len(top_consistent)), top_consistent['mean_rank'], color=colors2, alpha=0.8)
    ax2.set_yticks(range(len(top_consistent)))
    ax2.set_yticklabels(top_consistent['feature'])
    ax2.set_xlabel('Mean Rank in Top 20 Lists', fontsize=11, fontweight='bold')
    ax2.set_title('Average Ranking Position\n(Lower = More Important)', 
                  fontsize=12, fontweight='bold')
    ax2.invert_yaxis()
    ax2.invert_xaxis()
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'feature_consistency_visualization.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return csv_path, output_path

def generate_summary_report(all_results, dataset_type, output_dir):
    """Generate a comprehensive summary report"""
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append(f"MULTI-SEED FEATURE IMPORTANCE ANALYSIS")
    report_lines.append(f"Dataset: {dataset_type.upper()}")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Summary statistics
    total_combinations = 0
    successful_combinations = 0
    
    for (algo, strategy), (df, seeds) in all_results.items():
        total_combinations += 1
        if df is not None:
            successful_combinations += 1
    
    report_lines.append(f"Total Algorithm-Strategy Combinations Analyzed: {successful_combinations}/{total_combinations}")
    report_lines.append(f"Random Seeds Used: {SEEDS}")
    report_lines.append("")
    report_lines.append("-" * 80)
    
    # Per algorithm-strategy details
    for algo in ALGORITHMS:
        report_lines.append(f"\n{algo.upper()}")
        report_lines.append("-" * 40)
        
        for strategy in SAMPLING_STRATEGIES:
            if (algo, strategy) in all_results:
                df, seeds = all_results[(algo, strategy)]
                if df is not None and len(df) > 0:
                    report_lines.append(f"\n  Strategy: {strategy}")
                    report_lines.append(f"  Seeds Available: {len(seeds)} - {seeds}")
                    report_lines.append(f"  Total Features: {len(df)}")
                    report_lines.append(f"\n  Top 10 Features:")
                    
                    top_10 = df.head(10)
                    for idx, row in top_10.iterrows():
                        report_lines.append(
                            f"    {idx+1:2d}. {row['feature']:40s} "
                            f"Mean: {row['mean_importance']:.6f} "
                            f"Std: {row['std_importance']:.6f} "
                            f"CV: {row['cv_importance']:.4f}"
                        )
                else:
                    report_lines.append(f"\n  Strategy: {strategy} - No data available")
    
    report_lines.append("\n" + "=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)
    
    # Save report
    report_path = os.path.join(output_dir, 'feature_importance_summary_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    return report_path

def main(dataset_type):
    """Main aggregation workflow"""
    print(f"\n{'='*80}")
    print(f"AGGREGATING FEATURE IMPORTANCE FOR {dataset_type.upper()} DATASET")
    print(f"All 10 Seeds: {SEEDS}")
    print(f"{'='*80}\n")
    
    # Create output directory
    results_folder = f"{dataset_type.capitalize()}_Results"
    output_dir = os.path.join(results_folder, 'feature_importance_aggregated')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}\n")
    
    # Store all results
    all_results = {}
    
    # Process each algorithm and strategy
    for algo in ALGORITHMS:
        print(f"\n{'─'*60}")
        print(f"Processing: {algo.upper()}")
        print(f"{'─'*60}")
        
        for strategy in SAMPLING_STRATEGIES:
            print(f"\n  Strategy: {strategy}...", end=' ')
            
            df, seeds = aggregate_feature_importance_for_model(dataset_type, algo, strategy)
            
            if df is not None:
                print(f"✓ ({len(seeds)} seeds)")
                all_results[(algo, strategy)] = (df, seeds)
                
                # Save aggregated CSV
                csv_path = os.path.join(output_dir, f'aggregated_{algo}_{strategy}.csv')
                df.to_csv(csv_path, index=False)
                print(f"    Saved aggregated data: {csv_path}")
                
                # Create visualizations
                heatmap_path = create_importance_heatmap(df, algo, strategy, output_dir)
                if heatmap_path:
                    print(f"    Created heatmap: {heatmap_path}")
                
                barplot_path = create_top_features_barplot(df, algo, strategy, output_dir)
                if barplot_path:
                    print(f"    Created bar plot: {barplot_path}")
            else:
                print("✗ No data available")
    
    # Create cross-algorithm analyses
    print(f"\n{'='*80}")
    print("CREATING CROSS-ALGORITHM ANALYSES")
    print(f"{'='*80}\n")
    
    # Algorithm comparison
    print("Creating algorithm comparison...")
    comp_path = create_algorithm_comparison(all_results, dataset_type, output_dir)
    if comp_path:
        print(f"✓ Algorithm comparison saved: {comp_path}")
    
    # Consistency analysis
    print("Creating consistency analysis...")
    csv_path, viz_path = create_consistency_analysis(all_results, dataset_type, output_dir)
    print(f"✓ Consistency analysis saved:")
    print(f"  - CSV: {csv_path}")
    print(f"  - Visualization: {viz_path}")
    
    # Generate summary report
    print("\nGenerating summary report...")
    report_path = generate_summary_report(all_results, dataset_type, output_dir)
    print(f"✓ Summary report saved: {report_path}")
    
    print(f"\n{'='*80}")
    print("FEATURE IMPORTANCE AGGREGATION COMPLETE!")
    print(f"{'='*80}")
    print(f"\nAll results saved to: {output_dir}")
    print(f"Total files created: CSV files + Visualizations + Reports")
    
    return output_dir

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python aggregate_feature_importance.py [mix|single]")
        print("\nThis script aggregates feature importance across multiple seeds and algorithms.")
        print("It creates:")
        print("  - Aggregated CSV files with mean/std/CV for each feature")
        print("  - Heatmaps showing importance across seeds")
        print("  - Bar plots of top features with error bars")
        print("  - Cross-algorithm comparisons")
        print("  - Feature consistency analysis")
        print("  - Comprehensive summary report")
        sys.exit(1)
    
    dataset_type = sys.argv[1].lower()
    if dataset_type not in ['mix', 'single']:
        print("Error: Dataset type must be 'mix' or 'single'")
        sys.exit(1)
    
    main(dataset_type)
