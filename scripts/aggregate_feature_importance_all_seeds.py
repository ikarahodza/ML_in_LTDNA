"""
Aggregate Feature Importance Across All Seeds and Algorithms
This script aggregates feature importance across all seeds and algorithms for Mix_Results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from collections import defaultdict

# Configuration
SEEDS = [123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]
BASE_DIR = Path(r"c:\Users\DT User\Documents\GitHub\ML_in_LTDNA\Mix_Results")
OUTPUT_DIR = BASE_DIR / "multi_seed_aggregated" / "feature_importance"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Algorithm patterns to search for
ALGORITHMS = {
    'lgbm': 'lgbm',
    'xgboost': 'xgboost',
    'extratrees': 'extratrees'
}

SAMPLING_METHODS = ['original', 'smote', 'randomoversampler']

def collect_all_feature_importance_files():
    """Collect all feature importance CSV files from all seeds"""
    all_files = []
    
    for seed in SEEDS:
        seed_dir = BASE_DIR / f"seed_{seed}" / "figures"
        if not seed_dir.exists():
            print(f"Warning: {seed_dir} does not exist")
            continue
            
        # Find all feature importance CSV files
        fi_files = list(seed_dir.glob("feature_importance_*.csv"))
        
        for fi_file in fi_files:
            # Parse filename to extract algorithm and sampling method
            filename = fi_file.stem  # Remove .csv extension
            parts = filename.split('_')
            
            # Extract algorithm
            algorithm = None
            for algo_key, algo_name in ALGORITHMS.items():
                if algo_key in filename:
                    algorithm = algo_key
                    break
            
            # Extract sampling method
            sampling = None
            for method in SAMPLING_METHODS:
                if method in filename:
                    sampling = method
                    break
            
            if algorithm and sampling:
                all_files.append({
                    'path': fi_file,
                    'seed': seed,
                    'algorithm': algorithm,
                    'sampling': sampling,
                    'filename': filename
                })
    
    return all_files

def load_and_aggregate_feature_importance(files_info):
    """Load all feature importance files and aggregate them"""
    
    # Dictionary to store feature importances
    # Structure: {feature_name: [(seed, algorithm, sampling, importance), ...]}
    feature_data = defaultdict(list)
    
    # Load all files
    for file_info in files_info:
        try:
            df = pd.read_csv(file_info['path'])
            
            # Check if dataframe has required columns
            if 'feature' not in df.columns or 'importance' not in df.columns:
                print(f"Skipping {file_info['path']}: missing required columns")
                continue
            
            # Add to feature_data
            for _, row in df.iterrows():
                feature = row['feature']
                importance = row['importance']
                
                feature_data[feature].append({
                    'seed': file_info['seed'],
                    'algorithm': file_info['algorithm'],
                    'sampling': file_info['sampling'],
                    'importance': importance
                })
        
        except Exception as e:
            print(f"Error loading {file_info['path']}: {e}")
    
    return feature_data

def create_aggregated_dataframes(feature_data):
    """Create aggregated dataframes with statistics"""
    
    # Overall aggregation (across all seeds and algorithms)
    overall_stats = []
    for feature, data_list in feature_data.items():
        importances = [d['importance'] for d in data_list]
        overall_stats.append({
            'feature': feature,
            'mean_importance': np.mean(importances),
            'std_importance': np.std(importances),
            'median_importance': np.median(importances),
            'min_importance': np.min(importances),
            'max_importance': np.max(importances),
            'count': len(importances)
        })
    
    overall_df = pd.DataFrame(overall_stats).sort_values('mean_importance', ascending=False)
    
    # By algorithm aggregation
    by_algorithm = defaultdict(lambda: defaultdict(list))
    for feature, data_list in feature_data.items():
        for d in data_list:
            by_algorithm[d['algorithm']][feature].append(d['importance'])
    
    algorithm_dfs = {}
    for algorithm, features_dict in by_algorithm.items():
        algo_stats = []
        for feature, importances in features_dict.items():
            algo_stats.append({
                'feature': feature,
                'mean_importance': np.mean(importances),
                'std_importance': np.std(importances),
                'median_importance': np.median(importances),
                'count': len(importances)
            })
        algorithm_dfs[algorithm] = pd.DataFrame(algo_stats).sort_values('mean_importance', ascending=False)
    
    # By sampling method aggregation
    by_sampling = defaultdict(lambda: defaultdict(list))
    for feature, data_list in feature_data.items():
        for d in data_list:
            by_sampling[d['sampling']][feature].append(d['importance'])
    
    sampling_dfs = {}
    for sampling, features_dict in by_sampling.items():
        samp_stats = []
        for feature, importances in features_dict.items():
            samp_stats.append({
                'feature': feature,
                'mean_importance': np.mean(importances),
                'std_importance': np.std(importances),
                'median_importance': np.median(importances),
                'count': len(importances)
            })
        sampling_dfs[sampling] = pd.DataFrame(samp_stats).sort_values('mean_importance', ascending=False)
    
    return overall_df, algorithm_dfs, sampling_dfs

def create_visualizations(overall_df, algorithm_dfs, sampling_dfs):
    """Create comprehensive visualizations"""
    
    # 1. Top 20 Overall Features
    plt.figure(figsize=(12, 8))
    top_20 = overall_df.head(20)
    plt.barh(range(len(top_20)), top_20['mean_importance'], xerr=top_20['std_importance'])
    plt.yticks(range(len(top_20)), top_20['feature'])
    plt.xlabel('Mean Importance (± Std)', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title('Top 20 Features - Aggregated Across All Seeds and Algorithms', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'top_20_features_overall.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Top 30 Overall Features (extended view)
    plt.figure(figsize=(12, 10))
    top_30 = overall_df.head(30)
    plt.barh(range(len(top_30)), top_30['mean_importance'], xerr=top_30['std_importance'])
    plt.yticks(range(len(top_30)), top_30['feature'])
    plt.xlabel('Mean Importance (± Std)', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title('Top 30 Features - Aggregated Across All Seeds and Algorithms', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'top_30_features_overall.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Feature Importance by Algorithm (Top 15 from each)
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    for idx, (algo, df) in enumerate(algorithm_dfs.items()):
        top_15 = df.head(15)
        axes[idx].barh(range(len(top_15)), top_15['mean_importance'], xerr=top_15['std_importance'])
        axes[idx].set_yticks(range(len(top_15)))
        axes[idx].set_yticklabels(top_15['feature'], fontsize=9)
        axes[idx].set_xlabel('Mean Importance (± Std)', fontsize=10)
        axes[idx].set_title(f'{algo.upper()}', fontsize=12, fontweight='bold')
        axes[idx].invert_yaxis()
    
    plt.suptitle('Top 15 Features by Algorithm (Aggregated Across All Seeds)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'top_15_by_algorithm.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Feature Importance by Sampling Method (Top 15 from each)
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    for idx, (sampling, df) in enumerate(sampling_dfs.items()):
        top_15 = df.head(15)
        axes[idx].barh(range(len(top_15)), top_15['mean_importance'], xerr=top_15['std_importance'])
        axes[idx].set_yticks(range(len(top_15)))
        axes[idx].set_yticklabels(top_15['feature'], fontsize=9)
        axes[idx].set_xlabel('Mean Importance (± Std)', fontsize=10)
        axes[idx].set_title(f'{sampling.upper()}', fontsize=12, fontweight='bold')
        axes[idx].invert_yaxis()
    
    plt.suptitle('Top 15 Features by Sampling Method (Aggregated Across All Seeds)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'top_15_by_sampling.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Heatmap of Top Features Across Algorithms
    top_features = overall_df.head(25)['feature'].tolist()
    
    heatmap_data = []
    for feature in top_features:
        row = []
        for algo in ['lgbm', 'xgboost', 'extratrees']:
            if algo in algorithm_dfs:
                algo_df = algorithm_dfs[algo]
                feature_row = algo_df[algo_df['feature'] == feature]
                if not feature_row.empty:
                    row.append(feature_row['mean_importance'].values[0])
                else:
                    row.append(0)
            else:
                row.append(0)
        heatmap_data.append(row)
    
    plt.figure(figsize=(8, 12))
    heatmap_df = pd.DataFrame(heatmap_data, 
                               index=top_features, 
                               columns=['LGBM', 'XGBoost', 'ExtraTrees'])
    sns.heatmap(heatmap_df, annot=False, cmap='YlOrRd', fmt='.1f', cbar_kws={'label': 'Mean Importance'})
    plt.title('Top 25 Features Importance Heatmap by Algorithm', fontsize=14, fontweight='bold')
    plt.xlabel('Algorithm', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'heatmap_top25_by_algorithm.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Feature Stability Analysis (Coefficient of Variation)
    overall_df['cv'] = overall_df['std_importance'] / (overall_df['mean_importance'] + 1e-10)
    stable_features = overall_df.nsmallest(20, 'cv')
    
    plt.figure(figsize=(12, 8))
    colors = ['green' if cv < 0.3 else 'orange' if cv < 0.5 else 'red' for cv in stable_features['cv']]
    plt.barh(range(len(stable_features)), stable_features['mean_importance'], color=colors)
    plt.yticks(range(len(stable_features)), stable_features['feature'])
    plt.xlabel('Mean Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title('Top 20 Most Stable Features (Lowest Coefficient of Variation)', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='CV < 0.3 (Very Stable)'),
        Patch(facecolor='orange', label='0.3 ≤ CV < 0.5 (Moderate)'),
        Patch(facecolor='red', label='CV ≥ 0.5 (Variable)')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'most_stable_features.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved all visualizations to {OUTPUT_DIR}")

def save_results(overall_df, algorithm_dfs, sampling_dfs, files_info):
    """Save all aggregated results"""
    
    # Save overall aggregation
    overall_df.to_csv(OUTPUT_DIR / 'overall_feature_importance.csv', index=False)
    print(f"\n✓ Saved overall feature importance: {OUTPUT_DIR / 'overall_feature_importance.csv'}")
    
    # Save by algorithm
    for algo, df in algorithm_dfs.items():
        df.to_csv(OUTPUT_DIR / f'feature_importance_by_{algo}.csv', index=False)
        print(f"✓ Saved {algo} feature importance")
    
    # Save by sampling method
    for sampling, df in sampling_dfs.items():
        df.to_csv(OUTPUT_DIR / f'feature_importance_by_{sampling}.csv', index=False)
        print(f"✓ Saved {sampling} feature importance")
    
    # Save metadata
    metadata = {
        'seeds': SEEDS,
        'num_seeds': len(SEEDS),
        'algorithms': list(ALGORITHMS.keys()),
        'sampling_methods': SAMPLING_METHODS,
        'num_files_processed': len(files_info),
        'num_features': len(overall_df),
        'top_10_features': overall_df.head(10)[['feature', 'mean_importance', 'std_importance']].to_dict('records')
    }
    
    with open(OUTPUT_DIR / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"✓ Saved metadata: {OUTPUT_DIR / 'metadata.json'}")
    
    # Create summary report
    summary_lines = [
        "=" * 80,
        "FEATURE IMPORTANCE AGGREGATION SUMMARY",
        "=" * 80,
        f"\nSeeds Analyzed: {', '.join(map(str, SEEDS))}",
        f"Total Seeds: {len(SEEDS)}",
        f"Algorithms: {', '.join(ALGORITHMS.keys())}",
        f"Sampling Methods: {', '.join(SAMPLING_METHODS)}",
        f"Files Processed: {len(files_info)}",
        f"Unique Features: {len(overall_df)}",
        "\n" + "=" * 80,
        "TOP 20 FEATURES (Overall)",
        "=" * 80,
        overall_df.head(20).to_string(index=False),
        "\n" + "=" * 80,
        "TOP 10 MOST STABLE FEATURES (Lowest CV)",
        "=" * 80,
    ]
    
    overall_df['cv'] = overall_df['std_importance'] / (overall_df['mean_importance'] + 1e-10)
    stable = overall_df.nsmallest(10, 'cv')[['feature', 'mean_importance', 'std_importance', 'cv']]
    summary_lines.append(stable.to_string(index=False))
    
    summary_lines.extend([
        "\n" + "=" * 80,
        "TOP 10 FEATURES BY ALGORITHM",
        "=" * 80
    ])
    
    for algo, df in algorithm_dfs.items():
        summary_lines.append(f"\n{algo.upper()}:")
        summary_lines.append("-" * 80)
        summary_lines.append(df.head(10).to_string(index=False))
    
    summary_text = "\n".join(summary_lines)
    
    with open(OUTPUT_DIR / 'aggregation_summary.txt', 'w') as f:
        f.write(summary_text)
    
    print(f"✓ Saved summary report: {OUTPUT_DIR / 'aggregation_summary.txt'}")

def main():
    """Main execution function"""
    
    print("=" * 80)
    print("FEATURE IMPORTANCE AGGREGATION ACROSS ALL SEEDS")
    print("=" * 80)
    print(f"\nSeeds: {SEEDS}")
    print(f"Base Directory: {BASE_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("\n" + "=" * 80)
    
    # Step 1: Collect all feature importance files
    print("\n[1/5] Collecting feature importance files...")
    files_info = collect_all_feature_importance_files()
    print(f"Found {len(files_info)} feature importance files")
    
    if len(files_info) == 0:
        print("ERROR: No feature importance files found!")
        return
    
    # Step 2: Load and aggregate data
    print("\n[2/5] Loading and aggregating feature importance data...")
    feature_data = load_and_aggregate_feature_importance(files_info)
    print(f"Aggregated data for {len(feature_data)} unique features")
    
    # Step 3: Create aggregated dataframes
    print("\n[3/5] Creating aggregated statistics...")
    overall_df, algorithm_dfs, sampling_dfs = create_aggregated_dataframes(feature_data)
    print(f"Created overall aggregation with {len(overall_df)} features")
    print(f"Created algorithm-specific aggregations: {list(algorithm_dfs.keys())}")
    print(f"Created sampling-specific aggregations: {list(sampling_dfs.keys())}")
    
    # Step 4: Create visualizations
    print("\n[4/5] Creating visualizations...")
    create_visualizations(overall_df, algorithm_dfs, sampling_dfs)
    
    # Step 5: Save results
    print("\n[5/5] Saving results...")
    save_results(overall_df, algorithm_dfs, sampling_dfs, files_info)
    
    print("\n" + "=" * 80)
    print("AGGREGATION COMPLETE!")
    print("=" * 80)
    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print(f"\nTop 5 Features (Overall):")
    print(overall_df.head(5)[['feature', 'mean_importance', 'std_importance']].to_string(index=False))
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
