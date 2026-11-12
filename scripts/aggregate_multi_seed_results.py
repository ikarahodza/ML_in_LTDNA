"""
Multi-Seed Results Aggregator
Aggregates results across multiple random seeds and computes statistics.
Merges in TabPFN v3.5 results (combined_results_v35.json) alongside the
non-TabPFN models (combined_results.json).
Usage: python aggregate_multi_seed_results.py [mix|single]
"""
import json
import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict

# Configuration
SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]
METRICS_TO_AGGREGATE = [
    'accuracy', 'balanced_accuracy', 'f1_macro', 'log_loss',
    'focal_loss', 'weighted_log_loss', 'mse', 'mae',
    'informedness', 'markedness'
]

def load_seed_results(dataset_type, seed):
    """Load results for a specific seed, merging in TabPFN v3.5 results"""
    results_folder = f"{dataset_type.capitalize()}_Results_PoI"

    # Seed 42 uses default location (no seed subfolder)
    if seed == 42:
        json_path = os.path.join(results_folder, 'reports', 'combined_results.json')
        tabpfn_json_path = os.path.join(results_folder, 'reports', 'combined_results_v35.json')
    else:
        seed_folder = f"seed_{seed}"
        json_path = os.path.join(results_folder, seed_folder, 'reports', 'combined_results.json')
        tabpfn_json_path = os.path.join(results_folder, f"{seed_folder}_v35", 'reports', 'combined_results_v35.json')

    if not os.path.exists(json_path):
        print(f"⚠ Warning: Results not found for seed {seed}: {json_path}")
        return None

    with open(json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    if os.path.exists(tabpfn_json_path):
        with open(tabpfn_json_path, 'r', encoding='utf-8') as f:
            results.update(json.load(f))
    else:
        print(f"⚠ Warning: TabPFN v3.5 results not found for seed {seed}: {tabpfn_json_path}")

    return results

def aggregate_results(dataset_type):
    """Aggregate results across all seeds for a dataset"""
    print(f"\n{'='*80}")
    print(f"Aggregating results for {dataset_type.upper()} dataset")
    print(f"{'='*80}\n")
    
    # Collect all results across seeds
    all_results = {}
    available_seeds = []
    
    for seed in SEEDS:
        results = load_seed_results(dataset_type, seed)
        if results is not None:
            all_results[seed] = results
            available_seeds.append(seed)
    
    if not all_results:
        print(f"✗ No results found for {dataset_type} dataset")
        return None
    
    print(f"✓ Loaded results for {len(available_seeds)} seeds: {available_seeds}\n")
    
    # Structure: {model: {strategy: {metric: [values across seeds]}}}
    aggregated = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    # Collect values across seeds
    for seed, seed_results in all_results.items():
        for model_name, model_data in seed_results.items():
            if 'results' not in model_data:
                continue
            
            for strategy, metrics in model_data['results'].items():
                for metric in METRICS_TO_AGGREGATE:
                    if metric in metrics and metrics[metric] is not None:
                        aggregated[model_name][strategy][metric].append(metrics[metric])
    
    # Compute statistics
    summary = {}
    detailed_stats = []
    
    for model_name in sorted(aggregated.keys()):
        summary[model_name] = {}
        
        for strategy in sorted(aggregated[model_name].keys()):
            summary[model_name][strategy] = {}
            
            for metric in METRICS_TO_AGGREGATE:
                values = aggregated[model_name][strategy][metric]
                
                if len(values) == 0:
                    summary[model_name][strategy][metric] = {
                        'mean': None,
                        'std': None,
                        'min': None,
                        'max': None,
                        'n_seeds': 0
                    }
                else:
                    values_array = np.array(values)
                    summary[model_name][strategy][metric] = {
                        'mean': float(np.mean(values_array)),
                        'std': float(np.std(values_array, ddof=1 if len(values) > 1 else 0)),
                        'min': float(np.min(values_array)),
                        'max': float(np.max(values_array)),
                        'n_seeds': len(values)
                    }
                    
                    # Add to detailed stats for CSV
                    detailed_stats.append({
                        'model': model_name,
                        'strategy': strategy,
                        'metric': metric,
                        'mean': float(np.mean(values_array)),
                        'std': float(np.std(values_array, ddof=1 if len(values) > 1 else 0)),
                        'min': float(np.min(values_array)),
                        'max': float(np.max(values_array)),
                        'n_seeds': len(values)
                    })
    
    # Save results
    results_folder = f"{dataset_type.capitalize()}_Results_PoI"
    output_dir = os.path.join(results_folder, 'multi_seed_aggregated')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save JSON summary
    json_path = os.path.join(output_dir, 'aggregated_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved JSON summary: {json_path}")
    
    # Save CSV with detailed stats
    csv_path = os.path.join(output_dir, 'aggregated_results.csv')
    df = pd.DataFrame(detailed_stats)
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved CSV: {csv_path}")
    
    # Create readable summary text
    summary_text_path = os.path.join(output_dir, 'aggregated_results_summary.txt')
    with open(summary_text_path, 'w', encoding='utf-8') as f:
        f.write("MULTI-SEED RESULTS SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Dataset: {dataset_type.upper()}\n")
        f.write(f"Seeds used: {available_seeds}\n")
        f.write(f"Number of seeds: {len(available_seeds)}\n\n")
        
        for model_name in sorted(summary.keys()):
            f.write(f"\n{'='*80}\n")
            f.write(f"MODEL: {model_name}\n")
            f.write(f"{'='*80}\n\n")
            
            for strategy in sorted(summary[model_name].keys()):
                f.write(f"  Strategy: {strategy}\n")
                f.write(f"  {'-'*76}\n")
                
                for metric in METRICS_TO_AGGREGATE:
                    stats = summary[model_name][strategy][metric]
                    if stats['mean'] is not None:
                        f.write(f"    {metric:25s}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                               f"(min: {stats['min']:.4f}, max: {stats['max']:.4f}, n={stats['n_seeds']})\n")
                    else:
                        f.write(f"    {metric:25s}: N/A\n")
                f.write("\n")
    
    print(f"✓ Saved summary: {summary_text_path}")
    
    # Print top performers
    print(f"\n{'='*80}")
    print("TOP PERFORMERS (by mean balanced accuracy)")
    print(f"{'='*80}\n")
    
    rankings = []
    for model_name, strategies in summary.items():
        for strategy, metrics in strategies.items():
            bal_acc = metrics.get('balanced_accuracy', {})
            if bal_acc.get('mean') is not None:
                rankings.append({
                    'model': model_name,
                    'strategy': strategy,
                    'bal_acc_mean': bal_acc['mean'],
                    'bal_acc_std': bal_acc['std'],
                    'f1_mean': metrics.get('f1_macro', {}).get('mean', 0),
                    'f1_std': metrics.get('f1_macro', {}).get('std', 0)
                })
    
    rankings.sort(key=lambda x: x['bal_acc_mean'], reverse=True)
    
    for i, entry in enumerate(rankings[:10], 1):
        print(f"{i:2d}. {entry['model']:15s} | {entry['strategy']:20s} | "
              f"Bal Acc: {entry['bal_acc_mean']:.4f}±{entry['bal_acc_std']:.4f} | "
              f"F1: {entry['f1_mean']:.4f}±{entry['f1_std']:.4f}")
    
    return summary

def main():
    if len(sys.argv) > 1:
        dataset_type = sys.argv[1].lower()
        if dataset_type not in ['mix', 'single']:
            print(f"Error: Invalid dataset type '{dataset_type}'. Use 'mix' or 'single'.")
            sys.exit(1)
        datasets = [dataset_type]
    else:
        # Process both datasets
        datasets = ['mix', 'single']
    
    print("="*80)
    print("MULTI-SEED RESULTS AGGREGATOR")
    print("="*80)
    print(f"Datasets to process: {datasets}")
    print(f"Seeds to aggregate: {SEEDS}")
    print("="*80)
    
    for dataset in datasets:
        aggregate_results(dataset)
    
    print(f"\n{'='*80}")
    print("✓ AGGREGATION COMPLETE")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
