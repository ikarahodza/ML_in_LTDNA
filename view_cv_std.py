"""
View cross-validation results with standard deviations for all models
Shows model stability/variance across different data splits
"""
import json
import pandas as pd

def show_cv_results(dataset='Mix'):
    print("=" * 100)
    print(f"CROSS-VALIDATION RESULTS - {dataset.upper()} DATASET")
    print("Standard Deviation indicates model stability across 10 folds")
    print("=" * 100)
    
    # Load CV results
    cv_file = f'{dataset}_Results/cross_validation_results/cv_results.json'
    try:
        with open(cv_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"\nNo CV results found for {dataset} dataset.")
        print(f"Run: python {dataset}_Results\\analysis_scripts\\cross_validation.py")
        return
    
    # Create DataFrame
    results = []
    for model, model_data in data.items():
        metrics = model_data['metrics']
        results.append({
            'Model': model,
            'Mean Bal Acc': metrics['balanced_accuracy']['mean'],
            'Std Dev': metrics['balanced_accuracy']['std'],
            'Min': metrics['balanced_accuracy']['min'],
            'Max': metrics['balanced_accuracy']['max'],
            'Mean F1': metrics['f1_macro']['mean'],
            'Std F1': metrics['f1_macro']['std']
        })
    
    df = pd.DataFrame(results)
    df = df.sort_values('Mean Bal Acc', ascending=False)
    
    print("\nAll Models (sorted by mean balanced accuracy):\n")
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_columns', None)
    print(df.to_string(index=False))
    
    print("\n" + "=" * 100)
    print("STABILITY ANALYSIS:")
    print("=" * 100)
    
    # Most stable models
    most_stable = df.nsmallest(3, 'Std Dev')
    print("\nMost Stable Models (lowest std dev):")
    for _, row in most_stable.iterrows():
        print(f"  {row['Model']}: {row['Mean Bal Acc']:.4f} ± {row['Std Dev']:.4f}")
    
    # Least stable models
    least_stable = df.nlargest(3, 'Std Dev')
    print("\nLeast Stable Models (highest std dev):")
    for _, row in least_stable.iterrows():
        print(f"  {row['Model']}: {row['Mean Bal Acc']:.4f} ± {row['Std Dev']:.4f}")
    
    # Best performers with good stability
    stable_performers = df[(df['Std Dev'] < df['Std Dev'].mean()) & 
                           (df['Mean Bal Acc'] > df['Mean Bal Acc'].quantile(0.5))]
    print(f"\nBest Stable Performers (low variance, high accuracy - {len(stable_performers)} models):")
    for _, row in stable_performers.iterrows():
        print(f"  {row['Model']}: {row['Mean Bal Acc']:.4f} ± {row['Std Dev']:.4f}")
    
    print("\n" + "=" * 100)
    print("INTERPRETATION:")
    print("=" * 100)
    print("""
- Lower Std Dev = More consistent/stable across different data splits
- Higher Std Dev = More sensitive to training data, potential overfitting
- Good model: High mean + Low std dev (consistent high performance)
- Range (Max - Min) shows best/worst case performance
""")

# Show both datasets
print("\n")
show_cv_results('Mix')
print("\n\n")
show_cv_results('Single')

print("\n" + "=" * 100)
print("TO RUN CV ON MORE MODELS:")
print("=" * 100)
print("python Mix_Results\\analysis_scripts\\cross_validation.py")
print("python Single_Results\\analysis_scripts\\cross_validation.py")
print("\nVisualization files:")
print("  - Mix_Results/cross_validation_results/cv_boxplots.png")
print("  - Single_Results/cross_validation_results/cv_boxplots.png")
