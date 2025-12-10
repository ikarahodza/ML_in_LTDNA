"""
Comprehensive Error Analysis
Analyzes misclassification patterns across all models to identify:
- Which classes are most confused
- Feature value ranges associated with errors
- Systematic biases in predictions
- Error patterns across different resampling strategies
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from collections import defaultdict

# Setup paths
data_dir = Path('Mix_Results/data')
figures_dir = Path('figures')
figures_dir.mkdir(exist_ok=True)

# Find all error analysis files
error_files = sorted(data_dir.glob('error_analysis_*.csv'))

print(f"Found {len(error_files)} error analysis files")
print("=" * 80)

# Dictionary to store analysis results
error_analysis_results = {}
all_errors_combined = []

# Analyze each model's errors
for error_file in error_files:
    model_name = error_file.stem.replace('error_analysis_', '')
    print(f"\nAnalyzing: {model_name}")
    
    # Load error data
    df = pd.read_csv(error_file)
    
    # Standardize column names
    if 'True_Label' in df.columns:
        df.rename(columns={'True_Label': 'Actual_Class'}, inplace=True)
    if 'Predicted_Label' in df.columns:
        df.rename(columns={'Predicted_Label': 'Predicted_Class'}, inplace=True)
    
    # Basic error statistics
    n_errors = len(df)
    
    print(f"  Total errors: {n_errors}")
    
    # Confusion matrix from errors
    if 'Actual_Class' in df.columns and 'Predicted_Class' in df.columns:
        confusion_pairs = df.groupby(['Actual_Class', 'Predicted_Class']).size()
        print(f"  Top confusion pairs:")
        for (actual, pred), count in confusion_pairs.nlargest(5).items():
            print(f"    {actual} -> {pred}: {count} errors")
        
        # Store confusion data
        error_analysis_results[model_name] = {
            'n_errors': n_errors,
            'confusion_pairs': confusion_pairs.to_dict()
        }
        
        # Add to combined dataset
        df['Model'] = model_name
        all_errors_combined.append(df)

print("\n" + "=" * 80)
print("AGGREGATE ANALYSIS ACROSS ALL MODELS")
print("=" * 80)

# Combine all errors
if all_errors_combined:
    combined_df = pd.concat(all_errors_combined, ignore_index=True)
    
    # 1. Most confused class pairs across all models
    print("\n1. MOST COMMONLY CONFUSED CLASS PAIRS (All Models)")
    overall_confusion = combined_df.groupby(['Actual_Class', 'Predicted_Class']).size()
    print("\nTop 10 confusion pairs:")
    for (actual, pred), count in overall_confusion.nlargest(10).items():
        print(f"  {actual} -> {pred}: {count} total errors")
    
    # 2. Error rate by actual class
    print("\n2. ERROR DISTRIBUTION BY ACTUAL CLASS")
    error_by_class = combined_df['Actual_Class'].value_counts()
    print("\nErrors per class:")
    for cls, count in error_by_class.items():
        print(f"  Class {cls}: {count} errors")
    
    # 3. Feature analysis for misclassified samples
    print("\n3. FEATURE VALUE ANALYSIS FOR ERRORS")
    
    # Get feature columns (exclude metadata columns)
    metadata_cols = ['Actual_Class', 'Predicted_Class', 'Model', 'Sample_Index', 'sample', 
                     'Classification', 'Correct', 'Prob_Allele', 'Prob_Back_Stutter', 
                     'Prob_Forward_Stutter', 'Prob_Pull_Up', 'Confidence']
    feature_cols = [col for col in combined_df.columns if col not in metadata_cols]
    
    if feature_cols:
        print(f"\nAnalyzing {len(feature_cols)} features...")
        
        # Calculate feature statistics for errors
        feature_stats = {}
        for feature in feature_cols[:10]:  # Analyze top 10 features
            if pd.api.types.is_numeric_dtype(combined_df[feature]):
                stats = {
                    'mean': combined_df[feature].mean(),
                    'std': combined_df[feature].std(),
                    'min': combined_df[feature].min(),
                    'max': combined_df[feature].max(),
                    'median': combined_df[feature].median()
                }
                feature_stats[feature] = stats
        
        print("\nFeature statistics for misclassified samples (top 10 features):")
        for feature, stats in list(feature_stats.items())[:10]:
            print(f"  {feature}:")
            print(f"    Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
            print(f"    Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
    
    # 4. Error patterns by model type
    print("\n4. ERROR PATTERNS BY MODEL")
    model_error_counts = combined_df['Model'].value_counts()
    print("\nTotal errors by model:")
    for model, count in model_error_counts.items():
        print(f"  {model}: {count} errors")
    
    # 5. Systematic bias analysis
    print("\n5. SYSTEMATIC BIAS ANALYSIS")
    
    # Check for directional bias (over-prediction vs under-prediction)
    if pd.api.types.is_numeric_dtype(combined_df['Actual_Class']) and \
       pd.api.types.is_numeric_dtype(combined_df['Predicted_Class']):
        
        prediction_bias = combined_df['Predicted_Class'] - combined_df['Actual_Class']
        over_predictions = (prediction_bias > 0).sum()
        under_predictions = (prediction_bias < 0).sum()
        
        print(f"\nPrediction bias:")
        print(f"  Over-predictions (predicted > actual): {over_predictions}")
        print(f"  Under-predictions (predicted < actual): {under_predictions}")
        print(f"  Mean bias: {prediction_bias.mean():.4f}")
    
    # Extract resampling strategy from model name
    def get_resampling_strategy(model_name):
        if 'smote' in model_name.lower():
            return 'SMOTE'
        elif 'randomoversampler' in model_name.lower():
            return 'RandomOverSampler'
        elif 'original' in model_name.lower():
            return 'Original'
        elif 'no_resampling' in model_name.lower():
            return 'No Resampling'
        elif 'class_weights' in model_name.lower():
            return 'Class Weights'
        elif 'borderline' in model_name.lower():
            return 'Borderline SMOTE'
        elif 'svm' in model_name.lower():
            return 'SVM SMOTE'
        else:
            return 'Unknown'
    
    combined_df['Resampling'] = combined_df['Model'].apply(get_resampling_strategy)
    
    # VISUALIZATIONS
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    # Visualization 1: Confusion heatmap (aggregate)
    print("\nCreating aggregate confusion heatmap...")
    confusion_matrix = pd.crosstab(
        combined_df['Actual_Class'], 
        combined_df['Predicted_Class'],
        normalize='index'
    )
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt='.2%', cmap='YlOrRd', 
                cbar_kws={'label': 'Percentage of Errors'})
    plt.title('Aggregate Error Confusion Matrix\n(Normalized by Actual Class)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('Actual Class', fontsize=12)
    plt.tight_layout()
    plt.savefig(figures_dir / 'error_analysis_confusion_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: error_analysis_confusion_heatmap.png")
    
    # Visualization 2: Error count by model
    print("\nCreating error count comparison...")
    plt.figure(figsize=(12, 8))
    model_errors = combined_df['Model'].value_counts().sort_values()
    colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.8, len(model_errors)))
    model_errors.plot(kind='barh', color=colors)
    plt.xlabel('Number of Errors', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.title('Total Errors by Model', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / 'error_analysis_by_model.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: error_analysis_by_model.png")
    
    # Visualization 3: Error distribution by actual class
    print("\nCreating error distribution by class...")
    plt.figure(figsize=(10, 6))
    class_errors = combined_df['Actual_Class'].value_counts().sort_index()
    plt.bar(range(len(class_errors)), class_errors.values, 
            color=sns.color_palette("husl", len(class_errors)))
    plt.xlabel('Actual Class', fontsize=12)
    plt.ylabel('Number of Errors', fontsize=12)
    plt.title('Error Distribution by Actual Class', fontsize=14, fontweight='bold')
    plt.xticks(range(len(class_errors)), class_errors.index)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / 'error_analysis_by_class.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: error_analysis_by_class.png")
    
    # Visualization 4: Confusion pairs frequency (top 15)
    print("\nCreating confusion pairs analysis...")
    top_confusions = overall_confusion.nlargest(15)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos = np.arange(len(top_confusions))
    confusion_labels = [f"{actual} -> {pred}" for (actual, pred) in top_confusions.index]
    
    colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(top_confusions)))
    ax.barh(y_pos, top_confusions.values, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(confusion_labels)
    ax.set_xlabel('Number of Errors', fontsize=12)
    ax.set_title('Top 15 Most Common Confusion Pairs', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / 'error_analysis_confusion_pairs.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: error_analysis_confusion_pairs.png")
    
    # Visualization 5: Error rate by resampling strategy
    print("\nCreating error rate comparison by resampling strategy...")
    
    resampling_errors = combined_df.groupby('Resampling').size().sort_values()
    
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("Set2", len(resampling_errors))
    resampling_errors.plot(kind='barh', color=colors)
    plt.xlabel('Number of Errors', fontsize=12)
    plt.ylabel('Resampling Strategy', fontsize=12)
    plt.title('Error Count by Resampling Strategy', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / 'error_analysis_by_resampling.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: error_analysis_by_resampling.png")
    
    # Visualization 6: Feature correlation with errors (if features available)
    if feature_cols and len(feature_cols) >= 5:
        print("\nCreating feature analysis for errors...")
        
        # Select top 10 features by variance
        numeric_features = [col for col in feature_cols if pd.api.types.is_numeric_dtype(combined_df[col])]
        if numeric_features:
            top_features = combined_df[numeric_features[:10]].var().nlargest(10).index.tolist()
            
            fig, axes = plt.subplots(2, 5, figsize=(20, 8))
            axes = axes.flatten()
            
            for idx, feature in enumerate(top_features):
                if idx < len(axes):
                    ax = axes[idx]
                    combined_df[feature].hist(bins=30, ax=ax, color='coral', edgecolor='black', alpha=0.7)
                    ax.set_title(feature, fontsize=10)
                    ax.set_xlabel('Value', fontsize=9)
                    ax.set_ylabel('Frequency', fontsize=9)
                    ax.grid(alpha=0.3)
            
            plt.suptitle('Feature Value Distributions for Misclassified Samples', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(figures_dir / 'error_analysis_feature_distributions.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("  Saved: error_analysis_feature_distributions.png")
    
    # Save summary report
    print("\n" + "=" * 80)
    print("SAVING SUMMARY REPORT")
    print("=" * 80)
    
    summary_path = figures_dir / 'error_analysis_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("COMPREHENSIVE ERROR ANALYSIS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total models analyzed: {len(error_files)}\n")
        f.write(f"Total errors across all models: {len(combined_df)}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("1. TOP 10 CONFUSION PAIRS\n")
        f.write("=" * 80 + "\n")
        for (actual, pred), count in overall_confusion.nlargest(10).items():
            f.write(f"{actual} -> {pred}: {count} errors\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("2. ERRORS BY ACTUAL CLASS\n")
        f.write("=" * 80 + "\n")
        for cls, count in error_by_class.items():
            f.write(f"Class {cls}: {count} errors\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("3. ERRORS BY MODEL\n")
        f.write("=" * 80 + "\n")
        for model, count in model_error_counts.items():
            f.write(f"{model}: {count} errors\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("4. ERRORS BY RESAMPLING STRATEGY\n")
        f.write("=" * 80 + "\n")
        for strategy, count in resampling_errors.items():
            f.write(f"{strategy}: {count} errors\n")
        
        if pd.api.types.is_numeric_dtype(combined_df['Actual_Class']) and \
           pd.api.types.is_numeric_dtype(combined_df['Predicted_Class']):
            prediction_bias = combined_df['Predicted_Class'] - combined_df['Actual_Class']
            over_predictions = (prediction_bias > 0).sum()
            under_predictions = (prediction_bias < 0).sum()
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("5. PREDICTION BIAS\n")
            f.write("=" * 80 + "\n")
            f.write(f"Over-predictions: {over_predictions}\n")
            f.write(f"Under-predictions: {under_predictions}\n")
            f.write(f"Mean bias: {prediction_bias.mean():.4f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("KEY INSIGHTS\n")
        f.write("=" * 80 + "\n")
        
        # Generate key insights
        top_confusion = overall_confusion.nlargest(1).iloc[0]
        top_confusion_pair = overall_confusion.nlargest(1).index[0]
        
        f.write(f"\n* Most common confusion: Class {top_confusion_pair[0]} -> Class {top_confusion_pair[1]} ({top_confusion} errors)\n")
        
        worst_model = model_error_counts.idxmax()
        best_model = model_error_counts.idxmin()
        f.write(f"* Model with most errors: {worst_model} ({model_error_counts[worst_model]} errors)\n")
        f.write(f"* Model with fewest errors: {best_model} ({model_error_counts[best_model]} errors)\n")
        
        most_confused_class = error_by_class.idxmax()
        f.write(f"* Most frequently misclassified class: Class {most_confused_class} ({error_by_class[most_confused_class]} errors)\n")
        
        best_resampling = resampling_errors.idxmin()
        worst_resampling = resampling_errors.idxmax()
        f.write(f"* Best resampling strategy (fewest errors): {best_resampling}\n")
        f.write(f"* Worst resampling strategy (most errors): {worst_resampling}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("=" * 80 + "\n")
        f.write(f"\n1. Focus model improvement on reducing {top_confusion_pair[0]} -> {top_confusion_pair[1]} confusion\n")
        f.write(f"2. Investigate why {worst_model} has significantly more errors\n")
        f.write(f"3. Consider using {best_resampling} resampling for future models\n")
        f.write(f"4. Pay special attention to Class {most_confused_class} predictions\n")
        f.write("5. Review feature engineering opportunities based on error patterns\n")
    
    print(f"\nSummary saved to: {summary_path}")
    
    # Save detailed JSON results
    json_results = {
        'total_models': len(error_files),
        'total_errors': len(combined_df),
        'top_confusion_pairs': {f"{k[0]}_to_{k[1]}": int(v) for k, v in overall_confusion.nlargest(10).items()},
        'errors_by_class': {str(k): int(v) for k, v in error_by_class.items()},
        'errors_by_model': {k: int(v) for k, v in model_error_counts.items()},
        'errors_by_resampling': {k: int(v) for k, v in resampling_errors.items()},
        'best_model': best_model,
        'worst_model': worst_model,
        'most_confused_class': str(most_confused_class),
        'best_resampling_strategy': best_resampling,
        'worst_resampling_strategy': worst_resampling
    }
    
    json_path = figures_dir / 'error_analysis_results.json'
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Detailed results saved to: {json_path}")

print("\n" + "=" * 80)
print("ERROR ANALYSIS COMPLETE!")
print("=" * 80)
print("\nGenerated visualizations:")
print("  * error_analysis_confusion_heatmap.png")
print("  * error_analysis_by_model.png")
print("  * error_analysis_by_class.png")
print("  * error_analysis_confusion_pairs.png")
print("  * error_analysis_by_resampling.png")
print("  * error_analysis_feature_distributions.png")
print("\nGenerated reports:")
print("  * error_analysis_summary.txt")
print("  * error_analysis_results.json")
