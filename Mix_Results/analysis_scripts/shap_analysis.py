"""
SHAP Analysis for Feature Engineering
Analyzes SHAP plots to identify:
1. Most important features across models
2. Feature interactions
3. Non-linear relationships
4. Opportunities for feature engineering
"""

import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Setup paths
figures_dir = Path('Mix_Results/figures')
shap_output_dir = Path('figures')
shap_output_dir.mkdir(exist_ok=True)

# Since SHAP plots are images, we'll analyze the feature importance from the model results
# Load combined results to get feature importance data
results_path = Path('Mix_Results/reports/combined_results.json')

print("=" * 80)
print("SHAP ANALYSIS FOR FEATURE ENGINEERING")
print("=" * 80)

if not results_path.exists():
    print(f"\nError: Could not find {results_path}")
    print("Please ensure combined_results.json exists")
    exit(1)

with open(results_path, 'r') as f:
    results = json.load(f)

# Find all feature importance CSV files
feature_importance_files = sorted(Path('Mix_Results/figures').glob('feature_importance_*_importances_*.csv'))

print(f"\nFound {len(feature_importance_files)} feature importance files")

# Dictionary to store feature importance across models
all_feature_importance = {}

# Load and analyze feature importance
for fi_file in feature_importance_files:
    model_name = fi_file.stem.replace('feature_importance_', '').replace('_importances_feature_importances_', '')
    print(f"\nAnalyzing: {model_name}")
    
    df = pd.read_csv(fi_file)
    
    # Store top features
    if len(df) > 0:
        all_feature_importance[model_name] = df.head(20).to_dict('records')
        print(f"  Top 5 features:")
        for idx, row in df.head(5).iterrows():
            if 'feature' in row and 'importance' in row:
                print(f"    {row['feature']}: {row['importance']:.4f}")

# Aggregate feature importance across models
print("\n" + "=" * 80)
print("AGGREGATED FEATURE IMPORTANCE")
print("=" * 80)

if all_feature_importance:
    # Create a combined dataframe
    combined_importance = []
    for model, features in all_feature_importance.items():
        for feat_dict in features:
            if 'feature' in feat_dict and 'importance' in feat_dict:
                combined_importance.append({
                    'Model': model,
                    'Feature': feat_dict['feature'],
                    'Importance': feat_dict['importance']
                })
    
    if combined_importance:
        df_combined = pd.DataFrame(combined_importance)
        
        # Calculate mean importance per feature
        feature_avg_importance = df_combined.groupby('Feature')['Importance'].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False)
        
        print("\nTop 20 Features by Average Importance Across Models:")
        print("=" * 80)
        for idx, (feature, row) in enumerate(feature_avg_importance.head(20).iterrows(), 1):
            print(f"{idx:2d}. {feature:30s} Mean: {row['mean']:.6f} | Std: {row['std']:.6f} | Models: {int(row['count'])}")
        
        # Identify highly consistent features (low std, high importance)
        consistent_features = feature_avg_importance[(feature_avg_importance['mean'] > 0.01) & 
                                                    (feature_avg_importance['std'] < 0.01)]
        
        print("\n" + "=" * 80)
        print(f"HIGHLY CONSISTENT FEATURES ({len(consistent_features)} features)")
        print("(High importance, low variance across models)")
        print("=" * 80)
        for idx, (feature, row) in enumerate(consistent_features.head(10).iterrows(), 1):
            print(f"{idx:2d}. {feature:30s} Mean: {row['mean']:.6f} | Std: {row['std']:.6f}")
        
        # Identify features with high variance (model-specific importance)
        variable_features = feature_avg_importance[(feature_avg_importance['mean'] > 0.01) & 
                                                  (feature_avg_importance['std'] > 0.02)]
        
        print("\n" + "=" * 80)
        print(f"MODEL-SPECIFIC FEATURES ({len(variable_features)} features)")
        print("(High importance variance across models)")
        print("=" * 80)
        for idx, (feature, row) in enumerate(variable_features.head(10).iterrows(), 1):
            print(f"{idx:2d}. {feature:30s} Mean: {row['mean']:.6f} | Std: {row['std']:.6f}")
        
        # Visualizations
        print("\n" + "=" * 80)
        print("GENERATING VISUALIZATIONS")
        print("=" * 80)
        
        # Visualization 1: Top 20 features by average importance
        print("\nCreating top features bar chart...")
        plt.figure(figsize=(12, 8))
        top_20 = feature_avg_importance.head(20)
        colors = plt.cm.viridis(range(len(top_20)))
        
        plt.barh(range(len(top_20)), top_20['mean'].values, xerr=top_20['std'].values, 
                color=colors, edgecolor='black', alpha=0.8)
        plt.yticks(range(len(top_20)), top_20.index, fontsize=9)
        plt.xlabel('Average Feature Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title('Top 20 Features by Average Importance Across Models', fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(shap_output_dir / 'shap_analysis_top_features.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: shap_analysis_top_features.png")
        
        # Visualization 2: Feature importance heatmap across models
        print("\nCreating feature importance heatmap...")
        
        # Pivot to create model x feature matrix
        pivot_df = df_combined.pivot_table(index='Model', columns='Feature', values='Importance', fill_value=0)
        
        # Select top 30 features by average importance
        top_30_features = feature_avg_importance.head(30).index.tolist()
        pivot_top = pivot_df[top_30_features]
        
        plt.figure(figsize=(16, 10))
        sns.heatmap(pivot_top, cmap='YlOrRd', cbar_kws={'label': 'Importance'}, 
                   linewidths=0.5, fmt='.3f')
        plt.title('Feature Importance Heatmap (Top 30 Features)', fontsize=14, fontweight='bold')
        plt.xlabel('Feature', fontsize=12)
        plt.ylabel('Model', fontsize=12)
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        plt.savefig(shap_output_dir / 'shap_analysis_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: shap_analysis_heatmap.png")
        
        # Visualization 3: Feature importance consistency
        print("\nCreating feature consistency scatter plot...")
        plt.figure(figsize=(12, 8))
        
        scatter_data = feature_avg_importance[feature_avg_importance['mean'] > 0.001]
        plt.scatter(scatter_data['mean'], scatter_data['std'], 
                   s=100, alpha=0.6, c=range(len(scatter_data)), cmap='viridis')
        
        # Label top 10 features
        for idx, (feature, row) in enumerate(scatter_data.head(10).iterrows()):
            plt.annotate(feature, (row['mean'], row['std']), 
                        fontsize=8, alpha=0.7,
                        xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('Mean Feature Importance', fontsize=12)
        plt.ylabel('Standard Deviation of Importance', fontsize=12)
        plt.title('Feature Importance: Consistency vs Magnitude', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(shap_output_dir / 'shap_analysis_consistency.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: shap_analysis_consistency.png")
        
        # Visualization 4: Feature category analysis
        print("\nCreating feature category analysis...")
        
        # Categorize features
        def categorize_feature(feature_name):
            if 'PoI' in feature_name:
                return 'PoI'
            elif 'UpStream' in feature_name:
                return 'Upstream'
            elif 'DownStream' in feature_name:
                return 'Downstream'
            elif feature_name.startswith('C') and '_' in feature_name:
                return 'Context'
            else:
                return 'Other'
        
        df_combined['Category'] = df_combined['Feature'].apply(categorize_feature)
        category_importance = df_combined.groupby('Category')['Importance'].agg(['mean', 'sum', 'count'])
        category_importance = category_importance.sort_values('mean', ascending=False)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Mean importance by category
        ax1 = axes[0]
        colors = sns.color_palette("Set2", len(category_importance))
        ax1.bar(category_importance.index, category_importance['mean'], color=colors, edgecolor='black')
        ax1.set_xlabel('Feature Category', fontsize=12)
        ax1.set_ylabel('Mean Importance', fontsize=12)
        ax1.set_title('Mean Feature Importance by Category', fontsize=12, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Total contribution by category
        ax2 = axes[1]
        ax2.bar(category_importance.index, category_importance['sum'], color=colors, edgecolor='black')
        ax2.set_xlabel('Feature Category', fontsize=12)
        ax2.set_ylabel('Total Importance', fontsize=12)
        ax2.set_title('Total Feature Importance by Category', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Feature Category Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(shap_output_dir / 'shap_analysis_categories.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: shap_analysis_categories.png")

# Generate insights document
print("\n" + "=" * 80)
print("GENERATING INSIGHTS DOCUMENT")
print("=" * 80)

insights_path = shap_output_dir / 'shap_analysis_insights.txt'
with open(insights_path, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("SHAP ANALYSIS INSIGHTS FOR FEATURE ENGINEERING\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("=" * 80 + "\n")
    f.write("1. TOP 20 MOST IMPORTANT FEATURES\n")
    f.write("=" * 80 + "\n")
    for idx, (feature, row) in enumerate(feature_avg_importance.head(20).iterrows(), 1):
        f.write(f"{idx:2d}. {feature:30s} Mean: {row['mean']:.6f} | Std: {row['std']:.6f} | Models: {int(row['count'])}\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("2. HIGHLY CONSISTENT FEATURES\n")
    f.write("(High importance, low variance - reliable across models)\n")
    f.write("=" * 80 + "\n")
    if len(consistent_features) > 0:
        for idx, (feature, row) in enumerate(consistent_features.head(15).iterrows(), 1):
            f.write(f"{idx:2d}. {feature:30s} Mean: {row['mean']:.6f} | Std: {row['std']:.6f}\n")
    else:
        f.write("No features meet the consistency criteria.\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("3. MODEL-SPECIFIC FEATURES\n")
    f.write("(High variance - important for specific models)\n")
    f.write("=" * 80 + "\n")
    if len(variable_features) > 0:
        for idx, (feature, row) in enumerate(variable_features.head(15).iterrows(), 1):
            f.write(f"{idx:2d}. {feature:30s} Mean: {row['mean']:.6f} | Std: {row['std']:.6f}\n")
    else:
        f.write("No features meet the variability criteria.\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("4. FEATURE CATEGORY ANALYSIS\n")
    f.write("=" * 80 + "\n")
    for category, row in category_importance.iterrows():
        f.write(f"{category:15s} Mean: {row['mean']:.6f} | Total: {row['sum']:.4f} | Count: {int(row['count'])}\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("5. FEATURE ENGINEERING OPPORTUNITIES\n")
    f.write("=" * 80 + "\n\n")
    
    # Identify top features for interaction terms
    top_features = feature_avg_importance.head(10).index.tolist()
    
    f.write("A. INTERACTION TERMS (Top Feature Pairs)\n")
    f.write("-" * 80 + "\n")
    f.write("Create multiplicative interactions between top features:\n\n")
    for i in range(min(5, len(top_features))):
        for j in range(i+1, min(i+3, len(top_features))):
            f.write(f"  * {top_features[i]} x {top_features[j]}\n")
    
    f.write("\n\nB. POLYNOMIAL FEATURES\n")
    f.write("-" * 80 + "\n")
    f.write("Create squared and cubic terms for top non-linear features:\n\n")
    for feat in top_features[:5]:
        f.write(f"  * {feat}^2\n")
        f.write(f"  * {feat}^3\n")
    
    f.write("\n\nC. RATIO FEATURES\n")
    f.write("-" * 80 + "\n")
    f.write("Create ratio features between related features:\n\n")
    
    # Identify related feature groups
    upstream_features = [f for f in top_features if 'UpStream' in f]
    downstream_features = [f for f in top_features if 'DownStream' in f]
    
    if upstream_features and downstream_features:
        f.write(f"  * {upstream_features[0]} / {downstream_features[0]}\n")
    
    if len(top_features) >= 2:
        f.write(f"  * {top_features[0]} / {top_features[1]}\n")
        f.write(f"  * {top_features[0]} / mean(top_5_features)\n")
    
    f.write("\n\nD. AGGREGATION FEATURES\n")
    f.write("-" * 80 + "\n")
    f.write("Create summary statistics across feature groups:\n\n")
    
    if upstream_features:
        f.write("  * Mean of all UpStream features\n")
        f.write("  * Std of all UpStream features\n")
        f.write("  * Max-Min range of UpStream features\n")
    
    if downstream_features:
        f.write("  * Mean of all DownStream features\n")
        f.write("  * Std of all DownStream features\n")
        f.write("  * Max-Min range of DownStream features\n")
    
    f.write("\n\nE. CONTEXT-BASED FEATURES\n")
    f.write("-" * 80 + "\n")
    f.write("Create features based on local context:\n\n")
    f.write("  * Sum of upstream + downstream around PoI\n")
    f.write("  * Difference between upstream and downstream\n")
    f.write("  * Ratio of PoI to local average\n")
    
    f.write("\n\n" + "=" * 80 + "\n")
    f.write("6. RECOMMENDATIONS\n")
    f.write("=" * 80 + "\n\n")
    
    top_3_features = feature_avg_importance.head(3).index.tolist()
    
    f.write(f"1. PRIORITY INTERACTIONS: Focus on interactions between top 3 features:\n")
    for i, feat in enumerate(top_3_features, 1):
        f.write(f"   - {feat}\n")
    
    f.write(f"\n2. CONSISTENT FEATURES: The {len(consistent_features)} consistent features should be\n")
    f.write("   retained in any feature selection process.\n")
    
    f.write(f"\n3. MODEL-SPECIFIC: Consider ensemble methods that leverage the {len(variable_features)}\n")
    f.write("   model-specific features for different prediction scenarios.\n")
    
    most_important_category = category_importance.index[0]
    f.write(f"\n4. CATEGORY FOCUS: {most_important_category} features are most important overall.\n")
    f.write("   Consider engineering more features in this category.\n")
    
    f.write("\n5. NON-LINEAR RELATIONSHIPS: Based on SHAP plots, test polynomial\n")
    f.write("   transformations (square, cube) on top features.\n")
    
    f.write("\n6. VALIDATION: After creating new features, re-run cross-validation\n")
    f.write("   to measure performance improvement and avoid overfitting.\n")

print(f"\nInsights document saved to: {insights_path}")

# Save JSON summary
json_summary = {
    'top_20_features': [
        {'feature': feat, 'mean_importance': float(row['mean']), 
         'std_importance': float(row['std']), 'model_count': int(row['count'])}
        for feat, row in feature_avg_importance.head(20).iterrows()
    ],
    'consistent_features': [
        {'feature': feat, 'mean_importance': float(row['mean']), 
         'std_importance': float(row['std'])}
        for feat, row in consistent_features.head(15).iterrows()
    ] if len(consistent_features) > 0 else [],
    'variable_features': [
        {'feature': feat, 'mean_importance': float(row['mean']), 
         'std_importance': float(row['std'])}
        for feat, row in variable_features.head(15).iterrows()
    ] if len(variable_features) > 0 else [],
    'category_analysis': {
        category: {'mean': float(row['mean']), 'sum': float(row['sum']), 'count': int(row['count'])}
        for category, row in category_importance.iterrows()
    },
    'feature_engineering_suggestions': {
        'interaction_pairs': [(top_features[i], top_features[j]) 
                             for i in range(min(5, len(top_features))) 
                             for j in range(i+1, min(i+3, len(top_features)))],
        'polynomial_features': top_features[:5],
        'ratio_features': [(top_features[i], top_features[j]) 
                          for i in range(min(3, len(top_features))) 
                          for j in range(i+1, min(3, len(top_features)))]
    }
}

json_path = shap_output_dir / 'shap_analysis_results.json'
with open(json_path, 'w') as f:
    json.dump(json_summary, f, indent=2)

print(f"JSON summary saved to: {json_path}")

print("\n" + "=" * 80)
print("SHAP ANALYSIS COMPLETE!")
print("=" * 80)
print("\nGenerated files:")
print("  * shap_analysis_top_features.png")
print("  * shap_analysis_heatmap.png")
print("  * shap_analysis_consistency.png")
print("  * shap_analysis_categories.png")
print("  * shap_analysis_insights.txt")
print("  * shap_analysis_results.json")
print("\nNext steps:")
print("  1. Review shap_analysis_insights.txt for feature engineering ideas")
print("  2. Implement suggested feature interactions")
print("  3. Test new features with cross-validation")
print("  4. Compare performance before and after feature engineering")
