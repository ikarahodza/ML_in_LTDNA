"""
Comprehensive Visualization of Cross-Validation and Multi-Seed Results
Creates detailed interactive and static visualizations for model validation.

Features:
- Cross-validation performance comparisons
- Multi-seed stability analysis
- Model rankings with confidence intervals
- Strategy comparisons
- Interactive HTML dashboards
- Publication-ready plots
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATASET_TYPE = sys.argv[1] if len(sys.argv) > 1 else 'mix'
DATASET_TYPE = DATASET_TYPE.lower()

if DATASET_TYPE == 'mix':
    CV_RESULTS = 'cross_validation_results/cv_results.json'
    MULTI_SEED_RESULTS = 'Mix_Results/multi_seed_aggregated/aggregated_results.json'
    OUTPUT_DIR = 'Mix_Results/validation_visualizations'
elif DATASET_TYPE == 'single':
    CV_RESULTS = 'Single_Results/cross_validation_results/cv_results.json'
    MULTI_SEED_RESULTS = 'Single_Results/multi_seed_aggregated/aggregated_results.json'
    OUTPUT_DIR = 'Single_Results/validation_visualizations'
else:
    print(f"Error: Invalid dataset type '{DATASET_TYPE}'. Use 'mix' or 'single'")
    sys.exit(1)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300


def load_cv_results():
    """Load cross-validation results."""
    try:
        with open(CV_RESULTS, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"   ⚠ CV results not found: {CV_RESULTS}")
        return None


def load_multi_seed_results():
    """Load multi-seed aggregated results."""
    try:
        with open(MULTI_SEED_RESULTS, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"   ⚠ Multi-seed results not found: {MULTI_SEED_RESULTS}")
        return None


def parse_cv_results(cv_data):
    """Parse CV results into DataFrame."""
    rows = []
    for model_name, model_data in cv_data.items():
        ba = model_data['metrics']['balanced_accuracy']
        rows.append({
            'model': model_name,
            'type': model_data.get('type', 'unknown'),
            'resampling': model_data.get('resampling', 'unknown'),
            'mean': ba['mean'],
            'std': ba['std'],
            'min': ba['min'],
            'max': ba['max']
        })
    
    df = pd.DataFrame(rows).sort_values('mean', ascending=False)
    return df


def parse_multi_seed_results(ms_data):
    """Parse multi-seed results into DataFrame."""
    rows = []
    for model_name, strategies in ms_data.items():
        for strategy_name, metrics in strategies.items():
            ba = metrics['balanced_accuracy']
            rows.append({
                'model': model_name,
                'strategy': strategy_name,
                'model_strategy': f"{model_name}_{strategy_name}",
                'mean': ba['mean'],
                'std': ba['std'],
                'min': ba['min'],
                'max': ba['max'],
                'n_seeds': ba.get('n_seeds', 10)
            })
    
    df = pd.DataFrame(rows).sort_values('mean', ascending=False)
    return df


def create_cv_performance_plot(cv_df, output_dir):
    """Create comprehensive CV performance visualization."""
    print("\n[1] Creating CV performance plot...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
    
    # Plot 1: Top 20 models
    top_20 = cv_df.head(20)
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_20)))
    
    y_pos = np.arange(len(top_20))
    ax1.barh(y_pos, top_20['mean'], xerr=top_20['std'],
             color=colors, edgecolor='black', alpha=0.8, linewidth=1.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(top_20['model'], fontsize=9)
    ax1.set_xlabel('Balanced Accuracy (30-Fold CV)', fontsize=12, fontweight='bold')
    ax1.set_title('Top 20 Models - Cross-Validation Performance', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    ax1.axvline(x=0.8, color='red', linestyle='--', alpha=0.5, label='80% threshold')
    ax1.legend()
    
    # Plot 2: Stability analysis (std vs mean)
    scatter = ax2.scatter(cv_df['mean'], cv_df['std'],
                         s=100, alpha=0.6, c=cv_df['mean'],
                         cmap='RdYlGn', edgecolors='black', linewidth=1)
    
    # Label top 10
    for idx, row in cv_df.head(10).iterrows():
        ax2.annotate(row['model'], (row['mean'], row['std']),
                    fontsize=7, alpha=0.7,
                    xytext=(5, 5), textcoords='offset points')
    
    ax2.set_xlabel('Mean Balanced Accuracy', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Standard Deviation', fontsize=12, fontweight='bold')
    ax2.set_title('CV Stability Analysis (Mean vs Std)', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Mean Performance', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cv_performance_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: cv_performance_analysis.png")


def create_multi_seed_performance_plot(ms_df, output_dir):
    """Create comprehensive multi-seed performance visualization."""
    print("\n[2] Creating multi-seed performance plot...")
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    
    # Plot 1: Top 20 model-strategy combinations
    ax1 = axes[0, 0]
    top_20 = ms_df.head(20)
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_20)))
    
    y_pos = np.arange(len(top_20))
    ax1.barh(y_pos, top_20['mean'], xerr=top_20['std'],
             color=colors, edgecolor='black', alpha=0.8, linewidth=1.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(top_20['model_strategy'], fontsize=8)
    ax1.set_xlabel('Balanced Accuracy (Multi-Seed)', fontsize=11, fontweight='bold')
    ax1.set_title('Top 20 Model-Strategy Combinations', fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    ax1.axvline(x=0.8, color='red', linestyle='--', alpha=0.5, label='80% threshold')
    ax1.legend()
    
    # Plot 2: Model comparison (aggregate across strategies)
    ax2 = axes[0, 1]
    model_agg = ms_df.groupby('model').agg({
        'mean': 'mean',
        'std': 'mean'
    }).sort_values('mean', ascending=False).head(10)
    
    y_pos = np.arange(len(model_agg))
    colors = plt.cm.viridis(np.linspace(0, 1, len(model_agg)))
    ax2.barh(y_pos, model_agg['mean'], xerr=model_agg['std'],
             color=colors, edgecolor='black', alpha=0.8, linewidth=1.5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(model_agg.index, fontsize=10)
    ax2.set_xlabel('Mean Balanced Accuracy', fontsize=11, fontweight='bold')
    ax2.set_title('Top 10 Models (Averaged Across Strategies)', fontsize=12, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # Plot 3: Strategy comparison
    ax3 = axes[1, 0]
    strategy_agg = ms_df.groupby('strategy').agg({
        'mean': 'mean',
        'std': 'mean'
    }).sort_values('mean', ascending=False)
    
    colors = sns.color_palette("Set2", len(strategy_agg))
    strategy_agg.plot(kind='bar', y='mean', yerr='std', ax=ax3,
                     color=colors, edgecolor='black', alpha=0.8, linewidth=1.5,
                     legend=False, rot=45)
    ax3.set_xlabel('Resampling Strategy', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Mean Balanced Accuracy', fontsize=11, fontweight='bold')
    ax3.set_title('Strategy Performance Comparison', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # Plot 4: Stability analysis
    ax4 = axes[1, 1]
    scatter = ax4.scatter(ms_df['mean'], ms_df['std'],
                         s=100, alpha=0.6, c=ms_df['mean'],
                         cmap='RdYlGn', edgecolors='black', linewidth=1)
    
    # Label top 10
    for idx, row in ms_df.head(10).iterrows():
        ax4.annotate(row['model'], (row['mean'], row['std']),
                    fontsize=7, alpha=0.7,
                    xytext=(5, 5), textcoords='offset points')
    
    ax4.set_xlabel('Mean Balanced Accuracy', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Standard Deviation', fontsize=11, fontweight='bold')
    ax4.set_title('Multi-Seed Stability (Mean vs Std)', fontsize=12, fontweight='bold')
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/multi_seed_performance_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: multi_seed_performance_analysis.png")


def create_comparison_plot(cv_df, ms_df, output_dir):
    """Create side-by-side comparison of CV vs Multi-Seed."""
    print("\n[3] Creating CV vs Multi-Seed comparison...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # CV top 10
    cv_top = cv_df.head(10)
    y_pos = np.arange(len(cv_top))
    colors1 = plt.cm.Blues(np.linspace(0.4, 0.9, len(cv_top)))
    
    ax1.barh(y_pos, cv_top['mean'], xerr=cv_top['std'],
             color=colors1, edgecolor='black', alpha=0.8, linewidth=1.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(cv_top['model'], fontsize=9)
    ax1.set_xlabel('Balanced Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('30-Fold Cross-Validation\nTop 10 Models', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    ax1.set_xlim(0, 1)
    
    # Multi-seed top 10
    ms_top = ms_df.head(10)
    y_pos = np.arange(len(ms_top))
    colors2 = plt.cm.Greens(np.linspace(0.4, 0.9, len(ms_top)))
    
    ax2.barh(y_pos, ms_top['mean'], xerr=ms_top['std'],
             color=colors2, edgecolor='black', alpha=0.8, linewidth=1.5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(ms_top['model_strategy'], fontsize=9)
    ax2.set_xlabel('Balanced Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Multi-Seed Validation (10 seeds)\nTop 10 Combinations', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    ax2.set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cv_vs_multi_seed_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: cv_vs_multi_seed_comparison.png")


def create_interactive_dashboard(cv_df, ms_df, output_dir):
    """Create interactive Plotly dashboard."""
    print("\n[4] Creating interactive HTML dashboard...")
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Cross-Validation Performance (Top 20)',
                       'Multi-Seed Performance (Top 20)',
                       'CV Stability Analysis',
                       'Multi-Seed Stability Analysis'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "scatter"}]]
    )
    
    # CV bar chart
    cv_top = cv_df.head(20)
    fig.add_trace(
        go.Bar(
            y=cv_top['model'],
            x=cv_top['mean'],
            error_x=dict(type='data', array=cv_top['std']),
            orientation='h',
            marker=dict(color=cv_top['mean'], colorscale='RdYlGn',
                       showscale=False, line=dict(color='black', width=1)),
            hovertemplate='<b>%{y}</b><br>Mean: %{x:.4f}<br>Std: %{error_x.array:.4f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Multi-seed bar chart
    ms_top = ms_df.head(20)
    fig.add_trace(
        go.Bar(
            y=ms_top['model_strategy'],
            x=ms_top['mean'],
            error_x=dict(type='data', array=ms_top['std']),
            orientation='h',
            marker=dict(color=ms_top['mean'], colorscale='RdYlGn',
                       showscale=False, line=dict(color='black', width=1)),
            hovertemplate='<b>%{y}</b><br>Mean: %{x:.4f}<br>Std: %{error_x.array:.4f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # CV scatter
    fig.add_trace(
        go.Scatter(
            x=cv_df['mean'],
            y=cv_df['std'],
            mode='markers',
            marker=dict(size=10, color=cv_df['mean'], colorscale='RdYlGn',
                       showscale=True, colorbar=dict(x=0.46, len=0.4),
                       line=dict(color='black', width=1)),
            text=cv_df['model'],
            hovertemplate='<b>%{text}</b><br>Mean: %{x:.4f}<br>Std: %{y:.4f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Multi-seed scatter
    fig.add_trace(
        go.Scatter(
            x=ms_df['mean'],
            y=ms_df['std'],
            mode='markers',
            marker=dict(size=10, color=ms_df['mean'], colorscale='RdYlGn',
                       showscale=True, colorbar=dict(x=1.02, len=0.4),
                       line=dict(color='black', width=1)),
            text=ms_df['model_strategy'],
            hovertemplate='<b>%{text}</b><br>Mean: %{x:.4f}<br>Std: %{y:.4f}<extra></extra>'
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_xaxes(title_text="Balanced Accuracy", row=1, col=1)
    fig.update_xaxes(title_text="Balanced Accuracy", row=1, col=2)
    fig.update_xaxes(title_text="Mean Balanced Accuracy", row=2, col=1)
    fig.update_xaxes(title_text="Mean Balanced Accuracy", row=2, col=2)
    fig.update_yaxes(title_text="Standard Deviation", row=2, col=1)
    fig.update_yaxes(title_text="Standard Deviation", row=2, col=2)
    
    fig.update_layout(
        title_text=f"<b>Validation Results Dashboard - {DATASET_TYPE.upper()} Dataset</b>",
        title_font_size=20,
        showlegend=False,
        height=1000,
        width=1600
    )
    
    html_path = f"{output_dir}/validation_dashboard.html"
    fig.write_html(html_path)
    print(f"   ✓ Saved: validation_dashboard.html")


def create_ranking_table(cv_df, ms_df, output_dir):
    """Create detailed ranking table."""
    print("\n[5] Creating ranking table...")
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 12))
    
    # CV table
    ax1 = axes[0]
    ax1.axis('tight')
    ax1.axis('off')
    
    cv_table_data = cv_df.head(15)[['model', 'mean', 'std']].copy()
    cv_table_data['rank'] = range(1, len(cv_table_data) + 1)
    cv_table_data = cv_table_data[['rank', 'model', 'mean', 'std']]
    cv_table_data.columns = ['Rank', 'Model', 'Mean BA', 'Std BA']
    cv_table_data['Mean BA'] = cv_table_data['Mean BA'].apply(lambda x: f"{x:.4f}")
    cv_table_data['Std BA'] = cv_table_data['Std BA'].apply(lambda x: f"{x:.4f}")
    
    table1 = ax1.table(cellText=cv_table_data.values,
                      colLabels=cv_table_data.columns,
                      cellLoc='center',
                      loc='center',
                      bbox=[0, 0, 1, 1])
    table1.auto_set_font_size(False)
    table1.set_fontsize(9)
    table1.scale(1, 2)
    
    # Color header
    for (i, j), cell in table1.get_celld().items():
        if i == 0:
            cell.set_facecolor('#4472C4')
            cell.set_text_props(weight='bold', color='white')
        else:
            if i % 2 == 0:
                cell.set_facecolor('#E7E6E6')
    
    ax1.set_title('Cross-Validation Rankings\n(30-Fold CV)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Multi-seed table
    ax2 = axes[1]
    ax2.axis('tight')
    ax2.axis('off')
    
    ms_table_data = ms_df.head(15)[['model_strategy', 'mean', 'std']].copy()
    ms_table_data['rank'] = range(1, len(ms_table_data) + 1)
    ms_table_data = ms_table_data[['rank', 'model_strategy', 'mean', 'std']]
    ms_table_data.columns = ['Rank', 'Model-Strategy', 'Mean BA', 'Std BA']
    ms_table_data['Mean BA'] = ms_table_data['Mean BA'].apply(lambda x: f"{x:.4f}")
    ms_table_data['Std BA'] = ms_table_data['Std BA'].apply(lambda x: f"{x:.4f}")
    
    table2 = ax2.table(cellText=ms_table_data.values,
                      colLabels=ms_table_data.columns,
                      cellLoc='center',
                      loc='center',
                      bbox=[0, 0, 1, 1])
    table2.auto_set_font_size(False)
    table2.set_fontsize(9)
    table2.scale(1, 2)
    
    # Color header
    for (i, j), cell in table2.get_celld().items():
        if i == 0:
            cell.set_facecolor('#70AD47')
            cell.set_text_props(weight='bold', color='white')
        else:
            if i % 2 == 0:
                cell.set_facecolor('#E7E6E6')
    
    ax2.set_title('Multi-Seed Rankings\n(10 Seeds)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/validation_rankings_table.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: validation_rankings_table.png")


def create_summary_report(cv_df, ms_df, output_dir):
    """Create text summary report."""
    print("\n[6] Creating summary report...")
    
    report_path = f"{output_dir}/validation_summary.txt"
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"VALIDATION RESULTS SUMMARY - {DATASET_TYPE.upper()} DATASET\n")
        f.write("="*80 + "\n\n")
        
        f.write("="*80 + "\n")
        f.write("1. CROSS-VALIDATION RESULTS (30-Fold)\n")
        f.write("="*80 + "\n\n")
        f.write("Top 10 Models:\n")
        f.write("-"*80 + "\n")
        for i, row in cv_df.head(10).iterrows():
            f.write(f"{i+1:2d}. {row['model']:40s} {row['mean']:.4f} ± {row['std']:.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("2. MULTI-SEED VALIDATION RESULTS (10 Seeds)\n")
        f.write("="*80 + "\n\n")
        f.write("Top 10 Model-Strategy Combinations:\n")
        f.write("-"*80 + "\n")
        for i, row in ms_df.head(10).iterrows():
            f.write(f"{i+1:2d}. {row['model_strategy']:50s} {row['mean']:.4f} ± {row['std']:.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("3. STATISTICS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Cross-Validation:\n")
        f.write(f"  Total models tested: {len(cv_df)}\n")
        f.write(f"  Best performance: {cv_df.iloc[0]['model']} ({cv_df.iloc[0]['mean']:.4f})\n")
        f.write(f"  Most stable (lowest std): {cv_df.loc[cv_df['std'].idxmin(), 'model']} ")
        f.write(f"({cv_df['std'].min():.4f})\n\n")
        
        f.write(f"Multi-Seed:\n")
        f.write(f"  Total combinations tested: {len(ms_df)}\n")
        f.write(f"  Best performance: {ms_df.iloc[0]['model_strategy']} ({ms_df.iloc[0]['mean']:.4f})\n")
        f.write(f"  Most stable (lowest std): {ms_df.loc[ms_df['std'].idxmin(), 'model_strategy']} ")
        f.write(f"({ms_df['std'].min():.4f})\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("4. RECOMMENDATIONS\n")
        f.write("="*80 + "\n\n")
        
        cv_best = cv_df.iloc[0]
        ms_best = ms_df.iloc[0]
        
        f.write(f"Best CV Model: {cv_best['model']}\n")
        f.write(f"  Performance: {cv_best['mean']:.4f} ± {cv_best['std']:.4f}\n")
        f.write(f"  Stability: {'High' if cv_best['std'] < 0.05 else 'Moderate' if cv_best['std'] < 0.10 else 'Low'}\n\n")
        
        f.write(f"Best Multi-Seed Combination: {ms_best['model_strategy']}\n")
        f.write(f"  Performance: {ms_best['mean']:.4f} ± {ms_best['std']:.4f}\n")
        f.write(f"  Stability: {'High' if ms_best['std'] < 0.02 else 'Moderate' if ms_best['std'] < 0.05 else 'Low'}\n\n")
        
        f.write("Deployment Recommendation:\n")
        f.write(f"  Use {ms_best['model']} with {ms_best['strategy']} strategy\n")
        f.write(f"  Expected performance: {ms_best['mean']:.2%} ± {ms_best['std']:.2%}\n")
    
    print(f"   ✓ Saved: validation_summary.txt")


def main():
    """Main analysis pipeline."""
    print("="*80)
    print(f"VALIDATION RESULTS VISUALIZATION - {DATASET_TYPE.upper()} DATASET")
    print("="*80)
    
    # Load results
    print("\nLoading results...")
    cv_data = load_cv_results()
    ms_data = load_multi_seed_results()
    
    if cv_data is None and ms_data is None:
        print("\n✗ No validation results found!")
        return
    
    # Parse results
    cv_df = parse_cv_results(cv_data) if cv_data else None
    ms_df = parse_multi_seed_results(ms_data) if ms_data else None
    
    print(f"\nParsed results:")
    if cv_df is not None:
        print(f"  CV models: {len(cv_df)}")
    if ms_df is not None:
        print(f"  Multi-seed combinations: {len(ms_df)}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    if cv_df is not None:
        create_cv_performance_plot(cv_df, OUTPUT_DIR)
    
    if ms_df is not None:
        create_multi_seed_performance_plot(ms_df, OUTPUT_DIR)
    
    if cv_df is not None and ms_df is not None:
        create_comparison_plot(cv_df, ms_df, OUTPUT_DIR)
        create_interactive_dashboard(cv_df, ms_df, OUTPUT_DIR)
        create_ranking_table(cv_df, ms_df, OUTPUT_DIR)
        create_summary_report(cv_df, ms_df, OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("✓ VALIDATION VISUALIZATION COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    if cv_df is not None:
        print("  * cv_performance_analysis.png")
    if ms_df is not None:
        print("  * multi_seed_performance_analysis.png")
    if cv_df is not None and ms_df is not None:
        print("  * cv_vs_multi_seed_comparison.png")
        print("  * validation_dashboard.html (interactive)")
        print("  * validation_rankings_table.png")
        print("  * validation_summary.txt")


if __name__ == '__main__':
    main()
