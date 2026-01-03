"""
Advanced Interactive Visualizations for Multi-Seed ML Results
Creates comprehensive, publication-ready interactive dashboards using Plotly.

Features:
- Seed-by-seed performance tracking
- Model stability heatmaps
- Performance evolution plots
- Metric correlation matrices
- Strategy comparison radar charts
- Interactive filtering and drill-down
- Export-ready static versions
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
    RESULTS_DIR = 'Mix_Results'
elif DATASET_TYPE == 'single':
    RESULTS_DIR = 'Single_Results'
else:
    print(f"Error: Invalid dataset type '{DATASET_TYPE}'. Use 'mix' or 'single'")
    sys.exit(1)

OUTPUT_DIR = f'{RESULTS_DIR}/advanced_visualizations'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Seed list
SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300


def load_seed_results(seed):
    """Load results for a specific seed."""
    seed_file = f'{RESULTS_DIR}/seed_{seed}/reports/combined_results.json'
    if not Path(seed_file).exists():
        return None
    
    with open(seed_file, 'r') as f:
        return json.load(f)


def load_aggregated_results():
    """Load aggregated multi-seed results."""
    agg_file = f'{RESULTS_DIR}/multi_seed_aggregated/aggregated_results.json'
    if not Path(agg_file).exists():
        return None
    
    with open(agg_file, 'r') as f:
        return json.load(f)


def create_seed_performance_evolution():
    """Create interactive plot showing performance evolution across seeds."""
    print("\n[1] Creating seed performance evolution plot...")
    
    # Load all seed results
    all_data = []
    for seed in SEEDS:
        results = load_seed_results(seed)
        if results:
            for model_name, model_data in results.items():
                for strategy_name, strategy_data in model_data['results'].items():
                    all_data.append({
                        'seed': seed,
                        'model': model_name,
                        'strategy': strategy_name,
                        'model_strategy': f"{model_name}_{strategy_name}",
                        'balanced_accuracy': strategy_data['balanced_accuracy'],
                        'f1_macro': strategy_data['f1_macro'],
                        'accuracy': strategy_data['accuracy']
                    })
    
    df = pd.DataFrame(all_data)
    
    # Get top 10 model-strategy combinations by mean performance
    top_models = df.groupby('model_strategy')['balanced_accuracy'].mean().nlargest(10).index
    df_top = df[df['model_strategy'].isin(top_models)]
    
    # Create interactive line plot
    fig = go.Figure()
    
    colors = px.colors.qualitative.Plotly
    for i, model_strategy in enumerate(top_models):
        model_df = df_top[df_top['model_strategy'] == model_strategy]
        
        fig.add_trace(go.Scatter(
            x=model_df['seed'],
            y=model_df['balanced_accuracy'],
            mode='lines+markers',
            name=model_strategy,
            line=dict(width=2, color=colors[i % len(colors)]),
            marker=dict(size=8),
            hovertemplate='<b>%{fullData.name}</b><br>' +
                         'Seed: %{x}<br>' +
                         'Balanced Accuracy: %{y:.4f}<extra></extra>'
        ))
    
    fig.update_layout(
        title='<b>Performance Evolution Across Seeds - Top 10 Models</b>',
        xaxis_title='Random Seed',
        yaxis_title='Balanced Accuracy',
        hovermode='x unified',
        height=600,
        width=1400,
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02)
    )
    
    fig.write_html(f"{OUTPUT_DIR}/seed_performance_evolution.html")
    print(f"   ✓ Saved: seed_performance_evolution.html")


def create_model_stability_heatmap():
    """Create heatmap showing model stability across seeds."""
    print("\n[2] Creating model stability heatmap...")
    
    # Load aggregated results
    agg_data = load_aggregated_results()
    if not agg_data:
        print("   ⚠ No aggregated results found")
        return
    
    # Extract mean and std for each model-strategy
    rows = []
    for model_name, strategies in agg_data.items():
        for strategy_name, metrics in strategies.items():
            ba = metrics['balanced_accuracy']
            rows.append({
                'model': model_name,
                'strategy': strategy_name,
                'model_strategy': f"{model_name}_{strategy_name}",
                'mean': ba['mean'],
                'std': ba['std'],
                'cv': ba['std'] / ba['mean'] if ba['mean'] > 0 else 0  # Coefficient of variation
            })
    
    df = pd.DataFrame(rows).sort_values('mean', ascending=False)
    
    # Create static heatmap for mean performance
    top_20 = df.head(20)
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Heatmap 1: Mean Performance
    pivot_mean = top_20.pivot_table(
        values='mean',
        index='model',
        columns='strategy',
        aggfunc='first'
    )
    
    sns.heatmap(pivot_mean, annot=True, fmt='.3f', cmap='RdYlGn',
                vmin=0.5, vmax=1.0, ax=axes[0], cbar_kws={'label': 'Mean BA'})
    axes[0].set_title('Mean Balanced Accuracy by Model & Strategy', 
                     fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Strategy', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Model', fontsize=12, fontweight='bold')
    
    # Heatmap 2: Stability (lower std = more stable)
    pivot_std = top_20.pivot_table(
        values='std',
        index='model',
        columns='strategy',
        aggfunc='first'
    )
    
    sns.heatmap(pivot_std, annot=True, fmt='.3f', cmap='RdYlGn_r',
                vmin=0, vmax=0.1, ax=axes[1], cbar_kws={'label': 'Std Dev'})
    axes[1].set_title('Performance Stability (Lower = Better)', 
                     fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Strategy', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Model', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/model_stability_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: model_stability_heatmap.png")
    
    # Interactive heatmap
    fig = go.Figure(data=go.Heatmap(
        z=df.head(30)['mean'].values.reshape(-1, 1),
        x=['Balanced Accuracy'],
        y=df.head(30)['model_strategy'],
        colorscale='RdYlGn',
        text=np.round(df.head(30)['mean'].values.reshape(-1, 1), 4),
        texttemplate='%{text}',
        hovertemplate='Model: %{y}<br>Mean BA: %{z:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='<b>Top 30 Models - Mean Performance Heatmap</b>',
        height=900,
        width=600,
        yaxis_nticks=30
    )
    
    fig.write_html(f"{OUTPUT_DIR}/performance_heatmap_interactive.html")
    print(f"   ✓ Saved: performance_heatmap_interactive.html")


def create_metric_correlation_matrix():
    """Create correlation matrix for different metrics."""
    print("\n[3] Creating metric correlation matrix...")
    
    # Load seed results
    all_metrics = []
    for seed in SEEDS:
        results = load_seed_results(seed)
        if results:
            for model_name, model_data in results.items():
                for strategy_name, strategy_data in model_data['results'].items():
                    all_metrics.append({
                        'model_strategy': f"{model_name}_{strategy_name}",
                        'accuracy': strategy_data['accuracy'],
                        'balanced_accuracy': strategy_data['balanced_accuracy'],
                        'f1_macro': strategy_data['f1_macro'],
                        'log_loss': strategy_data.get('log_loss', np.nan),
                        'informedness': strategy_data.get('informedness', np.nan),
                        'markedness': strategy_data.get('markedness', np.nan)
                    })
    
    df = pd.DataFrame(all_metrics)
    
    # Calculate correlation matrix
    corr_matrix = df[['accuracy', 'balanced_accuracy', 'f1_macro', 
                       'log_loss', 'informedness', 'markedness']].corr()
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                center=0, vmin=-1, vmax=1, square=True,
                linewidths=1, cbar_kws={'label': 'Correlation'})
    plt.title('Metric Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/metric_correlation_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: metric_correlation_matrix.png")


def create_strategy_radar_chart():
    """Create radar chart comparing strategies."""
    print("\n[4] Creating strategy comparison radar chart...")
    
    agg_data = load_aggregated_results()
    if not agg_data:
        return
    
    # Aggregate by strategy across all models
    strategy_metrics = {}
    for model_name, strategies in agg_data.items():
        for strategy_name, metrics in strategies.items():
            if strategy_name not in strategy_metrics:
                strategy_metrics[strategy_name] = {
                    'balanced_accuracy': [],
                    'f1_macro': [],
                    'accuracy': []
                }
            
            strategy_metrics[strategy_name]['balanced_accuracy'].append(metrics['balanced_accuracy']['mean'])
            strategy_metrics[strategy_name]['f1_macro'].append(metrics['f1_macro']['mean'])
            strategy_metrics[strategy_name]['accuracy'].append(metrics['accuracy']['mean'])
    
    # Calculate mean for each strategy
    strategy_summary = []
    for strategy, metrics in strategy_metrics.items():
        strategy_summary.append({
            'strategy': strategy,
            'balanced_accuracy': np.mean(metrics['balanced_accuracy']),
            'f1_macro': np.mean(metrics['f1_macro']),
            'accuracy': np.mean(metrics['accuracy'])
        })
    
    df = pd.DataFrame(strategy_summary)
    
    # Create radar chart
    categories = ['Balanced Accuracy', 'F1 Macro', 'Accuracy']
    
    fig = go.Figure()
    
    for _, row in df.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row['balanced_accuracy'], row['f1_macro'], row['accuracy']],
            theta=categories,
            fill='toself',
            name=row['strategy']
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0.5, 1.0])
        ),
        title='<b>Strategy Performance Comparison (Averaged Across Models)</b>',
        showlegend=True,
        height=600,
        width=800
    )
    
    fig.write_html(f"{OUTPUT_DIR}/strategy_radar_chart.html")
    print(f"   ✓ Saved: strategy_radar_chart.html")


def create_violin_plots():
    """Create violin plots showing performance distributions."""
    print("\n[5] Creating performance distribution violin plots...")
    
    # Load all seed results
    all_data = []
    for seed in SEEDS:
        results = load_seed_results(seed)
        if results:
            for model_name, model_data in results.items():
                for strategy_name, strategy_data in model_data['results'].items():
                    all_data.append({
                        'model': model_name,
                        'strategy': strategy_name,
                        'model_strategy': f"{model_name}_{strategy_name}",
                        'balanced_accuracy': strategy_data['balanced_accuracy']
                    })
    
    df = pd.DataFrame(all_data)
    
    # Get top 10 models
    top_models = df.groupby('model_strategy')['balanced_accuracy'].mean().nlargest(10).index
    df_top = df[df['model_strategy'].isin(top_models)]
    
    # Sort by median performance
    model_order = df_top.groupby('model_strategy')['balanced_accuracy'].median().sort_values(ascending=False).index
    
    # Create violin plot
    fig = go.Figure()
    
    for model in model_order:
        model_data = df_top[df_top['model_strategy'] == model]['balanced_accuracy']
        
        fig.add_trace(go.Violin(
            y=model_data,
            name=model,
            box_visible=True,
            meanline_visible=True,
            hovertemplate='<b>%{fullData.name}</b><br>' +
                         'Value: %{y:.4f}<extra></extra>'
        ))
    
    fig.update_layout(
        title='<b>Performance Distribution Across Seeds - Top 10 Models</b>',
        yaxis_title='Balanced Accuracy',
        xaxis_title='Model-Strategy',
        height=600,
        width=1400,
        showlegend=False
    )
    
    fig.write_html(f"{OUTPUT_DIR}/performance_violin_plots.html")
    print(f"   ✓ Saved: performance_violin_plots.html")


def create_model_comparison_box_plot():
    """Create box plot comparing models."""
    print("\n[6] Creating model comparison box plots...")
    
    all_data = []
    for seed in SEEDS:
        results = load_seed_results(seed)
        if results:
            for model_name, model_data in results.items():
                for strategy_name, strategy_data in model_data['results'].items():
                    all_data.append({
                        'model': model_name,
                        'balanced_accuracy': strategy_data['balanced_accuracy']
                    })
    
    df = pd.DataFrame(all_data)
    
    # Calculate median for ordering
    model_order = df.groupby('model')['balanced_accuracy'].median().sort_values(ascending=False).index
    
    plt.figure(figsize=(14, 8))
    sns.boxplot(data=df, x='model', y='balanced_accuracy', order=model_order,
                palette='Set2', width=0.6)
    sns.stripplot(data=df, x='model', y='balanced_accuracy', order=model_order,
                  color='black', alpha=0.3, size=4)
    
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.ylabel('Balanced Accuracy', fontsize=12, fontweight='bold')
    plt.title('Model Performance Distribution (All Strategies Combined)', 
             fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/model_comparison_boxplot.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: model_comparison_boxplot.png")


def create_comprehensive_dashboard():
    """Create comprehensive interactive dashboard."""
    print("\n[7] Creating comprehensive interactive dashboard...")
    
    # Load all seed results
    all_data = []
    for seed in SEEDS:
        results = load_seed_results(seed)
        if results:
            for model_name, model_data in results.items():
                for strategy_name, strategy_data in model_data['results'].items():
                    all_data.append({
                        'seed': seed,
                        'model': model_name,
                        'strategy': strategy_name,
                        'model_strategy': f"{model_name}_{strategy_name}",
                        'balanced_accuracy': strategy_data['balanced_accuracy'],
                        'f1_macro': strategy_data['f1_macro'],
                        'accuracy': strategy_data['accuracy']
                    })
    
    df = pd.DataFrame(all_data)
    
    # Get top 15 for visualization
    top_15 = df.groupby('model_strategy')['balanced_accuracy'].mean().nlargest(15).index
    df_top = df[df['model_strategy'].isin(top_15)]
    
    # Create subplot dashboard
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Performance Evolution', 'Distribution by Model',
                       'Performance vs Stability', 'Strategy Comparison'),
        specs=[[{"type": "scatter"}, {"type": "box"}],
               [{"type": "scatter"}, {"type": "bar"}]]
    )
    
    # 1. Evolution plot
    colors = px.colors.qualitative.Plotly
    for i, model_strategy in enumerate(list(top_15)[:10]):
        model_df = df_top[df_top['model_strategy'] == model_strategy]
        fig.add_trace(
            go.Scatter(x=model_df['seed'], y=model_df['balanced_accuracy'],
                      mode='lines+markers', name=model_strategy,
                      line=dict(color=colors[i % len(colors)]),
                      showlegend=True),
            row=1, col=1
        )
    
    # 2. Box plot by model
    for model in df_top['model'].unique():
        model_data = df_top[df_top['model'] == model]
        fig.add_trace(
            go.Box(y=model_data['balanced_accuracy'], name=model, showlegend=False),
            row=1, col=2
        )
    
    # 3. Mean vs Std scatter
    stats = df_top.groupby('model_strategy').agg({
        'balanced_accuracy': ['mean', 'std']
    }).reset_index()
    stats.columns = ['model_strategy', 'mean', 'std']
    
    fig.add_trace(
        go.Scatter(x=stats['mean'], y=stats['std'],
                  mode='markers', text=stats['model_strategy'],
                  marker=dict(size=12, color=stats['mean'], 
                            colorscale='RdYlGn', showscale=True),
                  hovertemplate='<b>%{text}</b><br>Mean: %{x:.4f}<br>Std: %{y:.4f}<extra></extra>',
                  showlegend=False),
        row=2, col=1
    )
    
    # 4. Strategy comparison
    strategy_stats = df_top.groupby('strategy')['balanced_accuracy'].mean().sort_values(ascending=False)
    fig.add_trace(
        go.Bar(x=strategy_stats.index, y=strategy_stats.values,
              marker_color=strategy_stats.values, marker_colorscale='RdYlGn',
              showlegend=False),
        row=2, col=2
    )
    
    # Update layout
    fig.update_xaxes(title_text="Seed", row=1, col=1)
    fig.update_yaxes(title_text="Balanced Accuracy", row=1, col=1)
    fig.update_xaxes(title_text="Model", row=1, col=2)
    fig.update_yaxes(title_text="Balanced Accuracy", row=1, col=2)
    fig.update_xaxes(title_text="Mean BA", row=2, col=1)
    fig.update_yaxes(title_text="Std Dev", row=2, col=1)
    fig.update_xaxes(title_text="Strategy", row=2, col=2)
    fig.update_yaxes(title_text="Mean BA", row=2, col=2)
    
    fig.update_layout(
        title_text=f"<b>Multi-Seed Validation Dashboard - {DATASET_TYPE.upper()} Dataset</b>",
        title_font_size=20,
        height=1000,
        width=1600,
        showlegend=True
    )
    
    fig.write_html(f"{OUTPUT_DIR}/comprehensive_dashboard.html")
    print(f"   ✓ Saved: comprehensive_dashboard.html")


def main():
    """Main visualization pipeline."""
    print("="*80)
    print(f"ADVANCED VISUALIZATIONS - {DATASET_TYPE.upper()} DATASET")
    print("="*80)
    
    # Create visualizations
    create_seed_performance_evolution()
    create_model_stability_heatmap()
    create_metric_correlation_matrix()
    create_strategy_radar_chart()
    create_violin_plots()
    create_model_comparison_box_plot()
    create_comprehensive_dashboard()
    
    print("\n" + "="*80)
    print("✓ ADVANCED VISUALIZATIONS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  * seed_performance_evolution.html (interactive)")
    print("  * model_stability_heatmap.png")
    print("  * performance_heatmap_interactive.html")
    print("  * metric_correlation_matrix.png")
    print("  * strategy_radar_chart.html (interactive)")
    print("  * performance_violin_plots.html (interactive)")
    print("  * model_comparison_boxplot.png")
    print("  * comprehensive_dashboard.html (interactive, all-in-one)")


if __name__ == '__main__':
    main()
