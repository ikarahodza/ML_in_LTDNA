"""
Customizable Report Generator
Easy-to-modify script for creating model comparison reports
Edit the CONFIGURATION section to customize your reports
"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages

# ==============================================================================
# CONFIGURATION - EDIT THESE TO CUSTOMIZE YOUR REPORT
# ==============================================================================

# Output settings
OUTPUT_DIR_NAME = 'custom_report'
REPORT_TITLE = 'Model Performance Report'

# Figure settings
FIGURE_DPI = 300  # Resolution (300 for print, 150 for screen)
FIGURE_FORMAT = 'png'  # 'png', 'pdf', or 'svg'
FIGURE_FACECOLOR = 'white'

# Font settings
TITLE_FONTSIZE = 16
SUBTITLE_FONTSIZE = 14
LABEL_FONTSIZE = 12
TICK_FONTSIZE = 10
LEGEND_FONTSIZE = 10

# Color scheme - Edit these colors to your preference
MODEL_COLORS = {
    'tabpfn': '#e74c3c',           # Red
    'xgboost': '#3498db',          # Blue
    'lightgbm': '#2ecc71',         # Green
    'extratrees': '#9b59b6',       # Purple
    'randomforest': '#f39c12',     # Orange
    'mlp': '#1abc9c',              # Teal
    'svm': '#e67e22',              # Dark Orange
    'gnb': '#95a5a6',              # Gray
    'logistic_regression': '#34495e'  # Dark Gray
}

# Bar chart settings
BAR_WIDTH = 0.35
BAR_ALPHA_MAIN = 0.8       # Opacity for main bars
BAR_ALPHA_SECONDARY = 0.5  # Opacity for secondary bars
BAR_EDGE_COLOR = 'black'
BAR_EDGE_WIDTH = 1.5

# Grid settings
GRID_ALPHA = 0.3
GRID_LINESTYLE = '--'
GRID_LINEWIDTH = 0.5

# Number of top models to show
NUM_TOP_MODELS = 3

# Metrics to display
METRICS_TO_PLOT = [
    ('balanced_accuracy', 'Balanced Accuracy'),
    ('f1_macro', 'F1-Score'),
    ('accuracy', 'Accuracy'),
    ('misclass_rate', 'Misclassification Rate (%)')
]

# Which reports to generate (set to False to skip)
GENERATE_COMPARISON_CHART = True
GENERATE_INDIVIDUAL_METRICS = True
GENERATE_MODEL_BREAKDOWNS = True
GENERATE_HTML_REPORT = False
GENERATE_PDF_REPORT = True

# ==============================================================================
# HELPER FUNCTIONS - Modify these to change how things look
# ==============================================================================

def get_model_color(model_name):
    """Get color for a model - edit MODEL_COLORS dict above to change"""
    for key, color in MODEL_COLORS.items():
        if key in model_name.lower().replace('_', '').replace(' ', ''):
            return color
    return '#7f8c8d'  # Default gray

def get_model_display_name(model_name):
    """Format model name for display"""
    return model_name.replace('_', ' ').title()

# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_dataset_results(dataset_type):
    """Load results from a dataset"""
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_folder = f"{dataset_type.capitalize()}_Results"
    json_path = os.path.join(PROJECT_ROOT, results_folder, 'reports', 'combined_results.json')
    
    if not os.path.exists(json_path):
        print(f"⚠ Warning: Results not found for {dataset_type} dataset")
        return None
    
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_best_results(data, dataset_name):
    """Extract best result for each model"""
    results = []
    
    for model_name, model_data in data.items():
        if 'results' not in model_data:
            continue
        
        # Find best strategy by balanced accuracy
        best_strategy = None
        best_bal_acc = -1
        
        for strategy, metrics in model_data['results'].items():
            bal_acc = metrics.get('balanced_accuracy', 0)
            if bal_acc > best_bal_acc:
                best_bal_acc = bal_acc
                best_strategy = strategy
        
        if best_strategy:
            metrics = model_data['results'][best_strategy]
            accuracy = metrics.get('accuracy', 0)
            
            results.append({
                'model': model_name,
                'strategy': best_strategy,
                'dataset': dataset_name,
                'balanced_accuracy': metrics.get('balanced_accuracy', 0),
                'accuracy': accuracy,
                'f1_macro': metrics.get('f1_macro', 0),
                'log_loss': metrics.get('log_loss', None),
                'misclass_rate': (1 - accuracy) * 100
            })
    
    return results

# ==============================================================================
# VISUALIZATION FUNCTIONS - Edit these to change how charts look
# ==============================================================================

def create_metric_bar_chart(mix_results, single_results, top_models, metric_key, 
                            metric_name, output_path):
    """Create a bar chart for a single metric"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data
    heights_mix = []
    heights_single = []
    colors = []
    labels = []
    
    for model in top_models:
        mix_result = next((r for r in mix_results if r['model'] == model), None)
        single_result = next((r for r in single_results if r['model'] == model), None)
        
        heights_mix.append(mix_result[metric_key] if mix_result else 0)
        heights_single.append(single_result[metric_key] if single_result else 0)
        colors.append(get_model_color(model))
        labels.append(get_model_display_name(model))
    
    x = np.arange(len(top_models))
    
    # Create bars
    bars1 = ax.bar(x - BAR_WIDTH/2, heights_mix, BAR_WIDTH, label='Mix Dataset',
                   color=colors, alpha=BAR_ALPHA_MAIN, 
                   edgecolor=BAR_EDGE_COLOR, linewidth=BAR_EDGE_WIDTH)
    bars2 = ax.bar(x + BAR_WIDTH/2, heights_single, BAR_WIDTH, label='Single Dataset',
                   color=colors, alpha=BAR_ALPHA_SECONDARY,
                   edgecolor=BAR_EDGE_COLOR, linewidth=BAR_EDGE_WIDTH)
    
    # Add value labels on bars
    for bar1, bar2 in zip(bars1, bars2):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        
        if height1 > 0:
            text = f'{height1:.1f}%' if metric_key == 'misclass_rate' else f'{height1:.3f}'
            ax.text(bar1.get_x() + bar1.get_width()/2., height1/2,
                   text, ha='center', va='center', fontsize=TICK_FONTSIZE+1, 
                   fontweight='bold', color='white',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.8))
        
        if height2 > 0:
            text = f'{height2:.1f}%' if metric_key == 'misclass_rate' else f'{height2:.3f}'
            ax.text(bar2.get_x() + bar2.get_width()/2., height2/2,
                   text, ha='center', va='center', fontsize=TICK_FONTSIZE+1,
                   fontweight='bold', color='white',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.8))
    
    # Styling
    ax.set_ylabel(metric_name, fontsize=LABEL_FONTSIZE, fontweight='bold')
    ax.set_xlabel('Model', fontsize=LABEL_FONTSIZE, fontweight='bold')
    ax.set_title(f'{metric_name} - Top {len(top_models)} Models', 
                fontsize=TITLE_FONTSIZE, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=TICK_FONTSIZE, fontweight='bold')
    ax.legend(fontsize=LEGEND_FONTSIZE, loc='best', framealpha=0.9)
    ax.grid(axis='y', alpha=GRID_ALPHA, linestyle=GRID_LINESTYLE, linewidth=GRID_LINEWIDTH)
    
    # Set appropriate y-axis limits
    if metric_key == 'misclass_rate':
        max_val = max(max(heights_mix), max(heights_single))
        ax.set_ylim(0, max_val * 1.2)
    else:
        ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight', facecolor=FIGURE_FACECOLOR)
    plt.close()

def create_combined_comparison(mix_results, single_results, top_models, output_path):
    """Create combined chart with all metrics"""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)
    
    for idx, (metric_key, metric_name) in enumerate(METRICS_TO_PLOT):
        row = idx // 2
        col = idx % 2
        ax = fig.add_subplot(gs[row, col])
        
        # Prepare data
        heights_mix = []
        heights_single = []
        colors = []
        labels = []
        
        for model in top_models:
            mix_result = next((r for r in mix_results if r['model'] == model), None)
            single_result = next((r for r in single_results if r['model'] == model), None)
            
            heights_mix.append(mix_result[metric_key] if mix_result else 0)
            heights_single.append(single_result[metric_key] if single_result else 0)
            colors.append(get_model_color(model))
            labels.append(get_model_display_name(model))
        
        x = np.arange(len(top_models))
        
        # Create bars
        bars1 = ax.bar(x - BAR_WIDTH/2, heights_mix, BAR_WIDTH, label='Mix Dataset',
                      color=colors, alpha=BAR_ALPHA_MAIN,
                      edgecolor=BAR_EDGE_COLOR, linewidth=BAR_EDGE_WIDTH)
        bars2 = ax.bar(x + BAR_WIDTH/2, heights_single, BAR_WIDTH, label='Single Dataset',
                      color=colors, alpha=BAR_ALPHA_SECONDARY,
                      edgecolor=BAR_EDGE_COLOR, linewidth=BAR_EDGE_WIDTH)
        
        # Add value labels
        for bar1, bar2 in zip(bars1, bars2):
            height1 = bar1.get_height()
            height2 = bar2.get_height()
            
            if height1 > 0:
                text = f'{height1:.1f}%' if metric_key == 'misclass_rate' else f'{height1:.3f}'
                ax.text(bar1.get_x() + bar1.get_width()/2., height1/2,
                       text, ha='center', va='center', fontsize=10, fontweight='bold',
                       color='white', bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
            
            if height2 > 0:
                text = f'{height2:.1f}%' if metric_key == 'misclass_rate' else f'{height2:.3f}'
                ax.text(bar2.get_x() + bar2.get_width()/2., height2/2,
                       text, ha='center', va='center', fontsize=10, fontweight='bold',
                       color='white', bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
        
        # Styling
        ax.set_ylabel(metric_name, fontsize=LABEL_FONTSIZE, fontweight='bold')
        ax.set_xlabel('Model', fontsize=LABEL_FONTSIZE, fontweight='bold')
        ax.set_title(f'{metric_name}', fontsize=SUBTITLE_FONTSIZE, fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=TICK_FONTSIZE, fontweight='bold')
        ax.legend(fontsize=LEGEND_FONTSIZE, loc='best')
        ax.grid(axis='y', alpha=GRID_ALPHA, linestyle=GRID_LINESTYLE)
        
        # Set y-axis limits
        if metric_key == 'misclass_rate':
            max_val = max(max(heights_mix), max(heights_single))
            ax.set_ylim(0, max_val * 1.2)
        else:
            ax.set_ylim(0, 1.0)
    
    # Add summary table
    ax_table = fig.add_subplot(gs[2, :])
    create_summary_table(ax_table, mix_results, single_results, top_models)
    
    plt.suptitle(f'{REPORT_TITLE}\nTop {len(top_models)} Models Comparison',
                fontsize=TITLE_FONTSIZE+2, fontweight='bold', y=0.98)
    
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight', facecolor=FIGURE_FACECOLOR)
    plt.close()

def create_summary_table(ax, mix_results, single_results, top_models):
    """Create summary table for the combined chart"""
    ax.axis('off')
    
    table_data = [['Model', 'Dataset', 'Strategy', 'Bal Acc', 'F1-Score', 
                   'Accuracy', 'Misclass %']]
    
    for model in top_models:
        # Mix results
        mix_result = next((r for r in mix_results if r['model'] == model), None)
        if mix_result:
            table_data.append([
                get_model_display_name(model),
                'Mix',
                mix_result['strategy'],
                f"{mix_result['balanced_accuracy']:.4f}",
                f"{mix_result['f1_macro']:.4f}",
                f"{mix_result['accuracy']:.4f}",
                f"{mix_result['misclass_rate']:.2f}%"
            ])
        
        # Single results
        single_result = next((r for r in single_results if r['model'] == model), None)
        if single_result:
            table_data.append([
                '',
                'Single',
                single_result['strategy'],
                f"{single_result['balanced_accuracy']:.4f}",
                f"{single_result['f1_macro']:.4f}",
                f"{single_result['accuracy']:.4f}",
                f"{single_result['misclass_rate']:.2f}%"
            ])
        
        # Separator
        if model != top_models[-1]:
            table_data.append(['─'*15, '─'*10, '─'*20, '─'*8, '─'*8, '─'*8, '─'*10])
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.15, 0.10, 0.25, 0.10, 0.10, 0.10, 0.10])
    table.auto_set_font_size(False)
    table.set_fontsize(TICK_FONTSIZE)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(7):
        cell = table[(0, i)]
        cell.set_facecolor('#34495e')
        cell.set_text_props(weight='bold', color='white')
    
    # Color code model rows
    row_idx = 1
    for model in top_models:
        color = get_model_color(model)
        if row_idx < len(table_data):
            cell = table[(row_idx, 0)]
            cell.set_facecolor(color)
            cell.set_text_props(weight='bold', color='white')
        row_idx += 2 if row_idx + 1 < len(table_data) and table_data[row_idx + 1][0] == '' else 1
        if row_idx < len(table_data) and '─' in table_data[row_idx][0]:
            row_idx += 1

def create_model_breakdown(model, mix_result, single_result, output_path):
    """Create detailed breakdown for a single model"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    model_color = get_model_color(model)
    model_name = get_model_display_name(model)
    
    fig.suptitle(f'{model_name} - Detailed Performance Breakdown',
                fontsize=TITLE_FONTSIZE, fontweight='bold')
    
    # Mix results
    if mix_result:
        ax = axes[0]
        metrics = ['balanced_accuracy', 'f1_macro', 'accuracy']
        values = [mix_result[m] for m in metrics]
        labels = ['Balanced\nAccuracy', 'F1-Score', 'Accuracy']
        
        bars = ax.bar(labels, values, color=model_color, alpha=BAR_ALPHA_MAIN,
                     edgecolor=BAR_EDGE_COLOR, linewidth=BAR_EDGE_WIDTH*1.5)
        
        # Add values
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., val/2,
                   f'{val:.3f}', ha='center', va='center',
                   fontsize=LABEL_FONTSIZE, fontweight='bold', color='white',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.7))
        
        ax.set_ylim(0, 1.0)
        ax.set_ylabel('Score', fontsize=LABEL_FONTSIZE, fontweight='bold')
        ax.set_title(f'Mix Dataset - {mix_result["strategy"]}', 
                    fontsize=SUBTITLE_FONTSIZE, fontweight='bold')
        ax.grid(axis='y', alpha=GRID_ALPHA, linestyle=GRID_LINESTYLE)
        
        # Add misclass info
        ax.text(0.5, 0.95, f'Misclassified: {mix_result["misclass_rate"]:.2f}%',
               transform=ax.transAxes, ha='center', va='top',
               fontsize=TICK_FONTSIZE+1, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.2))
    
    # Single results
    if single_result:
        ax = axes[1]
        metrics = ['balanced_accuracy', 'f1_macro', 'accuracy']
        values = [single_result[m] for m in metrics]
        labels = ['Balanced\nAccuracy', 'F1-Score', 'Accuracy']
        
        bars = ax.bar(labels, values, color=model_color, alpha=BAR_ALPHA_SECONDARY,
                     edgecolor=BAR_EDGE_COLOR, linewidth=BAR_EDGE_WIDTH*1.5)
        
        # Add values
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., val/2,
                   f'{val:.3f}', ha='center', va='center',
                   fontsize=LABEL_FONTSIZE, fontweight='bold', color='white',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.7))
        
        ax.set_ylim(0, 1.0)
        ax.set_ylabel('Score', fontsize=LABEL_FONTSIZE, fontweight='bold')
        ax.set_title(f'Single Dataset - {single_result["strategy"]}',
                    fontsize=SUBTITLE_FONTSIZE, fontweight='bold')
        ax.grid(axis='y', alpha=GRID_ALPHA, linestyle=GRID_LINESTYLE)
        
        # Add misclass info
        ax.text(0.5, 0.95, f'Misclassified: {single_result["misclass_rate"]:.2f}%',
               transform=ax.transAxes, ha='center', va='top',
               fontsize=TICK_FONTSIZE+1, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.2))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight', facecolor=FIGURE_FACECOLOR)
    plt.close()

def create_pdf_report(output_path, mix_results, single_results, top_models):
    """Create comprehensive PDF report with all figures"""
    print(f"\n  Creating PDF report...")
    
    with PdfPages(output_path) as pdf:
        # Page 1: Title page
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.6, REPORT_TITLE, ha='center', fontsize=28, fontweight='bold')
        fig.text(0.5, 0.5, f'Top {len(top_models)} Models Comparison', 
                ha='center', fontsize=20)
        fig.text(0.5, 0.4, f'Generated: {datetime.now().strftime("%B %d, %Y at %H:%M")}',
                ha='center', fontsize=14, style='italic')
        
        # Add summary info
        summary_text = f"""
Models Analyzed: {len(mix_results)} (Mix) + {len(single_results)} (Single)
Top {len(top_models)} Models:
"""
        for i, model_data in enumerate(sorted([m for m in mix_results + single_results 
                                              if m['model'] in top_models],
                                             key=lambda x: x['balanced_accuracy'], reverse=True)[:NUM_TOP_MODELS], 1):
            summary_text += f"\n{i}. {get_model_display_name(model_data['model'])} - Bal Acc: {model_data['balanced_accuracy']:.4f}"
        
        fig.text(0.5, 0.25, summary_text, ha='center', fontsize=12, 
                family='monospace', va='top',
                bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.3))
        
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 2: Combined comparison
        if GENERATE_COMPARISON_CHART:
            print("    Adding combined comparison...")
            fig = plt.figure(figsize=(20, 12))
            gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)
            
            for idx, (metric_key, metric_name) in enumerate(METRICS_TO_PLOT):
                row = idx // 2
                col = idx % 2
                ax = fig.add_subplot(gs[row, col])
                
                heights_mix = []
                heights_single = []
                colors = []
                labels = []
                
                for model in top_models:
                    mix_result = next((r for r in mix_results if r['model'] == model), None)
                    single_result = next((r for r in single_results if r['model'] == model), None)
                    
                    heights_mix.append(mix_result[metric_key] if mix_result else 0)
                    heights_single.append(single_result[metric_key] if single_result else 0)
                    colors.append(get_model_color(model))
                    labels.append(get_model_display_name(model))
                
                x = np.arange(len(top_models))
                
                bars1 = ax.bar(x - BAR_WIDTH/2, heights_mix, BAR_WIDTH, label='Mix Dataset',
                              color=colors, alpha=BAR_ALPHA_MAIN,
                              edgecolor=BAR_EDGE_COLOR, linewidth=BAR_EDGE_WIDTH)
                bars2 = ax.bar(x + BAR_WIDTH/2, heights_single, BAR_WIDTH, label='Single Dataset',
                              color=colors, alpha=BAR_ALPHA_SECONDARY,
                              edgecolor=BAR_EDGE_COLOR, linewidth=BAR_EDGE_WIDTH)
                
                for bar1, bar2 in zip(bars1, bars2):
                    height1 = bar1.get_height()
                    height2 = bar2.get_height()
                    
                    if height1 > 0:
                        text = f'{height1:.1f}%' if metric_key == 'misclass_rate' else f'{height1:.3f}'
                        ax.text(bar1.get_x() + bar1.get_width()/2., height1/2,
                               text, ha='center', va='center', fontsize=10, fontweight='bold',
                               color='white', bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
                    
                    if height2 > 0:
                        text = f'{height2:.1f}%' if metric_key == 'misclass_rate' else f'{height2:.3f}'
                        ax.text(bar2.get_x() + bar2.get_width()/2., height2/2,
                               text, ha='center', va='center', fontsize=10, fontweight='bold',
                               color='white', bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
                
                ax.set_ylabel(metric_name, fontsize=LABEL_FONTSIZE, fontweight='bold')
                ax.set_xlabel('Model', fontsize=LABEL_FONTSIZE, fontweight='bold')
                ax.set_title(f'{metric_name}', fontsize=SUBTITLE_FONTSIZE, fontweight='bold', pad=10)
                ax.set_xticks(x)
                ax.set_xticklabels(labels, fontsize=TICK_FONTSIZE, fontweight='bold')
                ax.legend(fontsize=LEGEND_FONTSIZE, loc='best')
                ax.grid(axis='y', alpha=GRID_ALPHA, linestyle=GRID_LINESTYLE)
                
                if metric_key == 'misclass_rate':
                    max_val = max(max(heights_mix), max(heights_single))
                    ax.set_ylim(0, max_val * 1.2)
                else:
                    ax.set_ylim(0, 1.0)
            
            ax_table = fig.add_subplot(gs[2, :])
            create_summary_table(ax_table, mix_results, single_results, top_models)
            
            plt.suptitle(f'{REPORT_TITLE}\nTop {len(top_models)} Models Comparison',
                        fontsize=TITLE_FONTSIZE+2, fontweight='bold', y=0.98)
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        # Pages 3+: Individual model breakdowns
        if GENERATE_MODEL_BREAKDOWNS:
            print("    Adding model breakdowns...")
            for model in top_models:
                mix_result = next((r for r in mix_results if r['model'] == model), None)
                single_result = next((r for r in single_results if r['model'] == model), None)
                
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                model_color = get_model_color(model)
                model_name = get_model_display_name(model)
                
                fig.suptitle(f'{model_name} - Detailed Performance Breakdown',
                            fontsize=TITLE_FONTSIZE, fontweight='bold')
                
                if mix_result:
                    ax = axes[0]
                    metrics = ['balanced_accuracy', 'f1_macro', 'accuracy']
                    values = [mix_result[m] for m in metrics]
                    labels = ['Balanced\nAccuracy', 'F1-Score', 'Accuracy']
                    
                    bars = ax.bar(labels, values, color=model_color, alpha=BAR_ALPHA_MAIN,
                                 edgecolor=BAR_EDGE_COLOR, linewidth=BAR_EDGE_WIDTH*1.5)
                    
                    for bar, val in zip(bars, values):
                        ax.text(bar.get_x() + bar.get_width()/2., val/2,
                               f'{val:.3f}', ha='center', va='center',
                               fontsize=LABEL_FONTSIZE, fontweight='bold', color='white',
                               bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.7))
                    
                    ax.set_ylim(0, 1.0)
                    ax.set_ylabel('Score', fontsize=LABEL_FONTSIZE, fontweight='bold')
                    ax.set_title(f'Mix Dataset - {mix_result["strategy"]}', 
                                fontsize=SUBTITLE_FONTSIZE, fontweight='bold')
                    ax.grid(axis='y', alpha=GRID_ALPHA, linestyle=GRID_LINESTYLE)
                    
                    ax.text(0.5, 0.95, f'Misclassified: {mix_result["misclass_rate"]:.2f}%',
                           transform=ax.transAxes, ha='center', va='top',
                           fontsize=TICK_FONTSIZE+1, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.2))
                
                if single_result:
                    ax = axes[1]
                    metrics = ['balanced_accuracy', 'f1_macro', 'accuracy']
                    values = [single_result[m] for m in metrics]
                    labels = ['Balanced\nAccuracy', 'F1-Score', 'Accuracy']
                    
                    bars = ax.bar(labels, values, color=model_color, alpha=BAR_ALPHA_SECONDARY,
                                 edgecolor=BAR_EDGE_COLOR, linewidth=BAR_EDGE_WIDTH*1.5)
                    
                    for bar, val in zip(bars, values):
                        ax.text(bar.get_x() + bar.get_width()/2., val/2,
                               f'{val:.3f}', ha='center', va='center',
                               fontsize=LABEL_FONTSIZE, fontweight='bold', color='white',
                               bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.7))
                    
                    ax.set_ylim(0, 1.0)
                    ax.set_ylabel('Score', fontsize=LABEL_FONTSIZE, fontweight='bold')
                    ax.set_title(f'Single Dataset - {single_result["strategy"]}',
                                fontsize=SUBTITLE_FONTSIZE, fontweight='bold')
                    ax.grid(axis='y', alpha=GRID_ALPHA, linestyle=GRID_LINESTYLE)
                    
                    ax.text(0.5, 0.95, f'Misclassified: {single_result["misclass_rate"]:.2f}%',
                           transform=ax.transAxes, ha='center', va='top',
                           fontsize=TICK_FONTSIZE+1, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.2))
                
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
        
        # Add metadata
        d = pdf.infodict()
        d['Title'] = REPORT_TITLE
        d['Author'] = 'ML_in_LTDNA Pipeline'
        d['Subject'] = 'Model Performance Analysis'
        d['Keywords'] = 'Machine Learning, DNA Analysis, Model Comparison'
        d['CreationDate'] = datetime.now()
    
    print(f"  ✓ PDF created with {len(top_models) + 2} pages")
def main():
    print("="*80)
    print("CUSTOMIZABLE REPORT GENERATOR")
    print("="*80)
    print(f"\nReport Title: {REPORT_TITLE}")
    print(f"Output Directory: {OUTPUT_DIR_NAME}")
    print(f"Top Models: {NUM_TOP_MODELS}")
    print(f"Figure DPI: {FIGURE_DPI}")
    print("="*80)
    
    # Setup output directory
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, OUTPUT_DIR_NAME)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    print("\n[1/5] Loading results...")
    mix_data = load_dataset_results('mix')
    single_data = load_dataset_results('single')
    
    if not mix_data or not single_data:
        print("✗ Error: Could not load results from both datasets")
        return
    
    # Extract best results
    print("[2/5] Extracting best results...")
    mix_results = extract_best_results(mix_data, 'Mix')
    single_results = extract_best_results(single_data, 'Single')
    
    print(f"  Mix: {len(mix_results)} models")
    print(f"  Single: {len(single_results)} models")
    
    # Get top models
    all_results = mix_results + single_results
    model_best = {}
    for result in all_results:
        model = result['model']
        if model not in model_best or result['balanced_accuracy'] > model_best[model]['balanced_accuracy']:
            model_best[model] = result
    
    sorted_models = sorted(model_best.values(), key=lambda x: x['balanced_accuracy'], reverse=True)
    top_models = [m['model'] for m in sorted_models[:NUM_TOP_MODELS]]
    
    print(f"\nTop {NUM_TOP_MODELS} Models:")
    for i, model_data in enumerate(sorted_models[:NUM_TOP_MODELS], 1):
        print(f"  {i}. {get_model_display_name(model_data['model'])} "
              f"(Bal Acc: {model_data['balanced_accuracy']:.4f})")
    
    # Filter results for top models
    mix_top = [r for r in mix_results if r['model'] in top_models]
    single_top = [r for r in single_results if r['model'] in top_models]
    
    # Generate reports
    if GENERATE_COMPARISON_CHART:
        print("\n[3/5] Creating combined comparison chart...")
        output_path = os.path.join(OUTPUT_DIR, f'top{NUM_TOP_MODELS}_comparison.{FIGURE_FORMAT}')
        create_combined_comparison(mix_top, single_top, top_models, output_path)
        print(f"  ✓ Saved: top{NUM_TOP_MODELS}_comparison.{FIGURE_FORMAT}")
    
    if GENERATE_INDIVIDUAL_METRICS:
        print("\n[4/5] Creating individual metric charts...")
        for metric_key, metric_name in METRICS_TO_PLOT:
            output_path = os.path.join(OUTPUT_DIR, f'top{NUM_TOP_MODELS}_{metric_key}.{FIGURE_FORMAT}')
            create_metric_bar_chart(mix_top, single_top, top_models, metric_key, 
                                   metric_name, output_path)
            print(f"  ✓ Saved: top{NUM_TOP_MODELS}_{metric_key}.{FIGURE_FORMAT}")
    
    if GENERATE_MODEL_BREAKDOWNS:
        print("\n[5/5] Creating model breakdowns...")
        for model in top_models:
            mix_result = next((r for r in mix_top if r['model'] == model), None)
            single_result = next((r for r in single_top if r['model'] == model), None)
            
            safe_name = model.replace(' ', '_').lower()
            output_path = os.path.join(OUTPUT_DIR, f'{safe_name}_breakdown.{FIGURE_FORMAT}')
            create_model_breakdown(model, mix_result, single_result, output_path)
            print(f"  ✓ Saved: {safe_name}_breakdown.{FIGURE_FORMAT}")
    
    if GENERATE_PDF_REPORT:
        print("\n[6/6] Creating comprehensive PDF report...")
        pdf_path = os.path.join(OUTPUT_DIR, f'{OUTPUT_DIR_NAME}_report.pdf')
        create_pdf_report(pdf_path, mix_top, single_top, top_models)
        print(f"  ✓ Saved: {OUTPUT_DIR_NAME}_report.pdf")
    
    print("\n" + "="*80)
    print("✓ REPORT GENERATION COMPLETE")
    print("="*80)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nTo customize:")
    print("  1. Edit the CONFIGURATION section at the top of this script")
    print("  2. Modify colors, fonts, sizes, etc.")
    print("  3. Re-run the script to regenerate with new settings")
    print("="*80)

if __name__ == "__main__":
    main()
