"""
Generate Top Models Visualization Report
Combines results from both Mix and Single datasets
Shows top 3 models with consistent color scheme and misclassification rates
"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configuration
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'top_models_report')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Color scheme - one color per model family
MODEL_COLORS = {
    'tabpfn': '#e74c3c',      # Red
    'xgboost': '#3498db',     # Blue
    'lightgbm': '#2ecc71',    # Green
    'extratrees': '#9b59b6',  # Purple
    'randomforest': '#f39c12', # Orange
    'mlp': '#1abc9c',         # Teal
    'svm': '#e67e22',         # Dark Orange
    'gnb': '#95a5a6',         # Gray
    'logre': '#34495e'        # Dark Gray
}

def load_dataset_results(dataset_type):
    """Load results from a dataset"""
    results_folder = f"{dataset_type.capitalize()}_Results"
    json_path = os.path.join(PROJECT_ROOT, results_folder, 'reports', 'combined_results.json')
    
    if not os.path.exists(json_path):
        print(f"⚠ Warning: Results not found for {dataset_type} dataset: {json_path}")
        return None
    
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_best_results(data, dataset_name):
    """Extract best result for each model (handling TabPFN specially)"""
    results = []
    
    for model_name, model_data in data.items():
        if 'results' not in model_data:
            continue
        
        # For each model, find the best strategy
        best_strategy = None
        best_bal_acc = -1
        
        for strategy, metrics in model_data['results'].items():
            bal_acc = metrics.get('balanced_accuracy', 0)
            if bal_acc > best_bal_acc:
                best_bal_acc = bal_acc
                best_strategy = strategy
        
        if best_strategy:
            metrics = model_data['results'][best_strategy]
            
            # Calculate misclassification rate
            accuracy = metrics.get('accuracy', 0)
            misclass_rate = (1 - accuracy) * 100
            
            results.append({
                'model': model_name,
                'strategy': best_strategy,
                'dataset': dataset_name,
                'balanced_accuracy': metrics.get('balanced_accuracy', 0),
                'accuracy': metrics.get('accuracy', 0),
                'f1_macro': metrics.get('f1_macro', 0),
                'log_loss': metrics.get('log_loss', None),
                'misclass_rate': misclass_rate,
                'misclass_count': int(misclass_rate),  # Will calculate properly later
                'total_samples': 100  # Placeholder
            })
    
    return results

def get_model_color(model_name):
    """Get color for a model"""
    for key, color in MODEL_COLORS.items():
        if key in model_name.lower():
            return color
    return '#7f8c8d'  # Default gray

def create_top3_comparison(mix_results, single_results):
    """Create comparison visualization for top 3 models"""
    # Combine and rank by balanced accuracy
    all_results = mix_results + single_results
    
    # Group by model and get best across both datasets
    model_best = {}
    for result in all_results:
        model = result['model']
        if model not in model_best or result['balanced_accuracy'] > model_best[model]['balanced_accuracy']:
            model_best[model] = result
    
    # Sort and get top 3
    sorted_models = sorted(model_best.values(), key=lambda x: x['balanced_accuracy'], reverse=True)
    top3_models = [m['model'] for m in sorted_models[:3]]
    
    print(f"\n{'='*80}")
    print("TOP 3 MODELS (by Balanced Accuracy)")
    print(f"{'='*80}")
    for i, model_data in enumerate(sorted_models[:3], 1):
        print(f"{i}. {model_data['model']:15s} | Bal Acc: {model_data['balanced_accuracy']:.4f} | "
              f"F1: {model_data['f1_macro']:.4f} | Misclass: {model_data['misclass_rate']:.2f}%")
    print(f"{'='*80}\n")
    
    # Filter results for top 3 models only
    mix_top3 = [r for r in mix_results if r['model'] in top3_models]
    single_top3 = [r for r in single_results if r['model'] in top3_models]
    
    # Define metrics for plotting
    metrics = [
        ('balanced_accuracy', 'Balanced Accuracy', (0, 0)),
        ('f1_macro', 'F1-Score (Macro)', (0, 1)),
        ('accuracy', 'Accuracy', (1, 0)),
        ('misclass_rate', 'Misclassification Rate (%)', (1, 1))
    ]
    
    # First, create individual plots for each metric
    print("\n   Creating individual metric plots...")
    for metric_key, metric_name, _ in metrics:
        fig_individual, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data
        heights_mix = []
        heights_single = []
        colors = []
        labels = []
        
        for model in top3_models:
            mix_result = next((r for r in mix_top3 if r['model'] == model), None)
            single_result = next((r for r in single_top3 if r['model'] == model), None)
            
            heights_mix.append(mix_result[metric_key] if mix_result else 0)
            heights_single.append(single_result[metric_key] if single_result else 0)
            colors.append(get_model_color(model))
            labels.append(model.upper())
        
        x = np.arange(len(top3_models))
        width = 0.35
        
        # Create bars
        bars1 = ax.bar(x - width/2, heights_mix, width, label='Mix Dataset',
                      color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        bars2 = ax.bar(x + width/2, heights_single, width, label='Single Dataset',
                      color=colors, alpha=0.5, edgecolor='black', linewidth=2)
        
        # Add values inside bars
        for bar1, bar2 in zip(bars1, bars2):
            height1 = bar1.get_height()
            height2 = bar2.get_height()
            
            if height1 > 0:
                text = f'{height1:.1f}%' if metric_key == 'misclass_rate' else f'{height1:.3f}'
                ax.text(bar1.get_x() + bar1.get_width()/2., height1/2,
                       text, ha='center', va='center', fontsize=11, fontweight='bold',
                       color='white', bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.8))
            
            if height2 > 0:
                text = f'{height2:.1f}%' if metric_key == 'misclass_rate' else f'{height2:.3f}'
                ax.text(bar2.get_x() + bar2.get_width()/2., height2/2,
                       text, ha='center', va='center', fontsize=11, fontweight='bold',
                       color='white', bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.8))
        
        ax.set_ylabel(metric_name, fontsize=14, fontweight='bold')
        ax.set_xlabel('Model', fontsize=14, fontweight='bold')
        ax.set_title(f'{metric_name} - Top 3 Models', fontsize=16, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
        ax.legend(fontsize=11, loc='upper right', framealpha=0.9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Set y-axis limits
        if metric_key == 'misclass_rate':
            ax.set_ylim(0, max(max(heights_mix), max(heights_single)) * 1.2)
        else:
            ax.set_ylim(0, 1.0)
        
        plt.tight_layout()
        individual_file = os.path.join(OUTPUT_DIR, f'top3_{metric_key}.png')
        plt.savefig(individual_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   ✓ Saved: top3_{metric_key}.png")
        plt.close()
    
    # Now create the combined figure with subplots
    print("\n   Creating combined plot...")
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)
    
    for metric_key, metric_name, (row, col) in metrics:
        ax = fig.add_subplot(gs[row, col])
        
        # Prepare data
        x_positions = []
        heights_mix = []
        heights_single = []
        colors = []
        labels = []
        
        for i, model in enumerate(top3_models):
            # Find results for this model
            mix_result = next((r for r in mix_top3 if r['model'] == model), None)
            single_result = next((r for r in single_top3 if r['model'] == model), None)
            
            if mix_result:
                heights_mix.append(mix_result[metric_key])
            else:
                heights_mix.append(0)
            
            if single_result:
                heights_single.append(single_result[metric_key])
            else:
                heights_single.append(0)
            
            colors.append(get_model_color(model))
            labels.append(model.upper())
        
        x = np.arange(len(top3_models))
        width = 0.35
        
        # Create bars
        bars1 = ax.bar(x - width/2, heights_mix, width, label='Mix Dataset',
                      color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x + width/2, heights_single, width, label='Single Dataset',
                      color=colors, alpha=0.5, edgecolor='black', linewidth=1.5)
        
        # Add values and misclassification info inside bars
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            height1 = bar1.get_height()
            height2 = bar2.get_height()
            
            if height1 > 0:
                if metric_key == 'misclass_rate':
                    text = f'{height1:.1f}%'
                else:
                    text = f'{height1:.3f}'
                ax.text(bar1.get_x() + bar1.get_width()/2., height1/2,
                       text, ha='center', va='center', fontsize=10, fontweight='bold',
                       color='white', bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
            
            if height2 > 0:
                if metric_key == 'misclass_rate':
                    text = f'{height2:.1f}%'
                else:
                    text = f'{height2:.3f}'
                ax.text(bar2.get_x() + bar2.get_width()/2., height2/2,
                       text, ha='center', va='center', fontsize=10, fontweight='bold',
                       color='white', bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
        
        ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_title(f'{metric_name} - Top 3 Models', fontsize=14, fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Set y-axis limits
        if metric_key == 'misclass_rate':
            ax.set_ylim(0, max(max(heights_mix), max(heights_single)) * 1.2)
        else:
            ax.set_ylim(0, 1.0)
    
    # Add summary table in bottom half
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis('off')
    
    # Create summary table
    table_data = []
    table_data.append(['Model', 'Dataset', 'Strategy', 'Bal Acc', 'F1-Score', 'Accuracy', 'Misclass %'])
    
    for model in top3_models:
        # Mix results
        mix_result = next((r for r in mix_top3 if r['model'] == model), None)
        if mix_result:
            table_data.append([
                model.upper(),
                'Mix',
                mix_result['strategy'],
                f"{mix_result['balanced_accuracy']:.4f}",
                f"{mix_result['f1_macro']:.4f}",
                f"{mix_result['accuracy']:.4f}",
                f"{mix_result['misclass_rate']:.2f}%"
            ])
        
        # Single results
        single_result = next((r for r in single_top3 if r['model'] == model), None)
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
        
        # Add separator
        if model != top3_models[-1]:
            table_data.append(['─'*15, '─'*10, '─'*20, '─'*8, '─'*8, '─'*8, '─'*10])
    
    table = ax_table.table(cellText=table_data, cellLoc='center', loc='center',
                           colWidths=[0.15, 0.10, 0.25, 0.10, 0.10, 0.10, 0.10])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(7):
        cell = table[(0, i)]
        cell.set_facecolor('#34495e')
        cell.set_text_props(weight='bold', color='white')
    
    # Color code model rows
    row_idx = 1
    for model in top3_models:
        color = get_model_color(model)
        # Color the model name cell
        if row_idx < len(table_data):
            cell = table[(row_idx, 0)]
            cell.set_facecolor(color)
            cell.set_text_props(weight='bold', color='white')
        row_idx += 2 if row_idx + 1 < len(table_data) and table_data[row_idx + 1][0] == '' else 1
        if row_idx < len(table_data) and '─' in table_data[row_idx][0]:
            row_idx += 1
    
    plt.suptitle('Top 3 Model Performance Comparison\nMix vs Single Dataset',
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'top3_models_comparison.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: top3_models_comparison.png")
    plt.close()
    
    return top3_models

def create_detailed_breakdown(mix_results, single_results, top3_models):
    """Create detailed breakdown for each top model"""
    for model in top3_models:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'{model.upper()} - Detailed Performance Breakdown',
                    fontsize=16, fontweight='bold')
        
        model_color = get_model_color(model)
        
        # Mix results
        mix_result = next((r for r in mix_results if r['model'] == model), None)
        if mix_result:
            ax = axes[0]
            metrics = ['balanced_accuracy', 'f1_macro', 'accuracy']
            values = [mix_result[m] for m in metrics]
            labels = ['Balanced\nAccuracy', 'F1-Score', 'Accuracy']
            
            bars = ax.bar(labels, values, color=model_color, alpha=0.8,
                         edgecolor='black', linewidth=2)
            
            # Add values and misclass info
            for i, (bar, val) in enumerate(zip(bars, values)):
                ax.text(bar.get_x() + bar.get_width()/2., val/2,
                       f'{val:.3f}', ha='center', va='center',
                       fontsize=12, fontweight='bold', color='white',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.7))
            
            ax.set_ylim(0, 1.0)
            ax.set_ylabel('Score', fontsize=12, fontweight='bold')
            ax.set_title(f'Mix Dataset - {mix_result["strategy"]}', fontsize=13, fontweight='bold')
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Add misclass info
            ax.text(0.5, 0.95, f'Misclassified: {mix_result["misclass_rate"]:.2f}%',
                   transform=ax.transAxes, ha='center', va='top',
                   fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.2))
        
        # Single results
        single_result = next((r for r in single_results if r['model'] == model), None)
        if single_result:
            ax = axes[1]
            metrics = ['balanced_accuracy', 'f1_macro', 'accuracy']
            values = [single_result[m] for m in metrics]
            labels = ['Balanced\nAccuracy', 'F1-Score', 'Accuracy']
            
            bars = ax.bar(labels, values, color=model_color, alpha=0.6,
                         edgecolor='black', linewidth=2)
            
            # Add values
            for i, (bar, val) in enumerate(zip(bars, values)):
                ax.text(bar.get_x() + bar.get_width()/2., val/2,
                       f'{val:.3f}', ha='center', va='center',
                       fontsize=12, fontweight='bold', color='white',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.7))
            
            ax.set_ylim(0, 1.0)
            ax.set_ylabel('Score', fontsize=12, fontweight='bold')
            ax.set_title(f'Single Dataset - {single_result["strategy"]}', fontsize=13, fontweight='bold')
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Add misclass info
            ax.text(0.5, 0.95, f'Misclassified: {single_result["misclass_rate"]:.2f}%',
                   transform=ax.transAxes, ha='center', va='top',
                   fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.2))
        
        plt.tight_layout()
        safe_name = model.replace(' ', '_').lower()
        plt.savefig(os.path.join(OUTPUT_DIR, f'{safe_name}_breakdown.png'),
                   dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {safe_name}_breakdown.png")
        plt.close()

def generate_html_report(mix_results, single_results, top3_models):
    """Generate comprehensive HTML report"""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Top Models Performance Report</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 40px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: #333;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                padding: 40px;
                border-radius: 15px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 4px solid #3498db;
                padding-bottom: 15px;
                font-size: 2.5em;
                margin-top: 0;
            }}
            h2 {{
                color: #34495e;
                margin-top: 40px;
                font-size: 1.8em;
                border-left: 5px solid #e74c3c;
                padding-left: 15px;
            }}
            .timestamp {{
                color: #7f8c8d;
                font-size: 0.9em;
                margin-bottom: 30px;
            }}
            .summary {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 25px;
                border-radius: 10px;
                margin: 25px 0;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            }}
            .summary h3 {{
                margin-top: 0;
                font-size: 1.5em;
            }}
            .model-card {{
                border: 2px solid #ecf0f1;
                border-radius: 10px;
                padding: 20px;
                margin: 20px 0;
                background: #f8f9fa;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .model-header {{
                font-size: 1.4em;
                font-weight: bold;
                margin-bottom: 15px;
                padding: 10px;
                border-radius: 5px;
                color: white;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                background: white;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            th {{
                background: #34495e;
                color: white;
                padding: 15px;
                text-align: left;
                font-weight: bold;
            }}
            td {{
                padding: 12px 15px;
                border-bottom: 1px solid #ecf0f1;
            }}
            tr:hover {{
                background: #f8f9fa;
            }}
            .metric-value {{
                font-weight: bold;
                font-size: 1.1em;
            }}
            .metric-good {{
                color: #27ae60;
            }}
            .metric-warning {{
                color: #f39c12;
            }}
            .metric-bad {{
                color: #e74c3c;
            }}
            img {{
                max-width: 100%;
                height: auto;
                border-radius: 10px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                margin: 20px 0;
            }}
            .color-legend {{
                display: flex;
                flex-wrap: wrap;
                gap: 15px;
                margin: 20px 0;
                padding: 20px;
                background: #ecf0f1;
                border-radius: 10px;
            }}
            .color-item {{
                display: flex;
                align-items: center;
                gap: 8px;
            }}
            .color-box {{
                width: 30px;
                height: 30px;
                border-radius: 5px;
                border: 2px solid #333;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏆 Top Models Performance Report</h1>
            <div class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
            
            <div class="summary">
                <h3>📊 Executive Summary</h3>
                <p>This report presents the top 3 performing machine learning models across both Mix and Single datasets. For TabPFN and other models with multiple strategies, only the best performing strategy is shown.</p>
                <ul>
                    <li><strong>Datasets Analyzed:</strong> Mix and Single</li>
                    <li><strong>Models Evaluated:</strong> {len(set([r['model'] for r in mix_results + single_results]))}</li>
                    <li><strong>Top Models Selected:</strong> 3</li>
                    <li><strong>Selection Criteria:</strong> Balanced Accuracy</li>
                </ul>
            </div>
            
            <h2>🎨 Model Color Scheme</h2>
            <div class="color-legend">
"""
    
    for model in top3_models:
        color = get_model_color(model)
        html += f"""
                <div class="color-item">
                    <div class="color-box" style="background-color: {color};"></div>
                    <span><strong>{model.upper()}</strong></span>
                </div>
"""
    
    html += """
            </div>
            
            <h2>📈 Top 3 Models Comparison</h2>
            <img src="top3_models_comparison.png" alt="Top 3 Models Comparison">
            
            <h2>📋 Detailed Results</h2>
"""
    
    # Create detailed cards for each model
    for i, model in enumerate(top3_models, 1):
        color = get_model_color(model)
        
        mix_result = next((r for r in mix_results if r['model'] == model), None)
        single_result = next((r for r in single_results if r['model'] == model), None)
        
        html += f"""
            <div class="model-card">
                <div class="model-header" style="background-color: {color};">
                    #{i} - {model.upper()}
                </div>
                
                <img src="{model.replace(' ', '_').lower()}_breakdown.png" alt="{model} Breakdown">
                
                <table>
                    <thead>
                        <tr>
                            <th>Dataset</th>
                            <th>Strategy</th>
                            <th>Balanced Accuracy</th>
                            <th>F1-Score</th>
                            <th>Accuracy</th>
                            <th>Misclassified</th>
                        </tr>
                    </thead>
                    <tbody>
"""
        
        if mix_result:
            misclass_class = 'metric-good' if mix_result['misclass_rate'] < 10 else ('metric-warning' if mix_result['misclass_rate'] < 20 else 'metric-bad')
            html += f"""
                        <tr>
                            <td><strong>Mix</strong></td>
                            <td>{mix_result['strategy']}</td>
                            <td class="metric-value metric-good">{mix_result['balanced_accuracy']:.4f}</td>
                            <td class="metric-value">{mix_result['f1_macro']:.4f}</td>
                            <td class="metric-value">{mix_result['accuracy']:.4f}</td>
                            <td class="metric-value {misclass_class}">{mix_result['misclass_rate']:.2f}%</td>
                        </tr>
"""
        
        if single_result:
            misclass_class = 'metric-good' if single_result['misclass_rate'] < 10 else ('metric-warning' if single_result['misclass_rate'] < 20 else 'metric-bad')
            html += f"""
                        <tr>
                            <td><strong>Single</strong></td>
                            <td>{single_result['strategy']}</td>
                            <td class="metric-value metric-good">{single_result['balanced_accuracy']:.4f}</td>
                            <td class="metric-value">{single_result['f1_macro']:.4f}</td>
                            <td class="metric-value">{single_result['accuracy']:.4f}</td>
                            <td class="metric-value {misclass_class}">{single_result['misclass_rate']:.2f}%</td>
                        </tr>
"""
        
        html += """
                    </tbody>
                </table>
            </div>
"""
    
    html += """
            <h2>📝 Notes</h2>
            <ul>
                <li><strong>Balanced Accuracy:</strong> Average of recall scores per class, better for imbalanced datasets</li>
                <li><strong>F1-Score (Macro):</strong> Harmonic mean of precision and recall, averaged across classes</li>
                <li><strong>Misclassification Rate:</strong> Percentage of samples incorrectly classified</li>
                <li><strong>Color Coding:</strong> Each model has a consistent color throughout the report for easy identification</li>
            </ul>
        </div>
    </body>
    </html>
"""
    
    output_path = os.path.join(OUTPUT_DIR, 'top_models_report.html')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✓ Saved: top_models_report.html")
    return output_path

def main():
    print("="*80)
    print("TOP MODELS VISUALIZATION REPORT GENERATOR")
    print("="*80)
    
    # Load data from both datasets
    print("\n[1/5] Loading results...")
    mix_data = load_dataset_results('mix')
    single_data = load_dataset_results('single')
    
    if not mix_data or not single_data:
        print("✗ Error: Could not load results from both datasets")
        return
    
    # Extract best results
    print("\n[2/5] Extracting best results for each model...")
    mix_results = extract_best_results(mix_data, 'Mix')
    single_results = extract_best_results(single_data, 'Single')
    
    print(f"  Mix: {len(mix_results)} models")
    print(f"  Single: {len(single_results)} models")
    
    # Create visualizations
    print("\n[3/5] Creating top 3 comparison...")
    top3_models = create_top3_comparison(mix_results, single_results)
    
    print("\n[4/5] Creating detailed breakdowns...")
    create_detailed_breakdown(mix_results, single_results, top3_models)
    
    print("\n[5/5] Generating HTML report...")
    html_path = generate_html_report(mix_results, single_results, top3_models)
    
    print("\n" + "="*80)
    print("✓ REPORT GENERATION COMPLETE")
    print("="*80)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"HTML Report: {html_path}")
    print(f"\nOpen the HTML report in your browser to view the results!")
    print("="*80)

if __name__ == "__main__":
    main()
