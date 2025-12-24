"""
Generate HTML Report from Aggregated Multi-Seed Results
Creates bar charts showing top 3 models by strategy and overall best
"""
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_aggregated_results(dataset_type):
    """Load aggregated results for a dataset"""
    results_folder = f"{dataset_type.capitalize()}_Results"
    json_path = os.path.join(PROJECT_ROOT, results_folder, 'multi_seed_aggregated', 'aggregated_results.json')
    
    if not os.path.exists(json_path):
        print(f"✗ Aggregated results not found: {json_path}")
        return None
    
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_top_performers(data, n=3):
    """Extract top N performers by balanced accuracy"""
    performers = []
    
    for model_name, strategies in data.items():
        for strategy_name, metrics in strategies.items():
            bal_acc = metrics.get('balanced_accuracy', {})
            f1_macro = metrics.get('f1_macro', {})
            
            if bal_acc.get('mean') is not None:
                performers.append({
                    'model': model_name,
                    'strategy': strategy_name,
                    'bal_acc_mean': bal_acc['mean'],
                    'bal_acc_std': bal_acc['std'],
                    'f1_mean': f1_macro['mean'],
                    'f1_std': f1_macro['std']
                })
    
    # Sort by balanced accuracy mean
    performers.sort(key=lambda x: x['bal_acc_mean'], reverse=True)
    return performers[:n]

def get_best_models_overall(data, n=3):
    """Get top N models (best strategy for each model)"""
    best_per_model = {}
    
    for model_name, strategies in data.items():
        best_bal_acc = -1
        best_entry = None
        
        for strategy_name, metrics in strategies.items():
            bal_acc = metrics.get('balanced_accuracy', {})
            f1_macro = metrics.get('f1_macro', {})
            
            if bal_acc.get('mean') is not None and bal_acc['mean'] > best_bal_acc:
                best_bal_acc = bal_acc['mean']
                best_entry = {
                    'model': model_name,
                    'strategy': strategy_name,
                    'bal_acc_mean': bal_acc['mean'],
                    'bal_acc_std': bal_acc['std'],
                    'f1_mean': f1_macro['mean'],
                    'f1_std': f1_macro['std']
                }
        
        if best_entry:
            best_per_model[model_name] = best_entry
    
    # Sort and take top N
    sorted_models = sorted(best_per_model.values(), key=lambda x: x['bal_acc_mean'], reverse=True)
    return sorted_models[:n]

def create_bar_chart(performers, title, output_path, show_strategy=True):
    """Create bar chart for top performers"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Prepare data
    if show_strategy:
        labels = [f"{p['model'].replace('_', ' ').title()}\n({p['strategy']})" for p in performers]
    else:
        labels = [p['model'].replace('_', ' ').title() for p in performers]
    
    bal_acc_means = [p['bal_acc_mean'] for p in performers]
    bal_acc_stds = [p['bal_acc_std'] for p in performers]
    
    # Colors
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    x = np.arange(len(performers))
    bars = ax.bar(x, bal_acc_means, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    # Add error bars
    ax.errorbar(x, bal_acc_means, yerr=bal_acc_stds, fmt='none', ecolor='black', 
                capsize=10, capthick=2, elinewidth=2)
    
    # Add value labels on bars
    for i, (bar, performer) in enumerate(zip(bars, performers)):
        height = bar.get_height()
        mean_val = performer['bal_acc_mean']
        std_val = performer['bal_acc_std']
        
        ax.text(bar.get_x() + bar.get_width()/2., height/2,
               f'{mean_val:.4f}\n±{std_val:.4f}',
               ha='center', va='center', fontsize=14, fontweight='bold',
               color='white',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8))
        
        # Add F1 score below
        ax.text(bar.get_x() + bar.get_width()/2., -0.08,
               f"F1: {performer['f1_mean']:.4f}±{performer['f1_std']:.4f}",
               ha='center', va='top', fontsize=10, style='italic')
    
    # Styling
    ax.set_ylabel('Balanced Accuracy', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return output_path

def generate_html_report(mix_data, single_data, output_path):
    """Generate comprehensive HTML report"""
    
    # Get top performers
    mix_top_strategy = get_top_performers(mix_data, n=3)
    mix_top_overall = get_best_models_overall(mix_data, n=3)
    single_top_strategy = get_top_performers(single_data, n=3)
    single_top_overall = get_best_models_overall(single_data, n=3)
    
    # Create output directory
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate charts
    charts = {
        'mix_strategy': create_bar_chart(
            mix_top_strategy,
            'Mix Dataset - Top 3 Model Strategies',
            os.path.join(output_dir, 'mix_top3_strategies.png'),
            show_strategy=True
        ),
        'mix_overall': create_bar_chart(
            mix_top_overall,
            'Mix Dataset - Top 3 Models (Best Strategy Each)',
            os.path.join(output_dir, 'mix_top3_models.png'),
            show_strategy=False
        ),
        'single_strategy': create_bar_chart(
            single_top_strategy,
            'Single Dataset - Top 3 Model Strategies',
            os.path.join(output_dir, 'single_top3_strategies.png'),
            show_strategy=True
        ),
        'single_overall': create_bar_chart(
            single_top_overall,
            'Single Dataset - Top 3 Models (Best Strategy Each)',
            os.path.join(output_dir, 'single_top3_models.png'),
            show_strategy=False
        ),
    }
    
    # Generate HTML
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Seed Aggregated Results - Top 3 Performers</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            padding: 40px;
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }}
        .subtitle {{
            text-align: center;
            color: #7f8c8d;
            font-size: 1.2em;
            margin-bottom: 40px;
        }}
        .info-box {{
            background: #ecf0f1;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            border-left: 5px solid #3498db;
        }}
        .info-box h3 {{
            margin-top: 0;
            color: #2c3e50;
        }}
        .dataset-section {{
            margin-bottom: 50px;
            padding: 30px;
            background: #f8f9fa;
            border-radius: 10px;
            border: 2px solid #e9ecef;
        }}
        .dataset-title {{
            color: #16a085;
            font-size: 2em;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #16a085;
        }}
        .chart-container {{
            margin: 30px 0;
            text-align: center;
        }}
        .chart-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }}
        .chart-title {{
            font-size: 1.3em;
            font-weight: bold;
            color: #34495e;
            margin-bottom: 15px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 8px;
            overflow: hidden;
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
        .rank {{
            font-weight: bold;
            color: white;
            padding: 5px 10px;
            border-radius: 50%;
            display: inline-block;
            width: 30px;
            height: 30px;
            text-align: center;
            line-height: 30px;
        }}
        .rank-1 {{ background: #e74c3c; }}
        .rank-2 {{ background: #3498db; }}
        .rank-3 {{ background: #2ecc71; }}
        .metric {{
            font-family: 'Courier New', monospace;
            font-weight: bold;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            color: #7f8c8d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🏆 Multi-Seed Aggregated Results</h1>
        <div class="subtitle">Top 3 Performing Models Across 10 Random Seeds</div>
        
        <div class="info-box">
            <h3>📊 About This Report</h3>
            <p><strong>Seeds:</strong> 42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021</p>
            <p><strong>Metrics:</strong> Mean ± Standard Deviation across all seeds</p>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <!-- MIX DATASET -->
        <div class="dataset-section">
            <h2 class="dataset-title">Mix Dataset Results</h2>
            
            <div class="chart-container">
                <div class="chart-title">Top 3 Model-Strategy Combinations</div>
                <img src="{os.path.basename(charts['mix_strategy'])}" alt="Mix Top 3 Strategies">
            </div>
            
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Model</th>
                        <th>Strategy</th>
                        <th>Balanced Accuracy</th>
                        <th>F1-Score</th>
                    </tr>
                </thead>
                <tbody>
"""
    
    # Add Mix strategy table rows
    for i, perf in enumerate(mix_top_strategy, 1):
        html_content += f"""
                    <tr>
                        <td><span class="rank rank-{i}">{i}</span></td>
                        <td><strong>{perf['model'].replace('_', ' ').title()}</strong></td>
                        <td>{perf['strategy']}</td>
                        <td class="metric">{perf['bal_acc_mean']:.4f} ± {perf['bal_acc_std']:.4f}</td>
                        <td class="metric">{perf['f1_mean']:.4f} ± {perf['f1_std']:.4f}</td>
                    </tr>
"""
    
    html_content += """
                </tbody>
            </table>
            
            <div class="chart-container">
                <div class="chart-title">Top 3 Models Overall (Best Strategy)</div>
                <img src="{}" alt="Mix Top 3 Models">
            </div>
            
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Model</th>
                        <th>Best Strategy</th>
                        <th>Balanced Accuracy</th>
                        <th>F1-Score</th>
                    </tr>
                </thead>
                <tbody>
""".format(os.path.basename(charts['mix_overall']))
    
    # Add Mix overall table rows
    for i, perf in enumerate(mix_top_overall, 1):
        html_content += f"""
                    <tr>
                        <td><span class="rank rank-{i}">{i}</span></td>
                        <td><strong>{perf['model'].replace('_', ' ').title()}</strong></td>
                        <td>{perf['strategy']}</td>
                        <td class="metric">{perf['bal_acc_mean']:.4f} ± {perf['bal_acc_std']:.4f}</td>
                        <td class="metric">{perf['f1_mean']:.4f} ± {perf['f1_std']:.4f}</td>
                    </tr>
"""
    
    html_content += """
                </tbody>
            </table>
        </div>

        <!-- SINGLE DATASET -->
        <div class="dataset-section">
            <h2 class="dataset-title">Single Dataset Results</h2>
            
            <div class="chart-container">
                <div class="chart-title">Top 3 Model-Strategy Combinations</div>
                <img src="{}" alt="Single Top 3 Strategies">
            </div>
            
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Model</th>
                        <th>Strategy</th>
                        <th>Balanced Accuracy</th>
                        <th>F1-Score</th>
                    </tr>
                </thead>
                <tbody>
""".format(os.path.basename(charts['single_strategy']))
    
    # Add Single strategy table rows
    for i, perf in enumerate(single_top_strategy, 1):
        html_content += f"""
                    <tr>
                        <td><span class="rank rank-{i}">{i}</span></td>
                        <td><strong>{perf['model'].replace('_', ' ').title()}</strong></td>
                        <td>{perf['strategy']}</td>
                        <td class="metric">{perf['bal_acc_mean']:.4f} ± {perf['bal_acc_std']:.4f}</td>
                        <td class="metric">{perf['f1_mean']:.4f} ± {perf['f1_std']:.4f}</td>
                    </tr>
"""
    
    html_content += """
                </tbody>
            </table>
            
            <div class="chart-container">
                <div class="chart-title">Top 3 Models Overall (Best Strategy)</div>
                <img src="{}" alt="Single Top 3 Models">
            </div>
            
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Model</th>
                        <th>Best Strategy</th>
                        <th>Balanced Accuracy</th>
                        <th>F1-Score</th>
                    </tr>
                </thead>
                <tbody>
""".format(os.path.basename(charts['single_overall']))
    
    # Add Single overall table rows
    for i, perf in enumerate(single_top_overall, 1):
        html_content += f"""
                    <tr>
                        <td><span class="rank rank-{i}">{i}</span></td>
                        <td><strong>{perf['model'].replace('_', ' ').title()}</strong></td>
                        <td>{perf['strategy']}</td>
                        <td class="metric">{perf['bal_acc_mean']:.4f} ± {perf['bal_acc_std']:.4f}</td>
                        <td class="metric">{perf['f1_mean']:.4f} ± {perf['f1_std']:.4f}</td>
                    </tr>
"""
    
    html_content += """
                </tbody>
            </table>
        </div>
        
        <div class="footer">
            <p>Generated from multi-seed aggregated results | ML in LT-DNA Project</p>
        </div>
    </div>
</body>
</html>
"""
    
    # Write HTML
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return output_path

def main():
    print("="*80)
    print("AGGREGATED RESULTS REPORT GENERATOR")
    print("="*80)
    
    # Load data
    print("\n[1/3] Loading aggregated results...")
    mix_data = load_aggregated_results('mix')
    single_data = load_aggregated_results('single')
    
    if not mix_data or not single_data:
        print("✗ Failed to load aggregated results")
        return
    
    print("✓ Loaded results for both datasets")
    
    # Generate report
    print("\n[2/3] Generating charts and HTML report...")
    output_path = os.path.join(PROJECT_ROOT, 'aggregated_top3_report.html')
    report_path = generate_html_report(mix_data, single_data, output_path)
    
    print(f"✓ Charts generated")
    print(f"✓ HTML report created: {report_path}")
    
    print("\n[3/3] Opening report in browser...")
    import webbrowser
    webbrowser.open('file://' + os.path.abspath(report_path))
    
    print("\n" + "="*80)
    print("✓ REPORT GENERATION COMPLETE")
    print("="*80)
    print(f"\nReport saved to: {report_path}")
    print("="*80)

if __name__ == "__main__":
    main()
