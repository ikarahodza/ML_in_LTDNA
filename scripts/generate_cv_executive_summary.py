"""
Generate Executive Summary PDF from Cross-Validation Results (30-Fold)
Creates professional PDF report similar to train-test split reports
"""

import json
import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Set professional styling
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#fafafa'
plt.rcParams['font.size'] = 10

# Professional color scheme
COLOR_PRIMARY = '#5e81ac'
COLOR_SUCCESS = '#a3be8c'
COLOR_WARNING = '#ebcb8b'
COLOR_DANGER = '#bf616a'
COLOR_INFO = '#88c0d0'
CHART_COLORS = ['#5e81ac', '#a3be8c', '#ebcb8b', '#d08770', '#b48ead', '#88c0d0', '#81a1c1', '#bf616a']

def load_cv_results(dataset_type='Mix'):
    """Load cross-validation results."""
    # Try both possible paths
    cv_path1 = f'{dataset_type}_Results/cross_validation_results/cv_results.json'
    cv_path2 = 'cross_validation_results/cv_results.json'
    
    cv_path = cv_path1 if os.path.exists(cv_path1) else cv_path2
    
    if not os.path.exists(cv_path):
        print(f"✗ CV results not found in {cv_path1} or {cv_path2}")
        return None
    
    with open(cv_path, 'r') as f:
        return json.load(f)

def create_title_page(pdf, cv_data, dataset_type, cv_folds):
    """Create professional title page."""
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # Get top model
    best_model = max(cv_data.items(), 
                    key=lambda x: x[1]['metrics']['balanced_accuracy']['mean'])
    model_name, model_data = best_model
    best_bal_acc = model_data['metrics']['balanced_accuracy']['mean']
    best_std = model_data['metrics']['balanced_accuracy']['std']
    
    title_text = f"""
    
    
    Machine Learning for DNA Artifact Classification
    Low Template DNA (LTDNA) Analysis
    
    EXECUTIVE SUMMARY REPORT
    Cross-Validation Results ({cv_folds}-Fold)
    
    
    ═══════════════════════════════════════════════════════════════
    
    Dataset: {dataset_type} Profiles
    Date: {datetime.now().strftime("%B %d, %Y")}
    Validation: {cv_folds}-Fold Stratified Cross-Validation
    Models Evaluated: {len(cv_data)}
    
    
    ═══════════════════════════════════════════════════════════════
    
    
    KEY FINDINGS
    
    🏆 Best Model: {model_name.replace('_', ' ').title()}
       • Balanced Accuracy: {best_bal_acc:.4f} ± {best_std:.4f}
       • F1-Score (Macro): {model_data['metrics']['f1_macro']['mean']:.4f}
       • Stability: {'High' if best_std < 0.05 else 'Moderate' if best_std < 0.10 else 'Variable'}
    
    📊 Validation Approach:
       • {cv_folds}-fold stratified cross-validation
       • Ensures robust performance estimates
       • Accounts for data variability
       • Reduces overfitting bias
    
    
    ═══════════════════════════════════════════════════════════════
    """
    
    ax.text(0.5, 0.5, title_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='center', horizontalalignment='center',
            fontfamily='monospace', linespacing=1.5)
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def create_performance_comparison(pdf, cv_data, dataset_type, cv_folds):
    """Create performance comparison chart with error bars."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 10))
    
    # Sort by balanced accuracy
    sorted_models = sorted(cv_data.items(), 
                          key=lambda x: x[1]['metrics']['balanced_accuracy']['mean'],
                          reverse=True)
    
    model_names = [m[0].replace('_', '\n').title() for m in sorted_models]
    
    # Balanced Accuracy
    bal_acc_means = [m[1]['metrics']['balanced_accuracy']['mean'] for m in sorted_models]
    bal_acc_stds = [m[1]['metrics']['balanced_accuracy']['std'] for m in sorted_models]
    
    bars1 = ax1.barh(model_names, bal_acc_means, color=CHART_COLORS[:len(model_names)],
                     alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.errorbar(bal_acc_means, range(len(model_names)), xerr=bal_acc_stds,
                fmt='none', ecolor='black', capsize=5, capthick=2)
    
    ax1.set_xlabel('Balanced Accuracy', fontweight='bold', fontsize=12)
    ax1.set_title(f'{dataset_type} Dataset - Model Performance Comparison ({cv_folds}-Fold CV)',
                 fontweight='bold', fontsize=14, pad=15)
    ax1.set_xlim(0, 1.0)
    ax1.grid(axis='x', alpha=0.3)
    ax1.axvline(x=0.8, color='green', linestyle='--', alpha=0.5, label='Target: 0.80')
    ax1.legend()
    
    # Add value labels
    for i, (bar, mean, std) in enumerate(zip(bars1, bal_acc_means, bal_acc_stds)):
        ax1.text(mean + 0.02, bar.get_y() + bar.get_height()/2,
                f'{mean:.4f}\n±{std:.4f}',
                va='center', fontsize=9, fontweight='bold')
    
    # F1-Score
    f1_means = [m[1]['metrics']['f1_macro']['mean'] for m in sorted_models]
    f1_stds = [m[1]['metrics']['f1_macro']['std'] for m in sorted_models]
    
    bars2 = ax2.barh(model_names, f1_means, color=CHART_COLORS[:len(model_names)],
                     alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.errorbar(f1_means, range(len(model_names)), xerr=f1_stds,
                fmt='none', ecolor='black', capsize=5, capthick=2)
    
    ax2.set_xlabel('F1-Score (Macro)', fontweight='bold', fontsize=12)
    ax2.set_title(f'F1-Score Comparison ({cv_folds}-Fold CV)',
                 fontweight='bold', fontsize=14, pad=15)
    ax2.set_xlim(0, 1.0)
    ax2.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, mean, std) in enumerate(zip(bars2, f1_means, f1_stds)):
        ax2.text(mean + 0.02, bar.get_y() + bar.get_height()/2,
                f'{mean:.4f}\n±{std:.4f}',
                va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def create_stability_analysis(pdf, cv_data, dataset_type, cv_folds):
    """Create stability analysis showing variance across folds."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # Sort by standard deviation (most stable first)
    sorted_by_stability = sorted(cv_data.items(),
                                key=lambda x: x[1]['metrics']['balanced_accuracy']['std'])
    
    model_names = [m[0].replace('_', ' ').title() for m in sorted_by_stability]
    stds = [m[1]['metrics']['balanced_accuracy']['std'] for m in sorted_by_stability]
    means = [m[1]['metrics']['balanced_accuracy']['mean'] for m in sorted_by_stability]
    
    # Standard deviation chart
    colors = [COLOR_SUCCESS if s < 0.05 else COLOR_WARNING if s < 0.10 else COLOR_DANGER 
             for s in stds]
    bars = ax1.barh(model_names, stds, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax1.set_xlabel('Standard Deviation', fontweight='bold', fontsize=12)
    ax1.set_title(f'Model Stability Analysis\nLower = More Stable',
                 fontweight='bold', fontsize=14)
    ax1.axvline(x=0.05, color='green', linestyle='--', alpha=0.5, label='High Stability')
    ax1.axvline(x=0.10, color='orange', linestyle='--', alpha=0.5, label='Moderate Stability')
    ax1.legend()
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar, std in zip(bars, stds):
        ax1.text(std + 0.002, bar.get_y() + bar.get_height()/2,
                f'{std:.4f}', va='center', fontsize=9, fontweight='bold')
    
    # Stability vs Performance scatter
    ax2.scatter(means, stds, s=200, c=colors, alpha=0.7, edgecolors='black', linewidth=2)
    
    # Add labels
    for i, name in enumerate(model_names):
        ax2.annotate(name, (means[i], stds[i]), fontsize=8,
                    xytext=(5, 5), textcoords='offset points')
    
    ax2.set_xlabel('Mean Balanced Accuracy', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Standard Deviation', fontweight='bold', fontsize=12)
    ax2.set_title(f'Performance vs Stability Trade-off\n({cv_folds}-Fold CV)',
                 fontweight='bold', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.05, color='green', linestyle='--', alpha=0.3)
    ax2.axhline(y=0.10, color='orange', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def create_detailed_metrics_table(pdf, cv_data, dataset_type):
    """Create detailed metrics table."""
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.axis('off')
    
    # Prepare data
    sorted_models = sorted(cv_data.items(),
                          key=lambda x: x[1]['metrics']['balanced_accuracy']['mean'],
                          reverse=True)
    
    table_data = [['Rank', 'Model', 'Bal Acc (Mean±Std)', 'F1 (Mean±Std)', 
                   'Accuracy (Mean±Std)', 'Stability']]
    
    for rank, (model_name, model_data) in enumerate(sorted_models, 1):
        bal_acc_mean = model_data['metrics']['balanced_accuracy']['mean']
        bal_acc_std = model_data['metrics']['balanced_accuracy']['std']
        f1_mean = model_data['metrics']['f1_macro']['mean']
        f1_std = model_data['metrics']['f1_macro']['std']
        acc_mean = model_data['metrics']['accuracy']['mean']
        acc_std = model_data['metrics']['accuracy']['std']
        
        stability = 'High' if bal_acc_std < 0.05 else 'Moderate' if bal_acc_std < 0.10 else 'Variable'
        
        table_data.append([
            str(rank),
            model_name.replace('_', ' ').title(),
            f'{bal_acc_mean:.4f}±{bal_acc_std:.4f}',
            f'{f1_mean:.4f}±{f1_std:.4f}',
            f'{acc_mean:.4f}±{acc_std:.4f}',
            stability
        ])
    
    table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                    colWidths=[0.08, 0.25, 0.20, 0.20, 0.20, 0.12])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(6):
        cell = table[(0, i)]
        cell.set_facecolor('#34495e')
        cell.set_text_props(weight='bold', color='white')
    
    # Color code ranks
    for i in range(1, min(4, len(table_data))):
        cell = table[(i, 0)]
        if i == 1:
            cell.set_facecolor(COLOR_SUCCESS)
        elif i == 2:
            cell.set_facecolor(COLOR_INFO)
        elif i == 3:
            cell.set_facecolor(COLOR_WARNING)
        cell.set_text_props(weight='bold', color='white')
    
    ax.set_title(f'{dataset_type} Dataset - Detailed Performance Metrics',
                fontsize=14, fontweight='bold', pad=20)
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def generate_cv_executive_summary(dataset_type='Mix'):
    """Generate complete executive summary PDF from CV results."""
    print("="*80)
    print(f"GENERATING EXECUTIVE SUMMARY - {dataset_type} DATASET (CROSS-VALIDATION)")
    print("="*80)
    
    # Load CV results
    print("\n[1/5] Loading cross-validation results...")
    cv_data = load_cv_results(dataset_type)
    
    if not cv_data:
        print("✗ Failed to load CV results. Run cross_validation.py first!")
        return
    
    # Determine number of folds from data
    first_model = list(cv_data.values())[0]
    cv_folds = len(first_model['metrics']['balanced_accuracy']['values'])
    
    print(f"✓ Loaded results for {len(cv_data)} models ({cv_folds}-fold CV)")
    
    # Setup output
    reports_dir = f'{dataset_type}_Results/reports'
    os.makedirs(reports_dir, exist_ok=True)
    output_pdf = os.path.join(reports_dir, 
                             f'executive_summary_cv_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf')
    
    print(f"\n[2/5] Creating PDF: {output_pdf}")
    
    with PdfPages(output_pdf) as pdf:
        print("  [2a] Generating title page...")
        create_title_page(pdf, cv_data, dataset_type, cv_folds)
        
        print("  [2b] Creating performance comparison...")
        create_performance_comparison(pdf, cv_data, dataset_type, cv_folds)
        
        print("  [2c] Creating stability analysis...")
        create_stability_analysis(pdf, cv_data, dataset_type, cv_folds)
        
        print("  [2d] Creating detailed metrics table...")
        create_detailed_metrics_table(pdf, cv_data, dataset_type)
        
        # Add metadata
        d = pdf.infodict()
        d['Title'] = f'Executive Summary - {dataset_type} Dataset ({cv_folds}-Fold CV)'
        d['Author'] = 'ML in LTDNA Pipeline'
        d['Subject'] = 'Cross-Validation Results Summary'
        d['Keywords'] = 'Machine Learning, DNA Analysis, Cross-Validation'
        d['CreationDate'] = datetime.now()
    
    print(f"\n[3/5] ✓ PDF generated successfully")
    print(f"[4/5] Pages: 4")
    print(f"[5/5] Location: {output_pdf}")
    
    print("\n" + "="*80)
    print("✓ EXECUTIVE SUMMARY COMPLETE")
    print("="*80)
    
    return output_pdf

if __name__ == "__main__":
    import sys
    
    dataset_type = sys.argv[1] if len(sys.argv) > 1 else 'Mix'
    generate_cv_executive_summary(dataset_type)
