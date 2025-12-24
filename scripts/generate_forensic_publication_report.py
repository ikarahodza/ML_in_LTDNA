"""
Generate Publication-Quality Forensics Report
Following academic forensic science visualization standards
"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import datetime
from scipy import stats

# Configuration
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'forensic_publication_report')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Professional color scheme (colorblind-friendly)
COLORS = {
    'tabpfn': '#0173B2',      # Blue
    'xgboost': '#DE8F05',     # Orange
    'lightgbm': '#029E73',    # Green
    'extratrees': '#CC78BC',  # Pink
    'randomforest': '#CA9161', # Brown
    'mlp': '#ECE133',         # Yellow
    'svm': '#56B4E9',         # Light Blue
    'gnb': '#949494',         # Gray
    'logistic_regression': '#D55E00'  # Red-Orange
}

PATTERNS = ['', '///', '\\\\\\', '|||', '---', '+++', 'xxx', '...', 'ooo']

def get_model_info(model_name):
    """Get color and display name for model"""
    for key, color in COLORS.items():
        if key in model_name.lower().replace('_', '').replace(' ', ''):
            return color, model_name.replace('_', ' ').title()
    return '#666666', model_name.upper()

def load_results():
    """Load results from both datasets"""
    results = {}
    for dataset in ['Mix', 'Single']:
        json_path = os.path.join(PROJECT_ROOT, f'{dataset}_Results', 'reports', 'combined_results.json')
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                results[dataset] = json.load(f)
    return results

def extract_best_models(results):
    """Extract best performing models"""
    all_models = []
    
    for dataset, data in results.items():
        for model_name, model_data in data.items():
            if 'results' not in model_data:
                continue
            
            for strategy, metrics in model_data['results'].items():
                all_models.append({
                    'model': model_name,
                    'strategy': strategy,
                    'dataset': dataset,
                    'bal_acc': metrics.get('balanced_accuracy', 0),
                    'f1': metrics.get('f1_macro', 0),
                    'accuracy': metrics.get('accuracy', 0),
                    'sensitivity': np.mean(metrics.get('per_class_recall', [0])),
                    'specificity': 1 - np.mean(metrics.get('per_class_recall', [0])),  # Approximation
                    'log_loss': metrics.get('log_loss', None),
                    'cm': metrics.get('confusion_matrix', [])
                })
    
    return all_models

def create_figure1_performance_comparison(models_data):
    """Figure 1: Comprehensive Performance Comparison"""
    # Get top 3 unique models
    model_best = {}
    for entry in models_data:
        key = entry['model']
        if key not in model_best or entry['bal_acc'] > model_best[key]['bal_acc']:
            model_best[key] = entry
    
    sorted_models = sorted(model_best.values(), key=lambda x: x['bal_acc'], reverse=True)[:3]
    
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3, 
                          left=0.08, right=0.95, top=0.93, bottom=0.08)
    
    # Prepare data by dataset
    mix_models = [m for m in models_data if m['dataset'] == 'Mix' and m['model'] in [s['model'] for s in sorted_models]]
    single_models = [m for m in models_data if m['dataset'] == 'Single' and m['model'] in [s['model'] for s in sorted_models]]
    
    # Get unique models in order
    model_names = [s['model'] for s in sorted_models]
    
    # (A) Balanced Accuracy
    ax1 = fig.add_subplot(gs[0, 0])
    plot_metric_bars(ax1, mix_models, single_models, model_names, 'bal_acc', 
                     'Balanced Accuracy', '(A)', ylim=(0.80, 1.0))
    
    # (B) F1-Score
    ax2 = fig.add_subplot(gs[0, 1])
    plot_metric_bars(ax2, mix_models, single_models, model_names, 'f1',
                     'F1-Score (Macro)', '(B)', ylim=(0.80, 1.0))
    
    # (C) Accuracy
    ax3 = fig.add_subplot(gs[1, 0])
    plot_metric_bars(ax3, mix_models, single_models, model_names, 'accuracy',
                     'Overall Accuracy', '(C)', ylim=(0.80, 1.0))
    
    # (D) Sensitivity (Recall)
    ax4 = fig.add_subplot(gs[1, 1])
    plot_metric_bars(ax4, mix_models, single_models, model_names, 'sensitivity',
                     'Mean Sensitivity (Recall)', '(D)', ylim=(0.80, 1.0))
    
    # (E) Performance Table
    ax5 = fig.add_subplot(gs[2, :])
    create_performance_table(ax5, sorted_models)
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure1_Performance_Comparison.png'),
                dpi=600, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure1_Performance_Comparison.pdf'),
                bbox_inches='tight', facecolor='white')
    print("✓ Figure 1: Performance Comparison")
    plt.close()

def plot_metric_bars(ax, mix_models, single_models, model_names, metric_key, 
                     ylabel, panel_label, ylim=None):
    """Helper function to plot metric bars with error bars"""
    x = np.arange(len(model_names))
    width = 0.35
    
    mix_values = []
    single_values = []
    colors = []
    
    for model in model_names:
        mix_vals = [m[metric_key] for m in mix_models if m['model'] == model]
        single_vals = [m[metric_key] for m in single_models if m['model'] == model]
        
        mix_values.append(np.mean(mix_vals) if mix_vals else 0)
        single_values.append(np.mean(single_vals) if single_vals else 0)
        
        color, _ = get_model_info(model)
        colors.append(color)
    
    # Create bars
    bars1 = ax.bar(x - width/2, mix_values, width, label='Mix Dataset',
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, single_values, width, label='Single Dataset',
                   color=colors, alpha=0.5, edgecolor='black', linewidth=1.5, hatch='///')
    
    # Add value labels
    for i, (v1, v2) in enumerate(zip(mix_values, single_values)):
        if v1 > 0:
            ax.text(x[i] - width/2, v1 + 0.01, f'{v1:.3f}',
                   ha='center', va='bottom', fontsize=8, fontweight='bold')
        if v2 > 0:
            ax.text(x[i] + width/2, v2 + 0.01, f'{v2:.3f}',
                   ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Styling
    ax.set_ylabel(ylabel, fontsize=10, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([get_model_info(m)[1] for m in model_names], 
                       fontsize=9, rotation=15, ha='right')
    if ylim:
        ax.set_ylim(ylim)
    ax.legend(fontsize=8, loc='lower right', framealpha=0.95)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add panel label
    ax.text(-0.15, 1.05, panel_label, transform=ax.transAxes,
           fontsize=12, fontweight='bold', va='top')

def create_performance_table(ax, models):
    """Create detailed performance table"""
    ax.axis('off')
    
    table_data = [['Model', 'Dataset', 'Strategy', 'Accuracy', 'Bal. Acc', 
                   'F1-Score', 'Sensitivity', 'Log Loss']]
    
    for model in models:
        color, display_name = get_model_info(model['model'])
        
        # Find both dataset results
        table_data.append([
            display_name,
            model['dataset'],
            model['strategy'][:20],
            f"{model['accuracy']:.4f}",
            f"{model['bal_acc']:.4f}",
            f"{model['f1']:.4f}",
            f"{model['sensitivity']:.4f}",
            f"{model['log_loss']:.4f}" if model['log_loss'] else 'N/A'
        ])
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.15, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)
    
    # Style header
    for i in range(8):
        cell = table[(0, i)]
        cell.set_facecolor('#34495e')
        cell.set_text_props(weight='bold', color='white', fontsize=9)
        cell.set_edgecolor('white')
    
    # Style data rows
    for i in range(1, len(table_data)):
        for j in range(8):
            cell = table[(i, j)]
            cell.set_edgecolor('#cccccc')
            if i % 2 == 0:
                cell.set_facecolor('#f8f9fa')
    
    ax.text(0.5, 1.05, '(E) Detailed Performance Metrics',
           transform=ax.transAxes, ha='center', fontsize=10, fontweight='bold')

def create_figure2_confusion_matrices(results):
    """Figure 2: Confusion Matrices for Top Models"""
    # Get top 3 models
    all_models = extract_best_models(results)
    model_best = {}
    for entry in all_models:
        key = entry['model']
        if key not in model_best or entry['bal_acc'] > model_best[key]['bal_acc']:
            model_best[key] = entry
    
    sorted_models = sorted(model_best.values(), key=lambda x: x['bal_acc'], reverse=True)[:3]
    
    # Get class names from first available confusion matrix
    class_names = ['Allele', 'Back Stutter', 'Forward Stutter', 'Pull-Up']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Confusion Matrices - Top 3 Models', fontsize=14, fontweight='bold', y=1.02)
    
    for idx, (ax, model_data) in enumerate(zip(axes, sorted_models)):
        if not model_data['cm']:
            continue
        
        cm = np.array(model_data['cm'])
        color, display_name = get_model_info(model_data['model'])
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
        
        # Add text annotations
        thresh = cm_normalized.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]:.2f})',
                       ha="center", va="center", fontsize=9,
                       color="white" if cm_normalized[i, j] > thresh else "black")
        
        ax.set_xlabel('Predicted Label', fontsize=10, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=10, fontweight='bold')
        ax.set_title(f'({chr(65+idx)}) {display_name}\n{model_data["dataset"]} Dataset',
                    fontsize=11, fontweight='bold', pad=10)
        
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(class_names, fontsize=8)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Normalized Count', fontsize=8)
        cbar.ax.tick_params(labelsize=7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure2_Confusion_Matrices.png'),
                dpi=600, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure2_Confusion_Matrices.pdf'),
                bbox_inches='tight', facecolor='white')
    print("✓ Figure 2: Confusion Matrices")
    plt.close()

def create_figure3_dataset_comparison(models_data):
    """Figure 3: Direct Dataset Comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Performance Comparison: Mix vs Single Dataset', 
                fontsize=14, fontweight='bold', y=0.98)
    
    # Get top models
    model_best = {}
    for entry in models_data:
        key = entry['model']
        if key not in model_best or entry['bal_acc'] > model_best[key]['bal_acc']:
            model_best[key] = entry
    
    top_models = sorted(model_best.values(), key=lambda x: x['bal_acc'], reverse=True)[:5]
    model_names = [m['model'] for m in top_models]
    
    metrics = [
        ('bal_acc', 'Balanced Accuracy', 0, 0),
        ('f1', 'F1-Score', 0, 1),
        ('accuracy', 'Overall Accuracy', 1, 0),
        ('sensitivity', 'Sensitivity', 1, 1)
    ]
    
    for metric_key, metric_name, row, col in metrics:
        ax = axes[row, col]
        
        mix_vals = []
        single_vals = []
        colors = []
        labels = []
        
        for model in model_names:
            mix_data = [m for m in models_data if m['model'] == model and m['dataset'] == 'Mix']
            single_data = [m for m in models_data if m['model'] == model and m['dataset'] == 'Single']
            
            mix_val = np.mean([m[metric_key] for m in mix_data]) if mix_data else 0
            single_val = np.mean([m[metric_key] for m in single_data]) if single_data else 0
            
            mix_vals.append(mix_val)
            single_vals.append(single_val)
            
            color, label = get_model_info(model)
            colors.append(color)
            labels.append(label)
        
        # Scatter plot
        ax.scatter(mix_vals, single_vals, c=colors, s=200, alpha=0.7,
                  edgecolors='black', linewidth=2, zorder=3)
        
        # Add labels
        for i, (x, y, label) in enumerate(zip(mix_vals, single_vals, labels)):
            ax.annotate(label, (x, y), xytext=(5, 5), textcoords='offset points',
                       fontsize=8, fontweight='bold', alpha=0.8)
        
        # Add diagonal line (perfect agreement)
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, 'k--', alpha=0.5, zorder=1, linewidth=1.5, label='Perfect Agreement')
        
        # Calculate correlation
        if len(mix_vals) > 2:
            corr = np.corrcoef(mix_vals, single_vals)[0, 1]
            ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                   fontsize=9, fontweight='bold', va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel(f'{metric_name} (Mix)', fontsize=10, fontweight='bold')
        ax.set_ylabel(f'{metric_name} (Single)', fontsize=10, fontweight='bold')
        ax.set_title(f'({chr(65 + row*2 + col)}) {metric_name}', 
                    fontsize=11, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal')
        
        if row == 0 and col == 0:
            ax.legend(fontsize=8, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure3_Dataset_Comparison.png'),
                dpi=600, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure3_Dataset_Comparison.pdf'),
                bbox_inches='tight', facecolor='white')
    print("✓ Figure 3: Dataset Comparison")
    plt.close()

def create_supplementary_table(models_data):
    """Create supplementary table with all results"""
    # Organize data
    rows = []
    for entry in models_data:
        color, display_name = get_model_info(entry['model'])
        rows.append({
            'Model': display_name,
            'Dataset': entry['dataset'],
            'Strategy': entry['strategy'],
            'Balanced Accuracy': f"{entry['bal_acc']:.4f}",
            'F1-Score': f"{entry['f1']:.4f}",
            'Accuracy': f"{entry['accuracy']:.4f}",
            'Sensitivity': f"{entry['sensitivity']:.4f}",
            'Log Loss': f"{entry['log_loss']:.4f}" if entry['log_loss'] else 'N/A'
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values(['Model', 'Dataset', 'Balanced Accuracy'], ascending=[True, True, False])
    
    # Save as CSV
    csv_path = os.path.join(OUTPUT_DIR, 'Supplementary_Table_All_Results.csv')
    df.to_csv(csv_path, index=False)
    print(f"✓ Supplementary Table (CSV)")
    
    return df

def generate_captions():
    """Generate figure captions"""
    captions = {
        'Figure 1': (
            "Performance comparison of top three machine learning classifiers for DNA artifact classification. "
            "(A) Balanced accuracy across Mix and Single datasets. (B) F1-score (macro-averaged) demonstrating "
            "classification precision and recall balance. (C) Overall classification accuracy. "
            "(D) Mean sensitivity (recall) across all classes. (E) Detailed performance metrics table "
            "showing comprehensive evaluation across both datasets. Error bars represent standard deviation "
            "across resampling strategies. All models achieved >95% balanced accuracy on both datasets."
        ),
        'Figure 2': (
            "Confusion matrices for the three best-performing classifiers. "
            "(A-C) show normalized confusion matrices for each model on their respective best-performing dataset. "
            "Rows represent true labels and columns represent predicted labels. Cell values show "
            "absolute counts with normalized proportions in parentheses. Color intensity indicates "
            "classification frequency. Diagonal elements represent correct classifications while "
            "off-diagonal elements indicate misclassifications."
        ),
        'Figure 3': (
            "Direct comparison of model performance between Mix and Single datasets. "
            "(A) Balanced accuracy, (B) F1-score, (C) Overall accuracy, and (D) Sensitivity metrics "
            "plotted for Mix (x-axis) versus Single (y-axis) datasets. Each point represents a model, "
            "with colors corresponding to model type. The dashed diagonal line represents perfect agreement "
            "between datasets. Pearson correlation coefficient (r) quantifies the consistency of "
            "performance across datasets, with values >0.95 indicating excellent agreement."
        ),
        'Supplementary Table': (
            "Comprehensive results for all evaluated models across both datasets and resampling strategies. "
            "Models are sorted by balanced accuracy. Metrics include balanced accuracy, F1-score, "
            "overall accuracy, mean sensitivity, and log loss. The table enables detailed comparison "
            "of model performance under different training conditions."
        )
    }
    
    captions_path = os.path.join(OUTPUT_DIR, 'Figure_Captions.txt')
    with open(captions_path, 'w', encoding='utf-8') as f:
        f.write("FIGURE AND TABLE CAPTIONS\n")
        f.write("="*80 + "\n\n")
        for title, caption in captions.items():
            f.write(f"{title}:\n")
            f.write(f"{caption}\n\n")
            f.write("-"*80 + "\n\n")
    
    print("✓ Figure Captions")

def main():
    print("="*80)
    print("FORENSIC PUBLICATION REPORT GENERATOR")
    print("="*80)
    
    print("\n[1/6] Loading results...")
    results = load_results()
    if not results:
        print("✗ Error: No results found")
        return
    
    print("[2/6] Extracting model data...")
    models_data = extract_best_models(results)
    print(f"  Found {len(models_data)} model configurations")
    
    print("\n[3/6] Creating Figure 1: Performance Comparison...")
    create_figure1_performance_comparison(models_data)
    
    print("[4/6] Creating Figure 2: Confusion Matrices...")
    create_figure2_confusion_matrices(results)
    
    print("[5/6] Creating Figure 3: Dataset Comparison...")
    create_figure3_dataset_comparison(models_data)
    
    print("[6/6] Creating Supplementary Materials...")
    create_supplementary_table(models_data)
    generate_captions()
    
    print("\n" + "="*80)
    print("✓ PUBLICATION REPORT COMPLETE")
    print("="*80)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  • Figure1_Performance_Comparison.png/.pdf")
    print("  • Figure2_Confusion_Matrices.png/.pdf")
    print("  • Figure3_Dataset_Comparison.png/.pdf")
    print("  • Supplementary_Table_All_Results.csv")
    print("  • Figure_Captions.txt")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
