"""
Generate comprehensive PDF report comparing all trained ML models.
Reads combined_results.json and all generated figures to create a detailed analysis.
"""

import json
import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import matplotlib.image as mpimg
from pathlib import Path

# Configuration
REPORTS_DIR = 'reports'
FIGURES_DIR = 'figures'
DATA_DIR = 'data'
OUTPUT_PDF = os.path.join(REPORTS_DIR, f'comprehensive_model_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf')

# Load results
RESULTS_FILE = os.path.join(REPORTS_DIR, 'combined_results.json')

def load_results():
    """Load the combined results JSON file."""
    with open(RESULTS_FILE, 'r') as f:
        return json.load(f)

def create_title_page(pdf, results):
    """Create a title page with executive summary."""
    fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
    fig.suptitle('DNA Artifact Classification\nMachine Learning Model Comparison Report', 
                 fontsize=24, fontweight='bold', y=0.85)
    
    # Add metadata
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    date_str = datetime.now().strftime("%B %d, %Y")
    
    summary_text = f"""
    Report Generated: {date_str}
    
    PROJECT OVERVIEW
    ─────────────────────────────────────────────────────────────
    This report presents a comprehensive comparison of machine learning models
    for DNA artifact classification in Low Template DNA (LTDNA) analysis.
    
    CLASSIFICATION TASK
    ─────────────────────────────────────────────────────────────
    • Target Classes: Allele, Back_Stutter, Forward_Stutter, Pull_Up
    • Models Evaluated: {len(results)} models
    • Resampling Strategies: Original, SMOTE, RandomOverSampler
    
    MODELS ANALYZED
    ─────────────────────────────────────────────────────────────
    """
    
    model_names = {
        'extratrees': 'Extra Trees Classifier',
        'tabpfn': 'TabPFN (Tabular Prior-Fitted Networks)',
        'xgboost': 'XGBoost',
        'lgbm': 'LightGBM',
        'randomforest': 'Random Forest',
        'svm': 'Support Vector Machine',
        'logistic_regression': 'Logistic Regression',
        'mlp': 'Multi-Layer Perceptron',
        'gnb': 'Gaussian Naive Bayes'
    }
    
    for model_key in results.keys():
        model_name = model_names.get(model_key, model_key)
        summary_text += f"    • {model_name}\n"
    
    summary_text += """
    
    EVALUATION METRICS
    ─────────────────────────────────────────────────────────────
    • Accuracy & Balanced Accuracy
    • F1-Score (Macro)
    • Mean Squared Error (MSE) & Mean Absolute Error (MAE)
    • Log Loss
    • Per-Class Precision, Recall, F1-Score
    • Confusion Matrices
    • ROC & Precision-Recall Curves
    • Feature Importance (tree-based models)
    • SHAP Values (select models)
    
    """
    
    ax.text(0.5, 0.5, summary_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='center', horizontalalignment='center',
            family='monospace')
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def create_overall_comparison_table(pdf, results):
    """Create overall comparison table across all models and strategies."""
    fig, axes = plt.subplots(2, 1, figsize=(8.27, 11.69))
    fig.suptitle('Overall Model Performance Comparison', fontsize=16, fontweight='bold')
    
    # Prepare data
    data = []
    for model_name, model_data in results.items():
        if 'results' in model_data:
            for strategy, metrics in model_data['results'].items():
                data.append({
                    'Model': model_name.upper(),
                    'Strategy': strategy,
                    'Accuracy': metrics.get('accuracy', 0),
                    'Balanced Acc': metrics.get('balanced_accuracy', 0),
                    'F1 (macro)': metrics.get('f1_macro', 0),
                    'Log Loss': metrics.get('log_loss', 0) if metrics.get('log_loss') else 0
                })
    
    df = pd.DataFrame(data)
    
    # Top performers table
    axes[0].axis('tight')
    axes[0].axis('off')
    axes[0].set_title('Top 10 Model-Strategy Combinations (by Balanced Accuracy)', 
                      fontsize=12, fontweight='bold', pad=20)
    
    top_10 = df.nlargest(10, 'Balanced Acc')[['Model', 'Strategy', 'Accuracy', 'Balanced Acc', 'F1 (macro)', 'Log Loss']]
    
    table1 = axes[0].table(cellText=top_10.values,
                          colLabels=top_10.columns,
                          cellLoc='center',
                          loc='center',
                          bbox=[0, 0, 1, 1])
    table1.auto_set_font_size(False)
    table1.set_fontsize(9)
    table1.scale(1, 2)
    
    # Style header
    for i in range(len(top_10.columns)):
        table1[(0, i)].set_facecolor('#4CAF50')
        table1[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color rows
    for i in range(1, len(top_10) + 1):
        for j in range(len(top_10.columns)):
            if i % 2 == 0:
                table1[(i, j)].set_facecolor('#f0f0f0')
    
    # Per-model best strategy
    axes[1].axis('tight')
    axes[1].axis('off')
    axes[1].set_title('Best Strategy per Model (by Balanced Accuracy)', 
                      fontsize=12, fontweight='bold', pad=20)
    
    best_per_model = df.loc[df.groupby('Model')['Balanced Acc'].idxmax()]
    best_per_model = best_per_model.sort_values('Balanced Acc', ascending=False)
    
    table2 = axes[1].table(cellText=best_per_model.values,
                          colLabels=best_per_model.columns,
                          cellLoc='center',
                          loc='center',
                          bbox=[0, 0, 1, 1])
    table2.auto_set_font_size(False)
    table2.set_fontsize(9)
    table2.scale(1, 2)
    
    # Style header
    for i in range(len(best_per_model.columns)):
        table2[(0, i)].set_facecolor('#2196F3')
        table2[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color rows
    for i in range(1, len(best_per_model) + 1):
        for j in range(len(best_per_model.columns)):
            if i % 2 == 0:
                table2[(i, j)].set_facecolor('#f0f0f0')
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def create_metrics_visualization(pdf, results):
    """Create bar charts comparing key metrics across models."""
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.suptitle('Performance Metrics Comparison', fontsize=16, fontweight='bold')
    
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Prepare data
    data = []
    for model_name, model_data in results.items():
        if 'results' in model_data:
            for strategy, metrics in model_data['results'].items():
                data.append({
                    'Model': model_name.upper(),
                    'Strategy': strategy,
                    'Accuracy': metrics.get('accuracy', 0),
                    'Balanced Acc': metrics.get('balanced_accuracy', 0),
                    'F1 (macro)': metrics.get('f1_macro', 0),
                    'Log Loss': metrics.get('log_loss', 0) if metrics.get('log_loss') else 0
                })
    
    df = pd.DataFrame(data)
    
    # Get best per model
    best_per_model = df.loc[df.groupby('Model')['Balanced Acc'].idxmax()]
    best_per_model = best_per_model.sort_values('Balanced Acc', ascending=True)
    
    metrics_to_plot = [
        ('Accuracy', 'Accuracy', '#4CAF50'),
        ('Balanced Acc', 'Balanced Accuracy', '#2196F3'),
        ('F1 (macro)', 'F1-Score (Macro)', '#FF9800'),
        ('Log Loss', 'Log Loss (lower is better)', '#F44336')
    ]
    
    for idx, (metric, title, color) in enumerate(metrics_to_plot):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        
        values = best_per_model[metric].values
        models = best_per_model['Model'].values
        
        bars = ax.barh(models, values, color=color, alpha=0.7)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'{width:.4f}', ha='left', va='center', fontsize=8)
        
        ax.set_xlabel(title, fontsize=10)
        ax.set_ylabel('Model', fontsize=10)
        ax.set_title(f'{title} - Best Strategy per Model', fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def create_per_class_performance(pdf, results):
    """Create per-class performance comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(8.27, 11.69))
    fig.suptitle('Per-Class Performance Comparison (Best Strategy per Model)', 
                 fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    # Get class names from first model
    first_model = list(results.keys())[0]
    class_names = results[first_model].get('class_names', ['Class_0', 'Class_1', 'Class_2', 'Class_3'])
    
    metrics = ['per_class_precision', 'per_class_recall', 'per_class_f1', 'per_class_support']
    titles = ['Precision', 'Recall', 'F1-Score', 'Support']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]
        
        # Collect data
        model_data = {}
        for model_name, model_info in results.items():
            if 'results' in model_info:
                # Get best strategy
                best_strategy = max(model_info['results'].items(), 
                                   key=lambda x: x[1].get('balanced_accuracy', 0))
                values = best_strategy[1].get(metric, [])
                if values:
                    model_data[model_name.upper()] = values
        
        if metric == 'per_class_support':
            # For support, just show from one model (they should be the same)
            if model_data:
                first_key = list(model_data.keys())[0]
                values = model_data[first_key]
                ax.bar(class_names, values, color='#9C27B0', alpha=0.7)
                ax.set_ylabel('Count', fontsize=10)
                ax.set_title(f'{title} (Test Set)', fontsize=11, fontweight='bold')
                
                # Add value labels
                for i, v in enumerate(values):
                    ax.text(i, v, str(int(v)), ha='center', va='bottom', fontsize=9)
        else:
            # Create grouped bar chart
            x = np.arange(len(class_names))
            width = 0.8 / len(model_data)
            
            colors = plt.cm.tab10(np.linspace(0, 1, len(model_data)))
            
            for i, (model, values) in enumerate(model_data.items()):
                offset = (i - len(model_data)/2 + 0.5) * width
                ax.bar(x + offset, values, width, label=model, color=colors[i], alpha=0.7)
            
            ax.set_ylabel(title, fontsize=10)
            ax.set_title(f'{title} by Class', fontsize=11, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(class_names, rotation=45, ha='right')
            ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1, 1))
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def create_model_detail_page(pdf, model_name, model_data, figures_dir):
    """Create detailed page for each model showing all visualizations."""
    strategies = list(model_data['results'].keys()) if 'results' in model_data else []
    assets = model_data.get('assets', {})
    
    for strategy in strategies:
        strategy_assets = assets.get(strategy, {})
        
        # Create a page with confusion matrix, feature importance, and metrics
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.suptitle(f'{model_name.upper()} - {strategy} Strategy', 
                    fontsize=16, fontweight='bold')
        
        gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)
        
        # Metrics table
        ax_metrics = fig.add_subplot(gs[0, :])
        ax_metrics.axis('tight')
        ax_metrics.axis('off')
        ax_metrics.set_title('Performance Metrics', fontsize=12, fontweight='bold')
        
        metrics = model_data['results'][strategy]
        metrics_data = [
            ['Accuracy', f"{metrics.get('accuracy', 0):.4f}"],
            ['Balanced Accuracy', f"{metrics.get('balanced_accuracy', 0):.4f}"],
            ['F1-Score (Macro)', f"{metrics.get('f1_macro', 0):.4f}"],
            ['MSE', f"{metrics.get('mse', 0):.4f}"],
            ['MAE', f"{metrics.get('mae', 0):.4f}"],
            ['Log Loss', f"{metrics.get('log_loss', 0):.4f}" if metrics.get('log_loss') else 'N/A']
        ]
        
        table = ax_metrics.table(cellText=metrics_data, cellLoc='left',
                                loc='center', bbox=[0.2, 0, 0.6, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Confusion Matrix
        cm_path = strategy_assets.get('confusion_matrix')
        if cm_path and os.path.exists(cm_path):
            ax_cm = fig.add_subplot(gs[1, 0])
            img = mpimg.imread(cm_path)
            ax_cm.imshow(img)
            ax_cm.axis('off')
            ax_cm.set_title('Confusion Matrix', fontsize=11, fontweight='bold')
        
        # Feature Importance
        fi_path = strategy_assets.get('feature_importance')
        if fi_path and os.path.exists(fi_path):
            ax_fi = fig.add_subplot(gs[1, 1])
            img = mpimg.imread(fi_path)
            ax_fi.imshow(img)
            ax_fi.axis('off')
            ax_fi.set_title('Feature Importance', fontsize=11, fontweight='bold')
        
        # ROC Curves
        roc_path = strategy_assets.get('roc_curves')
        if roc_path and os.path.exists(roc_path):
            ax_roc = fig.add_subplot(gs[2, 0])
            img = mpimg.imread(roc_path)
            ax_roc.imshow(img)
            ax_roc.axis('off')
            ax_roc.set_title('ROC Curves', fontsize=11, fontweight='bold')
        
        # PR Curves
        pr_path = strategy_assets.get('pr_curves')
        if pr_path and os.path.exists(pr_path):
            ax_pr = fig.add_subplot(gs[2, 1])
            img = mpimg.imread(pr_path)
            ax_pr.imshow(img)
            ax_pr.axis('off')
            ax_pr.set_title('Precision-Recall Curves', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', dpi=150)
        plt.close()
        
        # SHAP plots on separate page if available
        shap_summary = strategy_assets.get('shap_summary')
        shap_bar = strategy_assets.get('shap_bar')
        
        if (shap_summary and os.path.exists(shap_summary)) or (shap_bar and os.path.exists(shap_bar)):
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.suptitle(f'{model_name.upper()} - {strategy} - SHAP Analysis', 
                        fontsize=16, fontweight='bold')
            
            if shap_summary and os.path.exists(shap_summary):
                if shap_bar and os.path.exists(shap_bar):
                    # Both plots
                    ax1 = fig.add_subplot(2, 1, 1)
                    img = mpimg.imread(shap_summary)
                    ax1.imshow(img)
                    ax1.axis('off')
                    ax1.set_title('SHAP Summary Plot', fontsize=12, fontweight='bold')
                    
                    ax2 = fig.add_subplot(2, 1, 2)
                    img = mpimg.imread(shap_bar)
                    ax2.imshow(img)
                    ax2.axis('off')
                    ax2.set_title('SHAP Bar Plot', fontsize=12, fontweight='bold')
                else:
                    # Only summary
                    ax = fig.add_subplot(1, 1, 1)
                    img = mpimg.imread(shap_summary)
                    ax.imshow(img)
                    ax.axis('off')
                    ax.set_title('SHAP Summary Plot', fontsize=12, fontweight='bold')
            elif shap_bar and os.path.exists(shap_bar):
                # Only bar
                ax = fig.add_subplot(1, 1, 1)
                img = mpimg.imread(shap_bar)
                ax.imshow(img)
                ax.axis('off')
                ax.set_title('SHAP Bar Plot', fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight', dpi=150)
            plt.close()

def create_conclusions_page(pdf, results):
    """Create conclusions and recommendations page."""
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.suptitle('Conclusions and Recommendations', fontsize=20, fontweight='bold', y=0.95)
    
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # Find best overall model
    best_overall = None
    best_bal_acc = 0
    
    for model_name, model_data in results.items():
        if 'results' in model_data:
            for strategy, metrics in model_data['results'].items():
                bal_acc = metrics.get('balanced_accuracy', 0)
                if bal_acc > best_bal_acc:
                    best_bal_acc = bal_acc
                    best_overall = (model_name, strategy, metrics)
    
    conclusions_text = f"""
    KEY FINDINGS
    {'═'*70}
    
    Best Performing Model: {best_overall[0].upper()} with {best_overall[1]} strategy
    • Balanced Accuracy: {best_overall[2]['balanced_accuracy']:.4f}
    • F1-Score (Macro): {best_overall[2]['f1_macro']:.4f}
    • Accuracy: {best_overall[2]['accuracy']:.4f}
    
    
    OBSERVATIONS
    {'═'*70}
    
    1. RESAMPLING IMPACT
       Different models respond differently to resampling strategies. SMOTE and
       RandomOverSampler can help with class imbalance, but may lead to overfitting
       in some cases.
    
    2. MODEL COMPLEXITY
       Tree-based models (Random Forest, XGBoost, LightGBM, Extra Trees) generally
       perform well on this dataset and provide interpretability through feature
       importance and SHAP values.
    
    3. CLASS IMBALANCE
       The dataset shows class imbalance. Balanced accuracy and per-class metrics
       provide better insights than overall accuracy alone.
    
    4. FEATURE IMPORTANCE
       SHAP values and feature importance plots reveal which features most strongly
       influence DNA artifact classification. Review these for domain validation.
    
    
    RECOMMENDATIONS
    {'═'*70}
    
    ✓ Use the best performing model for production deployment
    ✓ Monitor per-class performance, especially for minority classes
    ✓ Consider ensemble methods combining top performers
    ✓ Validate feature importance with domain experts
    ✓ Collect more data for underrepresented classes if possible
    ✓ Perform cross-validation for more robust performance estimates
    ✓ Test on independent validation set before deployment
    
    
    NEXT STEPS
    {'═'*70}
    
    → Hyperparameter tuning for top models
    → Cross-validation analysis
    → External validation dataset testing
    → Error analysis for misclassified samples
    → Feature engineering based on SHAP insights
    → Consider model calibration for probability estimates
    
    """
    
    ax.text(0.5, 0.5, conclusions_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='center', horizontalalignment='center',
            family='monospace')
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def generate_pdf_report():
    """Main function to generate the complete PDF report."""
    print("=" * 70)
    print("GENERATING COMPREHENSIVE PDF REPORT")
    print("=" * 70)
    
    # Load results
    print("\n[1/8] Loading results...")
    results = load_results()
    print(f"      Loaded {len(results)} models")
    
    # Create PDF
    print(f"\n[2/8] Creating PDF: {OUTPUT_PDF}")
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    with PdfPages(OUTPUT_PDF) as pdf:
        print("\n[3/8] Generating title page...")
        create_title_page(pdf, results)
        
        print("\n[4/8] Creating overall comparison tables...")
        create_overall_comparison_table(pdf, results)
        
        print("\n[5/8] Creating metrics visualizations...")
        create_metrics_visualization(pdf, results)
        
        print("\n[6/8] Creating per-class performance analysis...")
        create_per_class_performance(pdf, results)
        
        print("\n[7/8] Generating detailed model pages...")
        for idx, (model_name, model_data) in enumerate(results.items(), 1):
            print(f"      [{idx}/{len(results)}] Processing {model_name}...")
            create_model_detail_page(pdf, model_name, model_data, FIGURES_DIR)
        
        print("\n[8/8] Adding conclusions page...")
        create_conclusions_page(pdf, results)
        
        # Add metadata
        d = pdf.infodict()
        d['Title'] = 'DNA Artifact Classification - ML Model Comparison'
        d['Author'] = 'ML Pipeline'
        d['Subject'] = 'Comprehensive model performance analysis'
        d['Keywords'] = 'Machine Learning, DNA, LTDNA, Classification, Model Comparison'
        d['CreationDate'] = datetime.now()
    
    print("\n" + "=" * 70)
    print(f"✓ PDF REPORT GENERATED SUCCESSFULLY")
    print(f"  Location: {OUTPUT_PDF}")
    print(f"  Size: {os.path.getsize(OUTPUT_PDF) / 1024 / 1024:.2f} MB")
    print("=" * 70)

if __name__ == '__main__':
    generate_pdf_report()
