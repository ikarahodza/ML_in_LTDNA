"""
Convert Top Models HTML Report to PDF
Uses matplotlib to create a PDF from the visualizations
"""
import os
import sys
from pathlib import Path
import json
from datetime import datetime

try:
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib", "pillow"])
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_DIR = os.path.join(PROJECT_ROOT, 'top_models_report')
PDF_FILE = os.path.join(REPORT_DIR, 'top_models_report.pdf')

# Model colors
MODEL_COLORS = {
    'tabpfn': '#e74c3c', 'xgboost': '#3498db', 'lightgbm': '#2ecc71',
    'extratrees': '#9b59b6', 'randomforest': '#f39c12', 'mlp': '#1abc9c',
    'svm': '#e67e22', 'gnb': '#95a5a6', 'logre': '#34495e'
}

def get_model_color(model_name):
    for key, color in MODEL_COLORS.items():
        if key in model_name.lower():
            return color
    return '#7f8c8d'

def load_results():
    """Load results from both datasets"""
    results = {}
    for dataset in ['Mix', 'Single']:
        json_path = os.path.join(PROJECT_ROOT, f'{dataset}_Results', 'reports', 'combined_results.json')
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                results[dataset] = json.load(f)
    return results

def create_title_page(pdf):
    """Create a title page"""
    fig = plt.figure(figsize=(8.27, 11.69))  # A4 size
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.7, 'Top Models Performance Report',
           ha='center', va='center', fontsize=32, fontweight='bold',
           transform=ax.transAxes)
    
    # Subtitle
    ax.text(0.5, 0.6, 'Mix & Single Dataset Analysis',
           ha='center', va='center', fontsize=20,
           transform=ax.transAxes, color='#555')
    
    # Date
    ax.text(0.5, 0.5, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
           ha='center', va='center', fontsize=14,
           transform=ax.transAxes, color='#888')
    
    # Description
    desc = ('This report presents the top 3 performing machine learning models\n'
            'across both Mix and Single datasets. For models with multiple strategies,\n'
            'only the best performing strategy is shown.')
    ax.text(0.5, 0.35, desc,
           ha='center', va='center', fontsize=12,
           transform=ax.transAxes, style='italic',
           bbox=dict(boxstyle='round,pad=1', facecolor='#f0f0f0', alpha=0.8))
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def add_image_to_pdf(pdf, image_path, title=None):
    """Add an image to the PDF"""
    if not os.path.exists(image_path):
        return
    
    img = Image.open(image_path)
    
    fig = plt.figure(figsize=(8.27, 11.69))  # A4 size
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    ax.imshow(img)
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def create_summary_page(pdf, results):
    """Create a summary statistics page"""
    fig = plt.figure(figsize=(8.27, 11.69))  # A4 size
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.95, 'Performance Summary', ha='center', va='top',
           fontsize=20, fontweight='bold', transform=ax.transAxes)
    
    # Extract best models
    all_models = {}
    for dataset, data in results.items():
        for model_name, model_data in data.items():
            if 'results' not in model_data:
                continue
            
            best_bal_acc = max(metrics.get('balanced_accuracy', 0) 
                             for metrics in model_data['results'].values())
            
            if model_name not in all_models or best_bal_acc > all_models[model_name]['bal_acc']:
                best_strategy = max(model_data['results'].items(),
                                  key=lambda x: x[1].get('balanced_accuracy', 0))
                all_models[model_name] = {
                    'bal_acc': best_bal_acc,
                    'strategy': best_strategy[0],
                    'dataset': dataset
                }
    
    # Sort and get top models
    sorted_models = sorted(all_models.items(), key=lambda x: x[1]['bal_acc'], reverse=True)[:10]
    
    # Create table
    table_data = [['Rank', 'Model', 'Dataset', 'Strategy', 'Bal Acc']]
    colors = [['#34495e']*5]  # Header color
    
    for i, (model, data) in enumerate(sorted_models, 1):
        table_data.append([
            str(i),
            model.upper(),
            data['dataset'],
            data['strategy'],
            f"{data['bal_acc']:.4f}"
        ])
        model_color = get_model_color(model)
        colors.append([model_color if j == 1 else '#ffffff' for j in range(5)])
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    bbox=[0.1, 0.2, 0.8, 0.7], cellColours=colors)
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(5):
        cell = table[(0, i)]
        cell.set_text_props(weight='bold', color='white')
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def convert_report_to_pdf():
    """Convert the report to PDF"""
    print("="*80)
    print("CONVERTING REPORT TO PDF")
    print("="*80)
    
    print(f"\nOutput: {PDF_FILE}")
    
    # Load results
    print("\n[1/4] Loading results...")
    results = load_results()
    
    if not results:
        print("✗ Error: No results found")
        return False
    
    print(f"  Loaded: {', '.join(results.keys())}")
    
    # Create PDF
    print("\n[2/4] Creating PDF pages...")
    with PdfPages(PDF_FILE) as pdf:
        # Title page
        print("  - Title page")
        create_title_page(pdf)
        
        # Summary page
        print("  - Summary page")
        create_summary_page(pdf, results)
        
        # Main comparison
        print("\n[3/4] Adding visualizations...")
        comparison_img = os.path.join(REPORT_DIR, 'top3_models_comparison.png')
        if os.path.exists(comparison_img):
            print("  - Top 3 models comparison")
            add_image_to_pdf(pdf, comparison_img, "Top 3 Models Comparison")
        
        # Individual breakdowns
        print("\n[4/4] Adding model breakdowns...")
        for img_file in os.listdir(REPORT_DIR):
            if img_file.endswith('_breakdown.png'):
                model_name = img_file.replace('_breakdown.png', '').replace('_', ' ').title()
                print(f"  - {model_name}")
                add_image_to_pdf(pdf, os.path.join(REPORT_DIR, img_file),
                               f'{model_name} - Detailed Breakdown')
        
        # Set PDF metadata
        d = pdf.infodict()
        d['Title'] = 'Top Models Performance Report'
        d['Author'] = 'ML Pipeline'
        d['Subject'] = 'Machine Learning Model Comparison'
        d['Keywords'] = 'ML, Classification, Performance'
        d['CreationDate'] = datetime.now()
    
    file_size = os.path.getsize(PDF_FILE) / 1024 / 1024  # MB
    
    print("\n" + "="*80)
    print("✓ PDF GENERATION COMPLETE")
    print("="*80)
    print(f"\nPDF saved to: {PDF_FILE}")
    print(f"File size: {file_size:.2f} MB")
    print("="*80)
    
    return True

if __name__ == "__main__":
    try:
        success = convert_report_to_pdf()
        if success:
            print("\n✓ Done! Open the PDF file to view the report.")
        else:
            sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error during PDF conversion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
