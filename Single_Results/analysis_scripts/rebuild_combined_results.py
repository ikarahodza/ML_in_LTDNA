"""
Rebuild combined_results.json from existing individual HTML reports.
This script reconstructs the combined JSON by parsing the HTML report files.
"""

import json
import os
import re
from pathlib import Path
from bs4 import BeautifulSoup

# Get project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_DIR = os.path.dirname(SCRIPT_DIR)  # Single_Results
PROJECT_ROOT = os.path.dirname(ANALYSIS_DIR)

REPORTS_DIR = os.path.join(ANALYSIS_DIR, 'reports')
OUTPUT_JSON = os.path.join(REPORTS_DIR, 'combined_results.json')

print("=" * 80)
print("REBUILDING COMBINED RESULTS JSON FROM HTML REPORTS")
print("=" * 80)

# Load dataset to get class names
print("\n[1/3] Loading dataset to get class names...")
import pandas as pd
data_file = os.path.join(PROJECT_ROOT, 'Input_Data-Single-train.csv')
df = pd.read_csv(data_file)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(df['Artifact'])
class_names = le.classes_.tolist()
print(f"   Class names: {class_names}")

# Find all HTML reports
print("\n[2/3] Parsing HTML reports...")
html_files = list(Path(REPORTS_DIR).glob('comparison_report_*.html'))
html_files = [f for f in html_files if f.stem != 'comparison_report']  # Exclude generic one
print(f"   Found {len(html_files)} model report files")

combined_data = {}

for html_file in html_files:
    model_name = html_file.stem.replace('comparison_report_', '')
    print(f"   - Parsing {model_name}...")
    
    with open(html_file, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find the summary table
    summary_table = soup.find('h2', string='Summary by Strategy')
    if not summary_table:
        print(f"     ✗ Could not find summary table")
        continue
    
    table = summary_table.find_next('table')
    if not table:
        print(f"     ✗ Could not find data table")
        continue
    
    # Parse the table
    rows = table.find('tbody').find_all('tr')
    results = {}
    
    for row in rows:
        cells = row.find_all('td')
        if len(cells) >= 7:
            strategy = cells[0].get_text(strip=True)
            
            results[strategy] = {
                'accuracy': float(cells[1].get_text(strip=True)),
                'balanced_accuracy': float(cells[2].get_text(strip=True)),
                'f1_macro': float(cells[3].get_text(strip=True)),
                'mse': float(cells[4].get_text(strip=True)),
                'mae': float(cells[5].get_text(strip=True)),
                'log_loss': float(cells[6].get_text(strip=True))
            }
    
    # Now parse per-class metrics for each strategy
    for strategy in results.keys():
        header_text = f'Per-class Metrics - {strategy}'
        per_class_header = soup.find('h3', string=header_text)
        
        if per_class_header:
            per_class_table = per_class_header.find_next('table')
            if per_class_table:
                per_class_rows = per_class_table.find('tbody').find_all('tr')
                
                precisions = []
                recalls = []
                f1s = []
                supports = []
                
                for pc_row in per_class_rows:
                    pc_cells = pc_row.find_all('td')
                    if len(pc_cells) >= 5:
                        precisions.append(float(pc_cells[1].get_text(strip=True)))
                        recalls.append(float(pc_cells[2].get_text(strip=True)))
                        f1s.append(float(pc_cells[3].get_text(strip=True)))
                        supports.append(int(pc_cells[4].get_text(strip=True)))
                
                results[strategy]['per_class_precision'] = precisions
                results[strategy]['per_class_recall'] = recalls
                results[strategy]['per_class_f1'] = f1s
                results[strategy]['per_class_support'] = supports
    
    combined_data[model_name] = {
        'results': results,
        'class_names': class_names
    }
    print(f"     ✓ Found {len(results)} strategies")

# Save to JSON
print("\n[3/3] Saving combined_results.json...")
os.makedirs(REPORTS_DIR, exist_ok=True)
with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(combined_data, f, indent=2)

print(f"\n✓ Combined results saved to: {OUTPUT_JSON}")
print(f"  Total models: {len(combined_data)}")
print(f"  Total model-strategy combinations: {sum(len(v['results']) for v in combined_data.values())}")

print("\n" + "=" * 80)
print("REBUILD COMPLETE")
print("=" * 80)
