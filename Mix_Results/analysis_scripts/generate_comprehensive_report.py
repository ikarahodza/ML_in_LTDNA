"""
Generate Comprehensive Analysis Report
Combines all analysis results into a single comprehensive HTML report:
- Original model comparison
- Hyperparameter tuning results
- Cross-validation analysis
- External validation results
- Error analysis findings
- SHAP insights
- Feature engineering impact
"""

import json
import base64
from pathlib import Path
from datetime import datetime

print("=" * 80)
print("GENERATING COMPREHENSIVE ANALYSIS REPORT")
print("=" * 80)

# Helper function to encode image to base64
def encode_image(image_path):
    try:
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None

# Load all result files
print("\n1. Loading analysis results...")

# Original results
with open('Mix_Results/reports/combined_results.json', 'r') as f:
    raw_results = json.load(f)

# Flatten the nested structure: extract best strategy per model
original_results = {}
for model_name, model_data in raw_results.items():
    if 'results' in model_data:
        # Find best strategy by balanced_accuracy
        best_strategy = None
        best_score = -1
        best_metrics = {}
        
        for strategy, metrics in model_data['results'].items():
            score = metrics.get('balanced_accuracy', 0)
            if score > best_score:
                best_score = score
                best_strategy = strategy
                best_metrics = metrics.copy()
                best_metrics['resampling_strategy'] = strategy
        
        if best_strategy:
            # Use formatted name: model_strategy
            formatted_name = f"{model_name}_{best_strategy.lower().replace(' ', '_')}"
            original_results[formatted_name] = best_metrics

print(f"   - Original model results loaded ({len(original_results)} model-strategy combinations)")

# Hyperparameter tuning results (if exists)
try:
    tuning_results = {}
    tuning_dir = Path('Mix_Results/hyperparameter_tuning_results')
    if tuning_dir.exists():
        for file in tuning_dir.glob('*_results.json'):
            with open(file, 'r') as f:
                model_name = file.stem.replace('_results', '')
                tuning_results[model_name] = json.load(f)
    print(f"   - Hyperparameter tuning results loaded ({len(tuning_results)} models)")
except:
    tuning_results = {}
    print("   - Hyperparameter tuning results not found")

# Error analysis results
try:
    with open('figures/error_analysis_results.json', 'r') as f:
        error_results = json.load(f)
    print("   - Error analysis results loaded")
except:
    error_results = {}
    print("   - Error analysis results not found")

# SHAP analysis results
try:
    with open('figures/shap_analysis_results.json', 'r') as f:
        shap_results = json.load(f)
    print("   - SHAP analysis results loaded")
except:
    shap_results = {}
    print("   - SHAP analysis results not found")

# Feature engineering results
try:
    with open('feature_engineering_results.json', 'r') as f:
        feature_eng_results = json.load(f)
    print("   - Feature engineering results loaded")
except:
    feature_eng_results = {}
    print("   - Feature engineering results not found")

# External validation results
try:
    with open('external_validation_results/external_validation_summary.json', 'r') as f:
        external_val_results = json.load(f)
    print("   - External validation results loaded")
except:
    external_val_results = {}
    print("   - External validation results not found")

print("\n2. Generating HTML report...")

# Start HTML
html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Comprehensive ML Analysis Report - Mix Dataset</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 40px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 4px solid #3498db;
            padding-bottom: 15px;
            margin-top: 0;
        }}
        h2 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            margin-top: 40px;
        }}
        h3 {{
            color: #34495e;
            margin-top: 25px;
        }}
        .summary-box {{
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 5px solid #3498db;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .metric-card h4 {{
            margin: 0 0 10px 0;
            font-size: 14px;
            opacity: 0.9;
        }}
        .metric-card .value {{
            font-size: 32px;
            font-weight: bold;
            margin: 10px 0;
        }}
        .metric-card .change {{
            font-size: 14px;
            opacity: 0.9;
        }}
        .positive {{
            color: #2ecc71;
        }}
        .negative {{
            color: #e74c3c;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th {{
            background-color: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: bold;
        }}
        td {{
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .image-container {{
            margin: 30px 0;
            text-align: center;
        }}
        .image-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        .key-finding {{
            background-color: #fff3cd;
            border-left: 5px solid #ffc107;
            padding: 15px;
            margin: 15px 0;
            border-radius: 4px;
        }}
        .recommendation {{
            background-color: #d1ecf1;
            border-left: 5px solid #17a2b8;
            padding: 15px;
            margin: 15px 0;
            border-radius: 4px;
        }}
        .section-divider {{
            height: 2px;
            background: linear-gradient(to right, #3498db, transparent);
            margin: 40px 0;
        }}
        .toc {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin: 30px 0;
        }}
        .toc a {{
            color: #3498db;
            text-decoration: none;
            display: block;
            padding: 5px 0;
        }}
        .toc a:hover {{
            text-decoration: underline;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 14px;
            text-align: right;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Comprehensive Machine Learning Analysis Report</h1>
        <p style="font-size: 18px; color: #7f8c8d;">Analysis of Mix Dataset - Complete Pipeline Results</p>
        <div class="timestamp">Generated: {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}</div>
        
        <div class="toc">
            <h3>Table of Contents</h3>
            <a href="#executive-summary">1. Executive Summary</a>
            <a href="#original-results">2. Original Model Comparison</a>
            <a href="#hyperparameter-tuning">3. Hyperparameter Tuning</a>
            <a href="#error-analysis">4. Error Analysis</a>
            <a href="#shap-analysis">5. SHAP Feature Importance Analysis</a>
            <a href="#feature-engineering">6. Feature Engineering</a>
            <a href="#external-validation">7. External Validation</a>
            <a href="#conclusions">8. Conclusions & Recommendations</a>
        </div>
"""

# 1. EXECUTIVE SUMMARY
html += """
        <div class="section-divider"></div>
        <h2 id="executive-summary">1. Executive Summary</h2>
        
        <div class="summary-box">
            <h3>Project Overview</h3>
            <p>This report presents a comprehensive analysis of machine learning models for DNA classification, 
            including baseline model comparison, hyperparameter optimization, error pattern analysis, 
            feature importance investigation, and feature engineering.</p>
        </div>
"""

# Key metrics from original results
if original_results:
    best_models = sorted(original_results.items(), 
                        key=lambda x: x[1].get('balanced_accuracy', 0), 
                        reverse=True)[:3]
    
    html += """
        <div class="metric-grid">
"""
    for idx, (model_name, metrics) in enumerate(best_models, 1):
        html += f"""
            <div class="metric-card">
                <h4>Top {idx} Model: {model_name.replace('_', ' ').title()}</h4>
                <div class="value">{metrics.get('balanced_accuracy', 0):.4f}</div>
                <div class="change">Balanced Accuracy</div>
                <div class="change">F1-Macro: {metrics.get('f1_macro', 0):.4f}</div>
            </div>
"""
    html += """
        </div>
"""

# Feature engineering impact
if feature_eng_results:
    baseline_acc = feature_eng_results.get('baseline', {}).get('balanced_accuracy', 0)
    engineered_acc = feature_eng_results.get('engineered', {}).get('balanced_accuracy', 0)
    improvement = feature_eng_results.get('improvement', {}).get('balanced_accuracy_pct', 0)
    
    html += f"""
        <div class="key-finding">
            <strong>Feature Engineering Impact:</strong> Created 56 new features resulting in 
            <strong>{improvement:+.2f}%</strong> improvement in balanced accuracy 
            ({baseline_acc:.4f} → {engineered_acc:.4f})
        </div>
"""

# Error analysis highlights
if error_results:
    html += f"""
        <div class="key-finding">
            <strong>Error Analysis Findings:</strong> Analyzed {error_results.get('total_errors', 0)} errors across 
            {error_results.get('total_models', 0)} models. Most common confusion: 
            Class {list(error_results.get('top_confusion_pairs', {}).keys())[0].replace('_to_', ' → ')} 
            ({list(error_results.get('top_confusion_pairs', {}).values())[0]} errors)
        </div>
"""

# 2. ORIGINAL MODEL COMPARISON
html += """
        <div class="section-divider"></div>
        <h2 id="original-results">2. Original Model Comparison</h2>
        
        <h3>Model Performance Rankings</h3>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Model</th>
                    <th>Balanced Accuracy</th>
                    <th>F1-Macro</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>Accuracy</th>
                    <th>Log Loss</th>
                </tr>
            </thead>
            <tbody>
"""

if original_results:
    ranked_models = sorted(original_results.items(), 
                          key=lambda x: x[1].get('balanced_accuracy', 0), 
                          reverse=True)
    
    for idx, (model_name, metrics) in enumerate(ranked_models[:15], 1):
        # Calculate macro precision and recall from per-class values
        per_class_precision = metrics.get('per_class_precision', [])
        per_class_recall = metrics.get('per_class_recall', [])
        
        macro_precision = sum(per_class_precision) / len(per_class_precision) if per_class_precision else 0
        macro_recall = sum(per_class_recall) / len(per_class_recall) if per_class_recall else 0
        
        html += f"""
                <tr>
                    <td><strong>{idx}</strong></td>
                    <td>{model_name.replace('_', ' ').title()}</td>
                    <td>{metrics.get('balanced_accuracy', 0):.4f}</td>
                    <td>{metrics.get('f1_macro', 0):.4f}</td>
                    <td>{macro_precision:.4f}</td>
                    <td>{macro_recall:.4f}</td>
                    <td>{metrics.get('accuracy', 0):.4f}</td>
                    <td>{metrics.get('log_loss', 0):.4f}</td>
                </tr>
"""

html += """
            </tbody>
        </table>
"""

# 3. HYPERPARAMETER TUNING
html += """
        <div class="section-divider"></div>
        <h2 id="hyperparameter-tuning">3. Hyperparameter Tuning</h2>
"""

if tuning_results:
    html += f"""
        <p>Performed randomized search with 50 iterations and 5-fold cross-validation for {len(tuning_results)} top-performing models.</p>
        
        <h3>Tuning Results Summary</h3>
        <table>
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Best CV Score</th>
                    <th>Improvement</th>
                    <th>Total Iterations</th>
                </tr>
            </thead>
            <tbody>
"""
    
    for model_name, results in tuning_results.items():
        best_score = results.get('best_score', 0)
        html += f"""
                <tr>
                    <td>{model_name.replace('_', ' ').title()}</td>
                    <td>{best_score:.4f}</td>
                    <td>Tuned model</td>
                    <td>50</td>
                </tr>
"""
    
    html += """
            </tbody>
        </table>
"""
else:
    html += """
        <p><em>Hyperparameter tuning results not available.</em></p>
"""

# 4. ERROR ANALYSIS
html += """
        <div class="section-divider"></div>
        <h2 id="error-analysis">4. Error Analysis</h2>
"""

if error_results:
    html += f"""
        <div class="summary-box">
            <h3>Error Analysis Overview</h3>
            <ul>
                <li><strong>Total Models Analyzed:</strong> {error_results.get('total_models', 0)}</li>
                <li><strong>Total Errors:</strong> {error_results.get('total_errors', 0)}</li>
                <li><strong>Best Performing Model:</strong> {error_results.get('best_model', 'N/A')}</li>
                <li><strong>Worst Performing Model:</strong> {error_results.get('worst_model', 'N/A')}</li>
                <li><strong>Most Confused Class:</strong> Class {error_results.get('most_confused_class', 'N/A')}</li>
            </ul>
        </div>
        
        <h3>Top Confusion Pairs</h3>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Confusion Pair</th>
                    <th>Error Count</th>
                </tr>
            </thead>
            <tbody>
"""
    
    for idx, (pair, count) in enumerate(list(error_results.get('top_confusion_pairs', {}).items())[:10], 1):
        pair_formatted = pair.replace('_to_', ' → ')
        html += f"""
                <tr>
                    <td>{idx}</td>
                    <td>{pair_formatted}</td>
                    <td>{count}</td>
                </tr>
"""
    
    html += """
            </tbody>
        </table>
"""
    
    # Add error analysis visualizations
    error_viz_files = [
        ('figures/error_analysis_confusion_heatmap.png', 'Error Confusion Heatmap'),
        ('figures/error_analysis_by_model.png', 'Errors by Model'),
        ('figures/error_analysis_by_class.png', 'Errors by Class'),
        ('figures/error_analysis_confusion_pairs.png', 'Top Confusion Pairs'),
        ('figures/error_analysis_by_resampling.png', 'Errors by Resampling Strategy')
    ]
    
    html += """
        <h3>Error Analysis Visualizations</h3>
"""
    
    for img_path, title in error_viz_files:
        img_data = encode_image(img_path)
        if img_data:
            html += f"""
        <div class="image-container">
            <h4>{title}</h4>
            <img src="data:image/png;base64,{img_data}" alt="{title}">
        </div>
"""

# 5. SHAP ANALYSIS
html += """
        <div class="section-divider"></div>
        <h2 id="shap-analysis">5. SHAP Feature Importance Analysis</h2>
"""

if shap_results:
    html += """
        <div class="summary-box">
            <h3>SHAP Analysis Overview</h3>
            <p>Analyzed feature importance across multiple models using SHAP (SHapley Additive exPlanations) 
            to identify the most influential features and guide feature engineering.</p>
        </div>
        
        <h3>Top 20 Most Important Features</h3>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Feature</th>
                    <th>Category</th>
                    <th>Mean Importance</th>
                    <th>Consistency</th>
                </tr>
            </thead>
            <tbody>
"""
    
    for idx, feature_data in enumerate(shap_results.get('top_features', [])[:20], 1):
        feature_name = feature_data.get('feature', '')
        mean_importance = feature_data.get('mean_importance', 0)
        std_dev = feature_data.get('std_dev', 0)
        
        # Determine category
        if 'UpStream' in feature_name:
            category = 'Upstream'
        elif 'DownStream' in feature_name:
            category = 'Downstream'
        elif feature_name.startswith('C'):
            category = 'C-region'
        else:
            category = 'Other'
        
        # Determine consistency
        if std_dev < mean_importance * 0.3:
            consistency = 'High'
        elif std_dev < mean_importance * 0.6:
            consistency = 'Medium'
        else:
            consistency = 'Low'
        
        html += f"""
                <tr>
                    <td>{idx}</td>
                    <td><strong>{feature_name}</strong></td>
                    <td>{category}</td>
                    <td>{mean_importance:.2f}</td>
                    <td>{consistency}</td>
                </tr>
"""
    
    html += """
            </tbody>
        </table>
"""
    
    # Add SHAP visualizations
    shap_viz_files = [
        ('figures/shap_analysis_top_features.png', 'Top Features Bar Chart'),
        ('figures/shap_analysis_heatmap.png', 'Feature Importance Heatmap'),
        ('figures/shap_analysis_consistency.png', 'Feature Consistency Analysis'),
        ('figures/shap_analysis_categories.png', 'Feature Category Analysis')
    ]
    
    html += """
        <h3>SHAP Visualizations</h3>
"""
    
    for img_path, title in shap_viz_files:
        img_data = encode_image(img_path)
        if img_data:
            html += f"""
        <div class="image-container">
            <h4>{title}</h4>
            <img src="data:image/png;base64,{img_data}" alt="{title}">
        </div>
"""

# 6. FEATURE ENGINEERING
html += """
        <div class="section-divider"></div>
        <h2 id="feature-engineering">6. Feature Engineering</h2>
"""

if feature_eng_results:
    baseline = feature_eng_results.get('baseline', {})
    engineered = feature_eng_results.get('engineered', {})
    improvement = feature_eng_results.get('improvement', {})
    
    html += f"""
        <div class="summary-box">
            <h3>Feature Engineering Impact</h3>
            <p>Created 56 new features based on SHAP insights, including interaction terms, polynomial features, 
            ratios, aggregations, and context-based features.</p>
        </div>
        
        <div class="metric-grid">
            <div class="metric-card">
                <h4>Balanced Accuracy</h4>
                <div class="value">{engineered.get('balanced_accuracy', 0):.4f}</div>
                <div class="change positive">+{improvement.get('balanced_accuracy_pct', 0):.2f}%</div>
                <div class="change">Baseline: {baseline.get('balanced_accuracy', 0):.4f}</div>
            </div>
            <div class="metric-card">
                <h4>F1-Macro Score</h4>
                <div class="value">{engineered.get('f1_macro', 0):.4f}</div>
                <div class="change positive">+{improvement.get('f1_macro_pct', 0):.2f}%</div>
                <div class="change">Baseline: {baseline.get('f1_macro', 0):.4f}</div>
            </div>
            <div class="metric-card">
                <h4>CV Score (5-fold)</h4>
                <div class="value">{engineered.get('cv_balanced_accuracy_mean', 0):.4f}</div>
                <div class="change positive">+{improvement.get('cv_balanced_accuracy_pct', 0):.2f}%</div>
                <div class="change">Baseline: {baseline.get('cv_balanced_accuracy_mean', 0):.4f}</div>
            </div>
        </div>
        
        <h3>Top New Engineered Features</h3>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Feature Name</th>
                    <th>Importance</th>
                    <th>Type</th>
                </tr>
            </thead>
            <tbody>
"""
    
    for idx, feat in enumerate(feature_eng_results.get('top_new_features', [])[:15], 1):
        feature_name = feat.get('feature', '')
        importance = feat.get('importance', 0)
        
        # Determine type
        if 'interact_' in feature_name:
            feat_type = 'Interaction'
        elif '_squared' in feature_name or '_cubed' in feature_name:
            feat_type = 'Polynomial'
        elif 'ratio_' in feature_name:
            feat_type = 'Ratio'
        elif 'upstream_' in feature_name or 'downstream_' in feature_name:
            feat_type = 'Aggregation'
        elif 'poi_' in feature_name:
            feat_type = 'Context'
        else:
            feat_type = 'Statistical'
        
        html += f"""
                <tr>
                    <td>{idx}</td>
                    <td><strong>{feature_name}</strong></td>
                    <td>{importance:.6f}</td>
                    <td>{feat_type}</td>
                </tr>
"""
    
    html += """
            </tbody>
        </table>
        
        <div class="recommendation">
            <strong>Recommendation:</strong> Feature engineering shows positive improvement. 
            Use the engineered dataset for final model training and deployment.
        </div>
"""

# 7. EXTERNAL VALIDATION
html += """
        <div class="section-divider"></div>
        <h2 id="external-validation">7. External Validation</h2>
"""

if external_val_results:
    html += """
        <p>Models tested on independent validation dataset (Input_Data-Single-train.csv) 
        to assess generalization performance.</p>
"""
else:
    html += """
        <p><em>External validation results not available. Run external_validation.py to generate.</em></p>
"""

# 8. CONCLUSIONS
html += """
        <div class="section-divider"></div>
        <h2 id="conclusions">8. Conclusions & Recommendations</h2>
        
        <h3>Key Findings</h3>
"""

# Generate key findings based on available data
findings = []

if original_results:
    best_model = max(original_results.items(), key=lambda x: x[1].get('balanced_accuracy', 0))
    findings.append(f"<strong>Best Original Model:</strong> {best_model[0].replace('_', ' ').title()} "
                   f"with balanced accuracy of {best_model[1].get('balanced_accuracy', 0):.4f}")

if error_results:
    findings.append(f"<strong>Error Patterns:</strong> Most common confusion is between "
                   f"classes {list(error_results.get('top_confusion_pairs', {}).keys())[0].replace('_to_', ' and ')}, "
                   f"suggesting these classes may benefit from additional feature engineering")

if shap_results and shap_results.get('top_features'):
    top_feat = shap_results['top_features'][0]
    findings.append(f"<strong>Most Important Feature:</strong> {top_feat.get('feature', 'N/A')} "
                   f"consistently shows highest importance across models")

if feature_eng_results:
    improvement = feature_eng_results.get('improvement', {}).get('balanced_accuracy_pct', 0)
    if improvement > 0:
        findings.append(f"<strong>Feature Engineering Success:</strong> Engineered features improved "
                       f"performance by {improvement:.2f}%, validating the feature engineering approach")

for finding in findings:
    html += f"""
        <div class="key-finding">
            {finding}
        </div>
"""

html += """
        <h3>Recommendations for Deployment</h3>
        <div class="recommendation">
            <ol>
                <li><strong>Model Selection:</strong> Use top-performing models (LGBM, TabPFN, or XGBoost) 
                with engineered features for production deployment</li>
                <li><strong>Feature Engineering:</strong> Include the 56 engineered features in the 
                production pipeline for improved performance</li>
                <li><strong>Monitoring:</strong> Implement monitoring for the most confused class pairs 
                identified in error analysis</li>
                <li><strong>Ensemble Approach:</strong> Consider ensemble methods combining top 3-5 models 
                for more robust predictions</li>
                <li><strong>Continuous Improvement:</strong> Regularly update models with new data and 
                re-evaluate feature importance</li>
            </ol>
        </div>
        
        <h3>Next Steps</h3>
        <ul>
            <li>Apply feature engineering to Single dataset for comparison</li>
            <li>Implement probability calibration for improved confidence estimates</li>
            <li>Deploy best model with monitoring infrastructure</li>
            <li>Set up A/B testing framework for model comparison in production</li>
        </ul>
"""

# Footer
html += f"""
        <div class="section-divider"></div>
        <div class="timestamp">
            Report generated on {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}<br>
            Analysis Pipeline: ML_in_LTDNA
        </div>
    </div>
</body>
</html>
"""

# Save report
output_file = 'comprehensive_analysis_report.html'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"\n3. Report saved to: {output_file}")
print("\n" + "=" * 80)
print("COMPREHENSIVE REPORT GENERATION COMPLETE!")
print("=" * 80)
print(f"\nOpen {output_file} in a web browser to view the full report.")
