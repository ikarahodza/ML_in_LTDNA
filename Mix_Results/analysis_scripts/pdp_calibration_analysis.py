"""
Partial Dependence and Calibration Analysis
- PDP: Shows how features affect predictions on average
- Calibration: Validates if predicted probabilities match actual outcomes
"""
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings('ignore')

# Professional styling
sns.set_style("whitegrid")
COLOR_PRIMARY = '#5e81ac'
COLOR_SUCCESS = '#a3be8c'
COLOR_WARNING = '#ebcb8b'
COLOR_DANGER = '#bf616a'

# Configuration
DATASET_TYPE = 'Mix'  # 'Mix' or 'Single'
TOP_N_FEATURES = 6  # Number of features for PDP

print("=" * 100)
print(f"PARTIAL DEPENDENCE & CALIBRATION ANALYSIS - {DATASET_TYPE} DATASET")
print("=" * 100)

# Load data
print("\nLoading data and models...")
df = pd.read_csv(f'Input_Data-{DATASET_TYPE}-train.csv')
X = df.drop(['sample', 'PoI', 'Classification'], axis=1, errors='ignore')
y = df['Classification'].map({'Allele': 0, 'Back_Stutter': 1, 'Forward_Stutter': 2, 'Pull_Up': 3})

feature_names = X.columns.tolist()
class_names = ['Allele', 'Back_Stutter', 'Forward_Stutter', 'Pull_Up']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Load best individual model (PDP is too slow for ensemble)
if DATASET_TYPE == 'Mix':
    # Use LightGBM (best individual model) instead of Stacking (too slow for PDP)
    model_path = 'Mix_Results/models/lgbm_model_lgbm_randomoversampler.pkl'
    scaler_path = None
    model_name = "LightGBM (for PDP analysis)"
    
    # Also load stacking for calibration
    stacking_path = 'Mix_Results/models/stacking_ensemble.pkl'
    stacking_scaler_path = 'Mix_Results/models/stacking_ensemble_scaler.pkl'
else:
    model_path = 'Single_Results/models/model_logistic_regression_original.pkl'
    scaler_path = None
    model_name = "Logistic Regression"
    stacking_path = None

print(f"Loading model: {model_name}")
model = joblib.load(model_path)

if scaler_path and os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Scaler loaded and applied")
    X_for_analysis = X_train_scaled
else:
    X_train_scaled = X_train.values
    X_test_scaled = X_test.values
    scaler = None
    print("No scaler needed")
    X_for_analysis = X_train.values

# Create output directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
output_dir = os.path.join(project_root, f'{DATASET_TYPE}_Results', 'explainability_results')
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}")

# Get feature importance from SHAP or built-in
print("\nIdentifying top features for PDP...")
shap_results_path = f'{DATASET_TYPE}_Results/figures/shap_feature_importance.json'
if os.path.exists(shap_results_path):
    with open(shap_results_path, 'r') as f:
        shap_data = json.load(f)
    top_features = list(shap_data.keys())[:TOP_N_FEATURES]
    print(f"Using top {TOP_N_FEATURES} features from SHAP analysis")
else:
    # Use variance as proxy
    feature_variance = X_train.var().sort_values(ascending=False)
    top_features = feature_variance.index[:TOP_N_FEATURES].tolist()
    print(f"Using top {TOP_N_FEATURES} features by variance")

print(f"Selected features: {', '.join(top_features)}")

# ==============================================================================
# PARTIAL DEPENDENCE PLOTS
# ==============================================================================
print("\n" + "=" * 100)
print("GENERATING PARTIAL DEPENDENCE PLOTS")
print("=" * 100)

# Get feature indices
feature_indices = [feature_names.index(f) for f in top_features]

# Generate PDP for each class
for class_idx, class_name in enumerate(class_names):
    print(f"\nGenerating PDP for class: {class_name}")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Partial Dependence Plots - {class_name}\n{model_name} on {DATASET_TYPE} Dataset',
                 fontsize=14, fontweight='bold')
    
    axes = axes.ravel()
    
    for idx, (feat_idx, feat_name) in enumerate(zip(feature_indices, top_features)):
        ax = axes[idx]
        
        # Compute partial dependence
        pd_result = partial_dependence(model, X_for_analysis, [feat_idx], 
                                       method='auto', kind='average',
                                       grid_resolution=50)
        
        # Extract data
        pd_values = pd_result['average'][0]
        feature_values = pd_result['grid_values'][0]
        
        # For multi-class, get specific class probabilities
        if hasattr(model, 'predict_proba') and len(pd_values.shape) > 1:
            pd_values_class = pd_values[:, class_idx]
        else:
            pd_values_class = pd_values
        
        # Plot
        ax.plot(feature_values, pd_values_class, color=COLOR_PRIMARY, linewidth=2.5)
        ax.fill_between(feature_values, pd_values_class, alpha=0.3, color=COLOR_PRIMARY)
        ax.set_xlabel(feat_name, fontsize=10, fontweight='bold')
        ax.set_ylabel(f'P({class_name})', fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_title(feat_name, fontsize=11)
        
        # Add rug plot showing data distribution
        if isinstance(X_for_analysis, pd.DataFrame):
            rug_data = X_for_analysis.iloc[:, feat_idx].values
        else:
            rug_data = X_for_analysis[:, feat_idx]
        rug_sample = np.random.choice(rug_data, min(100, len(rug_data)), replace=False)
        ax.scatter(rug_sample, [pd_values_class.min()] * len(rug_sample), 
                  alpha=0.3, s=10, color='black', marker='|')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'pdp_{class_name.lower()}.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: pdp_{class_name.lower()}.png")

# Combined PDP for top 3 features (show class 0: Allele)
print(f"\nGenerating combined PDP for top 3 features...")
fig = plt.figure(figsize=(14, 10))
fig.suptitle(f'Partial Dependence Analysis - Top 3 Features (Allele Class)\n{model_name} on {DATASET_TYPE} Dataset',
             fontsize=14, fontweight='bold')

for feat_idx in range(min(3, len(top_features))):
    PartialDependenceDisplay.from_estimator(
        model, X_for_analysis, [feature_indices[feat_idx]],
        target=0,  # Allele class
        feature_names=feature_names,
        grid_resolution=50,
        ax=plt.subplot(1, 3, feat_idx + 1)
    )

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'pdp_combined_top3.png'), 
            dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: pdp_combined_top3.png")

# ==============================================================================
# CALIBRATION ANALYSIS
# ==============================================================================
print("\n" + "=" * 100)
print("GENERATING CALIBRATION PLOTS")
print("=" * 100)

# For calibration, use the stacking ensemble if available (best model)
if DATASET_TYPE == 'Mix' and stacking_path and os.path.exists(stacking_path):
    print(f"Loading Stacking Ensemble for calibration analysis...")
    cal_model = joblib.load(stacking_path)
    cal_scaler = joblib.load(stacking_scaler_path)
    X_test_cal = cal_scaler.transform(X_test)
    cal_model_name = "Stacking Ensemble"
else:
    cal_model = model
    X_test_cal = X_test_scaled
    cal_model_name = model_name

# Get predictions on test set
y_pred = cal_model.predict(X_test_cal)
y_proba = cal_model.predict_proba(X_test_cal)

# Overall calibration plot (all classes)
print("\nGenerating overall calibration plot...")
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle(f'Calibration Analysis - {cal_model_name} on {DATASET_TYPE} Dataset',
             fontsize=14, fontweight='bold')

calibration_data = {}

for class_idx, class_name in enumerate(class_names):
    ax = axes[class_idx // 2, class_idx % 2]
    
    # Binary problem: is it this class or not?
    y_true_binary = (y_test == class_idx).astype(int)
    y_prob_class = y_proba[:, class_idx]
    
    # Compute calibration curve
    try:
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true_binary, y_prob_class, n_bins=10, strategy='uniform'
        )
        
        # Plot calibration curve
        ax.plot(mean_predicted_value, fraction_of_positives, 's-', 
                color=COLOR_PRIMARY, linewidth=2, markersize=8,
                label='Model')
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
        
        # Calculate calibration error
        calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
        
        ax.set_xlabel('Mean Predicted Probability', fontsize=11)
        ax.set_ylabel('Fraction of Positives', fontsize=11)
        ax.set_title(f'{class_name}\nCalibration Error: {calibration_error:.4f}', 
                    fontsize=12, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(alpha=0.3)
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        
        # Store data
        calibration_data[class_name] = {
            'calibration_error': float(calibration_error),
            'mean_predicted': mean_predicted_value.tolist(),
            'fraction_positive': fraction_of_positives.tolist()
        }
        
        print(f"  {class_name}: Calibration Error = {calibration_error:.4f}")
        
    except Exception as e:
        ax.text(0.5, 0.5, f'Insufficient data\nfor {class_name}', 
               ha='center', va='center', fontsize=12)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        print(f"  {class_name}: Insufficient data")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'calibration_curves.png'), 
            dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: calibration_curves.png")

# Confidence histogram
print("\nGenerating confidence distribution...")
fig, ax = plt.subplots(figsize=(10, 6))

# Get max probability for each prediction (model confidence)
max_proba = y_proba.max(axis=1)
correct_mask = (y_pred == y_test.values)

# Plot distributions
ax.hist(max_proba[correct_mask], bins=20, alpha=0.7, 
        color=COLOR_SUCCESS, label='Correct', edgecolor='black')
ax.hist(max_proba[~correct_mask], bins=20, alpha=0.7, 
        color=COLOR_DANGER, label='Incorrect', edgecolor='black')

ax.set_xlabel('Prediction Confidence (Max Probability)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title(f'Confidence Distribution - {cal_model_name} on {DATASET_TYPE} Dataset\n'
             f'Correct: {correct_mask.sum()}/{len(y_test)} ({correct_mask.sum()/len(y_test)*100:.1f}%)',
             fontsize=13, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'confidence_distribution.png'), 
            dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: confidence_distribution.png")

# Save calibration data
with open(os.path.join(output_dir, 'calibration_results.json'), 'w') as f:
    json.dump(calibration_data, f, indent=2)

# ==============================================================================
# SUMMARY STATISTICS
# ==============================================================================
print("\n" + "=" * 100)
print("SUMMARY STATISTICS")
print("=" * 100)

print("\nCalibration Errors by Class:")
for class_name, data in calibration_data.items():
    if 'calibration_error' in data:
        print(f"  {class_name:20s}: {data['calibration_error']:.4f}")

avg_calibration_error = np.mean([d['calibration_error'] for d in calibration_data.values() 
                                  if 'calibration_error' in d])
print(f"\n  Average Calibration Error: {avg_calibration_error:.4f}")

print("\nConfidence Statistics:")
print(f"  Correct predictions:")
print(f"    Mean confidence: {max_proba[correct_mask].mean():.4f}")
print(f"    Std confidence:  {max_proba[correct_mask].std():.4f}")
if (~correct_mask).sum() > 0:
    print(f"  Incorrect predictions:")
    print(f"    Mean confidence: {max_proba[~correct_mask].mean():.4f}")
    print(f"    Std confidence:  {max_proba[~correct_mask].std():.4f}")

print("\n" + "=" * 100)
print("ANALYSIS COMPLETE")
print("=" * 100)
print(f"\nResults saved to: {output_dir}/")
print(f"  - Partial Dependence Plots (per class)")
print(f"  - Combined PDP for top 3 features")
print(f"  - Calibration curves (per class)")
print(f"  - Confidence distribution")
print(f"  - calibration_results.json")
print("\nInterpretation:")
print("  - PDP: Shows average effect of each feature on predictions")
print("  - Calibration: <0.05 = well-calibrated, >0.10 = poorly calibrated")
print("  - Confidence: Higher for correct predictions = model knows when it's right")
