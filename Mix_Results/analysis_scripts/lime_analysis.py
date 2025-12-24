"""
LIME (Local Interpretable Model-agnostic Explanations) Analysis
Explains individual predictions by showing feature contributions for specific samples
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from lime import lime_tabular
import os
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATASET_TYPE = 'Mix'  # 'Mix' or 'Single'
RESULTS_DIR = f'..'
INPUT_FILE = f'Input_Data-{DATASET_TYPE}-train.csv'
N_SAMPLES_TO_EXPLAIN = 10  # Number of predictions to explain
N_FEATURES = 10  # Top N features to show in explanations

# Load data
print("=" * 100)
print(f"LIME ANALYSIS - {DATASET_TYPE} DATASET")
print("=" * 100)
print("\nLoading data and models...")

df = pd.read_csv(INPUT_FILE)
X = df.drop(['sample', 'PoI', 'Classification'], axis=1, errors='ignore')
y = df['Classification'].map({'Allele': 0, 'Back_Stutter': 1, 'Forward_Stutter': 2, 'Pull_Up': 3})

feature_names = X.columns.tolist()
class_names = ['Allele', 'Back_Stutter', 'Forward_Stutter', 'Pull_Up']

# Split data (same as training)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Load best model and scaler
if DATASET_TYPE == 'Mix':
    model_path = 'Mix_Results/models/stacking_ensemble.pkl'
    scaler_path = 'Mix_Results/models/stacking_ensemble_scaler.pkl'
    model_name = "Stacking Ensemble"
else:
    model_path = 'Single_Results/models/logistic_regression_original.pkl'
    scaler_path = None
    model_name = "Logistic Regression"

print(f"Loading model: {model_name}")
model = joblib.load(model_path)

if scaler_path and os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Scaler loaded and applied")
else:
    X_train_scaled = X_train.values
    X_test_scaled = X_test.values
    scaler = None
    print("No scaler needed")

# Make predictions
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)

# Ensure predictions are numeric
if isinstance(y_pred[0], str):
    y_pred = np.array([class_names.index(p) for p in y_pred])

# Identify samples to explain
correct_indices = np.where(y_pred == y_test.values)[0]
incorrect_indices = np.where(y_pred != y_test.values)[0]

n_correct = min(N_SAMPLES_TO_EXPLAIN // 2, len(correct_indices))
n_incorrect = min(N_SAMPLES_TO_EXPLAIN // 2, len(incorrect_indices))

print(f"\nTotal test samples: {len(y_test)}")
print(f"Correct predictions: {len(correct_indices)} ({len(correct_indices)/len(y_test)*100:.1f}%)")
print(f"Incorrect predictions: {len(incorrect_indices)} ({len(incorrect_indices)/len(y_test)*100:.1f}%)")
print(f"\nExplaining {n_correct} correct and {n_incorrect} incorrect predictions...")

# Select samples to explain
np.random.seed(42)
samples_to_explain = []
samples_to_explain.extend(np.random.choice(correct_indices, n_correct, replace=False))
samples_to_explain.extend(np.random.choice(incorrect_indices, n_incorrect, replace=False))

# Initialize LIME explainer
print("\nInitializing LIME explainer...")

# Create wrapper for scaled predictions
if scaler is not None:
    def predict_fn(X):
        X_scaled = scaler.transform(X)
        return model.predict_proba(X_scaled)
else:
    def predict_fn(X):
        return model.predict_proba(X)

explainer = lime_tabular.LimeTabularExplainer(
    X_train.values,
    feature_names=feature_names,
    class_names=class_names,
    mode='classification',
    discretize_continuous=True,
    random_state=42
)

# Create output directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
output_dir = os.path.join(project_root, f'{DATASET_TYPE}_Results', 'lime_results')
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}")

# Store all explanations
all_explanations = []

print("\nGenerating LIME explanations...\n")

# Explain each sample
for idx, sample_idx in enumerate(samples_to_explain):
    true_class = y_test.iloc[sample_idx]
    pred_class = y_pred[sample_idx]
    pred_proba = y_proba[sample_idx]
    is_correct = true_class == pred_class
    
    print(f"Sample {idx+1}/{len(samples_to_explain)}: "
          f"True={class_names[true_class]}, "
          f"Pred={class_names[pred_class]}, "
          f"Confidence={pred_proba[pred_class]:.3f}, "
          f"{'✓ CORRECT' if is_correct else '✗ WRONG'}")
    
    # Generate LIME explanation
    exp = explainer.explain_instance(
        X_test.iloc[sample_idx].values,
        predict_fn,
        num_features=N_FEATURES,
        top_labels=4
    )
    
    # Extract feature contributions for predicted class
    feature_weights = dict(exp.as_list(label=pred_class))
    
    # Store explanation
    explanation_data = {
        'sample_id': int(sample_idx),
        'true_class': class_names[true_class],
        'predicted_class': class_names[pred_class],
        'confidence': float(pred_proba[pred_class]),
        'is_correct': bool(is_correct),
        'feature_contributions': feature_weights,
        'all_probabilities': {class_names[i]: float(pred_proba[i]) for i in range(4)}
    }
    all_explanations.append(explanation_data)
    
    # Create visualization
    fig = plt.figure(figsize=(12, 8))
    
    # Plot 1: Feature contributions
    ax1 = plt.subplot(2, 1, 1)
    features = list(feature_weights.keys())[:N_FEATURES]
    weights = list(feature_weights.values())[:N_FEATURES]
    colors = ['#a3be8c' if w > 0 else '#bf616a' for w in weights]
    
    y_pos = np.arange(len(features))
    ax1.barh(y_pos, weights, color=colors, alpha=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(features, fontsize=9)
    ax1.set_xlabel('Feature Contribution Weight', fontsize=10)
    ax1.set_title(f'LIME Explanation: Sample {sample_idx} '
                  f'{"✓ CORRECT" if is_correct else "✗ INCORRECT"}\n'
                  f'True: {class_names[true_class]} | Predicted: {class_names[pred_class]} '
                  f'(confidence: {pred_proba[pred_class]:.1%})',
                  fontsize=11, fontweight='bold')
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax1.grid(axis='x', alpha=0.3)
    
    # Plot 2: Class probabilities
    ax2 = plt.subplot(2, 1, 2)
    class_probs = [pred_proba[i] for i in range(4)]
    colors_bar = ['#5e81ac' if i == pred_class else '#d8dee9' for i in range(4)]
    if not is_correct:
        colors_bar[true_class] = '#ebcb8b'  # Highlight true class in incorrect predictions
    
    bars = ax2.bar(class_names, class_probs, color=colors_bar, alpha=0.7)
    ax2.set_ylabel('Probability', fontsize=10)
    ax2.set_title('Class Probabilities', fontsize=11, fontweight='bold')
    ax2.set_ylim([0, 1])
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save figure
    status = 'correct' if is_correct else 'incorrect'
    filename = f'lime_explanation_sample{sample_idx}_{status}.png'
    plt.savefig(f'{output_dir}/{filename}', dpi=150, bbox_inches='tight')
    plt.close()

# Save all explanations to JSON
with open(f'{output_dir}/lime_explanations.json', 'w') as f:
    json.dump(all_explanations, f, indent=2)

print(f"\n{'='*100}")
print("SUMMARY STATISTICS")
print(f"{'='*100}")

# Aggregate feature importance across all explanations
feature_importance_agg = {}
for exp in all_explanations:
    for feature, weight in exp['feature_contributions'].items():
        feature_base = feature.split('<=')[0].split('>')[0].strip()
        if feature_base not in feature_importance_agg:
            feature_importance_agg[feature_base] = []
        feature_importance_agg[feature_base].append(abs(weight))

# Calculate mean absolute importance
mean_importance = {feat: np.mean(weights) for feat, weights in feature_importance_agg.items()}
mean_importance_sorted = sorted(mean_importance.items(), key=lambda x: x[1], reverse=True)

print(f"\nTop 15 Most Important Features (average across {len(samples_to_explain)} explanations):")
print("-" * 100)
for i, (feat, imp) in enumerate(mean_importance_sorted[:15], 1):
    print(f"{i:2d}. {feat:40s} {imp:.4f}")

# Confidence analysis
correct_confidences = [exp['confidence'] for exp in all_explanations if exp['is_correct']]
incorrect_confidences = [exp['confidence'] for exp in all_explanations if not exp['is_correct']]

if correct_confidences:
    print(f"\nCorrect Predictions - Average Confidence: {np.mean(correct_confidences):.3f} (±{np.std(correct_confidences):.3f})")
if incorrect_confidences:
    print(f"Incorrect Predictions - Average Confidence: {np.mean(incorrect_confidences):.3f} (±{np.std(incorrect_confidences):.3f})")

# Create summary visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'LIME Analysis Summary - {DATASET_TYPE} Dataset ({model_name})', 
             fontsize=14, fontweight='bold')

# 1. Feature importance distribution
ax = axes[0, 0]
top_features = [f[0] for f in mean_importance_sorted[:15]]
top_importance = [f[1] for f in mean_importance_sorted[:15]]
ax.barh(range(len(top_features)), top_importance, color='#5e81ac', alpha=0.7)
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features, fontsize=9)
ax.set_xlabel('Mean Absolute Weight', fontsize=10)
ax.set_title('Top 15 Most Important Features\n(Averaged Across All Explanations)', fontsize=11)
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

# 2. Confidence distribution
ax = axes[0, 1]
if correct_confidences:
    ax.hist(correct_confidences, bins=10, alpha=0.7, color='#a3be8c', label='Correct', edgecolor='black')
if incorrect_confidences:
    ax.hist(incorrect_confidences, bins=10, alpha=0.7, color='#bf616a', label='Incorrect', edgecolor='black')
ax.set_xlabel('Prediction Confidence', fontsize=10)
ax.set_ylabel('Count', fontsize=10)
ax.set_title('Confidence Distribution', fontsize=11)
ax.legend()
ax.grid(alpha=0.3)

# 3. Class prediction distribution
ax = axes[1, 0]
pred_classes = [exp['predicted_class'] for exp in all_explanations]
true_classes = [exp['true_class'] for exp in all_explanations]
class_counts = pd.Series(pred_classes).value_counts()
ax.bar(class_counts.index, class_counts.values, color='#5e81ac', alpha=0.7)
ax.set_xlabel('Predicted Class', fontsize=10)
ax.set_ylabel('Count', fontsize=10)
ax.set_title('Prediction Distribution in Explained Samples', fontsize=11)
ax.grid(axis='y', alpha=0.3)

# 4. Confusion for explained samples
ax = axes[1, 1]
from sklearn.metrics import confusion_matrix
true_labels = [all_explanations[i]['true_class'] for i in range(len(all_explanations))]
pred_labels = [all_explanations[i]['predicted_class'] for i in range(len(all_explanations))]
cm = confusion_matrix([class_names.index(c) for c in true_labels],
                      [class_names.index(c) for c in pred_labels],
                      labels=range(4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, 
            yticklabels=class_names, ax=ax, cbar_kws={'label': 'Count'})
ax.set_xlabel('Predicted', fontsize=10)
ax.set_ylabel('True', fontsize=10)
ax.set_title('Confusion Matrix\n(Explained Samples Only)', fontsize=11)

plt.tight_layout()
plt.savefig(f'{output_dir}/lime_summary.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\n{'='*100}")
print("LIME ANALYSIS COMPLETE")
print(f"{'='*100}")
print(f"\nResults saved to: {output_dir}/")
print(f"  - {len(samples_to_explain)} individual explanation plots")
print(f"  - lime_explanations.json (all explanation data)")
print(f"  - lime_summary.png (aggregate analysis)")
print(f"\nTo view individual explanations, check the PNG files in {output_dir}/")
shap_dir = os.path.join(project_root, f'{DATASET_TYPE}_Results', 'figures')
print(f"\nCompare with SHAP results in: {shap_dir}/shap_*.png")
