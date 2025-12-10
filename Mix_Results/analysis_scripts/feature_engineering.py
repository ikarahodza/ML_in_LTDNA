"""
Feature Engineering Based on SHAP Analysis
Creates new features based on SHAP insights:
- Interaction terms between top features
- Polynomial features (squared, cubic)
- Ratio features
- Aggregation features
- Context-based features
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("FEATURE ENGINEERING PIPELINE")
print("=" * 80)

# Load original training data
print("\n1. Loading original training data...")
data = pd.read_csv('Input_Data-Mix-train.csv')
print(f"   Original data shape: {data.shape}")

# Separate features and target
X = data.drop(['Classification'], axis=1)
y = data['Classification']

# Drop non-numeric columns (like 'sample' identifier)
non_numeric_cols = X.select_dtypes(include=['object']).columns.tolist()
if non_numeric_cols:
    print(f"   Dropping non-numeric columns: {non_numeric_cols}")
    X = X.drop(columns=non_numeric_cols)

# Get feature column names
feature_cols = X.columns.tolist()
print(f"   Original features: {len(feature_cols)}")

# Store original feature names by category
upstream_features = [col for col in feature_cols if 'UpStream' in col]
downstream_features = [col for col in feature_cols if 'DownStream' in col]
c_features = [col for col in feature_cols if col.startswith('C') and '_' in col]
poi_feature = 'PoI' if 'PoI' in feature_cols else None

print(f"   - UpStream features: {len(upstream_features)}")
print(f"   - DownStream features: {len(downstream_features)}")
print(f"   - C features: {len(c_features)}")
print(f"   - PoI feature: {poi_feature is not None}")

# Create a copy for feature engineering
X_engineered = X.copy()

print("\n2. Creating new features...")
print("=" * 80)

# A. INTERACTION TERMS
print("\nA. Creating interaction terms (top feature pairs)...")
interaction_pairs = [
    ('UpStream_60', 'C2_UpStream_10'),
    ('UpStream_60', 'UpStream_67'),
    ('C2_UpStream_10', 'UpStream_67'),
    ('UpStream_52', 'DownStream_42'),
    ('C4_UpStream_1', 'UpStream_52'),
    ('UpStream_65', 'DownStream_40'),
    ('UpStream_98', 'DownStream_5'),
    ('UpStream_97', 'DownStream_4')
]

interaction_count = 0
for feat1, feat2 in interaction_pairs:
    if feat1 in X.columns and feat2 in X.columns:
        new_feature_name = f'interact_{feat1}_x_{feat2}'
        X_engineered[new_feature_name] = X[feat1] * X[feat2]
        interaction_count += 1

print(f"   Created {interaction_count} interaction features")

# B. POLYNOMIAL FEATURES
print("\nB. Creating polynomial features (squared and cubic)...")
top_features_for_poly = [
    'UpStream_60', 'C2_UpStream_10', 'UpStream_67', 
    'UpStream_52', 'DownStream_42', 'C4_UpStream_1',
    'UpStream_98', 'UpStream_65', 'DownStream_40'
]

poly_count = 0
for feat in top_features_for_poly:
    if feat in X.columns:
        # Squared
        X_engineered[f'{feat}_squared'] = X[feat] ** 2
        # Cubic
        X_engineered[f'{feat}_cubed'] = X[feat] ** 3
        poly_count += 2

print(f"   Created {poly_count} polynomial features")

# C. RATIO FEATURES
print("\nC. Creating ratio features...")
ratio_count = 0

# Ratios between important features
ratio_pairs = [
    ('UpStream_60', 'DownStream_42'),
    ('UpStream_60', 'C2_UpStream_10'),
    ('UpStream_67', 'DownStream_40'),
    ('C4_UpStream_1', 'DownStream_5')
]

for feat1, feat2 in ratio_pairs:
    if feat1 in X.columns and feat2 in X.columns:
        # Add small epsilon to avoid division by zero
        X_engineered[f'ratio_{feat1}_over_{feat2}'] = X[feat1] / (X[feat2] + 1e-8)
        ratio_count += 1

# Ratio to mean of top features
if all(f in X.columns for f in top_features_for_poly[:5]):
    top_5_mean = X[top_features_for_poly[:5]].mean(axis=1)
    for feat in top_features_for_poly[:3]:
        X_engineered[f'ratio_{feat}_over_top5mean'] = X[feat] / (top_5_mean + 1e-8)
        ratio_count += 1

print(f"   Created {ratio_count} ratio features")

# D. AGGREGATION FEATURES
print("\nD. Creating aggregation features...")
agg_count = 0

# UpStream aggregations
if upstream_features:
    X_engineered['upstream_mean'] = X[upstream_features].mean(axis=1)
    X_engineered['upstream_std'] = X[upstream_features].std(axis=1)
    X_engineered['upstream_max'] = X[upstream_features].max(axis=1)
    X_engineered['upstream_min'] = X[upstream_features].min(axis=1)
    X_engineered['upstream_range'] = X_engineered['upstream_max'] - X_engineered['upstream_min']
    X_engineered['upstream_median'] = X[upstream_features].median(axis=1)
    agg_count += 6

# DownStream aggregations
if downstream_features:
    X_engineered['downstream_mean'] = X[downstream_features].mean(axis=1)
    X_engineered['downstream_std'] = X[downstream_features].std(axis=1)
    X_engineered['downstream_max'] = X[downstream_features].max(axis=1)
    X_engineered['downstream_min'] = X[downstream_features].min(axis=1)
    X_engineered['downstream_range'] = X_engineered['downstream_max'] - X_engineered['downstream_min']
    X_engineered['downstream_median'] = X[downstream_features].median(axis=1)
    agg_count += 6

print(f"   Created {agg_count} aggregation features")

# E. CONTEXT-BASED FEATURES
print("\nE. Creating context-based features...")
context_count = 0

if upstream_features and downstream_features:
    # Sum of upstream + downstream
    X_engineered['upstream_downstream_sum'] = X[upstream_features].sum(axis=1) + X[downstream_features].sum(axis=1)
    
    # Difference between upstream and downstream means
    X_engineered['upstream_downstream_diff'] = X_engineered['upstream_mean'] - X_engineered['downstream_mean']
    
    # Ratio of upstream to downstream
    X_engineered['upstream_downstream_ratio'] = X_engineered['upstream_mean'] / (X_engineered['downstream_mean'] + 1e-8)
    
    context_count += 3

# PoI-based features
if poi_feature and poi_feature in X.columns:
    # Ratio of PoI to upstream mean
    X_engineered['poi_to_upstream_ratio'] = X[poi_feature] / (X_engineered['upstream_mean'] + 1e-8)
    
    # Ratio of PoI to downstream mean
    X_engineered['poi_to_downstream_ratio'] = X[poi_feature] / (X_engineered['downstream_mean'] + 1e-8)
    
    # PoI relative to local average
    local_avg = (X_engineered['upstream_mean'] + X_engineered['downstream_mean']) / 2
    X_engineered['poi_relative_to_local'] = X[poi_feature] / (local_avg + 1e-8)
    
    context_count += 3

print(f"   Created {context_count} context-based features")

# F. ADDITIONAL STATISTICAL FEATURES
print("\nF. Creating additional statistical features...")
stat_count = 0

# Top features variability
top_features_subset = [f for f in top_features_for_poly[:7] if f in X.columns]
if top_features_subset:
    X_engineered['top_features_mean'] = X[top_features_subset].mean(axis=1)
    X_engineered['top_features_std'] = X[top_features_subset].std(axis=1)
    X_engineered['top_features_cv'] = X_engineered['top_features_std'] / (X_engineered['top_features_mean'] + 1e-8)
    stat_count += 3

# Consistent features (from SHAP analysis)
consistent_features = [
    'UpStream_59', 'UpStream_64', 'DownStream_3', 'DownStream_6',
    'UpStream_61', 'DownStream_2', 'UpStream_100', 'DownStream_37'
]
consistent_features_present = [f for f in consistent_features if f in X.columns]
if consistent_features_present:
    X_engineered['consistent_features_mean'] = X[consistent_features_present].mean(axis=1)
    X_engineered['consistent_features_std'] = X[consistent_features_present].std(axis=1)
    stat_count += 2

print(f"   Created {stat_count} statistical features")

# Summary
print("\n" + "=" * 80)
print("FEATURE ENGINEERING SUMMARY")
print("=" * 80)
total_new_features = len(X_engineered.columns) - len(X.columns)
print(f"Original features: {len(X.columns)}")
print(f"New features created: {total_new_features}")
print(f"Total features: {len(X_engineered.columns)}")
print(f"\nBreakdown:")
print(f"  - Interaction terms: {interaction_count}")
print(f"  - Polynomial features: {poly_count}")
print(f"  - Ratio features: {ratio_count}")
print(f"  - Aggregation features: {agg_count}")
print(f"  - Context-based features: {context_count}")
print(f"  - Statistical features: {stat_count}")

# Save engineered dataset
print("\n3. Saving engineered dataset...")
engineered_data = pd.concat([X_engineered, y], axis=1)
output_path = 'Input_Data-Mix-train-engineered.csv'
engineered_data.to_csv(output_path, index=False)
print(f"   Saved to: {output_path}")

# Save feature names mapping
feature_mapping = {
    'original_features': feature_cols,
    'new_features': [col for col in X_engineered.columns if col not in feature_cols],
    'total_features': X_engineered.columns.tolist(),
    'feature_counts': {
        'original': len(feature_cols),
        'interaction': interaction_count,
        'polynomial': poly_count,
        'ratio': ratio_count,
        'aggregation': agg_count,
        'context': context_count,
        'statistical': stat_count,
        'total_new': total_new_features,
        'total': len(X_engineered.columns)
    }
}

with open('feature_engineering_mapping.json', 'w') as f:
    json.dump(feature_mapping, f, indent=2)
print(f"   Feature mapping saved to: feature_engineering_mapping.json")

# 4. BASELINE COMPARISON
print("\n" + "=" * 80)
print("4. BASELINE MODEL COMPARISON")
print("=" * 80)

# Split data
X_train_orig, X_test_orig, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_eng, X_test_eng, _, _ = train_test_split(
    X_engineered, y, test_size=0.2, random_state=42, stratify=y
)

# Train baseline model (original features)
print("\nTraining baseline model (original features)...")
rf_baseline = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_baseline.fit(X_train_orig, y_train)

y_pred_baseline = rf_baseline.predict(X_test_orig)
baseline_balanced_acc = balanced_accuracy_score(y_test, y_pred_baseline)
baseline_f1 = f1_score(y_test, y_pred_baseline, average='macro')

print(f"  Balanced Accuracy: {baseline_balanced_acc:.4f}")
print(f"  F1-Macro Score: {baseline_f1:.4f}")

# Train model with engineered features
print("\nTraining model with engineered features...")
rf_engineered = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_engineered.fit(X_train_eng, y_train)

y_pred_engineered = rf_engineered.predict(X_test_eng)
engineered_balanced_acc = balanced_accuracy_score(y_test, y_pred_engineered)
engineered_f1 = f1_score(y_test, y_pred_engineered, average='macro')

print(f"  Balanced Accuracy: {engineered_balanced_acc:.4f}")
print(f"  F1-Macro Score: {engineered_f1:.4f}")

# Calculate improvement
print("\n" + "=" * 80)
print("PERFORMANCE IMPROVEMENT")
print("=" * 80)
balanced_acc_improvement = (engineered_balanced_acc - baseline_balanced_acc) * 100
f1_improvement = (engineered_f1 - baseline_f1) * 100

print(f"\nBalanced Accuracy:")
print(f"  Baseline: {baseline_balanced_acc:.4f}")
print(f"  Engineered: {engineered_balanced_acc:.4f}")
print(f"  Improvement: {balanced_acc_improvement:+.2f}%")

print(f"\nF1-Macro Score:")
print(f"  Baseline: {baseline_f1:.4f}")
print(f"  Engineered: {engineered_f1:.4f}")
print(f"  Improvement: {f1_improvement:+.2f}%")

# 5. CROSS-VALIDATION COMPARISON
print("\n" + "=" * 80)
print("5. CROSS-VALIDATION COMPARISON")
print("=" * 80)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\nRunning 5-fold CV on original features...")
cv_scores_baseline = cross_val_score(
    RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    X, y, cv=cv, scoring='balanced_accuracy', n_jobs=-1
)
print(f"  Mean CV Balanced Accuracy: {cv_scores_baseline.mean():.4f} (+/- {cv_scores_baseline.std():.4f})")

print("\nRunning 5-fold CV on engineered features...")
cv_scores_engineered = cross_val_score(
    RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    X_engineered, y, cv=cv, scoring='balanced_accuracy', n_jobs=-1
)
print(f"  Mean CV Balanced Accuracy: {cv_scores_engineered.mean():.4f} (+/- {cv_scores_engineered.std():.4f})")

cv_improvement = (cv_scores_engineered.mean() - cv_scores_baseline.mean()) * 100
print(f"\nCV Improvement: {cv_improvement:+.2f}%")

# 6. FEATURE IMPORTANCE ANALYSIS
print("\n" + "=" * 80)
print("6. MOST IMPORTANT NEW FEATURES")
print("=" * 80)

# Get feature importance from engineered model
feature_importance = pd.DataFrame({
    'feature': X_engineered.columns,
    'importance': rf_engineered.feature_importances_
}).sort_values('importance', ascending=False)

# Filter for new features only
new_features_importance = feature_importance[
    ~feature_importance['feature'].isin(feature_cols)
].head(20)

print("\nTop 20 new engineered features by importance:")
for idx, row in new_features_importance.iterrows():
    print(f"  {row['feature']}: {row['importance']:.6f}")

# Save feature importance
feature_importance.to_csv('feature_importance_engineered.csv', index=False)
print(f"\nFull feature importance saved to: feature_importance_engineered.csv")

# Save results summary
results_summary = {
    'baseline': {
        'balanced_accuracy': float(baseline_balanced_acc),
        'f1_macro': float(baseline_f1),
        'cv_balanced_accuracy_mean': float(cv_scores_baseline.mean()),
        'cv_balanced_accuracy_std': float(cv_scores_baseline.std())
    },
    'engineered': {
        'balanced_accuracy': float(engineered_balanced_acc),
        'f1_macro': float(engineered_f1),
        'cv_balanced_accuracy_mean': float(cv_scores_engineered.mean()),
        'cv_balanced_accuracy_std': float(cv_scores_engineered.std())
    },
    'improvement': {
        'balanced_accuracy_pct': float(balanced_acc_improvement),
        'f1_macro_pct': float(f1_improvement),
        'cv_balanced_accuracy_pct': float(cv_improvement)
    },
    'top_new_features': new_features_importance.to_dict('records')
}

with open('feature_engineering_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print("\n" + "=" * 80)
print("FEATURE ENGINEERING COMPLETE!")
print("=" * 80)
print("\nGenerated files:")
print("  * Input_Data-Mix-train-engineered.csv")
print("  * feature_engineering_mapping.json")
print("  * feature_importance_engineered.csv")
print("  * feature_engineering_results.json")

if balanced_acc_improvement > 0:
    print(f"\n✓ Feature engineering improved performance by {balanced_acc_improvement:.2f}%!")
    print("  Recommendation: Use engineered features for final models")
else:
    print(f"\n× Feature engineering decreased performance by {abs(balanced_acc_improvement):.2f}%")
    print("  Recommendation: Review feature selection or use subset of new features")
