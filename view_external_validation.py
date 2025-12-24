"""
View detailed external validation results
Training on Single dataset, testing on Mix dataset
"""
import json
import pandas as pd

# Load results
with open('Single_Results/external_validation_results/external_validation_results.json', 'r') as f:
    data = json.load(f)

# Create DataFrame
df = pd.DataFrame.from_dict(data, orient='index')
df = df.sort_values('balanced_accuracy', ascending=False)

print("=" * 100)
print("EXTERNAL VALIDATION: Training on Single (200 samples) -> Testing on Mix (426 samples)")
print("=" * 100)
print("\nTop 10 Models:\n")
print(df[['balanced_accuracy', 'f1_macro', 'accuracy']].head(10).to_string())

print("\n" + "=" * 100)
print("\nKEY INSIGHTS:")
print("=" * 100)

# Best model
best_model = df.index[0]
best_acc = df.iloc[0]['balanced_accuracy']
print(f"\n1. BEST MODEL: {best_model}")
print(f"   - Balanced Accuracy: {best_acc:.2%}")
print(f"   - F1-Macro: {df.iloc[0]['f1_macro']:.2%}")
print(f"   - Accuracy: {df.iloc[0]['accuracy']:.2%}")

# Logistic Regression performance
lr_models = df[df.index.str.contains('logistic_regression')]
print(f"\n2. LOGISTIC REGRESSION MODELS:")
for idx, row in lr_models.iterrows():
    print(f"   - {idx}: {row['balanced_accuracy']:.2%} balanced accuracy")

# TabPFN performance
tabpfn_models = df[df.index.str.contains('tabpfn')]
print(f"\n3. TABPFN MODELS (Top performers):")
for idx, row in tabpfn_models.iterrows():
    print(f"   - {idx}: {row['balanced_accuracy']:.2%} balanced accuracy")

# Models that failed
failed = df[df['balanced_accuracy'] == 0.25]
print(f"\n4. MODELS WITH POOR GENERALIZATION ({len(failed)} models at 25% - random baseline):")
print(f"   - Tree-based models (RandomForest, ExtraTrees, XGBoost, LightGBM)")
print(f"   - These models severely overfit the Single dataset")

print("\n" + "=" * 100)
print("CONCLUSION:")
print("=" * 100)
print("""
- TabPFN shows best generalization (77.76% balanced accuracy)
- LogisticRegression maintains reasonable performance (70.12% with original)
- Simple models (LogReg, TabPFN) generalize better than complex tree ensembles
- Tree models (RF, ExtraTrees, XGBoost, LGBM) overfit Single dataset completely
- Performance drop from Single to Mix is expected (simpler → harder task)
- Single → Mix validation shows which models truly learned generalizable patterns
""")

print("\nFor visualizations, see:")
print("  - Single_Results/external_validation_results/external_validation_performance.png")
print("  - Single_Results/external_validation_results/train_vs_external_comparison.png")
print("  - Single_Results/reports/executive_summary_*.pdf (page 10)")
