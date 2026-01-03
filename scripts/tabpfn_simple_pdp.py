"""
TabPFN Simple Feature Effect Analysis
Manual implementation optimized for TabPFN's slow inference.
Uses grid sampling instead of full PDP to drastically reduce predictions needed.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATASET_TYPE = sys.argv[1] if len(sys.argv) > 1 else 'mix'
DATASET_TYPE = DATASET_TYPE.lower()

if DATASET_TYPE == 'mix':
    DATA_FILE = 'Input_Data-Mix-train.csv'
    MODELS_DIR = 'Mix_Results/models'
    OUTPUT_DIR = 'Mix_Results/tabpfn_explainability'
elif DATASET_TYPE == 'single':
    DATA_FILE = 'Input_Data-Single-train.csv'
    MODELS_DIR = 'Single_Results/models'
    OUTPUT_DIR = 'Single_Results/tabpfn_explainability'
else:
    print(f"Error: Invalid dataset type '{DATASET_TYPE}'. Use 'mix' or 'single'")
    sys.exit(1)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300


def load_data():
    """Load and prepare the dataset."""
    print(f"\n[1/4] Loading {DATASET_TYPE.upper()} dataset...")
    df = pd.read_csv(DATA_FILE)
    
    # Drop non-feature columns
    drop_cols = ['sample', 'PoI', 'Classification']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df['Classification']
    
    # 80-20 split (consistent with training)
    split_idx = int(0.8 * len(X))
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Features: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test


def find_tabpfn_model():
    """Find TabPFN model file."""
    print("\n[2/4] Finding TabPFN model...")
    
    model_files = list(Path(MODELS_DIR).glob('*tabpfn*.pkl'))
    
    if not model_files:
        print("   ✗ No TabPFN model found!")
        return None, None
    
    # Prioritize original, then classweights
    priority_order = ['original', 'classweights', 'randomoversampler']
    
    for priority in priority_order:
        for mf in model_files:
            if priority in mf.stem.lower():
                print(f"   ✓ Found: {mf.name}")
                model = joblib.load(mf)
                return model, mf.stem
    
    # Use first found
    print(f"   ✓ Found: {model_files[0].name}")
    model = joblib.load(model_files[0])
    return model, model_files[0].stem


def create_simple_pdp(model, X_test, feature_names):
    """Create simplified PDP using grid sampling (much faster for TabPFN)."""
    print("\n[3/4] Creating Simple Feature Effect Plots...")
    print("   Using optimized approach for TabPFN (grid sampling)")
    
    # Select top 6 features by variance
    variances = X_test.var()
    top_features = variances.nlargest(6).index.tolist()
    
    print(f"   Top 6 features: {', '.join(top_features)}")
    print(f"   Using 1 sample with 10 grid points per feature")
    
    # Use one representative sample
    baseline_sample = X_test.iloc[0].values.copy()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    results = {}
    
    for idx, feat_name in enumerate(top_features):
        print(f"   Processing {idx+1}/6: {feat_name}")
        
        try:
            feat_idx = feature_names.index(feat_name)
            
            # Create grid of values for this feature
            feature_min = X_test[feat_name].min()
            feature_max = X_test[feat_name].max()
            grid_values = np.linspace(feature_min, feature_max, 10)
            
            # Store predictions
            predictions = []
            
            # Vary this feature while keeping others constant
            for val in grid_values:
                sample = baseline_sample.copy()
                sample[feat_idx] = val
                pred_proba = model.predict_proba([sample])[0]
                predictions.append(pred_proba[0])  # Class 0 probability
            
            # Plot
            axes[idx].plot(grid_values, predictions, 'b-', linewidth=2, marker='o')
            axes[idx].set_xlabel(feat_name, fontsize=10)
            axes[idx].set_ylabel('Predicted Probability (Class 0)', fontsize=10)
            axes[idx].set_title(f'{feat_name}', fontsize=12, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)
            
            results[feat_name] = {
                'values': grid_values.tolist(),
                'predictions': predictions
            }
            
        except Exception as e:
            print(f"   ✗ Failed for {feat_name}: {e}")
            axes[idx].text(0.5, 0.5, f'Failed:\n{feat_name}', 
                         ha='center', va='center', transform=axes[idx].transAxes)
    
    plt.suptitle('TabPFN - Feature Effect Analysis (Grid Sampling Method)\n' +
                 'Shows how predicted probability changes with feature value',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/simple_feature_effects.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved: simple_feature_effects.png")
    
    return results


def save_summary(results, model_name):
    """Save summary report."""
    print("\n[4/4] Saving summary report...")
    
    summary_path = f"{OUTPUT_DIR}/simple_pdp_summary.txt"
    
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"TABPFN SIMPLE FEATURE EFFECT ANALYSIS - {DATASET_TYPE.upper()} DATASET\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: {DATASET_TYPE.upper()}\n\n")
        
        f.write("="*80 + "\n")
        f.write("METHOD\n")
        f.write("="*80 + "\n\n")
        
        f.write("Grid Sampling Feature Effect Analysis:\n")
        f.write("  - Uses 1 baseline sample with all other features held constant\n")
        f.write("  - Varies each feature across 10 equally-spaced grid points\n")
        f.write("  - Shows predicted probability for class 0\n")
        f.write("  - Optimized for TabPFN (only 60 predictions total vs 1000s for standard PDP)\n\n")
        
        f.write("="*80 + "\n")
        f.write("INTERPRETATION\n")
        f.write("="*80 + "\n\n")
        
        f.write("For each feature plot:\n")
        f.write("  - X-axis: Feature value range\n")
        f.write("  - Y-axis: Predicted probability for class 0\n")
        f.write("  - Upward slope: Higher feature values increase class 0 probability\n")
        f.write("  - Downward slope: Higher feature values decrease class 0 probability\n")
        f.write("  - Flat line: Feature has minimal effect on predictions\n")
        f.write("  - Non-linear curves: Non-linear relationship between feature and prediction\n\n")
        
        f.write("="*80 + "\n")
        f.write("FEATURE EFFECTS SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        for feat_name, data in results.items():
            preds = np.array(data['predictions'])
            trend = "increasing" if preds[-1] > preds[0] else "decreasing"
            change = abs(preds[-1] - preds[0])
            f.write(f"{feat_name}:\n")
            f.write(f"  Trend: {trend}\n")
            f.write(f"  Total change: {change:.4f}\n")
            f.write(f"  Range: [{preds.min():.4f}, {preds.max():.4f}]\n\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("NOTES\n")
        f.write("="*80 + "\n\n")
        f.write("- This is a simplified approach optimized for TabPFN's slow inference\n")
        f.write("- Results show marginal effects when other features are at baseline values\n")
        f.write("- For full PDP/ICE analysis, consider using faster models (RF, LightGBM, etc.)\n")
        f.write("- For individual sample explanations, use tabpfn_lime_explanations.py\n")
    
    print(f"   ✓ Saved: simple_pdp_summary.txt")


def main():
    """Main pipeline."""
    print("="*80)
    print(f"TABPFN SIMPLE FEATURE EFFECT ANALYSIS - {DATASET_TYPE.upper()} DATASET")
    print("="*80)
    print("\nOptimized approach for TabPFN (fast execution)")
    print("Uses grid sampling instead of full PDP\n")
    
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    feature_names = X_train.columns.tolist()
    
    # Find and load model
    model, model_name = find_tabpfn_model()
    if model is None:
        print("\n✗ No TabPFN model found! Exiting.")
        return
    
    # Create simple PDP plots
    results = create_simple_pdp(model, X_test, feature_names)
    
    # Save summary
    save_summary(results, model_name)
    
    print("\n" + "="*80)
    print("✓ TABPFN FEATURE EFFECT ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  * simple_feature_effects.png")
    print("  * simple_pdp_summary.txt")
    print("\nFor LIME explanations: python scripts\\tabpfn_lime_explanations.py " + DATASET_TYPE)


if __name__ == '__main__':
    main()
