"""
TabPFN Explainability Analysis - PDP and ICE Plots
Uses Partial Dependence Plots and Individual Conditional Expectation for model interpretation.

Features:
- Partial Dependence Plots (PDP) showing feature effects
- Individual Conditional Expectation (ICE) plots
- Interactive and static visualizations
- No internet connection required

Note: Run tabpfn_lime_explanations.py separately for LIME analysis (requires internet).
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.inspection import PartialDependenceDisplay
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATASET_TYPE = sys.argv[1] if len(sys.argv) > 1 else 'mix'
DATASET_TYPE = DATASET_TYPE.lower()

if DATASET_TYPE == 'mix':
    DATA_FILE = 'Input_Data-Mix-train.csv'
    MODELS_DIR = 'Mix_Results_PoI'
    OUTPUT_DIR = 'Mix_Results_PoI/tabpfn_explainability'
elif DATASET_TYPE == 'single':
    DATA_FILE = 'Input_Data-Single-train.csv'
    MODELS_DIR = 'Single_Results_PoI'
    OUTPUT_DIR = 'Single_Results_PoI/tabpfn_explainability'
else:
    print(f"Error: Invalid dataset type '{DATASET_TYPE}'. Use 'mix' or 'single'")
    sys.exit(1)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300


def load_data():
    """Load and prepare the dataset."""
    print(f"\n[1/6] Loading {DATASET_TYPE.upper()} dataset...")
    df = pd.read_csv(DATA_FILE)
    
    # Drop non-feature columns
    drop_cols = ['sample', 'PoI', 'Classification']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df['Classification']
    
    from sklearn.model_selection import train_test_split as _tts
    X_train, X_test, y_train, y_test = _tts(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Features: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test


def find_tabpfn_model():
    """Find TabPFN model file (current _v35 TabPFN-3.5 checkpoint only)."""
    print("\n[2/6] Finding TabPFN model...")

    model_files = [p for p in Path(MODELS_DIR).glob('models/*tabpfn*.pkl') if p.stem.endswith('_v35')]

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


def create_pdp_plots(model, X_train, X_test, feature_names):
    """Create Partial Dependence Plots for top features using manual calculation."""
    print("\n[3/4] Creating Partial Dependence Plots...")
    print("   Using batch prediction for speed (30 samples, 15 grid points)")
    
    try:
        # Use variance-based feature selection (faster than importance)
        variances = X_test.var()
        top_features = variances.nlargest(6).index.tolist()
        
        print(f"   Creating PDP for features: {', '.join(top_features)}")
        print(f"   Progress: ", end='', flush=True)
        
        # Create PDP plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        # Use 30 samples for good averaging
        X_subset = X_test.iloc[:30]
        
        for idx, feat_name in enumerate(top_features):
            try:
                feat_idx = feature_names.index(feat_name)
                
                # Get feature range
                feat_values = X_subset.iloc[:, feat_idx]
                grid = np.linspace(feat_values.min(), feat_values.max(), 15)  # 15 points for smooth curves
                
                # Calculate PDP with BATCH prediction (much faster!)
                # Create all variations at once
                all_variations = []
                for val in grid:
                    X_temp = X_subset.copy()
                    X_temp.iloc[:, feat_idx] = val
                    all_variations.append(X_temp.values)
                
                # Stack and predict ALL at once (15 grid points * 30 samples = 450 predictions)
                X_batch = np.vstack(all_variations)
                
                try:
                    # Single batch prediction - MUCH faster!
                    preds_batch = model.predict_proba(X_batch)[:, 0]
                except:
                    # Fallback to binary predictions
                    preds_batch = model.predict(X_batch).astype(float)
                
                # Calculate mean for each grid point
                pdp_values = preds_batch.reshape(len(grid), len(X_subset)).mean(axis=1)
                
                # Plot PDP
                axes[idx].plot(grid, pdp_values, linewidth=2.5, color='#1f77b4')
                axes[idx].set_xlabel(feat_name, fontsize=10)
                axes[idx].set_ylabel('Predicted Probability', fontsize=10)
                axes[idx].set_title(f'PDP: {feat_name}', fontsize=12, fontweight='bold')
                axes[idx].grid(True, alpha=0.3)
                
                print(f"{idx+1}/6 ", end='', flush=True)
            except Exception as e:
                print(f"\n   ✗ Failed PDP for {feat_name}: {str(e)[:50]}")
                axes[idx].text(0.5, 0.5, f'Failed:\n{feat_name}', 
                             ha='center', va='center', transform=axes[idx].transAxes)
                axes[idx].set_title(f'PDP: {feat_name} (Failed)', fontsize=12)
        
        print("\n   ✓ All PDPs completed")
        
        plt.suptitle('TabPFN - Partial Dependence Plots (Top 6 Features)',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/partial_dependence_plots.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved: partial_dependence_plots.png")
        
    except Exception as e:
        print(f"   ✗ PDP creation failed: {e}")
        import traceback
        traceback.print_exc()


def create_ice_plots(model, X_train, X_test, feature_names):
    """Create Individual Conditional Expectation plots using manual calculation."""
    print("\n[4/4] Creating ICE plots...")
    print("   Using batch prediction for speed (10 samples, 12 grid points)")
    
    try:
        # Use variance-based feature selection
        variances = X_test.var()
        top_features = variances.nlargest(6).index.tolist()
        
        print(f"   Creating ICE for features: {', '.join(top_features)}")
        print(f"   Progress: ", end='', flush=True)
        
        # Create ICE plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        # Use 10 instances to show individual variation
        X_subset = X_test.iloc[:10]
        
        for idx, feat_name in enumerate(top_features):
            try:
                feat_idx = feature_names.index(feat_name)
                
                # Get feature range
                feat_values = X_subset.iloc[:, feat_idx]
                grid = np.linspace(feat_values.min(), feat_values.max(), 12)  # 12 points for smooth curves
                
                # Calculate ICE with BATCH prediction (much faster!)
                # Create all variations at once for all samples
                all_variations = []
                sample_indices = []
                for sample_idx in range(len(X_subset)):
                    for val in grid:
                        X_temp = X_subset.iloc[[sample_idx]].copy()
                        X_temp.iloc[0, feat_idx] = val
                        all_variations.append(X_temp.values[0])
                        sample_indices.append(sample_idx)
                
                # Stack and predict ALL at once (12 grid points * 10 samples = 120 predictions)
                X_batch = np.array(all_variations)
                
                try:
                    # Single batch prediction - MUCH faster!
                    preds_batch = model.predict_proba(X_batch)[:, 0]
                except:
                    # Fallback to binary predictions
                    preds_batch = model.predict(X_batch).astype(float)
                
                # Reshape to [n_samples, n_grid_points]
                ice_lines = preds_batch.reshape(len(X_subset), len(grid))
                
                # Plot individual ICE lines
                for ice_line in ice_lines:
                    axes[idx].plot(grid, ice_line, color='gray', alpha=0.3, linewidth=1)
                
                # Calculate and plot average (PDP)
                pdp_values = ice_lines.mean(axis=0)
                axes[idx].plot(grid, pdp_values, linewidth=2.5, color='#1f77b4', 
                              label='Average (PDP)')
                
                axes[idx].set_xlabel(feat_name, fontsize=10)
                axes[idx].set_ylabel('Predicted Probability', fontsize=10)
                axes[idx].set_title(f'ICE + PDP: {feat_name}', fontsize=12, fontweight='bold')
                axes[idx].grid(True, alpha=0.3)
                axes[idx].legend(loc='best')
                
                print(f"{idx+1}/6 ", end='', flush=True)
            except Exception as e:
                print(f"\n   ✗ Failed ICE for {feat_name}: {str(e)[:50]}")
                axes[idx].text(0.5, 0.5, f'Failed:\n{feat_name}', 
                             ha='center', va='center', transform=axes[idx].transAxes)
                axes[idx].set_title(f'ICE: {feat_name} (Failed)', fontsize=12)
        
        print("\n   ✓ All ICE plots completed")
        
        plt.suptitle('TabPFN - ICE Plots + PDP (Top 6 Features)\nGray lines: Individual samples | Blue line: Average',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/ice_plots.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved: ice_plots.png")
        
    except Exception as e:
        print(f"   ✗ ICE creation failed: {e}")
        import traceback
        traceback.print_exc()


def create_summary_report(model_name):
    """Create text summary of explainability analysis."""
    print("\n[5/5] Creating summary report...")
    
    report_path = f"{OUTPUT_DIR}/pdp_ice_summary.txt"
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"TABPFN EXPLAINABILITY ANALYSIS - PDP & ICE - {DATASET_TYPE.upper()} DATASET\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: {DATASET_TYPE.upper()}\n\n")
        
        f.write("="*80 + "\n")
        f.write("METHODS USED\n")
        f.write("="*80 + "\n\n")
        
        f.write("1. PARTIAL DEPENDENCE PLOTS (PDP)\n")
        f.write("   - Shows average effect of a feature on predictions\n")
        f.write("   - Marginalizes over all other features\n")
        f.write("   - Global feature effect visualization\n\n")
        
        f.write("2. INDIVIDUAL CONDITIONAL EXPECTATION (ICE)\n")
        f.write("   - Shows effect of a feature for individual samples\n")
        f.write("   - Displays heterogeneity in feature effects\n")
        f.write("   - Gray lines = individual samples, Blue line = average (PDP)\n\n")
        
        f.write("="*80 + "\n")
        f.write("FEATURE SELECTION\n")
        f.write("="*80 + "\n\n")
        f.write("Features for PDP/ICE were selected by variance (top 6).\n")
        f.write("High variance features tend to be more informative for interpretation.\n\n")
        
        f.write("="*80 + "\n")
        f.write("INTERPRETATION GUIDELINES\n")
        f.write("="*80 + "\n\n")
        
        f.write("PDP:\n")
        f.write("  - Shows how predicted outcome changes with feature value\n")
        f.write("  - Flat line = feature has no effect\n")
        f.write("  - Steep slope = strong feature effect\n")
        f.write("  - Non-linear curves = non-linear relationships\n\n")
        
        f.write("ICE:\n")
        f.write("  - Gray lines diverging = heterogeneous feature effects\n")
        f.write("  - Parallel gray lines = homogeneous effects\n")
        f.write("  - Wide spread = feature interactions present\n")
    
    print(f"   ✓ Saved: explainability_summary.txt")


def main():
    """Main explainability pipeline."""
    print("="*80)
    print(f"TABPFN EXPLAINABILITY ANALYSIS - PDP & ICE - {DATASET_TYPE.upper()} DATASET")
    print("="*80)
    print("\nMethods: PDP (Partial Dependence), ICE (Individual Conditional Expectation)")
    print("Note: No internet connection required\n")
    
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    feature_names = X_train.columns.tolist()
    
    # Find and load model
    model, model_name = find_tabpfn_model()
    if model is None:
        print("\n✗ No TabPFN model found! Exiting.")
        return
    
    # Run explainability methods
    create_pdp_plots(model, X_train, X_test, feature_names)
    create_ice_plots(model, X_train, X_test, feature_names)
    create_summary_report(model_name)
    
    print("\n" + "="*80)
    print("✓ TABPFN PDP & ICE ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  * partial_dependence_plots.png")
    print("  * ice_plots.png")
    print("  * pdp_ice_summary.txt")
    print("\nFor LIME explanations, run: python scripts\\tabpfn_lime_explanations.py " + DATASET_TYPE)
    print("(Note: LIME requires internet connection)")


if __name__ == '__main__':
    main()
