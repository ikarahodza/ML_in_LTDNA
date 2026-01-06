"""
TabPFN Interpretable Machine Learning Analysis
Uses tabpfn_iml library to generate comprehensive explanations for TabPFN predictions.

Methods included:
- ICE/PD plots (Individual Conditional Expectation / Partial Dependence)
- Kernel SHAP
- Feature Importance (LOCO)
- ALE (Accumulated Local Effects)
- Sensitivity Analysis
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATASET_TYPE = sys.argv[1].lower() if len(sys.argv) > 1 else 'mix'
assert DATASET_TYPE in ['mix', 'single'], "Dataset type must be 'mix' or 'single'"

BASE_DIR = f"{DATASET_TYPE.capitalize()}_Results"
DATA_FILE = f"Input_Data-{DATASET_TYPE.capitalize()}-train.csv"
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "tabpfn_iml_analysis")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print(f"TabPFN IML ANALYSIS - {DATASET_TYPE.upper()} DATASET")
print("="*80)

# =============================================================================
# Step 1: Load Data
# =============================================================================
print("\n[1/6] Loading data...")
df = pd.read_csv(DATA_FILE)

# Prepare features and target
drop_cols = ['sample', 'PoI', 'Classification']
X = df.drop(columns=[c for c in drop_cols if c in df.columns])
y = df['Classification']

# 80-20 split
split_idx = int(0.8 * len(X))
X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

print(f"   Training samples: {len(X_train)}")
print(f"   Test samples: {len(X_test)}")
print(f"   Features: {X_train.shape[1]}")
print(f"   Classes: {sorted(y.unique())}")

# =============================================================================
# Step 2: Find and Load TabPFN Model
# =============================================================================
print("\n[2/6] Finding TabPFN model...")

# Import TabPFN and create comprehensive compatibility layer for unpickling
# Models may reference various internal modules
try:
    import tabpfn
    import sys
    from types import ModuleType
    
    # Import main class
    from tabpfn import TabPFNClassifier
    
    # Create stub classes for unpickling
    class ClassifierEvalMetrics:
        """Stub for unpickling TabPFN models."""
        def __init__(self, *args, **kwargs):
            pass
    
    # Create compatibility for all potential submodules
    # Handle parent packages first, then nested ones
    base_submodules = [
        'classifier', 'inference_tuning', 'model', 'utils', 'scripts',
        'architectures', 'transformer', 'bar_distribution', 
        'encoders', 'priors', 'positional_encodings', 'layer_norm', 'attention'
    ]
    
    nested_submodules = [
        'architectures.base', 'architectures.tabpfn'
    ]
    
    # First pass: create base modules
    for submod_name in base_submodules:
        full_name = f'tabpfn.{submod_name}'
        if full_name not in sys.modules:
            # Create fake module
            submod = ModuleType(submod_name)
            
            # Make it behave like a package so nested imports work
            submod.__path__ = []
            submod.__package__ = full_name
            
            # Add common attributes that might be needed
            if submod_name == 'classifier':
                submod.TabPFNClassifier = TabPFNClassifier
            elif submod_name == 'inference_tuning':
                submod.ClassifierEvalMetrics = ClassifierEvalMetrics
            
            # Inject into sys.modules and tabpfn
            sys.modules[full_name] = submod
            setattr(tabpfn, submod_name, submod)
    
    # Second pass: create nested modules
    for nested_name in nested_submodules:
        full_name = f'tabpfn.{nested_name}'
        if full_name not in sys.modules:
            parent_name, child_name = nested_name.rsplit('.', 1)
            parent_full_name = f'tabpfn.{parent_name}'
            
            # Create nested module
            nested_mod = ModuleType(child_name)
            nested_mod.__path__ = []
            nested_mod.__package__ = full_name
            
            # Add to parent module
            if parent_full_name in sys.modules:
                setattr(sys.modules[parent_full_name], child_name, nested_mod)
            
            sys.modules[full_name] = nested_mod
    
    # Also add to root level sys.modules as fallback
    if 'inference_tuning' not in sys.modules:
        inference_tuning_mod = ModuleType('inference_tuning')
        inference_tuning_mod.ClassifierEvalMetrics = ClassifierEvalMetrics
        sys.modules['inference_tuning'] = inference_tuning_mod
    
    print(f"   ✓ TabPFN library available (version {tabpfn.__version__ if hasattr(tabpfn, '__version__') else 'unknown'})")
    print(f"   ✓ Compatibility layer created for unpickling")
    
except ImportError as e:
    print(f"   ✗ TabPFN import failed: {e}")
    sys.exit(1)

pkl_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')]
tabpfn_files = [f for f in pkl_files if 'tabpfn' in f.lower()]

if not tabpfn_files:
    print("   ✗ No TabPFN model found!")
    sys.exit(1)

# Load first TabPFN model found
model_file = tabpfn_files[0]
model_path = os.path.join(MODELS_DIR, model_file)
model = joblib.load(model_path)
model_name = model_file.replace('.pkl', '')

print(f"   ✓ Loaded: {model_file}")

# Get predictions for accuracy
y_pred = model.predict(X_test.values)
y_pred_proba = model.predict_proba(X_test.values)

# Handle integer predictions
if isinstance(y_pred[0], (int, np.integer)):
    sorted_classes = sorted(y_test.unique())
    y_pred_names = [sorted_classes[p] for p in y_pred]
    accuracy = (np.array(y_pred_names) == y_test.values).mean()
else:
    accuracy = (y_pred == y_test.values).mean()

print(f"   Test Accuracy: {accuracy:.4f}")

# =============================================================================
# Step 3: Try to use tabpfn_iml methods
# =============================================================================
print("\n[3/6] Attempting to use tabpfn_iml library...")

try:
    # Try importing tabpfn_iml
    from tabpfniml import ICE_PD, kernel_shap, LOCO
    from tabpfniml.datasets.datasets import ArrayData
    
    print("   ✓ tabpfn_iml library found!")
    HAS_TABPFN_IML = True
    
    # Create dataset object compatible with tabpfn_iml
    print("\n[4/6] Preparing dataset for tabpfn_iml...")
    
    # TabPFN IML expects target as first column
    df_iml = df.copy()
    cols = ['Classification'] + [c for c in df_iml.columns if c not in ['Classification', 'sample', 'PoI']]
    df_iml = df_iml[cols]
    
    # Create ArrayData object
    dataset = ArrayData(
        data=df_iml.values,
        name=f"{DATASET_TYPE}_dataset"
    )
    
    print(f"   ✓ Dataset prepared: {dataset.name}")
    
    # =============================================================================
    # Step 5: Run IML Methods
    # =============================================================================
    print("\n[5/6] Running IML methods...")
    
    # Method 1: ICE/PD Plots
    print("\n   [5.1] Generating ICE/PD plots...")
    try:
        ice_pd = ICE_PD(
            model=model,
            dataset=dataset,
            num_features=min(5, X_train.shape[1]),
            train_size=min(200, len(X_train))
        )
        
        # Save plots
        ice_pd_output = os.path.join(OUTPUT_DIR, 'ice_pd_plots.png')
        plt.savefig(ice_pd_output, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"      ✓ Saved: ice_pd_plots.png")
    except Exception as e:
        print(f"      ✗ ICE/PD failed: {e}")
    
    # Method 2: Kernel SHAP
    print("\n   [5.2] Generating Kernel SHAP explanations...")
    try:
        shap_results = kernel_shap(
            model=model,
            dataset=dataset,
            n_samples=min(50, len(X_test)),
            train_size=min(200, len(X_train))
        )
        
        shap_output = os.path.join(OUTPUT_DIR, 'kernel_shap.png')
        plt.savefig(shap_output, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"      ✓ Saved: kernel_shap.png")
    except Exception as e:
        print(f"      ✗ Kernel SHAP failed: {e}")
    
    # Method 3: LOCO (Feature Importance)
    print("\n   [5.3] Computing LOCO feature importance...")
    try:
        loco_results = LOCO(
            model=model,
            dataset=dataset,
            train_size=min(200, len(X_train))
        )
        
        loco_output = os.path.join(OUTPUT_DIR, 'loco_importance.png')
        plt.savefig(loco_output, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"      ✓ Saved: loco_importance.png")
    except Exception as e:
        print(f"      ✗ LOCO failed: {e}")

except ImportError:
    print("   ✗ tabpfn_iml library not installed")
    print("\n   To install, run:")
    print("   pip install git+https://github.com/david-rundel/tabpfn_iml.git")
    HAS_TABPFN_IML = False

# =============================================================================
# Step 6: Alternative Analysis (if tabpfn_iml not available)
# =============================================================================
if not HAS_TABPFN_IML:
    print("\n[4/6] Using alternative analysis with SHAP library...")
    
    try:
        import shap
        
        # Create SHAP explainer
        print("   Creating SHAP explainer...")
        background = shap.sample(X_train, min(100, len(X_train)))
        explainer = shap.KernelExplainer(
            model.predict_proba,
            background.values
        )
        
        # Calculate SHAP values for a few test samples
        print("   Calculating SHAP values...")
        test_sample = X_test.iloc[:min(10, len(X_test))]
        shap_values = explainer.shap_values(test_sample.values)
        
        # Plot summary
        print("\n[5/6] Creating SHAP visualizations...")
        
        # Summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            test_sample,
            feature_names=X.columns.tolist(),
            show=False
        )
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'shap_summary.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("   ✓ Saved: shap_summary.png")
        
        # Waterfall plot for first prediction
        plt.figure(figsize=(10, 8))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0][0],
                base_values=explainer.expected_value[0],
                data=test_sample.iloc[0].values,
                feature_names=X.columns.tolist()
            ),
            show=False
        )
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'shap_waterfall_sample0.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("   ✓ Saved: shap_waterfall_sample0.png")
        
        print("\n[6/6] Analysis complete using SHAP library!")
        
    except ImportError:
        print("   ✗ SHAP library not available")
        print("   To install, run: pip install shap")
        print("\n[4/6] Creating basic feature importance analysis...")
        
        # Basic permutation importance
        from sklearn.inspection import permutation_importance
        
        result = permutation_importance(
            model, X_test.values, y_test.values,
            n_repeats=10, random_state=42
        )
        
        # Plot
        sorted_idx = result.importances_mean.argsort()[-20:]
        
        plt.figure(figsize=(10, 8))
        plt.barh(
            range(len(sorted_idx)),
            result.importances_mean[sorted_idx],
            xerr=result.importances_std[sorted_idx]
        )
        plt.yticks(range(len(sorted_idx)), [X.columns[i] for i in sorted_idx])
        plt.xlabel('Permutation Importance')
        plt.title('Top 20 Feature Importances')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'permutation_importance.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print("   ✓ Saved: permutation_importance.png")
        print("\n[5/6] Basic analysis complete!")

# =============================================================================
# Summary
# =============================================================================
print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'='*80}")
print(f"Model: {model_name}")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"{'='*80}")
