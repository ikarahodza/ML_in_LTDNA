"""
TabPFN LIME Explanations
Generates LIME explanations for individual predictions.
Note: Requires internet connection (LIME may download data).

Run separately from other explainability methods due to network dependency.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
from lime.lime_tabular import LimeTabularExplainer
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATASET_TYPE = sys.argv[1] if len(sys.argv) > 1 else 'mix'
DATASET_TYPE = DATASET_TYPE.lower()

if DATASET_TYPE == 'mix':
    DATA_FILE = 'Input_Data-Mix-train.csv'
    MODELS_DIR = 'Mix_Results'
    OUTPUT_DIR = 'Mix_Results/tabpfn_explainability'
elif DATASET_TYPE == 'single':
    DATA_FILE = 'Input_Data-Single-train.csv'
    MODELS_DIR = 'Single_Results'
    OUTPUT_DIR = 'Single_Results/tabpfn_explainability'
else:
    print(f"Error: Invalid dataset type '{DATASET_TYPE}'. Use 'mix' or 'single'")
    sys.exit(1)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    """Load and prepare the dataset."""
    print(f"\n[1/3] Loading {DATASET_TYPE.upper()} dataset...")
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
    print("\n[2/3] Finding TabPFN model...")
    
    # Search in seed subdirectories
    model_files = list(Path(MODELS_DIR).glob('**/models/*tabpfn*.pkl'))
    
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


def create_lime_explanations(model, X_train, X_test, y_test, feature_names):
    """Generate LIME explanations for individual predictions."""
    print("\n[3/3] Generating LIME explanations...")
    print("   ⚠ Note: LIME may require internet connection for initialization")
    
    try:
        # Get class names
        class_names = sorted(list(set(y_test)))
        
        # Create LIME explainer
        explainer = LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=feature_names,
            class_names=class_names,
            mode='classification',
            random_state=42
        )
        
        print(f"   ✓ Explainer created (Classes: {class_names})")
        print(f"   Generating LIME for 5 samples (500 perturbations each = faster)")
        print(f"   Progress: ", end='', flush=True)
        
        # Generate explanations for first 5 samples
        for i in range(min(5, len(X_test))):
            try:
                # Get prediction (use .values to avoid feature name warning)
                sample = X_test.iloc[i].values
                pred_proba = model.predict_proba([sample])[0]
                pred_class = model.predict([sample])[0]
                actual_class = y_test.iloc[i]
                
                # Generate LIME explanation with reduced samples for speed
                exp = explainer.explain_instance(
                    sample,
                    model.predict_proba,
                    num_features=10,
                    num_samples=500,  # Reduced from default 5000 for speed
                    top_labels=1  # Only explain predicted class for speed
                )
                
                # Get feature importance for predicted class
                pred_class_idx = 0  # top_labels=1 returns only the top predicted class
                feature_importance = dict(exp.as_list(label=pred_class_idx))
                
                # Create visualization
                features = list(feature_importance.keys())[:10]
                values = list(feature_importance.values())[:10]
                
                # Create Plotly figure
                colors = ['green' if v > 0 else 'red' for v in values]
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=features,
                    x=values,
                    orientation='h',
                    marker_color=colors,
                    hovertemplate='<b>%{y}</b><br>Contribution: %{x:.4f}<extra></extra>'
                ))
                
                fig.update_layout(
                    title=f'<b>LIME Explanation - Sample {i}</b><br>' +
                          f'Predicted: {pred_class} | Actual: {actual_class}',
                    xaxis_title='Feature Contribution',
                    yaxis_title='Feature',
                    height=500,
                    width=800
                )
                
                html_path = f"{OUTPUT_DIR}/lime_sample_{i}.html"
                fig.write_html(html_path)
                print(f"{i+1}/5 ", end='', flush=True)
                
            except Exception as e:
                print(f"\n   ✗ Failed sample {i}: {e}")
        
        print("\n   ✓ LIME explanations complete")
        
    except Exception as e:
        print(f"   ✗ LIME failed: {e}")


def main():
    """Main LIME pipeline."""
    print("="*80)
    print(f"TABPFN LIME EXPLANATIONS - {DATASET_TYPE.upper()} DATASET")
    print("="*80)
    print("\n⚠ Note: This script requires internet connection")
    print("   Run this separately from PDP/ICE analysis\n")
    
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    feature_names = X_train.columns.tolist()
    
    # Find and load model
    model, model_name = find_tabpfn_model()
    if model is None:
        print("\n✗ No TabPFN model found! Exiting.")
        return
    
    # Generate LIME explanations
    create_lime_explanations(model, X_train, X_test, y_test, feature_names)
    
    print("\n" + "="*80)
    print("✓ TABPFN LIME ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  * lime_sample_0.html to lime_sample_4.html")
    print("\nNext: Run tabpfn_explainability.py for PDP and ICE plots (no internet required)")


if __name__ == '__main__':
    main()
