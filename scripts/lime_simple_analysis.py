"""
Simple LIME Explainability Analysis
Generates LIME explanations for Mix or Single dataset models.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from lime import lime_tabular
import plotly.graph_objects as go

# Configuration
DATASET_TYPE = sys.argv[1].lower() if len(sys.argv) > 1 else 'mix'
assert DATASET_TYPE in ['mix', 'single'], "Dataset type must be 'mix' or 'single'"

BASE_DIR = f"{DATASET_TYPE.capitalize()}_Results"
DATA_FILE = f"Input_Data-{DATASET_TYPE.capitalize()}-train.csv"
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "lime_results")

def load_data():
    """Load and prepare the dataset."""
    print(f"\n[1/5] Loading {DATASET_TYPE.upper()} dataset...")
    df = pd.read_csv(DATA_FILE)
    
    # Drop non-feature columns and split
    drop_cols = ['sample', 'PoI', 'Classification']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df['Classification']
    
    # 80-20 split (same as original)
    split_idx = int(0.8 * len(X))
    X_train = X.iloc[:split_idx].values
    X_test = X.iloc[split_idx:].values
    y_train = y.iloc[:split_idx].values
    y_test = y.iloc[split_idx:].values
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Features: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test, list(X.columns)

def load_best_model():
    """Load the best performing model from Mix_Results/models."""
    print(f"\n[2/5] Loading model...")
    
    # Priority order: ExtraTrees, RandomForest, XGBoost, LightGBM
    priority_patterns = [
        'extratrees', 'randomforest', 'xgboost', 'lightgbm'
    ]
    
    pkl_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')]
    print(f"   Found {len(pkl_files)} .pkl files")
    
    # Try to load by priority
    for pattern in priority_patterns:
        matching_files = [f for f in pkl_files if pattern in f.lower()]
        for model_file in matching_files:
            try:
                model_path = os.path.join(MODELS_DIR, model_file)
                model = joblib.load(model_path)
                model_name = model_file.replace('_model_', '_').replace('.pkl', '')
                print(f"   ✓ Loaded: {model_file}")
                return model, model_name
            except Exception as e:
                print(f"   ✗ Failed to load {model_file}: {e}")
    
    # Fallback: create demo model
    print("   ⚠ No models found. Creating demo RandomForest...")
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    return model, 'demo_rf'

def main():
    """Main analysis pipeline."""
    print("="*80)
    print(f"LIME ANALYSIS - {DATASET_TYPE.upper()} DATASET")
    print("="*80)
    
    # Load data
    X_train, X_test, y_train, y_test, feature_names = load_data()
    
    # Load model
    model, model_name = load_best_model()
    
    # Train if demo
    if 'demo' in model_name:
        print("\n[3/5] Training demo model...")
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        print(f"   ✓ Model trained (Test accuracy: {acc:.4f})")
    else:
        print("\n[3/5] Model already trained")
    
    # Create LIME explainer
    print("\n[4/5] Creating LIME explainer...")
    class_names = sorted(list(set(y_train)))
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=class_names,
        mode='classification',
        random_state=42
    )
    print(f"   ✓ Explainer created (Classes: {class_names})")
    
    # Generate explanations for first 5 samples
    print("   Generating LIME explanations for first 5 test samples...")
    
    # Create output directory
    model_output_dir = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    
    print("\n[5/5] Saving visualizations...")
    for i in range(min(5, len(X_test))):
        try:
            # Get explanation
            exp = explainer.explain_instance(
                X_test[i],
                model.predict_proba,
                num_features=10,
                num_samples=500
            )
            
            # Get prediction
            pred_proba = model.predict_proba([X_test[i]])[0]
            pred_class = model.predict([X_test[i]])[0]
            actual_class = y_test[i]
            
            # Get top features
            exp_list = exp.as_list()
            features = [f[0] for f in exp_list]
            values = [f[1] for f in exp_list]
            
            # Create Plotly figure
            colors = ['green' if v > 0 else 'red' for v in values]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=values,
                y=features,
                orientation='h',
                marker=dict(color=colors),
                text=[f'{v:+.3f}' for v in values],
                textposition='outside'
            ))
            
            fig.update_layout(
                title=f'LIME Explanation - Sample {i}<br>' +
                      f'Predicted: {pred_class} | Actual: {actual_class}',
                xaxis_title='Feature Contribution',
                yaxis_title='Feature',
                height=500,
                showlegend=False
            )
            
            # Save
            html_path = os.path.join(model_output_dir, f'lime_sample_{i}.html')
            fig.write_html(html_path)
            print(f"   ✓ Saved: lime_sample_{i}.html")
            
        except Exception as e:
            print(f"   ✗ Failed to generate sample {i}: {e}")
    
    print(f"\n{'='*80}")
    print(f"✓ Analysis complete! Results saved to: {model_output_dir}")
    print(f"{'='*80}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
