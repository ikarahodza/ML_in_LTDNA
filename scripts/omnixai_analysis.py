"""
OmniXAI Explainability Analysis - Simplified Version
Comprehensive model interpretability using LIME
Requires Python 3.10 environment with OmniXAI installed
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from omnixai.data.tabular import Tabular
from omnixai.explainers.tabular import TabularExplainer
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATASET_TYPE = sys.argv[1].lower() if len(sys.argv) > 1 else 'mix'
RESULTS_DIR = f'{DATASET_TYPE.capitalize()}_Results'
DATA_FILE = f'Input_Data-{DATASET_TYPE.capitalize()}-train.csv'
OUTPUT_DIR = os.path.join(RESULTS_DIR, 'omnixai_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Class names
CLASS_NAMES = ['Allele', 'Back_Stutter', 'Forward_Stutter', 'Pull_Up']

def load_data():
    """Load and prepare dataset."""
    print(f"\n[1/5] Loading {DATASET_TYPE.upper()} dataset...")
    df = pd.read_csv(DATA_FILE)
    
    # Drop non-feature columns
    X = df.drop(['sample', 'PoI', 'Classification'], axis=1, errors='ignore')
    y = df['Classification']
    
    # Encode labels
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Features: {X.shape[1]}")
    
    return X_train, X_test, y_train, y_test, X.columns.tolist()

def load_best_model():
    """Load one best performing model."""
    print("\n[2/5] Loading model...")
    models_dir = os.path.join(RESULTS_DIR, 'models')
    
    if os.path.exists(models_dir):
        available = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
        print(f"   Found {len(available)} .pkl files")
        
        # Priority list
        priority = ['extratrees', 'randomforest', 'xgboost', 'lightgbm']
        
        for p in priority:
            for filename in available:
                if p in filename.lower():
                    try:
                        model_path = os.path.join(models_dir, filename)
                        model = joblib.load(model_path)
                        model_name = filename.replace('_best_model.pkl', '').replace('.pkl', '')
                        print(f"   ✓ Loaded: {model_name}")
                        return model, model_name
                    except Exception as e:
                        print(f"   ✗ Failed to load {filename}: {e}")
    
    # Fallback: create demo model
    print("   ⚠ No models found. Creating demo RandomForest...")
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    return model, 'demo_rf'

def main():
    """Main analysis pipeline."""
    print("="*80)
    print(f"OMNIXAI LIME ANALYSIS - {DATASET_TYPE.upper()} DATASET")
    print("="*80)
    
    # Load data
    X_train, X_test, y_train, y_test, feature_names = load_data()
    
    # Load model
    model, model_name = load_best_model()
    
    # Train if demo
    if 'demo' in model_name:
        print("\n[3/5] Training demo model...")
        model.fit(X_train, y_train)
        print("   ✓ Model trained")
    else:
        print("\n[3/5] Model already trained")
    
    # Create explainer
    print("\n[4/5] Creating LIME explainer and generating explanations...")
    try:
        # Convert to Tabular format
        train_data = Tabular(
            data=X_train,
            feature_columns=feature_names,
            target_column=None
        )
        test_data = Tabular(
            data=X_test,
            feature_columns=feature_names,
            target_column=None
        )
        
        # Create explainer - OmniXAI will use default LIME settings
        explainer = TabularExplainer(
            explainers=['lime'],
            mode='classification',
            data=train_data,
            model=model
        )
        print("   ✓ Explainer created")
        
        # Generate explanations for first 5 samples
        print("   Generating LIME explanations for first 5 test samples...")
        lime_exp = explainer.explain(X=test_data[:5])
        print("   ✓ Explanations generated")
        
        # Save visualizations
        print("\n[5/5] Saving visualizations...")
        model_output_dir = os.path.join(OUTPUT_DIR, model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        
        for i in range(5):
            try:
                lime_fig = lime_exp.plotly_plot(index=i)
                lime_path = os.path.join(model_output_dir, f'lime_sample_{i}.html')
                lime_fig.write_html(lime_path)
                print(f"   ✓ Saved: lime_sample_{i}.html")
            except Exception as e:
                print(f"   ✗ Failed to save sample {i}: {e}")
        
        print(f"\n✓ ANALYSIS COMPLETE")
        print(f"   Results directory: {model_output_dir}")
        print(f"   Open the HTML files in your browser to view explanations")
        
    except Exception as e:
        print(f"\n✗ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
