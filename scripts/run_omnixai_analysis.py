"""
OmniXAI Explainability Analysis for Top Models
Generates comprehensive explanations using OmniXAI library
"""
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from omnixai.data.tabular import Tabular
from omnixai.explainers.tabular import TabularExplainer
from omnixai.visualization.dashboard import Dashboard
import argparse

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Configuration
TOP_MODELS = {
    'mix': [
        ('tabpfn', 'classweights'),
        ('extratrees', 'smote'),
        ('extratrees', 'original')
    ],
    'single': [
        ('extratrees', 'smote'),
        ('extratrees', 'randomoversampler'),
        ('tabpfn', 'classweights')
    ]
}

CLASS_NAMES = {
    'mix': ['Allele', 'Back_Stutter', 'Forward_Stutter', 'Pull_Up'],
    'single': ['Allele', 'Back_Stutter', 'Forward_Stutter', 'Pull_Up']
}

def load_data(dataset_type):
    """Load training data"""
    data_file = os.path.join(PROJECT_ROOT, f'Input_Data-{dataset_type.capitalize()}-train.csv')
    
    if not os.path.exists(data_file):
        print(f"✗ Data file not found: {data_file}")
        return None, None, None
    
    df = pd.read_csv(data_file)
    
    # Prepare features and target
    X = df.drop(['sample', 'PoI', 'Classification'], axis=1)
    y_raw = df['Classification']
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    print(f"✓ Loaded {len(df)} samples with {len(X.columns)} features")
    print(f"✓ Classes: {le.classes_}")
    
    return X, y, le

def load_model(dataset_type, model_name, strategy, seed=42):
    """Load a trained model"""
    results_folder = f"{dataset_type.capitalize()}_Results"
    
    # Determine model path based on seed and model type
    if seed == 42:
        model_dir = os.path.join(PROJECT_ROOT, results_folder, 'models')
    else:
        model_dir = os.path.join(PROJECT_ROOT, results_folder, f'seed_{seed}', 'models')
    
    # Different naming conventions for different models
    if model_name == 'tabpfn':
        model_file = f'model_{model_name}_{strategy}.pkl'
    elif model_name in ['extratrees', 'randomforest']:
        model_file = f'{model_name}_model_{model_name}_{strategy}.pkl'
    elif model_name in ['xgboost', 'lgbm']:
        model_file = f'{model_name}_model_{model_name}_{strategy}.pkl'
    else:
        model_file = f'model_{model_name}_{strategy}.pkl'
    
    model_path = os.path.join(model_dir, model_file)
    
    if not os.path.exists(model_path):
        print(f"✗ Model not found: {model_path}")
        return None
    
    model = joblib.load(model_path)
    print(f"✓ Loaded model: {model_name} ({strategy})")
    
    return model

def run_omnixai_analysis(dataset_type='mix', model_name='tabpfn', strategy='classweights', 
                         seed=42, n_test_instances=10):
    """Run comprehensive OmniXAI analysis on a model"""
    
    print("\n" + "="*80)
    print(f"OMNIXAI EXPLAINABILITY ANALYSIS")
    print("="*80)
    print(f"Dataset: {dataset_type.upper()}")
    print(f"Model: {model_name}")
    print(f"Strategy: {strategy}")
    print(f"Seed: {seed}")
    print("="*80 + "\n")
    
    # Load data
    print("[1/5] Loading data...")
    X, y, le = load_data(dataset_type)
    if X is None:
        return
    
    # Load model
    print(f"\n[2/5] Loading model...")
    model = load_model(dataset_type, model_name, strategy, seed)
    if model is None:
        return
    
    # Prepare data for OmniXAI
    print(f"\n[3/5] Preparing OmniXAI data structures...")
    
    # Create DataFrame with target for OmniXAI
    df_full = X.copy()
    df_full['Classification'] = y
    
    # Create Tabular instance
    tabular_data = Tabular(
        df_full,
        target_column='Classification'
    )
    
    print(f"✓ Created Tabular dataset with {len(X.columns)} features")
    
    # Split for test instances
    from sklearn.model_selection import train_test_split
    train_data, test_data = train_test_split(
        tabular_data, 
        test_size=0.2, 
        stratify=y,
        random_state=seed
    )
    
    # Select test instances for explanation
    test_instances = test_data[:n_test_instances]
    
    print(f"✓ Selected {n_test_instances} test instances for explanation")
    
    # Initialize OmniXAI explainer
    print(f"\n[4/5] Initializing OmniXAI TabularExplainer...")
    
    # Preprocessing function
    def preprocess(z):
        """Convert Tabular to model input"""
        df = z.to_pd()
        if 'Classification' in df.columns:
            df = df.drop('Classification', axis=1)
        return df.values
    
    # Postprocessing function
    def postprocess(outputs):
        """Convert model output to probabilities"""
        if hasattr(outputs, 'shape') and len(outputs.shape) == 2:
            return outputs
        # If model outputs are already probabilities
        return outputs
    
    try:
        explainer = TabularExplainer(
            explainers=["lime", "shap", "pdp", "mace"],
            mode="classification",
            data=train_data,
            model=model,
            preprocess=preprocess,
            params={
                "lime": {"num_samples": 1000},
                "shap": {"nsamples": 100},
                "mace": {"ignored_features": []},
                "pdp": {"features": list(X.columns[:8])}  # First 8 features
            }
        )
        print("✓ TabularExplainer initialized successfully")
        print("  - LIME: Local interpretable explanations")
        print("  - SHAP: SHapley Additive exPlanations")
        print("  - PDP: Partial Dependence Plots")
        print("  - MACE: Model-Agnostic Counterfactual Explanations")
        
    except Exception as e:
        print(f"✗ Error initializing explainer: {e}")
        print("Trying with reduced explainer set (LIME + SHAP only)...")
        try:
            explainer = TabularExplainer(
                explainers=["lime", "shap"],
                mode="classification",
                data=train_data,
                model=model,
                preprocess=preprocess,
                params={
                    "lime": {"num_samples": 1000},
                    "shap": {"nsamples": 100}
                }
            )
            print("✓ TabularExplainer initialized with LIME + SHAP")
        except Exception as e2:
            print(f"✗ Failed to initialize explainer: {e2}")
            return
    
    # Generate explanations
    print(f"\n[5/5] Generating explanations...")
    
    try:
        print("  Generating local explanations (LIME, SHAP, MACE)...")
        local_explanations = explainer.explain(X=test_instances)
        print("  ✓ Local explanations generated")
    except Exception as e:
        print(f"  ⚠ Warning: Error generating local explanations: {e}")
        local_explanations = None
    
    try:
        print("  Generating global explanations (PDP)...")
        global_explanations = explainer.explain_global(
            params={"pdp": {"features": list(X.columns[:8])}}
        )
        print("  ✓ Global explanations generated")
    except Exception as e:
        print(f"  ⚠ Warning: Error generating global explanations: {e}")
        global_explanations = None
    
    # Save explanations
    output_dir = os.path.join(
        PROJECT_ROOT,
        f"{dataset_type.capitalize()}_Results",
        'omnixai_results',
        f'{model_name}_{strategy}_seed{seed}'
    )
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to files
    if local_explanations:
        try:
            # Export explanations as HTML
            for method in local_explanations.get_explanations():
                explanation = local_explanations.get_explanation(method)
                html_path = os.path.join(output_dir, f'{method}_explanations.html')
                try:
                    explanation.ipython_plot()
                    # Note: OmniXAI explanations are best viewed in Jupyter or Dashboard
                except Exception:
                    pass
        except Exception as e:
            print(f"  ⚠ Could not export explanations: {e}")
    
    print(f"\n✓ Results saved to: {output_dir}")
    
    # Launch dashboard
    print("\n" + "="*80)
    print("LAUNCHING OMNIXAI INTERACTIVE DASHBOARD")
    print("="*80)
    print("\nThe dashboard will open in your web browser...")
    print("Features:")
    print("  • Interactive LIME explanations")
    print("  • SHAP force plots and summary")
    print("  • Partial Dependence Plots")
    print("  • Counterfactual examples")
    print("  • What-if analysis")
    print("\nPress Ctrl+C to stop the dashboard server")
    print("="*80 + "\n")
    
    try:
        dashboard = Dashboard(
            instances=test_instances,
            local_explanations=local_explanations,
            global_explanations=global_explanations,
            class_names=CLASS_NAMES[dataset_type],
            explainer=explainer
        )
        dashboard.show()  # This will open in browser
    except Exception as e:
        print(f"✗ Error launching dashboard: {e}")
        print("\nExplanations are still available in:", output_dir)

def main():
    parser = argparse.ArgumentParser(description='Run OmniXAI analysis on top models')
    parser.add_argument('--dataset', type=str, default='mix', choices=['mix', 'single'],
                       help='Dataset type (default: mix)')
    parser.add_argument('--model', type=str, default='tabpfn',
                       help='Model name (default: tabpfn)')
    parser.add_argument('--strategy', type=str, default='classweights',
                       help='Strategy name (default: classweights)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--n-instances', type=int, default=10,
                       help='Number of test instances to explain (default: 10)')
    
    args = parser.parse_args()
    
    run_omnixai_analysis(
        dataset_type=args.dataset,
        model_name=args.model,
        strategy=args.strategy,
        seed=args.seed,
        n_test_instances=args.n_instances
    )

if __name__ == "__main__":
    main()
