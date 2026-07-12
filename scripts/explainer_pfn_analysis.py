import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ExplainerPFN')))
import pandas as pd
import numpy as np
import joblib
from explainerpfn.base import ExplainerPFN
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

def load_data(file_path):
    """Loads data from a CSV file."""
    print(f"Loading data from {file_path}...")
    return pd.read_csv(file_path)

def load_model(model_path):
    """Loads a pre-trained model."""
    print(f"Loading model from {model_path}...")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None
    return joblib.load(model_path)

def main():
    """Main function to run the ExplainerPFN analysis."""
    # --- Configuration ---
    # Using a pre-existing model from the Mix_Results directory
    MODEL_PATH = 'Mix_Results/models_v2/model_tabpfn_original_v2.pkl'
    DATA_PATH = 'Input_Data-Mix-train.csv'
    RESULTS_DIR = 'explainerpfn_results'
    
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- Load Model and Data ---
    model = load_model(MODEL_PATH)
    if model is None:
        return

    data = load_data(DATA_PATH)
    
    # Assuming the target column is 'Classification' and others are features
    # Adjust if your data has a different structure
    features = data.drop(columns=['sample', 'PoI', 'Classification'])
    target = data['Classification']

    # For demonstration, we'll use a subset of the data
    # ExplainerPFN fits on background data and explains instances
    X_background = features.head(1000)
    # ExplainerPFN expects a 1D scalar target; use max class probability (model confidence)
    y_background_preds = model.predict_proba(X_background).max(axis=1)

    X_explain = features.tail(100)
    y_explain_preds = model.predict_proba(X_explain).max(axis=1)

    print("Initializing ExplainerPFN...")
    explainer = ExplainerPFN(
        n_estimators=1,
        device="auto",
        fit_mode="fit_with_cache"
    )

    print("Fitting ExplainerPFN on background data...")
    explainer.fit(X_background, y_background_preds)

    print("Generating feature importance explanations...")
    explanations = explainer.predict(X_explain, y_explain_preds)

    print("Applying corrections to explanations...")
    corrected_explanations = explainer.apply_correction(
        y_explain_preds,
        explanations,
        kind=["statistical", "additive"]
    )
    
    # --- Save and Display Results ---
    results_df = pd.DataFrame(corrected_explanations, columns=features.columns)
    
    # Save the raw explanation values
    results_path = os.path.join(RESULTS_DIR, 'explainerpfn_feature_importance.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Saved raw feature importance values to {results_path}")

    # Calculate and save summary
    summary = results_df.abs().mean().sort_values(ascending=False)
    summary_path = os.path.join(RESULTS_DIR, 'explainerpfn_summary.csv')
    summary.to_csv(summary_path)
    print(f"Saved summary of feature importance to {summary_path}")

    print("\n--- Top 10 Most Important Features (based on mean absolute value) ---")
    print(summary.head(10))
    print("\nAnalysis complete.")

if __name__ == '__main__':
    main()
