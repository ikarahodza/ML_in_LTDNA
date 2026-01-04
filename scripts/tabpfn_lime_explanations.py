"""
TabPFN LIME Explanations (FAST & SAFE VERSION)

Key changes:
- Subsample training data (critical for TabPFN)
- Explain ONE sample only
- Reduce LIME perturbations
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
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
DATASET_TYPE = sys.argv[1] if len(sys.argv) > 1 else "mix"
DATASET_TYPE = DATASET_TYPE.lower()

if DATASET_TYPE == "mix":
    DATA_FILE = "Input_Data-Mix-train.csv"
    MODELS_DIR = "Mix_Results"
    OUTPUT_DIR = "Mix_Results/tabpfn_explainability"
elif DATASET_TYPE == "single":
    DATA_FILE = "Input_Data-Single-train.csv"
    MODELS_DIR = "Single_Results"
    OUTPUT_DIR = "Single_Results/tabpfn_explainability"
else:
    print(f"Invalid dataset type: {DATASET_TYPE}")
    sys.exit(1)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
def load_data():
    print(f"\n[1/3] Loading {DATASET_TYPE.upper()} dataset...")
    df = pd.read_csv(DATA_FILE)

    drop_cols = ["sample", "PoI", "Classification"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df["Classification"]

    split_idx = int(0.8 * len(X))
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Features: {X_train.shape[1]}")

    return X_train, X_test, y_train, y_test

# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------
def find_tabpfn_model():
    print("\n[2/3] Finding TabPFN model...")
    model_files = list(Path(MODELS_DIR).glob("**/models/*tabpfn*.pkl"))

    if not model_files:
        print("   ✗ No TabPFN model found")
        return None, None

    priority = ["original", "classweights", "randomoversampler"]
    for p in priority:
        for mf in model_files:
            if p in mf.stem.lower():
                print(f"   ✓ Found: {mf.name}")
                return joblib.load(mf), mf.stem

    print(f"   ✓ Found: {model_files[0].name}")
    return joblib.load(model_files[0]), model_files[0].stem

# ---------------------------------------------------------------------
# LIME explanation (FAST)
# ---------------------------------------------------------------------
def create_lime_explanation(model, X_train, X_test, y_test, feature_names):
    print("\n[3/3] Generating LIME explanation (FAST MODE)")

    # ---- CRITICAL: subsample training data ----
    X_train_small = X_train.sample(
        n=min(200, len(X_train)),
        random_state=42
    )

    class_names = sorted(list(set(y_test)))

    explainer = LimeTabularExplainer(
        training_data=X_train_small.values,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
        random_state=42
    )

    # ---- Wrap predict_proba safely ----
    def tabpfn_predict(X):
        return model.predict_proba(np.asarray(X))

    # ---- Explain ONE sample only ----
    i = 0
    sample = X_test.iloc[i].values

    pred_proba = model.predict_proba([sample])[0]
    pred_class = model.predict([sample])[0]
    actual_class = y_test.iloc[i]

    print("   → Explaining sample 0")
    print("   → 100 perturbations (safe for TabPFN)")

    exp = explainer.explain_instance(
        sample,
        tabpfn_predict,
        num_features=10,
        num_samples=100,
        top_labels=1
    )

    feature_importance = dict(exp.as_list(label=0))

    features = list(feature_importance.keys())
    values = list(feature_importance.values())
    colors = ["green" if v > 0 else "red" for v in values]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=features,
        x=values,
        orientation="h",
        marker_color=colors
    ))

    fig.update_layout(
        title=f"<b>LIME Explanation (TabPFN)</b><br>"
              f"Predicted: {pred_class} | Actual: {actual_class}",
        xaxis_title="Feature Contribution",
        yaxis_title="Feature",
        height=500,
        width=800
    )

    out_path = f"{OUTPUT_DIR}/lime_sample_0.html"
    fig.write_html(out_path)

    print(f"   ✓ Saved: {out_path}")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    print("=" * 80)
    print(f"TABPFN LIME EXPLANATION — {DATASET_TYPE.upper()}")
    print("=" * 80)

    X_train, X_test, y_train, y_test = load_data()
    feature_names = X_train.columns.tolist()

    model, model_name = find_tabpfn_model()
    if model is None:
        return

    create_lime_explanation(
        model, X_train, X_test, y_test, feature_names
    )

    print("\n✓ LIME ANALYSIS COMPLETE")

if __name__ == "__main__":
    main()
