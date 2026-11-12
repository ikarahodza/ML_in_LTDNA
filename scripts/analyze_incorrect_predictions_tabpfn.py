"""
Incorrect Predictions Analysis - TabPFN Only
Analyzes incorrect predictions for TabPFN model showing:
- Top 5 incorrect predictions (or all if less than 5)
- Prediction probabilities for each class
- Actual vs Predicted classes

Note: Limited to 5 samples due to TabPFN's slower inference speed.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATASET_TYPE = sys.argv[1].lower() if len(sys.argv) > 1 else 'mix'
assert DATASET_TYPE in ['mix', 'single'], "Dataset type must be 'mix' or 'single'"

BASE_DIR = f"{DATASET_TYPE.capitalize()}_Results_PoI"
DATA_FILE = f"Input_Data-{DATASET_TYPE.capitalize()}-train.csv"
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "incorrect_predictions_analysis")

# Maximum number of incorrect predictions to analyze for TabPFN
MAX_INCORRECT = 5

def load_data():
    """Load and prepare the dataset."""
    print(f"\n[1/4] Loading {DATASET_TYPE.upper()} dataset...")
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

def find_tabpfn_models():
    """Find all TabPFN model files."""
    print(f"\n[2/4] Finding TabPFN models...")
    
    # =============================================================================
    # OPTION 1: Load TabPFN models from main results (DEFAULT)
    # =============================================================================
    if not os.path.exists(MODELS_DIR):
        print(f"   ✗ Models directory not found: {MODELS_DIR}")
        return []
    
    # Get TabPFN model files (current _v35 TabPFN-3.5 checkpoint only; skips superseded _v3)
    pkl_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')]
    tabpfn_files = [f for f in pkl_files if 'tabpfn' in f.lower() and f[:-4].endswith('_v35')]

    if not tabpfn_files:
        print(f"   ✗ No TabPFN-3.5 (_v35) models found in {MODELS_DIR}")
        return []
    
    print(f"   ✓ Found {len(tabpfn_files)} TabPFN model(s)")
    for f in tabpfn_files:
        print(f"      - {f}")
    
    return tabpfn_files
    
    # =============================================================================
    # OPTION 2: Load TabPFN models from specific seed folder
    # Uncomment this section and comment out OPTION 1 above
    # =============================================================================
    # SEED_NUMBER = '123'  # Change this to desired seed: 123, 456, 789, 1011, etc.
    # SEED_MODELS_DIR = os.path.join(BASE_DIR, f'seed_{SEED_NUMBER}', 'models')
    # 
    # if not os.path.exists(SEED_MODELS_DIR):
    #     print(f"   ✗ Seed models directory not found: {SEED_MODELS_DIR}")
    #     return []
    # 
    # pkl_files = [f for f in os.listdir(SEED_MODELS_DIR) if f.endswith('.pkl')]
    # tabpfn_files = [f for f in pkl_files if 'tabpfn' in f.lower()]
    # 
    # if not tabpfn_files:
    #     print(f"   ✗ No TabPFN models found in {SEED_MODELS_DIR}")
    #     return []
    # 
    # print(f"   ✓ Found {len(tabpfn_files)} TabPFN model(s) from seed {SEED_NUMBER}")
    # for f in tabpfn_files:
    #     print(f"      - {f}")
    # 
    # # Update MODELS_DIR global for loading
    # global MODELS_DIR
    # MODELS_DIR = SEED_MODELS_DIR
    # 
    # return tabpfn_files
    
    # =============================================================================
    # OPTION 3: Load TabPFN models from ALL seed folders (aggregated analysis)
    # Uncomment this section and comment out OPTION 1 above
    # =============================================================================
    # import glob
    # all_tabpfn_models = []
    # seed_folders = glob.glob(os.path.join(BASE_DIR, 'seed_*', 'models'))
    # 
    # print(f"   Found {len(seed_folders)} seed folders")
    # 
    # for seed_dir in sorted(seed_folders):
    #     seed_name = os.path.basename(os.path.dirname(seed_dir))
    #     pkl_files = [f for f in os.listdir(seed_dir) if f.endswith('.pkl')]
    #     tabpfn_files = [f for f in pkl_files if 'tabpfn' in f.lower()]
    #     
    #     # Store with seed folder info
    #     for tabpfn_file in tabpfn_files:
    #         all_tabpfn_models.append((seed_dir, tabpfn_file, seed_name))
    # 
    # print(f"   ✓ Found {len(all_tabpfn_models)} TabPFN models across all seeds")
    # 
    # # Note: This requires modifying the main loop to handle tuple format
    # return all_tabpfn_models

def create_probability_barchart(detail, model_name, error_num, output_dir):
    """Create a bar chart for prediction probabilities (matching LIME format)."""
    # Get probabilities and ensure class names (not integers) are used
    prob_dict = detail['probabilities']
    
    # If keys are integers, we need to map them to class names
    prob_items = list(prob_dict.items())
    if prob_items and isinstance(list(prob_dict.keys())[0], str) and list(prob_dict.keys())[0].isdigit():
        # Keys are string integers like '0', '1', '2', '3'
        # Map to actual class names using sorted order
        sorted_class_names = ['Allele', 'Back_Stutter', 'Forward_Stutter', 'Pull_Up']
        prob_dict_mapped = {sorted_class_names[int(k)]: v for k, v in prob_dict.items()}
        prob_items = sorted(prob_dict_mapped.items(), key=lambda x: x[0])
    else:
        prob_items = sorted(prob_dict.items(), key=lambda x: x[0])
    
    classes = [item[0] for item in prob_items]
    probabilities = [item[1] for item in prob_items]
    
    # Create colors: green for actual, red for predicted (incorrect), gray for others
    colors_bar = []
    for class_name in classes:
        if class_name == detail['actual_class']:
            colors_bar.append('#27ae60')  # Green for actual/correct class
        elif class_name == detail['predicted_class']:
            colors_bar.append('#e74c3c')  # Red for predicted/incorrect class
        else:
            colors_bar.append('#d8dee9')  # Light gray for other classes
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(classes, probabilities, color=colors_bar, alpha=0.7)
    ax.set_ylabel('Probability', fontsize=10)
    ax.set_title(f'{model_name} (TabPFN) - Incorrect Prediction #{error_num}\n'
                 f'Sample {detail["sample_index"]}: '
                 f'True={detail["actual_class"]} | '
                 f'Predicted={detail["predicted_class"]} '
                 f'(confidence: {detail["confidence"]:.1%})',
                 fontsize=11, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}\n({height*100:.1f}%)',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save
    png_path = os.path.join(output_dir, f'error_{error_num}_sample_{detail["sample_index"]}.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return png_path

def analyze_tabpfn_errors(model, model_name, X_test, y_test, class_names, output_base_dir):
    """Analyze incorrect predictions for TabPFN model."""
    print(f"\n   Analyzing {model_name}...")
    print(f"   ⚠ This may take a while due to TabPFN's inference time...")
    
    try:
        # Get predictions
        print(f"   → Getting predictions...")
        y_pred = model.predict(X_test)
        
        print(f"   → Getting prediction probabilities...")
        y_pred_proba = model.predict_proba(X_test)
        
        # Get model's class labels (in the order of predict_proba output)
        if hasattr(model, 'classes_'):
            model_classes = model.classes_
        else:
            model_classes = class_names
        
        # Ensure y_test and y_pred are in same format for comparison
        # If model predicts integers, map them to class names
        if isinstance(y_pred[0], (int, np.integer)):
            # Model predicts integers, map to class names using sorted class_names
            # The integers correspond to sorted class names
            sorted_class_names = sorted(class_names)
            y_pred_mapped = np.array([sorted_class_names[p] for p in y_pred])
            y_test_comparable = y_test  # Keep as strings
        else:
            # Model predicts strings
            y_pred_mapped = y_pred
            y_test_comparable = y_test
        
        # Find incorrect predictions
        incorrect_mask = y_pred_mapped != y_test_comparable
        incorrect_indices = np.where(incorrect_mask)[0]
        
        n_incorrect = len(incorrect_indices)
        print(f"      Total incorrect: {n_incorrect} / {len(y_test)}")
        
        if n_incorrect == 0:
            print(f"      ✓ Perfect predictions! No errors to analyze.")
            return None
        
        # Limit to MAX_INCORRECT (5 for TabPFN)
        analyze_count = min(MAX_INCORRECT, n_incorrect)
        analyze_indices = incorrect_indices[:analyze_count]
        
        print(f"   → Analyzing {analyze_count} incorrect predictions...")
        
        # Collect error details
        error_details = []
        for idx in analyze_indices:
            actual = y_test[idx]
            predicted = y_pred_mapped[idx]  # Use mapped version for display
            probas = y_pred_proba[idx]
            
            # Create probability dictionary using model's classes
            prob_dict = {}
            for i, class_name in enumerate(model_classes):
                prob_dict[str(class_name)] = float(probas[i])
            
            # Find predicted class index in model's classes and get confidence
            pred_idx = np.where(model_classes == y_pred[idx])[0][0]  # Use original y_pred index
            confidence = float(probas[pred_idx])
            
            error_details.append({
                'sample_index': int(idx),
                'actual_class': str(actual),
                'predicted_class': str(predicted),
                'probabilities': prob_dict,
                'confidence': confidence
            })
        
        print(f"      ✓ Analyzed {analyze_count} incorrect predictions")
        
        # Calculate overall accuracy using mapped values
        accuracy = (y_pred_mapped == y_test_comparable).sum() / len(y_test)
        
        # Create visualizations
        model_viz_dir = os.path.join(output_base_dir, model_name)
        os.makedirs(model_viz_dir, exist_ok=True)
        
        print(f"   → Creating visualizations...")
        for i, detail in enumerate(error_details, 1):
            try:
                create_probability_barchart(detail, model_name, i, model_viz_dir)
            except Exception as e:
                print(f"      ⚠ Failed to create chart for error #{i}: {e}")
        
        print(f"      ✓ Saved {len(error_details)} visualizations to {model_viz_dir}")
        
        result = {
            'model_name': model_name,
            'total_samples': len(y_test),
            'correct_predictions': int((y_pred_mapped == y_test_comparable).sum()),
            'incorrect_predictions': n_incorrect,
            'accuracy': float(accuracy),
            'analyzed_count': analyze_count,
            'error_details': error_details,
            'visualization_dir': model_viz_dir
        }
        
        return result
        
    except Exception as e:
        print(f"      ✗ Error analyzing {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def format_results_text(result):
    """Format results as human-readable text."""
    if result is None:
        return ""
    
    text = []
    text.append("=" * 80)
    text.append(f"MODEL: {result['model_name']} (TabPFN)")
    text.append("=" * 80)
    text.append(f"Total Test Samples: {result['total_samples']}")
    text.append(f"Correct Predictions: {result['correct_predictions']}")
    text.append(f"Incorrect Predictions: {result['incorrect_predictions']}")
    text.append(f"Accuracy: {result['accuracy']:.4f}")
    text.append(f"\nShowing {result['analyzed_count']} incorrect prediction(s):")
    text.append(f"(Limited to {MAX_INCORRECT} due to TabPFN's slower inference)")
    text.append("-" * 80)
    
    for i, detail in enumerate(result['error_details'], 1):
        text.append(f"\nIncorrect Prediction #{i}:")
        text.append(f"  Sample Index: {detail['sample_index']}")
        text.append(f"  Actual Class: {detail['actual_class']}")
        text.append(f"  Predicted Class: {detail['predicted_class']} (Confidence: {detail['confidence']:.4f})")
        text.append(f"  Probability Distribution:")
        
        # Sort probabilities by value (descending)
        sorted_probs = sorted(detail['probabilities'].items(), key=lambda x: x[1], reverse=True)
        for class_name, prob in sorted_probs:
            marker = " <-- PREDICTED" if class_name == detail['predicted_class'] else ""
            marker += " <-- ACTUAL" if class_name == detail['actual_class'] else ""
            text.append(f"    {class_name}: {prob:.4f} ({prob*100:.2f}%){marker}")
    
    text.append("\n")
    return "\n".join(text)

def main():
    """Main analysis pipeline."""
    print("=" * 80)
    print(f"INCORRECT PREDICTIONS ANALYSIS - TabPFN")
    print(f"Dataset: {DATASET_TYPE.upper()}")
    print(f"Analyzing up to {MAX_INCORRECT} incorrect predictions")
    print("=" * 80)
    
    # Load data
    X_train, X_test, y_train, y_test, feature_names = load_data()
    class_names = np.unique(np.concatenate([y_train, y_test]))
    print(f"   Classes: {class_names}")
    
    # Find all TabPFN models
    model_files = find_tabpfn_models()
    
    if not model_files:
        print("\n✗ No TabPFN models found to analyze!")
        return
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Analyze each TabPFN model
    print(f"\n[3/4] Analyzing {len(model_files)} TabPFN model(s)...")
    all_results = []
    
    for model_file in sorted(model_files):
        try:
            # Load model
            model_path = os.path.join(MODELS_DIR, model_file)
            print(f"\n   Loading {model_file}...")
            model = joblib.load(model_path)
            print(f"   ✓ Model loaded successfully")
            
            # Clean model name
            model_name = model_file.replace('.pkl', '').replace('_model', '')
            
            # Analyze errors
            result = analyze_tabpfn_errors(model, model_name, X_test, y_test, class_names, OUTPUT_DIR)
            
            if result:
                all_results.append(result)
                
        except Exception as e:
            print(f"   ✗ Failed to process {model_file}: {e}")
            import traceback
            traceback.print_exc()
    
    if not all_results:
        print("\n✗ No results to save!")
        return
    
    # Save results
    print(f"\n[4/4] Saving results...")
    
    # Save JSON for all TabPFN models
    json_path = os.path.join(OUTPUT_DIR, 'incorrect_predictions_tabpfn_all.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"   ✓ Saved JSON: {json_path}")
    
    # Save formatted text report
    txt_path = os.path.join(OUTPUT_DIR, 'incorrect_predictions_tabpfn_report.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"INCORRECT PREDICTIONS ANALYSIS REPORT - TabPFN\n")
        f.write(f"Dataset: {DATASET_TYPE.upper()}\n")
        f.write(f"Date: {pd.Timestamp.now()}\n")
        f.write(f"Max incorrect predictions analyzed per model: {MAX_INCORRECT}\n")
        f.write(f"(Limited due to TabPFN's slower inference speed)\n")
        f.write(f"\n")
        
        for result in all_results:
            f.write(format_results_text(result))
    
    print(f"   ✓ Saved report: {txt_path}")
    
    # Print summary
    print(f"\n{'=' * 80}")
    print(f"SUMMARY")
    print(f"{'=' * 80}")
    print(f"Models analyzed: {len(all_results)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"\nModel Accuracies:")
    for result in sorted(all_results, key=lambda x: x['accuracy'], reverse=True):
        print(f"  {result['model_name']}: {result['accuracy']:.4f} ({result['incorrect_predictions']} errors)")
    print(f"{'=' * 80}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
