"""
LIME TabPFN Comparison
Generate LIME explanations specifically for TabPFN models.
Separate from other models because TabPFN is MUCH slower (1-2 sec per prediction).
Shows 5 incorrect predictions per TabPFN strategy for comparison.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid tkinter threading issues
import matplotlib.pyplot as plt
import seaborn as sns
from lime.lime_tabular import LimeTabularExplainer
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

# Configuration
DATASET_TYPE = sys.argv[1] if len(sys.argv) > 1 else 'mix'
DATASET_TYPE = DATASET_TYPE.lower()

if DATASET_TYPE == 'mix':
    DATA_FILE = 'Input_Data-Mix-train.csv'
    MODELS_DIR = 'Mix_Results/models'
    OUTPUT_DIR = 'Mix_Results/lime_tabpfn_comparison'
elif DATASET_TYPE == 'single':
    DATA_FILE = 'Input_Data-Single-train.csv'
    MODELS_DIR = 'Single_Results/models'
    OUTPUT_DIR = 'Single_Results/lime_tabpfn_comparison'
else:
    print(f"Error: Invalid dataset type '{DATASET_TYPE}'. Use 'mix' or 'single'")
    sys.exit(1)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    """Load and prepare the dataset."""
    print(f"\n[1/4] Loading {DATASET_TYPE.upper()} dataset...")
    df = pd.read_csv(DATA_FILE)
    
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
    
    return X_train, X_test, y_train, y_test


def select_tabpfn_models():
    """Select ONLY TabPFN model + strategy combinations."""
    print("\n[2/4] Selecting TabPFN models to compare...")
    
    # Find all model files
    model_files = list(Path(MODELS_DIR).glob('*.pkl'))
    
    # Filter to ONLY TabPFN models
    selected_models = [
        f for f in model_files 
        if 'tabpfn' in f.name.lower()
    ]
    
    # Sort by model name for organized output
    selected_models = sorted(selected_models, key=lambda x: x.name)
    
    print(f"   Selected {len(selected_models)} TabPFN model+strategy combinations:")
    for model_file in selected_models:
        print(f"     - {model_file.name}")
    
    # TabPFN is SLOW - estimate realistic time
    total_predictions = len(selected_models) * 5 * 500  # models * samples * lime perturbations
    estimated_minutes = (total_predictions * 1.5) / 60  # 1.5 sec per prediction
    
    print(f"\n   ⚠ WARNING: TabPFN is VERY SLOW for LIME!")
    print(f"   Estimated time: {estimated_minutes:.0f}-{estimated_minutes*1.3:.0f} minutes")
    print(f"   ({len(selected_models)} models × 5 samples × 500 perturbations × ~1.5 sec/prediction)")
    print(f"   Consider running overnight or reducing to 2-3 samples if needed")
    
    return selected_models


def get_incorrect_samples(model, model_name, X_test, y_test, n_samples=5):
    """Find incorrect predictions with varying confidence levels."""
    try:
        # Get predictions and probabilities
        y_pred = model.predict(X_test.values)
        y_proba = model.predict_proba(X_test.values)
        
        # Get class to index mapping
        class_to_idx = {cls: idx for idx, cls in enumerate(model.classes_)}
        
        # Find incorrect predictions with their confidence levels
        incorrect_data = []
        for i in range(len(y_test)):
            if y_pred[i] != y_test.iloc[i]:
                # Map predicted class name to index
                pred_idx = class_to_idx.get(y_pred[i], 0)
                confidence = y_proba[i][pred_idx] * 100  # Confidence in predicted class
                incorrect_data.append({'idx': i, 'confidence': confidence})
        
        if not incorrect_data:
            return []
        
        # Sort by confidence
        incorrect_data.sort(key=lambda x: x['confidence'])
        
        # Select samples from different confidence ranges
        selected = []
        total = len(incorrect_data)
        
        if total <= n_samples:
            # If we have fewer incorrect predictions than requested, take all
            selected = [item['idx'] for item in incorrect_data]
        else:
            # Divide into bins and take samples from each
            # Low confidence (20-40%), Medium (40-60%), High (60-80%), Very High (80-100%)
            bins = [
                [item for item in incorrect_data if 20 <= item['confidence'] < 40],
                [item for item in incorrect_data if 40 <= item['confidence'] < 60],
                [item for item in incorrect_data if 60 <= item['confidence'] < 80],
                [item for item in incorrect_data if 80 <= item['confidence'] <= 100]
            ]
            
            # Remove empty bins
            bins = [b for b in bins if b]
            
            if not bins:
                # If no bins match (edge case), just take evenly spaced samples
                step = total // n_samples
                selected = [incorrect_data[i * step]['idx'] for i in range(n_samples)]
            else:
                # Take samples from each bin proportionally
                samples_per_bin = max(1, n_samples // len(bins))
                for bin_data in bins:
                    if len(selected) >= n_samples:
                        break
                    # Take evenly spaced samples from this bin
                    bin_size = len(bin_data)
                    if bin_size <= samples_per_bin:
                        selected.extend([item['idx'] for item in bin_data])
                    else:
                        step = bin_size // samples_per_bin
                        selected.extend([bin_data[i * step]['idx'] for i in range(samples_per_bin)])
                
                # If we still need more samples, add from any remaining
                if len(selected) < n_samples:
                    all_indices = set([item['idx'] for item in incorrect_data])
                    remaining = list(all_indices - set(selected))
                    selected.extend(remaining[:n_samples - len(selected)])
        
        return selected[:n_samples]
        
    except Exception as e:
        print(f"     ✗ Error getting predictions for {model_name}: {str(e)[:50]}")
        return []


def create_lime_explanation(explainer, model, model_name, sample_idx, sample, 
                            actual, predicted, probabilities, class_names, feature_names):
    """Generate LIME explanation plot matching the user's desired format."""
    try:
        # Generate LIME explanation
        exp = explainer.explain_instance(
            sample,
            model.predict_proba,
            num_features=10,
            num_samples=500,  # Reduced for speed but still slow for TabPFN
            top_labels=len(class_names)  # Get all classes
        )
        
        # Get feature importance for predicted class
        # Find the label index corresponding to predicted class
        try:
            pred_class_idx = list(class_names).index(predicted)
        except ValueError:
            pred_class_idx = 0
        
        feature_importance = exp.as_list(label=pred_class_idx)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), 
                                        gridspec_kw={'height_ratios': [2, 1]})
        
        # Determine if correct or incorrect
        is_correct = actual == predicted
        status = "✓ CORRECT" if is_correct else "✗ INCORRECT"
        
        # Get confidence for predicted class
        confidence = probabilities[pred_class_idx] * 100
        
        # Get class names for display
        actual_name = class_names[actual]
        predicted_name = class_names[predicted]
        
        # Main title
        fig.suptitle(f'LIME Explanation: Sample {sample_idx} {status}\n' +
                    f'True: {actual_name} | Predicted: {predicted_name} (confidence: {confidence:.1f}%)',
                    fontsize=14, fontweight='bold',
                    color='green' if is_correct else 'red')
        
        # TOP PANEL: Feature Contributions
        features, weights = zip(*feature_importance) if feature_importance else ([], [])
        features = list(features)[:10]  # Top 10
        weights = list(weights)[:10]
        
        # Reverse order for bottom-to-top display
        features = features[::-1]
        weights = weights[::-1]
        
        colors = ['green' if w > 0 else 'red' for w in weights]
        ax1.barh(range(len(features)), weights, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_yticks(range(len(features)))
        ax1.set_yticklabels(features, fontsize=9)
        ax1.set_xlabel('Feature Contribution Weight', fontsize=11, fontweight='bold')
        ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax1.grid(True, alpha=0.3, axis='x')
        
        # BOTTOM PANEL: Class Probabilities
        colors_prob = []
        for i, cls in enumerate(class_names):
            if cls == predicted and cls == actual:
                colors_prob.append('#4CAF50')  # Green - correct prediction
            elif cls == predicted:
                colors_prob.append('#2196F3')  # Blue - predicted (but wrong)
            elif cls == actual:
                colors_prob.append('#FF9800')  # Orange - actual class
            else:
                colors_prob.append('#E0E0E0')  # Gray - other classes
        
        bars = ax2.bar(class_names, probabilities, color=colors_prob, 
                      alpha=0.8, edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('Probability', fontsize=11, fontweight='bold')
        ax2.set_xlabel('Class', fontsize=11, fontweight='bold')
        ax2.set_title('Class Probabilities', fontsize=12, fontweight='bold')
        ax2.set_ylim([0, 1])
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add probability labels on bars
        for i, (bar, prob) in enumerate(zip(bars, probabilities)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{prob:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        # Save combined plot
        model_dir = f"{OUTPUT_DIR}/{model_name}"
        os.makedirs(model_dir, exist_ok=True)
        
        png_path = f"{model_dir}/lime_sample_{sample_idx}.png"
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Create separate probability bar chart
        fig_prob, ax_prob = plt.subplots(figsize=(10, 6))
        
        # Title for standalone probability chart
        fig_prob.suptitle(f'Class Probabilities: Sample {sample_idx} {status}\n' +
                         f'True: {actual_name} | Predicted: {predicted_name} (confidence: {confidence:.1f}%)',
                         fontsize=14, fontweight='bold',
                         color='green' if is_correct else 'red')
        
        # Same color scheme as before
        bars_prob = ax_prob.bar(class_names, probabilities, color=colors_prob,
                               alpha=0.8, edgecolor='black', linewidth=1.5)
        ax_prob.set_ylabel('Probability', fontsize=12, fontweight='bold')
        ax_prob.set_xlabel('Class', fontsize=12, fontweight='bold')
        ax_prob.set_ylim([0, 1])
        ax_prob.grid(True, alpha=0.3, axis='y')
        
        # Add probability labels on bars
        for i, (bar, prob) in enumerate(zip(bars_prob, probabilities)):
            height = bar.get_height()
            ax_prob.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{prob:.3f}',
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        
        # Save separate probability chart
        prob_path = f"{model_dir}/lime_sample_{sample_idx}_probabilities.png"
        plt.savefig(prob_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return True
        
    except Exception as e:
        print(f"\n     ✗ Error: {str(e)[:60]}")
        return False


def compare_models(model_files, X_train, X_test, y_train, y_test, feature_names):
    """Generate LIME explanations for TabPFN models."""
    print("\n[3/4] Generating LIME explanations for TabPFN models...")
    print("   ⚠ This will take A LONG TIME (TabPFN is slow!)")
    
    class_names = sorted(list(set(y_test)))
    
    # Create LIME explainer (same for all models)
    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        class_names=class_names,
        mode='classification',
        random_state=42
    )
    
    summary_data = []
    
    for model_idx, model_file in enumerate(model_files):
        model_name = model_file.stem.replace('model_', '').replace('_model', '')
        
        print(f"\n   [{model_idx+1}/{len(model_files)}] Processing: {model_name}")
        
        try:
            # Load model
            model = joblib.load(model_file)
            print(f"     ✓ Model loaded")
            
            # Get incorrect predictions
            incorrect_indices = get_incorrect_samples(model, model_name, X_test, y_test, n_samples=5)
            
            if not incorrect_indices:
                print(f"     ⚠ No incorrect predictions found (perfect model!)")
                continue
            
            print(f"     Found {len(incorrect_indices)} incorrect predictions")
            print(f"     Generating LIME explanations (this is SLOW): ", end='', flush=True)
            
            # Generate LIME for each incorrect sample
            for i, idx in enumerate(incorrect_indices):
                print(f"\n     Sample {i+1}/{len(incorrect_indices)}: ", end='', flush=True)
                
                sample = X_test.iloc[idx].values
                actual_value = y_test.iloc[idx]
                predicted_value = model.predict([sample])[0]
                probabilities = model.predict_proba([sample])[0]
                
                # Convert class names to indices for the function
                class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
                actual = class_to_idx.get(actual_value, 0)
                predicted = class_to_idx.get(predicted_value, 0)
                
                success = create_lime_explanation(
                    explainer, model, model_name, idx, sample,
                    actual, predicted, probabilities, class_names, feature_names
                )
                
                if success:
                    print(f"✓", end='', flush=True)
                else:
                    print(f"✗", end='', flush=True)
                
                # Add to summary regardless (to track attempts)
                summary_data.append({
                    'Model': model_name,
                    'Sample_Index': idx,
                    'Actual': actual,
                    'Predicted': predicted,
                    'Success': success
                })
            
            print("")  # New line after all samples
            
        except Exception as e:
            print(f"     ✗ Error processing {model_name}: {str(e)[:60]}")
            continue
    
    # Save summary
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f"{OUTPUT_DIR}/tabpfn_comparison_summary.csv", index=False)
        print(f"\n   ✓ Saved comparison summary")
    
    return summary_data


def create_summary_report(summary_data):
    """Create text summary of comparison."""
    print("\n[4/4] Creating summary report...")
    
    report_path = f"{OUTPUT_DIR}/tabpfn_comparison_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"LIME TABPFN COMPARISON - {DATASET_TYPE.upper()} DATASET\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total explanations generated: {len(summary_data)}\n")
        
        # Group by model
        df = pd.DataFrame(summary_data)
        models = df['Model'].unique()
        
        f.write(f"TabPFN strategies compared: {len(models)}\n\n")
        
        for model in models:
            model_data = df[df['Model'] == model]
            successful = model_data['Success'].sum()
            f.write(f"\n{model.upper()}\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Incorrect predictions analyzed: {len(model_data)}\n")
            f.write(f"  Successful explanations: {successful}/{len(model_data)}\n")
            f.write(f"  Sample indices: {', '.join(map(str, model_data['Sample_Index'].values))}\n")
            f.write(f"  Predictions: {' | '.join([f'{p}(actual:{a})' for p, a in zip(model_data['Predicted'], model_data['Actual'])])}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("FOLDER STRUCTURE\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"{OUTPUT_DIR}/\n")
        for model in models:
            f.write(f"  - {model}/\n")
            model_data = df[df['Model'] == model]
            for idx in model_data['Sample_Index'].values:
                f.write(f"      - lime_sample_{idx}.png\n")
        f.write("  - tabpfn_comparison_summary.csv\n")
        f.write("  - tabpfn_comparison_report.txt\n\n")
        
        f.write("="*80 + "\n")
        f.write("HOW TO COMPARE TABPFN STRATEGIES\n")
        f.write("="*80 + "\n\n")
        
        f.write("1. Compare the same sample across different TabPFN strategies:\n")
        f.write("   - See if resampling (SMOTE, RandomOverSampler) changes feature importance\n")
        f.write("   - Check if class weights affect predictions\n")
        f.write("   - Look for consistency in feature usage\n\n")
        
        f.write("2. Compare different samples for the same strategy:\n")
        f.write("   - Identify patterns in mistakes\n")
        f.write("   - See which features are consistently important\n\n")
        
        f.write("3. Compare to other models (in lime_model_comparison/):\n")
        f.write("   - See if TabPFN uses different features than tree models\n")
        f.write("   - Check prediction confidence differences\n\n")
        
        f.write("Feature Contribution:\n")
        f.write("  - Green bars = feature supports the predicted class\n")
        f.write("  - Red bars = feature opposes the predicted class\n")
        f.write("  - Longer bars = stronger influence\n")
    
    print(f"   ✓ Saved: tabpfn_comparison_report.txt")


def main():
    """Main pipeline."""
    print("="*80)
    print(f"LIME TABPFN COMPARISON - {DATASET_TYPE.upper()} DATASET")
    print("="*80)
    print("\nGenerates LIME explanations for TabPFN models ONLY")
    print("Shows 5 incorrect predictions per TabPFN strategy for comparison")
    print("⚠ WARNING: This is VERY SLOW due to TabPFN's architecture!\n")
    
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    feature_names = X_train.columns.tolist()
    
    # Select TabPFN models
    model_files = select_tabpfn_models()
    
    if not model_files:
        print("\n✗ No TabPFN models found! Exiting.")
        return
    
    # Confirm before proceeding
    print("\n" + "="*80)
    response = input("Continue with TabPFN LIME analysis? This will take a long time. (y/n): ")
    if response.lower() != 'y':
        print("Cancelled by user.")
        return
    
    # Generate comparisons
    summary_data = compare_models(model_files, X_train, X_test, y_train, y_test, feature_names)
    
    if not summary_data:
        print("\n✗ No explanations generated! Exiting.")
        return
    
    # Create summary
    create_summary_report(summary_data)
    
    print("\n" + "="*80)
    print("✓ LIME TABPFN COMPARISON COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("\nFolder structure:")
    print("  Each TabPFN strategy has its own folder with LIME explanation PNG images")
    print("  tabpfn_comparison_summary.csv - table of all predictions")
    print("  tabpfn_comparison_report.txt - detailed report")
    print("\nEach image shows:")
    print("  - Top: Feature contributions (green=positive, red=negative)")
    print("  - Bottom: Class probabilities for ALL classes")
    print("\nFor other models, run: python scripts\\lime_multi_model_comparison.py " + DATASET_TYPE)


if __name__ == '__main__':
    main()
