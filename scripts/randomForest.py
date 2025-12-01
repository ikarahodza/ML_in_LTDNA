import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_recall_fscore_support, log_loss,
    balanced_accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error
)
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import joblib
import warnings
import os
import helper_functions as hf

warnings.filterwarnings("ignore", category=UserWarning)

data_mix = 'Input_Data-Mix-train.csv'
data_single = 'Input_Data-Single-train.csv'
CSV_FILE_PATH = data_mix  
TARGET_COLUMN = "Classification"  
OUTPUT_MODEL_PATH = "trained_rf_model.pkl"  

FEATURE_COLUMNS = None  

DROP_COLUMNS = ['sample', 'PoI']

# Train/test split ratio
TEST_SIZE = 0.3
RANDOM_STATE = 42

RESAMPLING_STRATEGY = 'compare_all'   # Options: None, 'smote', 'class_weights', 'smote_svm'
SMOTE_SAMPLING_STRATEGY = 'auto'  # Options: 'auto', 'minority', 'not majority', etc.
SMOTE_K_NEIGHBORS = 5  # Number of neighbors for SMOTE  

HYPERPARAMETERS = {
    'n_estimators': [10, 20, 50],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', 10]
}

MODEL_NAME = 'randomforest'
DATA_DIR = 'data'
FIG_DIR = 'figures'
MODEL_DIR = 'models'
REPORTS_DIR = 'reports'

def load_and_prepare_data(csv_path, target_col, feature_cols=None):
    """Load CSV and prepare features and target."""
    print(f"Loading data from: {csv_path}")
    data = pd.read_csv(csv_path)
    
    print(f"Dataset shape: {data.shape}")
    print(f"\nFirst few rows:")
    print(data.head())
    
    # Separate features and target
    if feature_cols is None:
        # Use all columns except target
        X = data.drop(columns=[target_col])
    else:
        X = data[feature_cols]

    # Explicitly drop columns that should not be used as features
    explicitly_dropped = []
    for c in DROP_COLUMNS:
        if c in X.columns:
            X = X.drop(columns=[c])
            explicitly_dropped.append(c)
    if explicitly_dropped:
        print(f"Dropping explicit feature columns: {explicitly_dropped}")

    # Keep only numeric feature columns (drop identifiers/strings like `sample`)
    numeric_X = X.select_dtypes(include=[np.number])
    dropped = [c for c in X.columns if c not in numeric_X.columns]
    if dropped:
        print(f"Dropping non-numeric feature columns: {dropped}")
    X = numeric_X
    
    y_raw = data[target_col]
    
    # Encode target labels to numeric values
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target classes (encoded): {np.unique(y)}")
    print(f"Target classes (original): {le.classes_}")
    print(f"Class distribution:\n{pd.Series(y).value_counts()}")
    
    return X, y, data, le

def calculate_class_weights(y):
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    weight_dict = dict(zip(classes, class_weights))
    print("Class distribution:")
    for cls in classes:
        count = np.sum(y == cls)
        pct = (count / len(y)) * 100
        weight = weight_dict[cls]
        print(f"  {cls}: {count} samples ({pct:.2f}%) -> weight: {weight:.4f}")
    print("="*60)

    return weight_dict

def apply_smote(X_train, y_train, strategy='auto', k_neighbors=5, smote_variant='standardd'):
    from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE
    if smote_variant == 'borderline':
        smote = BorderlineSMOTE(sampling_strategy=strategy, k_neighbors=k_neighbors, random_state=RANDOM_STATE)
    elif smote_variant == 'svm':
        smote = SVMSMOTE(sampling_strategy=strategy, k_neighbors=k_neighbors, random_state=RANDOM_STATE)
    else:  
        smote = SMOTE(sampling_strategy=strategy, k_neighbors=k_neighbors, random_state=RANDOM_STATE)
    
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

def train_random_forest(X_train, y_train, hyperparams, class_weight_param=None):
    """Train Random Forest with GridSearchCV."""
    print("\n" + "="*60)
    print("Starting hyperparameter search...")
    print("="*60)
    
    rf = RandomForestClassifier(random_state=RANDOM_STATE, class_weight=class_weight_param)
    grid_search = GridSearchCV(
        rf, 
        hyperparams, 
        cv=5, 
        scoring='balanced_accuracy',
        n_jobs=-1,
        verbose=2
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def compute_metrics(y_true, y_pred, y_prob):
    return hf.compute_metrics(y_true, y_pred, y_prob)

def evaluate_model(model, X_test, y_test, class_names):
    """Evaluate model performance."""
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    # Precision, Recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted'
    )
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Log Loss
    loss = log_loss(y_test, y_pred_proba)
    print(f"Log Loss: {loss:.4f}")
    
    # Classification Report
    print("\n" + "-"*60)
    print("Classification Report:")
    print("-"*60)
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Confusion Matrix
    print("\n" + "-"*60)
    print("Confusion Matrix:")
    print("-"*60)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Advanced metrics (binary-specific extras included when applicable)
    adv_metrics = hf.compute_metrics(y_test, y_pred, y_pred_proba)
    print("\nAdvanced Metrics:")
    for k, v in adv_metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")
    
    return y_pred, y_pred_proba, cm

def plot_confusion_matrix(cm, class_names, save_path='confusion_matrix_rf.png'):
    return hf.plot_confusion_matrix(cm, class_names, save_path=save_path)

def plot_feature_importance(model, feature_names, top_n=20, save_path='feature_importance_rf.png'):
    return hf.plot_feature_importance(model, feature_names, top_n=top_n, save_path=save_path)

def save_predictions(data, y_test, y_pred, y_pred_proba, test_indices, class_names, save_path='predictions_rf.csv'):
    return hf.save_predictions(data, y_test, y_pred, y_pred_proba, test_indices, class_names, save_path=save_path)

def plot_roc_curves(y_test, y_pred_proba, class_names, save_path='roc_curves_rf.png'):
    """Wrapper to call helper ROC plotting (returns saved path)."""
    return hf.plot_roc_curves(y_test, y_pred_proba, class_names, save_path=save_path)

def plot_precision_recall_curves(y_test, y_pred_proba, class_names, save_path='pr_curves_rf.png'):
    """Wrapper to call helper PR plotting (returns saved path)."""
    return hf.plot_precision_recall_curves(y_test, y_pred_proba, class_names, save_path=save_path)

def analyze_errors(data, y_test, y_pred, y_pred_proba, test_indices, class_names, save_path = 'error_analysis_rf.csv'):
    return hf.analyze_errors(data, y_test, y_pred, y_pred_proba, test_indices, class_names, save_path=save_path)

def generate_html_report(results, class_names, output_path=None, assets=None):
    if output_path is None:
        output_path = os.path.join(REPORTS_DIR, f'comparison_report_{MODEL_NAME}.html')
    hf.generate_model_html_report(MODEL_NAME, results, class_names, output_path=output_path, dataset=CSV_FILE_PATH, fig_dir=FIG_DIR, reports_dir=REPORTS_DIR, assets=assets)

if __name__ == "__main__":
    print("\n" + "="*60)
    print("RANDOM FOREST TRAINING PIPELINE")
    print("="*60)
    
    # Load data
    X, y, original_data, le = load_and_prepare_data(
        CSV_FILE_PATH, 
        TARGET_COLUMN, 
        FEATURE_COLUMNS
    )
    
    # Split data
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, 
        np.arange(len(X)),
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    if RESAMPLING_STRATEGY == 'compare_all':
        strategies_to_test = [
            ('No Resampling', 'none', None),
            ('Class Weights', 'class_weights', None),
            ('SMOTE', 'smote', 'standard'),
            ('Borderline SMOTE', 'smote_borderline', 'borderline'),
            ('SVM SMOTE', 'smote_svm', 'svm')
        ]
    else : 
        strategy_map = {
            'none': ('No Resampling', 'none', None),
            'class_weights': ('Class Weights', 'class_weights', None),
            'smote': ('SMOTE', 'smote', 'standard'),
            'smote_borderline': ('Borderline SMOTE', 'smote_borderline', 'borderline'),
            'smote_svm': ('SVM SMOTE', 'smote_svm', 'svm')
        }
    
        if RESAMPLING_STRATEGY in strategy_map: 
            strategies_to_test = [strategy_map[RESAMPLING_STRATEGY]]
        else: 
            print(f"WARNING: Unknown strategy '{RESAMPLING_STRATEGY}'. Using 'none'.")
            strategies_to_test = [('No Resampling', 'none', None)]
    
    results = {}
    models = {}

    for strategy_name, strategy_type, smote_variant in strategies_to_test:
        X_train_processed = X_train.copy()
        y_train_processed = y_train.copy()
        class_weight_param = None

        if strategy_type == 'class_weights': 
            class_weight_param = 'balanced'
            weight_dict = calculate_class_weights(y_train_processed)
        
        elif strategy_type == 'none': 
            pass

        elif strategy_type in ['smote', 'smote_borderline', 'smote_svm']:
            try: 
                X_train_processed, y_train_processed = apply_smote(
                    X_train_processed, 
                    y_train_processed, 
                    strategy=SMOTE_SAMPLING_STRATEGY, 
                    k_neighbors=SMOTE_K_NEIGHBORS,
                    smote_variant=smote_variant
                )
            except Exception as e:
                print(f"WARNING: SMOTE failed: {e}")
                print("Falling back to original training data")
                X_train_processed = X_train.copy()
                y_train_processed = y_train.copy()
        model = train_random_forest(X_train_processed, y_train_processed, HYPERPARAMETERS, class_weight_param)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        # Calculate per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None, zero_division=0)

        # Store results
        results[strategy_name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'mse': mean_squared_error(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'log_loss': log_loss(y_test, y_pred_proba),
            'per_class_precision': precision.tolist(),
            'per_class_recall': recall.tolist(),
            'per_class_f1': f1.tolist(),
            'per_class_support': support.tolist(),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        models[strategy_name] = model

        # Print quick summary
        print(f"\nAccuracy: {results[strategy_name]['accuracy']:.4f}")
        print(f"Balanced Accuracy: {results[strategy_name]['balanced_accuracy']:.4f}")
        print(f"F1 (macro): {results[strategy_name]['f1_macro']:.4f}")
        print(f"Log Loss: {results[strategy_name]['log_loss']:.4f}")
    
    class_names = le.classes_.tolist()
    if len(strategies_to_test) > 1:
        print("FINAL COMPARISON - ALL STRATEGIES")
        print("\n{:<25s} {:>10s} {:>12s} {:>10s} {:>10s}".format(
            "Strategy", "Accuracy", "Bal.Acc", "F1(macro)", "LogLoss"
        ))
        print("-"*80)
        for strategy_name in results.keys():
            r = results[strategy_name]
            print("{:<25s} {:>10.4f} {:>12.4f} {:>10.4f} {:>10.4f}".format(
                strategy_name,
                r['accuracy'],
                r['balanced_accuracy'],
                r['f1_macro'],
                r['log_loss']
            ))
        
        print("PER CLASS RECALL COMPARISON")
        print("\n{:<25s}".format("Strategy"), end="")
        for class_name in class_names:
            print(f"{class_name:>15s}", end="")
        print()
        print("-"*80)
        
        for strategy_name in results.keys():
            print(f"{strategy_name:<25s}", end="")
            for recall_val in results[strategy_name]['per_class_recall']:
                print(f"{recall_val:>15.4f}", end="")
            print()

        best_strategy = max(results.keys(), key=lambda k: results[k]['balanced_accuracy'])
        print(f"\n BEST STRATEGY (by Balanced Accuracy): {best_strategy}")
        print(f"   Balanced Accuracy: {results[best_strategy]['balanced_accuracy']:.4f}")
        
        print("SAVING INDIVIDUAL STRATEGU OUTPUTS")
        assets = {}

        for strategy_name in results.keys():
            # Create clean filename from strategy name and include model name
            safe_name = f"{MODEL_NAME}_{strategy_name.replace(' ', '_').lower()}"

            print(f"\nSaving outputs for: {strategy_name}")

            # Save confusion matrix
            cm_path = os.path.join(FIG_DIR, f'confusion_matrix_{safe_name}.png')
            cm_path = plot_confusion_matrix(results[strategy_name]['confusion_matrix'], class_names, save_path=cm_path)

            # Save feature importance
            fi_path = os.path.join(FIG_DIR, f'feature_importance_{safe_name}.png')
            try:
                fi_path, fi_csv = plot_feature_importance(models[strategy_name], X.columns.tolist(), save_path=fi_path)
            except Exception:
                fi_path = plot_feature_importance(models[strategy_name], X.columns.tolist(), save_path=fi_path)

            # Save ROC curves
            roc_path = os.path.join(FIG_DIR, f'roc_curves_{safe_name}.png')
            try:
                roc_path = plot_roc_curves(y_test, results[strategy_name]['y_pred_proba'], class_names, save_path=roc_path)
            except Exception:
                roc_path = None

            # Save Precision-Recall curves
            pr_path = os.path.join(FIG_DIR, f'pr_curves_{safe_name}.png')
            try:
                pr_path = plot_precision_recall_curves(y_test, results[strategy_name]['y_pred_proba'], class_names, save_path=pr_path)
            except Exception:
                pr_path = None

            # Save predictions
            pred_path = os.path.join(DATA_DIR, f'predictions_{safe_name}.csv')
            pred_path = save_predictions(
                original_data, y_test, 
                results[strategy_name]['y_pred'], 
                results[strategy_name]['y_pred_proba'], 
                test_idx, class_names, 
                save_path=pred_path
            )

            # Save error analysis
            error_path = os.path.join(DATA_DIR, f'error_analysis_{safe_name}.csv')
            error_path = analyze_errors(original_data, y_test, results[strategy_name]['y_pred'], 
                           results[strategy_name]['y_pred_proba'], test_idx, class_names, 
                           save_path=error_path)

            # Save model
            model_path = os.path.join(MODEL_DIR, f'model_{safe_name}.pkl')
            os.makedirs(MODEL_DIR, exist_ok=True)
            joblib.dump(models[strategy_name], model_path)
            print(f"Model saved to: {model_path}")

            # record assets for report
            assets[strategy_name] = {
                'confusion_matrix': cm_path,
                'feature_importance': fi_path,
                'roc_curves': roc_path,
                'pr_curves': pr_path,
                'predictions': pred_path,
                'error_analysis': error_path,
                'model': model_path
            }
        
        print("\n✓ All individual strategy outputs saved!")

        os.makedirs(REPORTS_DIR, exist_ok=True)
        generate_html_report(results, class_names, output_path=os.path.join(REPORTS_DIR, 'comparison_report.html'), assets=assets)

        best_model_to_save = models[best_strategy]
        y_pred = results[best_strategy]['y_pred']
        y_pred_proba = results[best_strategy]['y_pred_proba']
        cm = results[best_strategy]['confusion_matrix']
    
    else: 
        strategy_name = list(results.keys())[0]
        print(f"\n✓ Using strategy: {strategy_name}")
        best_model_to_save = models[strategy_name]
        y_pred = results[strategy_name]['y_pred']
        y_pred_proba = results[strategy_name]['y_pred_proba']
        cm = results[strategy_name]['confusion_matrix']
        
        safe_name = f"{MODEL_NAME}_{strategy_name.replace(' ', '_').lower()}"

        print(f"\nSaving outputs for: {strategy_name}")

        cm_path = os.path.join(FIG_DIR, f'confusion_matrix_{safe_name}.png')
        cm_path = plot_confusion_matrix(cm, class_names, save_path=cm_path)

        fi_path = os.path.join(FIG_DIR, f'feature_importance_{safe_name}.png')
        try:
            fi_path, fi_csv = plot_feature_importance(best_model_to_save, X.columns.tolist(), save_path=fi_path)
        except Exception:
            fi_path = plot_feature_importance(best_model_to_save, X.columns.tolist(), save_path=fi_path)

        # ROC/PR if available
        try:
            roc_path = plot_roc_curves(y_test, y_pred_proba, class_names, save_path=os.path.join(FIG_DIR, f'roc_curves_{safe_name}.png'))
        except Exception:
            roc_path = None
        try:
            pr_path = plot_precision_recall_curves(y_test, y_pred_proba, class_names, save_path=os.path.join(FIG_DIR, f'pr_curves_{safe_name}.png'))
        except Exception:
            pr_path = None

        pred_path = os.path.join(DATA_DIR, f'predictions_{safe_name}.csv')
        pred_path = save_predictions(
            original_data, y_test, y_pred, 
            y_pred_proba, test_idx, class_names, 
            save_path=pred_path
        )

        model_path = os.path.join(MODEL_DIR, f'model_{safe_name}.pkl')
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(best_model_to_save, model_path)
        print(f"Model saved to: {model_path}")

        assets = {strategy_name: {
            'confusion_matrix': cm_path,
            'feature_importance': fi_path,
            'roc_curves': roc_path,
            'pr_curves': pr_path,
            'predictions': pred_path,
            'model': model_path
        }}

        # generate per-model report including assets
        os.makedirs(REPORTS_DIR, exist_ok=True)
        generate_html_report(results, class_names, output_path=os.path.join(REPORTS_DIR, 'comparison_report.html'), assets=assets)
    
    # Also save the generic output model if requested, but include model name
    # Also save the generic output model if requested, but include model dir
    # Append to combined results JSON and regenerate combined HTML
    try:
        hf.append_model_results_to_json(MODEL_NAME, results, class_names, assets=assets)
        hf.generate_combined_html()
    except Exception as e:
        print(f"Warning: failed to append/generate combined report: {e}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    generic_output = OUTPUT_MODEL_PATH
    joblib.dump(best_model_to_save, generic_output)
    print(f"\nModel saved to: {generic_output}")

    # Plot confusion matrix and save with model+strategy name
    plot_confusion_matrix(cm, class_names, save_path=os.path.join(FIG_DIR, f'confusion_matrix_{safe_name}.png'))

    # Plot feature importance
    plot_feature_importance(best_model_to_save, X.columns.tolist(), save_path=os.path.join(FIG_DIR, f'feature_importance_{safe_name}.png'))

    # Save predictions
    save_predictions(
        original_data, y_test, y_pred, y_pred_proba, 
        test_idx, class_names, save_path=os.path.join(DATA_DIR, f'predictions_{safe_name}.csv')
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    