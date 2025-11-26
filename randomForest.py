import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, log_loss
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score
import joblib
import warnings
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, log_loss
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

warnings.filterwarnings("ignore", category=UserWarning)

CSV_FILE_PATH = "Input_Data-Mix-train.csv"  
TARGET_COLUMN = "Classification"  
OUTPUT_MODEL_PATH = "trained_rf_model.pkl"  

FEATURE_COLUMNS = None  # use all columns except target; non-numeric columns will be dropped

DROP_COLUMNS = ['sample', 'PoI']

# Train/test split ratio
TEST_SIZE = 0.3
RANDOM_STATE = 42

RESAMPLING_STRATEGY = 'compare_all'   # Options: None, 'smote', 'class_weights', 'smote_svm'
SMOTE_SAMPLING_STRATEGY = 'auto'  # Options: 'auto', 'minority', 'not majority', etc.
SMOTE_K_NEIGHBORS = 5  # Number of neighbors for SMOTE  

# Hyperparameter search space (adjust based on your needs)
HYPERPARAMETERS = {
    'n_estimators': [10, 20, 50],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', 10]
}

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
    
    y = data[target_col]
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target classes: {y.unique()}")
    print(f"Class distribution:\n{y.value_counts()}")
    
    return X, y, data

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
    else:  # standard SMOTE
        smote = SMOTE(sampling_strategy=strategy, k_neighbors=k_neighbors, random_state=RANDOM_STATE)
    
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    return X_resampled, y_resampled
    
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

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
    """Compute advanced metrics including binary-specific measures when applicable."""
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision (macro)": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall (macro)": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1 Score (macro)": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Log Loss": log_loss(y_true, y_prob),
    }

    # If binary classification, add sensitivity/specificity/precision(NPV)
    if cm.shape == (2, 2):
        TN, FP, FN, TP = cm.ravel()
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        precision_pos = TP / (TP + FP) if (TP + FP) > 0 else 0
        npv = TN / (TN + FN) if (TN + FN) > 0 else 0

        metrics.update({
            "Sensitivity (TPR)": sensitivity,
            "Specificity (TNR)": specificity,
            "Precision (positive)": precision_pos,
            "NPV": npv,
        })

    return metrics

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
    adv_metrics = compute_metrics(y_test, y_pred, y_pred_proba)
    print("\nAdvanced Metrics:")
    for k, v in adv_metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")
    
    return y_pred, y_pred_proba, cm

def plot_confusion_matrix(cm, class_names, save_path='confusion_matrix.png'):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12, fontweight='bold')
    
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix saved to: {save_path}")
    plt.close()

def plot_feature_importance(model, feature_names, top_n=20, save_path='feature_importance.png'):
    """Plot and save feature importance."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(10, 8))
    plt.title(f'Top {top_n} Feature Importances', fontsize=16, fontweight='bold')
    plt.barh(range(top_n), importances[indices], color='steelblue')
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Importance', fontsize=12, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Feature importance plot saved to: {save_path}")
    plt.close()

def save_predictions(data, y_test, y_pred, y_pred_proba, test_indices, 
                    class_names, save_path='predictions.csv'):
    """Save predictions to CSV."""
    results = data.iloc[test_indices].copy()
    results['True_Label'] = y_test.values
    results['Predicted_Label'] = y_pred
    
    # Add probability columns
    for i, class_name in enumerate(class_names):
        results[f'Prob_{class_name}'] = y_pred_proba[:, i]
    
    results.to_csv(save_path, index=False)
    print(f"\nPredictions saved to: {save_path}")

def plot_roc_curves(y_test, y_pred_proba, class_names, save_path='roc_curves.png'):
    """Plot and save ROC curves for each class."""
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    from sklearn.multiclass import OneVsRestClassifier

    y_test_bin = label_binarize(y_test, classes=range(len(class_names)))
    
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curve for each class
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=2, label=f'{class_name} (AUC = {roc_auc:.3f})')
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('ROC Curves - Multi-class', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"ROC curves saved to: {save_path}")
    plt.close()

def plot_precision_recall_curves(y_test, y_pred_proba, class_names, save_path='pr_curves.png'):
    """Plot precision-recall curves for each class."""
    from sklearn.preprocessing import label_binarize
    # Binarize the labels
    y_test_bin = label_binarize(y_test, classes=range(len(class_names)))
    
    plt.figure(figsize=(10, 8))
    
    # Plot PR curve for each class
    for i, class_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_pred_proba[:, i])
        ap = average_precision_score(y_test_bin[:, i], y_pred_proba[:, i])
        plt.plot(recall, precision, linewidth=2, label=f'{class_name} (AP = {ap:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12, fontweight='bold')
    plt.ylabel('Precision', fontsize=12, fontweight='bold')
    plt.title('Precision-Recall Curves', fontsize=16, fontweight='bold')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Precision-Recall curves saved to: {save_path}")
    plt.close()

def analyze_errors(data, y_test, y_pred, y_pred_proba, test_indices, class_names, save_path = 'error_analysis.csv'):
    """Save detailed analysis of misclassified samples."""
    errors = data.iloc[test_indices].copy()
    errors['True_Label'] = y_test.values
    errors['Predicted_Label'] = y_pred
    errors['Correct'] = (y_test.values == y_pred)
    
    # Add prediction probabilities
    for i, class_name in enumerate(class_names):
        errors[f'Prob_{class_name}'] = y_pred_proba[:, i]
    
    # Add confidence (max probability)
    errors['Confidence'] = y_pred_proba.max(axis=1)
    
    # Only keep misclassified samples
    misclassified = errors[~errors['Correct']].copy()
    
    # Sort by confidence (most confident errors first)
    misclassified = misclassified.sort_values('Confidence', ascending=False)
    
    misclassified.to_csv(save_path, index=False)
    print(f"Error analysis saved to: {save_path} ({len(misclassified)} misclassifications)")
    
    return misclassified

def generate_html_report(results, class_names, output_path ='comparison_report.html'):
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")    

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Random Forest Model Comparison Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                background-color: #f5f5f5;
            }}
            h1 {{
                color: #2c3e50;
                text-align: center;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #34495e;
                margin-top: 30px;
                border-left: 4px solid #3498db;
                padding-left: 10px;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
                background-color: white;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            th {{
                background-color: #3498db;
                color: white;
                padding: 12px;
                text-align: center;
                font-weight: bold;
            }}
            td {{
                border: 1px solid #ddd;
                padding: 10px;
                text-align: center;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            tr:hover {{
                background-color: #f0f0f0;
            }}
            .best {{
                background-color: #2ecc71 !important;
                color: white;
                font-weight: bold;
            }}
            .info {{
                background-color: #ecf0f1;
                padding: 15px;
                border-radius: 5px;
                margin: 20px 0;
            }}
            .timestamp {{
                text-align: center;
                color: #7f8c8d;
                font-style: italic;
                margin-bottom: 30px;
            }}
        </style>
    </head>
    <body>
        <h1>🌲 Random Forest Model Comparison Report</h1>
        <div class="timestamp">Generated: {timestamp}</div>
        
        <div class="info">
            <strong>Dataset:</strong> {CSV_FILE_PATH}<br>
            <strong>Target Column:</strong> {TARGET_COLUMN}<br>
            <strong>Test Size:</strong> {TEST_SIZE * 100}%<br>
            <strong>Random State:</strong> {RANDOM_STATE}
        </div>
    """
    
    # Find best strategy
    best_strategy = max(results.keys(), key=lambda k: results[k]['balanced_accuracy'])
    
    # Overall Metrics Table
    html += "<h2>📊 Overall Performance Metrics</h2>"
    html += "<table><tr><th>Strategy</th><th>Accuracy</th><th>Balanced Accuracy</th><th>F1 (macro)</th><th>Log Loss</th></tr>"
    
    for name, r in results.items():
        row_class = ' class="best"' if name == best_strategy else ''
        html += f"<tr{row_class}>"
        html += f"<td><strong>{name}</strong></td>"
        html += f"<td>{r['accuracy']:.4f}</td>"
        html += f"<td>{r['balanced_accuracy']:.4f}</td>"
        html += f"<td>{r['f1_macro']:.4f}</td>"
        html += f"<td>{r['log_loss']:.4f}</td>"
        html += "</tr>"
    
    html += "</table>"
    html += f"<p><strong>🏆 Best Strategy:</strong> {best_strategy} (Balanced Accuracy: {results[best_strategy]['balanced_accuracy']:.4f})</p>"
    
    # Per-Class Recall Table
    html += "<h2>🎯 Per-Class Recall</h2>"
    html += "<table><tr><th>Strategy</th>"
    for cn in class_names:
        html += f"<th>{cn}</th>"
    html += "</tr>"
    
    for name, r in results.items():
        row_class = ' class="best"' if name == best_strategy else ''
        html += f"<tr{row_class}><td><strong>{name}</strong></td>"
        for recall in r['per_class_recall']:
            color = ''
            if recall >= 0.9:
                color = ' style="background-color: #2ecc71; color: white;"'
            elif recall >= 0.7:
                color = ' style="background-color: #f39c12; color: white;"'
            elif recall < 0.5:
                color = ' style="background-color: #e74c3c; color: white;"'
            html += f"<td{color}>{recall:.4f}</td>"
        html += "</tr>"
    
    html += "</table>"
    
    # Summary Statistics
    html += "<h2>📈 Summary Statistics</h2>"
    html += "<table>"
    html += "<tr><th>Metric</th><th>Best</th><th>Worst</th><th>Average</th></tr>"
    
    metrics_to_summarize = ['accuracy', 'balanced_accuracy', 'f1_macro']
    for metric in metrics_to_summarize:
        values = [r[metric] for r in results.values()]
        html += f"<tr><td><strong>{metric.replace('_', ' ').title()}</strong></td>"
        html += f"<td>{max(values):.4f}</td>"
        html += f"<td>{min(values):.4f}</td>"
        html += f"<td>{np.mean(values):.4f}</td></tr>"
    
    html += "</table>"
    
    # Recommendations
    html += "<h2>💡 Recommendations</h2>"
    html += "<div class='info'>"
    html += f"<strong>Best Overall:</strong> {best_strategy}<br><br>"
    
    # Find best for minority class (last class, usually the rarest)
    minority_recalls = {name: r['per_class_recall'][-1] for name, r in results.items()}
    best_minority = max(minority_recalls.keys(), key=lambda k: minority_recalls[k])
    html += f"<strong>Best for Minority Class ({class_names[-1]}):</strong> {best_minority} "
    html += f"(Recall: {minority_recalls[best_minority]:.4f})<br><br>"
    
    html += "<strong>Use Case Guidance:</strong><br>"
    html += "• <strong>Balanced Performance:</strong> Use the strategy with highest Balanced Accuracy<br>"
    html += "• <strong>Minority Class Detection:</strong> Prioritize high recall for rare classes<br>"
    html += "• <strong>Production Deployment:</strong> Consider both performance and training time<br>"
    html += "</div>"
    
    html += """
    </body>
    </html>
    """
    
    # Write HTML report using UTF-8 encoding to avoid platform encoding errors
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"\n📄 HTML report saved to: {output_path}")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("RANDOM FOREST TRAINING PIPELINE")
    print("="*60)
    
    # Load data
    X, y, original_data = load_and_prepare_data(
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

        # Store results
        results[strategy_name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'log_loss': log_loss(y_test, y_pred_proba),
            'per_class_recall': recall_score(y_test, y_pred, average=None),
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
    
    class_names = [str(cls) for cls in np.unique(y)]
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
        print(f"\n🏆 BEST STRATEGY (by Balanced Accuracy): {best_strategy}")
        print(f"   Balanced Accuracy: {results[best_strategy]['balanced_accuracy']:.4f}")
        
        print("SAVING INDIVIDUAL STRATEGU OUTPUTS")

        for strategy_name in results.keys():
            # Create clean filename from strategy name
            safe_name = strategy_name.replace(" ", "_").lower()
            
            print(f"\nSaving outputs for: {strategy_name}")
            
            # Save confusion matrix
            cm_path = f'confusion_matrix_{safe_name}.png'
            plot_confusion_matrix(results[strategy_name]['confusion_matrix'], class_names, save_path=cm_path)
            
            # Save feature importance
            fi_path = f'feature_importance_{safe_name}.png'
            plot_feature_importance(models[strategy_name], X.columns.tolist(), save_path=fi_path)

            # Save ROC curves
            roc_path = f'roc_curves_{safe_name}.png'
            plot_roc_curves(y_test, results[strategy_name]['y_pred_proba'], class_names, save_path=roc_path)
            
            # Save Precision-Recall curves
            pr_path = f'pr_curves_{safe_name}.png'
            plot_precision_recall_curves(y_test, results[strategy_name]['y_pred_proba'], class_names, save_path=pr_path)
            
            # Save predictions
            pred_path = f'predictions_{safe_name}.csv'
            save_predictions(
                original_data, y_test, 
                results[strategy_name]['y_pred'], 
                results[strategy_name]['y_pred_proba'], 
                test_idx, class_names, 
                save_path=pred_path
            )

            # Save error analysis
            error_path = f'error_analysis_{safe_name}.csv'
            analyze_errors(original_data, y_test, results[strategy_name]['y_pred'], 
                    results[strategy_name]['y_pred_proba'], test_idx, class_names, 
                    save_path=error_path)
            
            # Save model
            model_path = f'model_{safe_name}.pkl'
            joblib.dump(models[strategy_name], model_path)
            print(f"Model saved to: {model_path}")
        
        print("\n✓ All individual strategy outputs saved!")

        generate_html_report(results, class_names, output_path='comparison_report.html')

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
        
        safe_name = strategy_name.replace(" ", "_").lower()

        print(f"\nSaving outputs for: {strategy_name}")

        cm_path = f'confusion_matrix_{safe_name}.png'
        plot_confusion_matrix(cm, class_names, save_path=cm_path)

        fi_path = f'feature_importance_{safe_name}.png'
        plot_feature_importance(best_model_to_save, X.columns.tolist(), save_path=fi_path)

        pred_path = f'predictions_{safe_name}.csv'
        save_predictions(
            original_data, y_test, y_pred, 
            y_pred_proba, test_idx, class_names, 
            save_path=pred_path
        )

        model_path = f'model_{safe_name}.pkl'
        joblib.dump(best_model_to_save, model_path)
        print(f"Model saved to: {model_path}")
    
    joblib.dump(best_model_to_save, OUTPUT_MODEL_PATH)
    print(f"\nModel saved to: {OUTPUT_MODEL_PATH}")
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, class_names)
    
    # Plot feature importance
    plot_feature_importance(best_model_to_save, X.columns.tolist())
    
    # Save predictions
    save_predictions(
        original_data, y_test, y_pred, y_pred_proba, 
        test_idx, class_names
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    