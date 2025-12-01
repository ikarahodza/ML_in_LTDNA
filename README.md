# Machine Learning in Low Template DNA Analysis

## Project Overview

This project implements and compares five machine learning algorithms for classifying DNA artifacts in low template DNA (LTDNA) analysis. The goal is to accurately distinguish between true alleles and various artifacts (back stutter, forward stutter, and pull-up) to improve forensic DNA interpretation.

## Classification Categories

- **Allele**: True DNA signal
- **Back_Stutter**: Artifact appearing one repeat unit shorter
- **Forward_Stutter**: Artifact appearing one repeat unit longer  
- **Pull_Up**: Artifact from spectral overlap between dye channels

## Models Implemented

1. **Extra Trees Classifier** - Ensemble method with extremely randomized trees
2. **TabPFN** - Transformer-based prior-fitted network for tabular data
3. **XGBoost** - Gradient boosting framework
4. **LightGBM** - Gradient boosting framework optimized for speed
5. **Random Forest** - Ensemble method with bootstrap aggregating

## Resampling Strategies

Each model is evaluated with three strategies to handle class imbalance:

- **Original**: No resampling
- **SMOTE**: Synthetic Minority Over-sampling Technique
- **RandomOverSampler**: Random duplication of minority class samples

## Project Structure

```
ML_in_LTDNA/
├── scripts/
│   ├── ExTraTrees.py          # Extra Trees model
│   ├── TabPFN.py               # TabPFN model
│   ├── XGBoost.py              # XGBoost model
│   ├── lightGBM.py             # LightGBM model
│   ├── randomForest.py         # Random Forest model
│   └── helper_functions.py     # Shared utility functions
├── data/
│   ├── predictions_*.csv       # Model predictions with probabilities
│   └── error_analysis_*.csv    # Misclassification analysis
├── figures/
│   ├── confusion_matrix_*.png  # Confusion matrices
│   ├── feature_importance_*.png # Feature importance plots
│   ├── roc_curves_*.png        # ROC curves
│   └── pr_curves_*.png         # Precision-Recall curves
├── models/
│   └── *.pkl                   # Saved trained models
├── reports/
│   ├── comparison_report_*.html      # Individual model reports
│   └── combined_report_all_models.html # Combined comparison report
├── Input_Data-Mix-train.csv    # Training dataset (mixed profiles)
└── Input_Data-Single-train.csv # Training dataset (single profiles)
```

## Requirements

### Python Packages
```
pandas
numpy
scikit-learn
xgboost
lightgbm
tabpfn
imbalanced-learn
matplotlib
seaborn
joblib
jinja2
```

Install dependencies:
```bash
pip install -r requirements_pinned.txt
```

### TabPFN API Key
TabPFN requires an API key. Set it via:
- Command line: `--tabpfn-key YOUR_KEY`
- Environment variable: `TABPFN_KEY` or `TABPFN_API_KEY`
- File: Create `tabpfn.key` in project root

## Usage

### Running Individual Models

From the project root directory:

```bash
# Extra Trees
python .\scripts\ExTraTrees.py

# TabPFN
python .\scripts\TabPFN.py

# XGBoost
python .\scripts\XGBoost.py

# LightGBM
python .\scripts\lightGBM.py

# Random Forest
python .\scripts\randomForest.py
```

### Running All Models

```bash
python .\scripts\ExTraTrees.py
python .\scripts\TabPFN.py
python .\scripts\XGBoost.py
python .\scripts\lightGBM.py
python .\scripts\randomForest.py
```

## Output

### Generated Files

Each model generates:
- **Confusion Matrix**: Visual representation of classification performance
- **Feature Importance**: Relative importance of features (except TabPFN)
- **Predictions CSV**: Test set predictions with probabilities
- **Error Analysis CSV**: Detailed misclassification analysis
- **ROC Curves**: Receiver Operating Characteristic curves
- **PR Curves**: Precision-Recall curves
- **Saved Model**: Serialized trained model (.pkl)

### Reports

**Individual Reports** (`reports/comparison_report_<model>.html`):
- Summary table comparing resampling strategies
- Per-class performance metrics (precision, recall, F1-score, support)
- Embedded visualizations (confusion matrices, feature importance)

**Combined Report** (`reports/combined_report_all_models.html`):
- Side-by-side comparison of all models
- Consistent metrics across all strategies
- All visualizations in one document

## Metrics Evaluated

- **Accuracy**: Overall classification accuracy
- **Balanced Accuracy**: Accuracy accounting for class imbalance
- **F1 Score (Macro)**: Harmonic mean of precision and recall
- **MSE**: Mean Squared Error (for encoded labels)
- **MAE**: Mean Absolute Error (for encoded labels)
- **Log Loss**: Logarithmic loss based on probabilities
- **Per-Class Metrics**: Precision, Recall, F1-Score, Support for each class

## Key Features

✅ Consistent evaluation across all models  
✅ Multiple resampling strategies for imbalanced data  
✅ Comprehensive performance metrics  
✅ Professional HTML reports with visualizations  
✅ Error analysis for misclassifications  
✅ Model persistence for future use  
✅ Modular code with shared helper functions  

## Notes

- **TabPFN Limitations**: 
  - Maximum 10,000 training samples
  - Maximum 1,000 features
  - No feature importance (transformer-based model)
  - May be slow with SMOTE oversampling

- **Data Requirements**: 
  - Training data should include columns: `sample`, `PoI`, `Classification`
  - `Classification` column should contain: Allele, Back_Stutter, Forward_Stutter, Pull_Up

- **Reproducibility**: 
  - Random state set to 42 for all models
  - 80/20 train-test split with stratification

## Future Improvements

- [ ] Hyperparameter optimization for each model
- [ ] Cross-validation for more robust evaluation
- [ ] Feature selection/engineering
- [ ] Additional resampling techniques (BorderlineSMOTE, ADASYN)
- [ ] Model interpretation (SHAP values)
- [ ] More models implemented 

## Authors

Research project for low template DNA analysis using machine learning.
