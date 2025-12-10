# Machine Learning in Low Template DNA Analysis

## Project Overview

This project implements and compares nine machine learning algorithms for classifying DNA artifacts in low template DNA (LTDNA) analysis. The goal is to accurately distinguish between true alleles and various artifacts (back stutter, forward stutter, and pull-up) to improve forensic DNA interpretation.

**Status:** Comprehensive analysis complete with hyperparameter tuning, error analysis, SHAP interpretability, and feature engineering.

**Best Model:** LightGBM with RandomOverSampler - **85.08% Balanced Accuracy**

## Classification Categories

- **Allele**: True DNA signal
- **Back_Stutter**: Artifact appearing one repeat unit shorter
- **Forward_Stutter**: Artifact appearing one repeat unit longer  
- **Pull_Up**: Artifact from spectral overlap between dye channels

## Models Implemented

1. **LightGBM** ⭐ - Gradient boosting optimized for speed (Best: 85.08% balanced accuracy)
2. **TabPFN** - Transformer-based prior-fitted network (Best: 83.45% balanced accuracy)
3. **XGBoost** - Gradient boosting framework (Best: 83.41% balanced accuracy)
4. **Random Forest** - Ensemble with bootstrap aggregating (Best: 82.32% balanced accuracy)
5. **Extra Trees Classifier** - Extremely randomized trees (Best: 78.49% balanced accuracy)
6. **Support Vector Machine (SVM)** - Kernel-based classification
7. **Multi-Layer Perceptron (MLP)** - Neural network classifier
8. **Logistic Regression** - Linear classification model
9. **Gaussian Naive Bayes (GNB)** - Probabilistic classifier

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
│   ├── svm.py                  # Support Vector Machine
│   ├── mlp.py                  # Multi-Layer Perceptron
│   ├── logre.py                # Logistic Regression
│   ├── gnb.py                  # Gaussian Naive Bayes
│   └── helper_functions.py     # Shared utility functions
├── Mix_Results/                # Complete Mix training results
│   ├── analysis_scripts/       # Advanced analysis scripts
│   │   ├── hyperparameter_tuning.py
│   │   ├── error_analysis.py
│   │   ├── shap_analysis.py
│   │   ├── feature_engineering.py
│   │   ├── cross_validation.py
│   │   ├── external_validation.py
│   │   └── generate_comprehensive_pdf.py
│   ├── data/                   # Predictions & error analysis CSVs
│   ├── figures/                # All visualizations (160+ images)
│   ├── models/                 # Trained models (29 variations)
│   ├── reports/                # HTML & PDF reports
│   ├── hyperparameter_tuning_results/  # Best parameters found
│   ├── cross_validation_results/       # CV scores
│   ├── external_validation_results/    # Test on Single dataset
│   └── documentation/          # Analysis documentation
├── Input_Data-Mix-train.csv    # Training dataset (426 samples, 307 features)
├── Input_Data-Single-train.csv # Validation dataset
└── README.md                   # This file
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
python .\scripts\svm.py
python .\scripts\mlp.py
python .\scripts\logre.py
python .\scripts\gnb.py
```

### Running Advanced Analysis

```bash
# Hyperparameter tuning (5 top models, 50 iterations each)
python Mix_Results\analysis_scripts\hyperparameter_tuning.py

# Error analysis (643 errors across all models)
python Mix_Results\analysis_scripts\error_analysis.py

# SHAP analysis (feature importance & interpretability)
python Mix_Results\analysis_scripts\shap_analysis.py

# Feature engineering (create 56 new features)
python Mix_Results\analysis_scripts\feature_engineering.py

# Cross-validation analysis
python Mix_Results\analysis_scripts\cross_validation.py

# External validation on Single dataset
python Mix_Results\analysis_scripts\external_validation.py

# Generate comprehensive PDF report
python Mix_Results\analysis_scripts\generate_comprehensive_pdf.py
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

**Individual Model Reports** (`Mix_Results/reports/comparison_report_<model>.html`):
- Summary table comparing resampling strategies
- Per-class performance metrics (precision, recall, F1-score, support)
- Embedded visualizations (confusion matrices, feature importance)

**Combined Report** (`Mix_Results/reports/combined_report_all_models.html`):
- Side-by-side comparison of all models
- Consistent metrics across all strategies
- All visualizations in one document

**Comprehensive Analysis Report** (`Mix_Results/reports/comprehensive_analysis_YYYYMMDD_HHMMSS.pdf`):
- **21-page PDF** with complete analysis
- Executive summary with top 10 models ranked
- Performance metrics visualization
- Hyperparameter tuning results (5 models)
- Error analysis (6 pages with visualizations)
- SHAP analysis (4 pages with feature importance)
- Feature engineering summary and impact
- Conclusions and deployment recommendations

**Interactive HTML Report** (`Mix_Results/reports/comprehensive_analysis_report.html`):
- All analyses in single interactive document
- Embedded visualizations
- Model comparison tables
- SHAP and error analysis findings

## Metrics Evaluated

- **Accuracy**: Overall classification accuracy
- **Balanced Accuracy**: Accuracy accounting for class imbalance
- **F1 Score (Macro)**: Harmonic mean of precision and recall
- **MSE**: Mean Squared Error (for encoded labels)
- **MAE**: Mean Absolute Error (for encoded labels)
- **Log Loss**: Logarithmic loss based on probabilities
- **Per-Class Metrics**: Precision, Recall, F1-Score, Support for each class

## Advanced Analysis Completed

### Hyperparameter Tuning
- **Models Tuned**: LightGBM, XGBoost, Random Forest, Extra Trees, TabPFN
- **Method**: RandomizedSearchCV with 50 iterations, 5-fold cross-validation
- **Optimization Metric**: Balanced Accuracy
- **Results**: Best parameters saved in `Mix_Results/hyperparameter_tuning_results/`

### Error Analysis (643 errors analyzed)
- **Top Confusion Pattern**: Class 0→1 (134 errors, 20.8%)
- **Second Pattern**: Class 0→3 (134 errors, 20.8%)
- **Findings**: Adjacent classes most confused; resampling affects error distribution
- **Visualizations**: 6 detailed plots showing error patterns by model, class, and strategy

### SHAP Analysis (Feature Importance & Interpretability)
- **Top Feature**: `UpStream_60` (mean importance: 122.17)
- **Top 10 Features** identified with consistent importance across models
- **Key Insight**: Upstream/downstream features dominate, peak measurements critical
- **Feature Interactions**: Captured by tree-based models
- **Visualizations**: 4 comprehensive SHAP plots in `Mix_Results/figures/`

### Feature Engineering
- **Original Features**: 307
- **New Features Created**: 56
  - 8 interaction terms
  - 18 polynomial features (degree 2)
  - 7 ratio features
  - 12 aggregation features
  - 6 context window features
  - 5 statistical features
- **Total Features**: 363
- **Performance Impact**:
  - Balanced Accuracy: 0.7610 → 0.7721 (+1.11%)
  - F1-Score (macro): 0.7701 → 0.7886 (+1.85%)
  - CV Score improvement: +2.29%

### Cross-Validation
- **Method**: 5-fold StratifiedKFold
- **Purpose**: Assess model stability and generalization
- **Results**: Mean and std deviation in `Mix_Results/cross_validation_results/`

### External Validation
- **Test Set**: Input_Data-Single-train.csv
- **Purpose**: Validate models on independent single-source profiles
- **Findings**: Performance documented in `Mix_Results/external_validation_results/`

## Results Summary

### Top 5 Model-Strategy Combinations

| Rank | Model | Strategy | Balanced Accuracy | F1-Macro | Accuracy |
|------|-------|----------|------------------|----------|----------|
| 1 | LightGBM | RandomOverSampler | 0.8508 | 0.8368 | 0.8488 |
| 2 | TabPFN | Original | 0.8345 | 0.8508 | 0.8953 |
| 3 | XGBoost | SMOTE | 0.8341 | 0.8159 | 0.8488 |
| 4 | Random Forest | SMOTE | 0.8232 | 0.7724 | 0.8047 |
| 5 | Extra Trees | SMOTE | 0.7849 | 0.7729 | 0.8372 |

### Key Findings

**Model Performance:**
- Gradient boosting methods (LightGBM, XGBoost) consistently top performers
- TabPFN competitive without hyperparameter tuning
- Tree ensembles provide strong baseline performance

**Resampling Impact:**
- RandomOverSampler best for LightGBM (85.08% balanced accuracy)
- SMOTE effective for XGBoost and Random Forest
- Original data sufficient for TabPFN (83.45% balanced accuracy)

**Feature Insights:**
- UpStream_60 most important feature (SHAP: 122.17)
- Peak height and area features critical predictors
- Positional features significantly contribute
- Feature engineering improves F1 by 1.85%

**Error Patterns:**
- Most confusion between Class 0→1 and Class 0→3
- 643 errors analyzed across 29 model configurations
- Resampling strategies affect error distribution patterns

## Key Features

✅ 9 machine learning models compared  
✅ 3 resampling strategies per model (27 configurations)  
✅ Hyperparameter tuning with 5-fold CV  
✅ Error analysis on 643 misclassifications  
✅ SHAP-based feature importance analysis  
✅ Feature engineering (+56 features, +1.85% F1)  
✅ Comprehensive HTML & PDF reports  
✅ Cross-validation for stability assessment  
✅ External validation on independent dataset  
✅ Model persistence for deployment  

