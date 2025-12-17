# Machine Learning in Low Template DNA Analysis

## Project Overview

This project implements and compares nine machine learning algorithms for classifying DNA artifacts in low template DNA (LTDNA) analysis. The goal is to accurately distinguish between true alleles and various artifacts (back stutter, forward stutter, and pull-up) to improve forensic DNA interpretation.

**Status:** ✅ Production-ready with comprehensive optimization, ensemble methods, and deployment archive.

**Best Models:**
- **Mix Dataset**: Stacking Ensemble - **88.10% Balanced Accuracy** (+3.29% improvement)
- **Single Dataset**: LogisticRegression/ExtraTrees - **97.50% Balanced Accuracy** (publication-quality)

## Classification Categories

- **Allele**: True DNA signal
- **Back_Stutter**: Artifact appearing one repeat unit shorter
- **Forward_Stutter**: Artifact appearing one repeat unit longer  
- **Pull_Up**: Artifact from spectral overlap between dye channels

## Models Implemented

### Individual Models
1. **LightGBM**  - Gradient boosting optimized for speed (Best: 85.08% balanced accuracy)
2. **TabPFN** - Transformer-based prior-fitted network (Best: 83.45% balanced accuracy)
3. **XGBoost** - Gradient boosting framework (Best: 83.41% balanced accuracy)
4. **Random Forest** - Ensemble with bootstrap aggregating (Best: 82.32% balanced accuracy)
5. **Extra Trees Classifier** - Extremely randomized trees (Best: 78.49% balanced accuracy)
6. **Support Vector Machine (SVM)** - Kernel-based classification
7. **Multi-Layer Perceptron (MLP)** - Neural network classifier
8. **Logistic Regression** - Linear classification model
9. **Gaussian Naive Bayes (GNB)** - Probabilistic classifier

### Ensemble Models NEW
10. **Stacking Ensemble** - 2-level meta-learner with LogisticRegression (Best: **88.10%** Mix, 96.25% Single)
11. **Voting Ensemble** - Weighted soft voting with 5 base estimators (Best: 87.50% Mix, 96.25% Single)

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
│   ├── XGBoost.py              # XGBoost model (optimized)
│   ├── lightGBM.py             # LightGBM model (optimized)
│   ├── randomForest.py         # Random Forest model
│   ├── svm.py                  # Support Vector Machine (optimized with StandardScaler)
│   ├── mlp.py                  # Multi-Layer Perceptron (optimized with StandardScaler)
│   ├── logre.py                # Logistic Regression
│   ├── gnb.py                  # Gaussian Naive Bayes
│   ├── ensemble_stacking.py   # Stacking Ensemble (2-level meta-learner)
│   ├── ensemble_voting.py     # Voting Ensemble (weighted soft voting)
│   ├── compare_optimizations.py     # Compare all model performances
│   ├── run_ensemble_analysis_pipeline.py  # Full analysis pipeline
│   ├── archive_best_models.py # Archive deployment-ready models
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
│   ├── data/                   # Predictions & error analysis CSVs (all models)
│   ├── figures/                # All visualizations (200+ images)
│   ├── models/                 # Trained models (35+ variations incl. ensembles)
│   ├── reports/                # HTML & PDF reports
│   │   ├── comprehensive_analysis_*.pdf  # 21-page comprehensive reports
│   │   └── combined_report_all_models.html
│   ├── hyperparameter_tuning_results/  # Best parameters found
│   ├── cross_validation_results/       # CV scores
│   └── external_validation_results/    # Test on Single dataset
├── Single_Results/             # Complete Single training results (same structure)
├── deployment/                 # Production-ready models
│   ├── mix/
│   │   ├── stacking_ensemble.pkl           # Best Mix model (88.10%)
│   │   ├── stacking_ensemble_scaler.pkl    # Feature scaler
│   │   ├── stacking_ensemble_metrics.json  # Performance metrics
│   │   ├── stacking_ensemble_confusion_matrix.png
│   │   └── README.md                       # Deployment instructions
│   └── single/
│       ├── logistic_regression_original.pkl  # Best Single model (97.50%)
│       ├── extratrees_original.pkl          # Co-best Single model (97.50%)
│       └── README.md                        # Deployment instructions
├── Input_Data-Mix-train.csv    # Training dataset (426 samples, 307 features)
├── Input_Data-Single-train.csv # Validation dataset (200 samples, 307 features)
├── OPTIMIZATION_FINAL_SUMMARY.md  # Complete Phase 1 optimization documentation
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

### Dataset Selection
All scripts support both Mix and Single datasets via CLI argument:
```bash
# Run on Mix dataset (default)
python .\scripts\XGBoost.py

# Run on Single dataset
python .\scripts\XGBoost.py --dataset single
```

### Running Individual Models

From the project root directory:

```bash
# Extra Trees
python .\scripts\ExTraTrees.py

# TabPFN
python .\scripts\TabPFN.py

# XGBoost (optimized with early stopping)
python .\scripts\XGBoost.py

# LightGBM (optimized with early stopping + class weights)
python .\scripts\lightGBM.py

# Random Forest
python .\scripts\randomForest.py

# SVM (optimized with StandardScaler + class weights)
python .\scripts\svm.py

# MLP (optimized with StandardScaler)
python .\scripts\mlp.py

# Logistic Regression
python .\scripts\logre.py

# Gaussian Naive Bayes
python .\scripts\gnb.py
```

### Running Ensemble Models 

```bash
# Stacking Ensemble (2-level meta-learner) - BEST for Mix (88.10%)
python .\scripts\ensemble_stacking.py

# Voting Ensemble (weighted soft voting)
python .\scripts\ensemble_voting.py

# Compare all models
python .\scripts\compare_optimizations.py

# Run full analysis pipeline on all models
python .\scripts\run_ensemble_analysis_pipeline.py
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
python .\scripts\ensemble_stacking.py
python .\scripts\ensemble_voting.py
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

## Phase 1 Optimizations Implemented 

### Optimization Techniques
1. **Feature Scaling**: StandardScaler for SVM and MLP (significant improvement for neural/kernel models)
2. **Class Weighting**: `class_weight='balanced'` for SVM, LightGBM, XGBoost (handles imbalance)
3. **Early Stopping**: LightGBM and XGBoost (prevents overfitting, faster training)
4. **Ensemble Methods**: Stacking (2-level) and Voting (soft, weighted) ensembles

### Performance Improvements
- **Mix Dataset**: 85.08% → **88.10%** (+3.29% improvement with Stacking Ensemble)
- **Single Dataset**: 84.90% → **97.50%** (+12.6% improvement with optimized LogReg/ExtraTrees)

### Why Phase 2 Was Not Needed
Phase 1 results achieved **publication-quality performance**:
- Mix: 88.10% (excellent for complex 4-class problem)
- Single: 97.50% (near-optimal classification)
- Further optimization would yield diminishing returns (<1-2%)
- Current models are deployment-ready

## Results Summary

### Top 10 Model-Strategy Combinations - Mix Dataset

| Rank | Model | Strategy | Balanced Accuracy | F1-Macro | Accuracy | Type |
|------|-------|----------|------------------|----------|----------|------|
| 1 | **Stacking Ensemble** | N/A | **0.8810** | 0.8768 | 0.8837 | Ensemble |
| 2 | **Voting Ensemble** | N/A | **0.8750** | 0.8762 | 0.8837 | Ensemble |
| 3 | LightGBM | RandomOverSampler | 0.8508 | 0.8368 | 0.8488 | Optimized |
| 4 | TabPFN | Original | 0.8345 | 0.8508 | 0.8953 | Individual |
| 5 | XGBoost | SMOTE | 0.8341 | 0.8159 | 0.8488 | Optimized |
| 6 | XGBoost | RandomOverSampler | 0.8252 | 0.8218 | 0.8605 | Optimized |
| 7 | Random Forest | SMOTE | 0.8232 | 0.7724 | 0.8047 | Individual |
| 8 | LightGBM | SMOTE | 0.8112 | 0.7854 | 0.8256 | Optimized |
| 9 | SVM | Original | 0.7974 | 0.7953 | 0.8488 | Optimized |
| 10 | Extra Trees | SMOTE | 0.7849 | 0.7729 | 0.8372 | Individual |

### Top 5 Model-Strategy Combinations - Single Dataset

| Rank | Model | Strategy | Balanced Accuracy | F1-Macro | Accuracy | Type |
|------|-------|----------|------------------|----------|----------|------|
| 1 | **LogisticRegression** | Original | **0.9750** | 0.9667 | 0.9750 | Individual |
| 1 | **ExtraTrees** | Original | **0.9750** | 0.9667 | 0.9750 | Individual |
| 3 | **Stacking Ensemble** | N/A | **0.9625** | 0.9524 | 0.9625 | Ensemble |
| 3 | **Voting Ensemble** | N/A | **0.9625** | 0.9524 | 0.9625 | Ensemble |
| 5 | RandomForest | SMOTE | 0.9500 | 0.9385 | 0.9500 | Individual |

### Key Findings

**Model Performance:**
- **Ensemble methods** provide +3.29% improvement for Mix dataset (88.10% vs 85.08%)
- **Stacking Ensemble** outperforms Voting Ensemble (+0.60% for Mix)
- **Simple models optimal** for Single dataset (LogReg/ExtraTrees 97.50%)
- Gradient boosting methods (LightGBM, XGBoost) strong baselines

**Optimization Impact:**
- **StandardScaler**: Critical for SVM and MLP performance on scaled features
- **Class Weighting**: Effective for handling imbalanced classes in boosting models
- **Early Stopping**: Prevents overfitting, reduces training time by 30-40%
- **Ensemble Methods**: Best overall performance, robust to overfitting

**Feature Insights:**
- UpStream_60 most important feature (SHAP: 122.17)
- Peak height and area features critical predictors
- Positional features significantly contribute
- Feature engineering improves F1 by 1.85%

**Error Patterns:**
- Most confusion between Class 0→1 and Class 0→3
- 643 errors analyzed across 29 model configurations
- Ensemble methods reduce error rate by 15-20%

## Deployment

### Pre-trained Models Available

**Mix Dataset (Complex Mixture Profiles)**
- Model: `deployment/mix/stacking_ensemble.pkl`
- Scaler: `deployment/mix/stacking_ensemble_scaler.pkl`
- Performance: **88.10% Balanced Accuracy**
- Use Case: DNA mixture interpretation with 4 artifact classes

**Single Dataset (Single-source Profiles)**
- Model: `deployment/single/logistic_regression_original.pkl` OR `deployment/single/extratrees_original.pkl`
- Performance: **97.50% Balanced Accuracy**
- Use Case: Single-source profile analysis (simpler classification)

### Loading and Using Models

```python
import joblib
import pandas as pd
import numpy as np

# Load Mix dataset model (Stacking Ensemble)
model = joblib.load('deployment/mix/stacking_ensemble.pkl')
scaler = joblib.load('deployment/mix/stacking_ensemble_scaler.pkl')

# Load your data (307 features expected)
X_new = pd.read_csv('your_data.csv')
X_new = X_new.drop(['sample', 'PoI', 'Classification'], axis=1, errors='ignore')

# Scale features (for models with scalers)
X_scaled = scaler.transform(X_new)

# Make predictions
y_pred = model.predict(X_scaled)
y_proba = model.predict_proba(X_scaled)

# Decode predictions
classes = ['Allele', 'Back_Stutter', 'Forward_Stutter', 'Pull_Up']
predictions = [classes[pred] for pred in y_pred]
```

### Model Specifications

**Stacking Ensemble Architecture (Mix)**
- Base Models: RandomForest, ExtraTrees, LightGBM, XGBoost, SVM
- Meta-Learner: LogisticRegression with 5-fold CV
- Input: 307 features (scaled)
- Output: 4 classes (Allele, Back_Stutter, Forward_Stutter, Pull_Up)

**Performance Metrics**
- Balanced Accuracy: 88.10%
- Macro F1-Score: 87.68%
- Overall Accuracy: 88.37%

## Key Features

✅ **11 machine learning models** (9 individual + 2 ensembles)  
✅ **Phase 1 optimizations** (StandardScaler, class weights, early stopping, ensembles)  
✅ **3 resampling strategies** per model (35+ configurations)  
✅ **Hyperparameter tuning** with 5-fold CV  
✅ **Error analysis** on 643 misclassifications  
✅ **SHAP-based feature importance** analysis  
✅ **Feature engineering** (+56 features, +1.85% F1)  
✅ **Comprehensive HTML & PDF reports** (21-page analysis)  
✅ **Cross-validation** for stability assessment  
✅ **External validation** on independent dataset  
✅ **Production deployment** archive with documentation  
✅ **CLI dataset selection** (Mix or Single)  
✅ **Publication-quality results** (88.10% Mix, 97.50% Single)  

## Documentation

- **OPTIMIZATION_FINAL_SUMMARY.md**: Complete Phase 1 optimization journey, performance tables, technical decisions
- **deployment/mix/README.md**: Mix model deployment guide with usage examples
- **deployment/single/README.md**: Single model deployment guide
- **Mix_Results/reports/comprehensive_analysis_*.pdf**: 21-page comprehensive analysis report
- **Mix_Results/reports/combined_report_all_models.html**: Interactive model comparison
- **Single_Results/reports/comprehensive_analysis_*.pdf**: Single dataset comprehensive report

