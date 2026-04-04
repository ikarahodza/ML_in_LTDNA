# Machine Learning in Low Template DNA Analysis

## Project Overview

This project implements and compares nine machine learning algorithms for classifying DNA artifacts in low template DNA (LTDNA) analysis. The goal is to accurately distinguish between true alleles and various artifacts (back stutter, forward stutter, and pull-up) to improve forensic DNA interpretation.

**Status:** ✅ Production-ready with comprehensive optimization, multi-seed validation, cross-validation, and deployment archive.

**Best Models (Strict Multi-Seed Validation - 10 Seeds):**
- **Mix Dataset**: TabPFN (ClassWeights) - **84.73% ± 4.51%**
- **Single Dataset**: Extra Trees (SMOTE) - **98.61% ± 1.92%**

**Legacy Fallback Note:**
- Older summaries may include low-seed entries (for example, RandomForest Borderline SMOTE on Single).
- Use strict 10-seed policy for all primary comparisons and deployment decisions.

**Best Models (30-Fold Cross-Validation - Seed 42):**
- **Mix Dataset**: ExtraTrees (Baseline) - **86.53% ± 12.90%**
- **Single Dataset**: ExtraTrees (Baseline) - **98.75% ± 4.95%**

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
│   ├── aggregate_multi_seed_results.py  # Aggregate results across 10 seeds
│   ├── generate_aggregated_report.py    # HTML report for multi-seed top 3
│   ├── generate_cv_executive_summary.py # Executive summary PDFs from CV
│   ├── visualize_cv_top_models.py       # Top 3 CV models bar charts
│   └── helper_functions.py     # Shared utility functions
├── Mix_Results/                # Complete Mix training results
│   ├── analysis_scripts/       # Advanced analysis scripts
│   │   ├── hyperparameter_tuni         # 30-fold stratified CV
│   │   ├── external_validation.py
│   │   ├── generate_comprehensive_pdf.py
│   │   └── generate_executive_summary_pdf.py
│   ├── data/                   # Predictions & error analysis CSVs (all models)
│   ├── figures/                # All visualizations (200+ images)
│   │   └── top3_cv_models.png  # Top 3 CV models bar charts
│   ├── models/                 # Trained models (35+ variations incl. ensembles)
│   ├── reports/                # HTML & PDF reports
│   │   ├── comprehensive_analysis_*.pdf  # 21-page comprehensive reports
│   │   ├── executive_summary_*.pdf       # 12-page presentation PDFs
│   │   ├── executive_summary_cv_*.pdf    # CV executive summaries (30-fold)
│   │   └── combined_report_all_models.html
│   ├── hyperparameter_tuning_results/  # Best parameters found
│   ├── cross_validation_results/       # 30-fold CV scores & visualizations
│   │   ├── cv_results.json             # Full CV metrics
│   │   ├── cv_summary.txt              # Text summary
│   │   └── cv_boxplots.png             # Performance distributions
│   ├── external_validation_results/    # Test on Single dataset as Mix)
│   ├── multi_seed_aggregated/  # Aggregated results across 10 seeds
│   └── cross_validation_results/  # 30-fold CV results
│   └── multi_seed_aggregated/          # Aggregated results across 10 seeds
│       └── aggregated_results.json     # Mean ± std across all seeds PDFs
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
├── aggregated_top3_report.html # Multi-seed top 3 models (HTML report)
├── check_overfitting.py        # Overfitting analysis script
├── view_cv_std.py              # View cross-validation standard deviations
├── view_external_validation.py # View external validation results
├── OPTIMIZATION_FINAL_SUMMARY.md  # Complete Phase 1 optimization documentation
├── PIPELINE_OPTIMIZATION_REPORT.md # Pipeline optimization details
├── check_overfitting.py        # Overfitting analysis script
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
python .\scripts\XGBoost.py mix

# Run on Single dataset
python .\scripts\XGBoost.py single
```

### Running Individual Models

From the project root directory:

```bash
# Extra Trees
python .\scripts\ExTraTrees.py mix          # Mix (default)
python .\scripts\ExTraTrees.py single       # Single

# TabPFN
python .\scripts\TabPFN.py mix              # Mix (default)
python .\scripts\TabPFN.py single           # Single

# XGBoost (optimized with early stopping)
python .\scripts\XGBoost.py mix             # Mix (default)
python .\scripts\XGBoost.py single          # Single

# LightGBM (optimized with early stopping + class weights)
python .\scripts\lightGBM.py mix            # Mix (default)
python .\scripts\lightGBM.py single         # Single

# Random Forest
python .\scripts\randomForest.py mix        # Mix (default)
python .\scripts\randomForest.py single     # Single

# SVM (optimized with StandardScaler + class weights)
python .\scripts\svm.py mix                 # Mix (default)
python .\scripts\svm.py single              # Single

# MLP (optimized with StandardScaler)
python .\scripts\mlp.py mix                 # Mix (default)
python .\scripts\mlp.py single              # Single

# Logistic Regression
python .\scripts\logre.py mix               # Mix (default)
python .\scripts\logre.py single            # Single

# Gaussian Naive Bayes
python .\scripts\gnb.py mix                 # Mix (default)
python .\scripts\gnb.py single              # Single
```

### Running Ensemble Models 

```bash
# Stacking Ensemble (2-level meta-learner) - BEST for Mix (88.10%)
python .\scripts\ensemble_stacking.py              # Mix (default)
python .\scripts\ensemble_stacking.py single       # Single

# Voting Ensemble (weighted soft voting)
python .\scripts\ensemble_voting.py                # Mix (default)
python .\scripts\ensemble_voting.py single         # Single

# Compare all models
python .\scripts\compare_optimizations.py          # Mix (default)
python .\scripts\compare_optimizations.py single   # Single

# Run full analysis pipeline on all models
python .\scripts\run_ensemble_analysis_pipeline.py # Mix (default)
python .\scripts\run_ensemble_analysis_pipeline.py single
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

# External validation (Mix models tested on Single dataset)
python Mix_Results\analysis_scripts\external_validation.py

# External validation (Single models tested on Mix dataset)
python Single_Results\analysis_scripts\external_validation.py

# Generate comprehensive PDF report - Mix (21 pages, all analyses)
python Mix_Results\analysis_scripts\generate_comprehensive_pdf.py

# Generate comprehensive PDF report - Single (21 pages, all analyses)
python Single_Results\analysis_scripts\generate_comprehensive_pdf.py

# Generate executive summary PDF - Mix (12 pages, visualizations only)

# Multi-seed validation (10 seeds: 42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021)
python scripts\aggregate_multi_seed_results.py          # Aggregate results across seeds
python scripts\generate_aggregated_report.py            # Generate HTML report with top 3

# Cross-validation executive summaries (30-fold)
python scripts\generate_cv_executive_summary.py mix     # Mix CV summary PDF
python scripts\generate_cv_executive_summary.py single  # Single CV summary PDF

# Visualize top 3 CV models
python scripts\visualize_cv_top_models.py mix           # Mix top 3 bar charts
python scripts\visualize_cv_top_models.py single        # Single top 3 bar charts
python Mix_Results\analysis_scripts\generate_executive_summary_pdf.py

# Generate executive summary PDF - Single (12 pages, visualizations only)
python Single_Results\analysis_scripts\generate_executive_summary_pdf.py
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

**Executive Summary Report** (`Mix_Results/reports/executive_summary_YYYYMMDD_HHMMSS.pdf`):
- **~12-page concise PDF** designed for presentations
- Professional soft color palette (Nordic-inspired)
- Title page with key findings
- Model comparison visualizations (sorted bar charts, radar plots)
- Top 3 confusion matrices, ROC curves, and PR curves
- SHAP feature importance visualization
- **External validation page**: Training vs validation performance comparison
- Key insights and recommendations
- No data tables - visualizations only for clarity

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
  - 8 interaction te (30-Fold Stratified)
- **Method**: 30-fold StratifiedKFold
- **Purpose**: Assess model stability and generalization with robust estimates
- **Seed**: 42 (baseline seed)
- **Results**: Mean and std deviation in `Mix_Results/cross_validation_results/`
- **Key Findings**:
  - **Mix Dataset**: ExtraTrees (86.53% ± 12.90% balanced accuracy)
  - **Single Dataset**: ExtraTrees (98.75% ± 4.95% balanced accuracy)
  - Higher std on Mix reflects harder classification task
  - Executive summary PDFs available: `executive_summary_cv_*.pdf
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
- **Test Set**: Input_Data-Single-train.csv (Mix models) / Input_Data-Mix-train.csv (Single models)
- **Purpose**: Validate models on independent dataset to assess generalization
- **Key Findings**:
  - **Mix → Single**: Top models achieve 70-75% balanced accuracy (expected drop due to dataset differences)
  - **Single → Mix**: Models maintain 70.12% balanced accuracy on harder Mix classification
  - LogisticRegression (Single) achieves 70.12% on Mix validation, confirming generalization
  - External validation confirms models are not overfitted despite high test accuracy
- **Results**: Documented in `Mix_Results/external_validation_results/` and visualized in executive summary PDFs

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
- Single: 97.50% (near-perfect on held-out single-source profiles)

### Validation Approaches

Three complementary validation strategies were performed:

1. **Train-Test Split (Seed 42)**: Initial model evaluation
2. **Multi-Seed Validation (10 Seeds)**: Robustness to random initialization
3. **30-Fold Cross-Validation (Seed 42)**: Additional stability and variance assessment

### Evaluation Policy Split (Important)

To avoid mixing incomparable settings, the project now reports multi-seed results in two tracks:

1. **Strict 10-seed policy (current default for ranking and reporting)**
  - Uses `--min-seeds 10 --top3-policy strict`
  - Requires all selected top-3 entries to have full 10-seed support
2. **Legacy fallback policy (historical reference only)**
  - Older reports may include model-strategy entries with fewer seeds
  - Useful for audit trail, but not recommended for primary model selection

### Strict 10-Seed Results (Recommended)

These values come from `top_models_report_seed_average/top3_seedavg_mean_summary.json` with strict policy enabled.

#### Mix Dataset (Top-3)

| Rank | Model | Strategy | Balanced Accuracy | Std Dev | Seeds |
|------|-------|----------|-------------------|---------|-------|
| 1 | **TabPFN** | ClassWeights | **0.8473** | ±0.0451 | 10 |
| 2 | **LightGBM** | SMOTE | **0.8020** | ±0.0526 | 10 |
| 3 | **XGBoost** | SMOTE | **0.7895** | ±0.0571 | 10 |

#### Single Dataset (Top-3)

| Rank | Model | Strategy | Balanced Accuracy | Std Dev | Seeds |
|------|-------|----------|-------------------|---------|-------|
| 1 | **Extra Trees** | SMOTE | **0.9861** | ±0.0192 | 10 |
| 2 | **TabPFN** | ClassWeights | **0.9811** | ±0.0236 | 10 |
| 3 | **Logistic Regression** | Original | **0.9691** | ±0.0232 | 10 |

### Legacy Fallback Results (Historical Only)

The following were reported in older summaries and may include low-seed entries.

#### Mix Dataset (Legacy Top-3)

| Rank | Model | Strategy | Balanced Accuracy | Std Dev |
|------|-------|----------|-------------------|---------|
| 1 | **TabPFN** | ClassWeights | **0.8473** | ±0.0451 |
| 2 | **TabPFN** | RandomUnderSampler | **0.8360** | ±0.0471 |
| 3 | **ExtraTrees** | SMOTE | **0.8172** | ±0.0239 |

#### Single Dataset (Legacy Top-3)

| Rank | Model | Strategy | Balanced Accuracy | Std Dev | Notes |
|------|-------|----------|-------------------|---------|-------|
| 1 | **ExtraTrees** | SMOTE | **0.9861** | ±0.0192 | Full 10-seed support |
| 2 | **RandomForest** | Borderline SMOTE | **0.9853** | ±0.0000 | Low seed support (not strict-comparable) |
| 3 | **ExtraTrees** | RandomOverSampler | **0.9841** | ±0.0186 | Full 10-seed support |

### Reproducible Commands

Use the unified CLI:

```bash
# Strict 10-seed (recommended)
python scripts/ml_cli.py run top3-seedavg -- --min-seeds 10 --top3-policy strict --coverage-gate warn --schema-validation warn

# Legacy-style fallback exploration (historical comparison)
python scripts/ml_cli.py run top3-seedavg -- --min-seeds 10 --top3-policy flexible --allow-fallback --coverage-gate warn --schema-validation warn
```

### Key Findings

- **Strict ranking recommendation (Mix)**: TabPFN (ClassWeights)
- **Strict ranking recommendation (Single)**: Extra Trees (SMOTE)
- **Cross-validation confirms stability trends**, with higher variance on Mix than Single due to task complexity
- **Legacy fallback results remain documented** for historical traceability, but strict policy should drive deployment decisions

## Key Features

### Core Documentation
- **OPTIMIZATION_FINAL_SUMMARY.md**: Complete Phase 1 optimization journey
- **PIPELINE_OPTIMIZATION_REPORT.md**: Pipeline optimization details
- **deployment/mix/README.md**: Mix model deployment guide
- **deployment/single/README.md**: Single model deployment guide

### Reports & Visualizations
- **aggregated_top3_report.html**: Multi-seed top 3 models (interactive HTML)
- **top_models_report_seed_average/top3_seedavg_mean_report.html**: Strict seed-average top-3 report
- **Mix_Results/reports/comprehensive_analysis_*.pdf**: 21-page technical report
- **Mix_Results/reports/executive_summary_*.pdf**: Presentation PDFs (train-test)
- **Mix_Results/reports/executive_summary_cv_*.pdf**: CV executive summaries (30-fold)
- **Mix_Results/figures/top3_cv_models.png**: Top 3 CV models bar charts
- **Single_Results/reports/executive_summary_cv_*.pdf**: Single CV summaries
- **Single_Results/figures/top3_cv_models.png**: Single top 3 visualizations

### Analysis Results
- **Mix_Results/multi_seed_aggregated/aggregated_results.json**: Mean ± std across 10 seeds
- **Single_Results/multi_seed_aggregated/aggregated_results.json**: Mean ± std across 10 seeds
- **top_models_report_seed_average/top3_seedavg_mean_summary.json**: Strict policy summary + coverage gate + stats
- **top_models_report_seed_average/top3_seedavg_mean_comparison_mix.json**: Mix pairwise statistics
- **top_models_report_seed_average/top3_seedavg_mean_comparison_single.json**: Single pairwise statistics

### Utility Scripts
- **view_cv_std.py**: View cross-validation standard deviations
- **view_external_validation.py**: View external validation results
- **check_overfitting.py**: Overfitting analysis for Single dataset results
- **scripts/ml_cli.py**: Unified CLI for running core scripts

## Documentation

- **OPTIMIZATION_FINAL_SUMMARY.md**: Complete Phase 1 optimization journey, performance tables, technical decisions
- **deployment/mix/README.md**: Mix model deployment guide with usage examples
- **deployment/single/README.md**: Single model deployment guide
- **Mix_Results/reports/comprehensive_analysis_*.pdf**: 21-page comprehensive analysis report (all details)
- **Mix_Results/reports/executive_summary_*.pdf**: 12-page presentation report (visualizations only, soft colors)
- **Single_Results/reports/comprehensive_analysis_*.pdf**: Single dataset comprehensive report
- **Single_Results/reports/executive_summary_*.pdf**: Single dataset executive summary
- **Mix_Results/reports/combined_report_all_models.html**: Interactive model comparison
- **check_overfitting.py**: Overfitting analysis for Single dataset results

