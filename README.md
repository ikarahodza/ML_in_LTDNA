# Transformer-based in-context learning outperforms traditional classifiers for peak classification in low-template DNA mixture electropherograms

**Ilma Karahodža** — University of Sarajevo

Companion code for the FSI:Genetics article extending Tan et al. (2025) to two-person LTDNA mixture electropherograms using TabPFN-2.5 and eight additional classifiers.

---

## Key Results

| Dataset | Best Model | Balanced Accuracy (10 seeds) |
|---|---|---|
| Mix (two-person mixtures) | TabPFN-2.5 (Class Weights) | **0.847 ± 0.045** |
| Single-source | Extra Trees (SMOTE) | **0.986 ± 0.019** |

TabPFN-2.5 achieves a 15.3% relative improvement over the best Tan et al. (2025) baseline (MLP: 0.735) on the Mix dataset without any hyperparameter tuning.

---

## Dataset

426 labelled peaks from capillary electrophoresis profiles of 11 verified two-person DNA mixtures (Tan et al., 2025):

| Class | n | % |
|---|---|---|
| Allele | 223 | 52.3% |
| Back Stutter | 89 | 20.9% |
| Pull-Up | 81 | 19.0% |
| Forward Stutter | 33 | 7.7% |

305 peak-level features per signal. Stratified 80:20 splits, 10 random seeds (42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021).

---

## Models

Nine classifiers evaluated across six class-imbalance strategies:

**Tree ensemble / gradient boosting**
- Extra Trees (`scripts/ExTraTrees.py`)
- LightGBM (`scripts/lightGBM.py`)
- XGBoost (`scripts/XGBoost.py`)
- Random Forest (`scripts/randomForest.py`)

**Transformer**
- TabPFN-2.5 (`scripts/TabPFN.py`) — prior-data fitted network, in-context inference

**Traditional baselines (Tan et al. reproduction)**
- Support Vector Machine (`scripts/svm.py`)
- Multilayer Perceptron (`scripts/mlp.py`)
- Logistic Regression (`scripts/logre.py`)
- Gaussian Naive Bayes (`scripts/gnb.py`)

**Resampling strategies**: Original, Class Weights, SMOTE, Borderline-SMOTE, RandomOverSampler, RandomUnderSampler

---

## Explainability

- **SHAP** — TreeExplainer (ET, RF), native XGBoost, LinearExplainer (LR), KernelExplainer (SVM, MLP, GNB, TabPFN)
- **DiCE** — diverse counterfactual explanations for all models
- **Permutation importance** — TabPFN (top SHAP features)
- **Prototype analysis** — nearest training instances for TabPFN predictions

---

## Project Structure

```
ML_in_LTDNA/
├── scripts/
│   ├── ExTraTrees.py                      # Extra Trees
│   ├── TabPFN.py                          # TabPFN-2.5
│   ├── XGBoost.py                         # XGBoost
│   ├── lightGBM.py                        # LightGBM
│   ├── randomForest.py                    # Random Forest
│   ├── svm.py                             # SVM
│   ├── mlp.py                             # MLP
│   ├── logre.py                           # Logistic Regression
│   ├── gnb.py                             # Gaussian Naive Bayes
│   ├── helper_functions.py                # Shared utilities
│   ├── model_colors.py                    # Consistent colour scheme for figures
│   ├── MLtoPeak-STEP1.py                  # Data preprocessing
│   ├── run_multi_seed_training.py         # Run all models across 10 seeds
│   ├── aggregate_multi_seed_results.py    # Aggregate seed results → JSON
│   ├── generate_model_comparison_figures.py  # Figure 2 and Figure 3
│   ├── generate_multiseed_violin_plot.py  # Figure 1
│   ├── generate_pr_curves.py              # Figure 5
│   ├── redraw_publication_figures.py      # Figure 4 and Figure 6
│   ├── compute_tabpfn_shap.py             # SHAP for TabPFN
│   ├── shap_comprehensive_analysis.py     # SHAP for tree models
│   ├── dice_explainability.py             # DiCE counterfactuals
│   └── tabpfn_prototypes_and_permutation.py  # Permutation importance + prototypes
│
├── Mix_Results/
│   ├── data/                              # Seed-42 predictions CSVs (all models)
│   ├── figures/                           # Publication figures (Figures 1–7)
│   ├── models/                            # Trained model pickles (seed 42)
│   ├── reports/
│   │   ├── combined_results.json          # Seed-42 per-model results
│   │   └── combined_report_all_models.html
│   ├── multi_seed_aggregated/
│   │   └── aggregated_results.json        # Mean ± SD across 10 seeds
│   ├── seed_123/ … seed_2021/             # Per-seed results (seeds 2–10)
│   ├── shap_explainability/               # SHAP bar plots and CSVs
│   ├── tabpfn_prototypes_and_permutation/ # Permutation importance
│   ├── dice_explainability/               # DiCE aggregated feature changes
│   └── cross_validation_results/          # 30-fold CV (secondary)
│
├── Single_Results/                        # Same structure — single-source dataset
│
├── Input_Data-Mix-train.csv               # Mix dataset (426 signals, 305 features)
├── Input_Data-Single-train.csv            # Single-source dataset (200 signals)
├── cas-refs.bib                           # BibTeX references for the paper
└── requirements_pinned.txt               # Pinned Python dependencies
```

---

## Requirements

```
python >= 3.11
scikit-learn >= 1.6.1
tabpfn >= 6.4.1
xgboost
lightgbm
imbalanced-learn
pandas
numpy
matplotlib
seaborn
shap
dice-ml
joblib
```

Install:
```bash
pip install -r requirements_pinned.txt
```

**TabPFN API key**: Required for TabPFN inference. Set via environment variable:
```bash
export TABPFN_API_KEY=your_key_here
```

Do not commit your API key to the repository.

---

## Usage

### Run a single model (seed 42, Mix dataset)
```bash
python scripts/ExTraTrees.py mix
python scripts/TabPFN.py mix
python scripts/XGBoost.py mix
python scripts/lightGBM.py mix
python scripts/randomForest.py mix
python scripts/svm.py mix
python scripts/mlp.py mix
python scripts/logre.py mix
python scripts/gnb.py mix
```

Replace `mix` with `single` for the single-source dataset.

### Run multi-seed training (10 seeds, all models)
```bash
python scripts/run_multi_seed_training.py mix
python scripts/aggregate_multi_seed_results.py
```

### Regenerate publication figures
```bash
python scripts/generate_multiseed_violin_plot.py       # Figure 1
python scripts/generate_model_comparison_figures.py    # Figure 2, 3
python scripts/redraw_publication_figures.py           # Figure 4, 6
python scripts/generate_pr_curves.py                   # Figure 5
```

### Run explainability
```bash
python scripts/shap_comprehensive_analysis.py mix
python scripts/compute_tabpfn_shap.py
python scripts/dice_explainability.py mix
python scripts/tabpfn_prototypes_and_permutation.py
```

---

## Citation
Underlying dataset:

> Tan, M., et al. (2025). Explainable artificial intelligence in forensic DNA analysis: Alleles identification in challenging electropherograms using supervised machine learning methods. *Forensic Science International: Genetics*, 78, 103289. https://doi.org/10.1016/j.fsigen.2025.103289
