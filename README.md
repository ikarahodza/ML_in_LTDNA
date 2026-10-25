# Transformer-based in-context learning outperforms traditional classifiers for peak classification in low-template DNA mixture electropherograms

## Key Results

| Dataset | Best Model | Balanced Accuracy (10 seeds) |
|---|---|---|
| Mix (two-person mixtures) | TabPFN-3.0 (Class Weights) | **0.890 ± 0.051** |
| Single-source | Logistic Regression (RandomOverSampler) | **0.981 ± 0.016** |

TabPFN-3.0 achieves a 21.1% relative improvement over the best Tan et al. (2025) baseline (MLP: 0.735) on the Mix dataset without any hyperparameter tuning.

Results above include the PoI (Point of Interest) feature — see [PoI feature](#poi-feature) below.

---

## Dataset

426 labelled peaks from capillary electrophoresis profiles of 11 verified two-person DNA mixtures (Tan et al., 2025):

| Class | n | % |
|---|---|---|
| Allele | 223 | 52.3% |
| Back Stutter | 89 | 20.9% |
| Pull-Up | 81 | 19.0% |
| Forward Stutter | 33 | 7.7% |

305 peak-level features per signal (306 with the PoI feature included — see below). Stratified 80:20 splits, 10 random seeds (42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021).

---

## PoI feature

`PoI` (Point of Interest) is an additional column in the raw input CSVs identifying the specific peak location flagged in the electropherogram signal. It is excluded from the feature set by default; pass `--include-poi` to any training script to keep it as a 306th predictive feature. Results in this README (Key Results table, `Mix_Results_PoI/`, `Single_Results_PoI/`) are from PoI-included runs, which currently supersede the earlier PoI-excluded runs.

Including PoI improves the Mix dataset result (TabPFN: 0.847 → 0.890) and changes the best Single-source model (Extra Trees → Logistic Regression), though it also caused SVM to collapse to chance-level balanced accuracy (0.25) on the Single-source PoI runs — worth investigating before relying on that combination.

---

## Models

Nine classifiers evaluated across six class-imbalance strategies:

**Tree ensemble / gradient boosting**
- Extra Trees (`scripts/ExTraTrees.py`)
- LightGBM (`scripts/lightGBM.py`)
- XGBoost (`scripts/XGBoost.py`)
- Random Forest (`scripts/randomForest.py`)

**Transformer**
- TabPFN-3.0 (`scripts/TabPFN.py`) — prior-data fitted network, in-context inference

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
│   ├── TabPFN.py                          # TabPFN-3.0
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
│   ├── run_tabpfn_all_seeds.py            # Run TabPFN-3.0 across 10 seeds (own script; --include-poi supported)
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
├── Mix_Results_PoI/                       # Mix dataset results, PoI feature included (current/primary)
│   ├── data/                              # Seed-42 predictions + error-analysis CSVs (all models)
│   ├── figures/                           # Publication figures (Figures 1–7)
│   ├── models/                            # Trained model pickles (seed 42)
│   ├── reports/
│   │   ├── combined_results.json          # Seed-42 results, non-TabPFN models
│   │   ├── combined_results_v3.json       # Seed-42 results, TabPFN-3.0
│   │   └── combined_report_all_models*.html
│   ├── multi_seed_aggregated/
│   │   └── aggregated_results.json        # Mean ± SD across 10 seeds (non-TabPFN models)
│   ├── seed_123/ … seed_2021/             # Per-seed results, non-TabPFN models (seeds 2–10)
│   ├── seed_123_v3/ … seed_2021_v3/       # Per-seed results, TabPFN-3.0 (seeds 2–10)
│   ├── shap_explainability/               # SHAP bar plots and CSVs
│   ├── permutation_importance/            # Permutation importance (all models)
│   ├── tabpfn_prototypes_and_permutation/ # TabPFN prototype + permutation importance
│   └── dice_explainability/               # DiCE aggregated feature changes
│
├── Single_Results_PoI/                    # Same structure — single-source dataset, PoI included
│
├── Input_Data-Mix-train.csv               # Mix dataset (426 signals, 305 features + PoI)
├── Input_Data-Single-train.csv            # Single-source dataset (200 signals)
└── requirements_pinned.txt               # Pinned Python dependencies
```

> The PoI-excluded result trees (`Mix_Results/`, `Single_Results/`) have been deleted from the working copy — the `_PoI` trees above are now the only results tree. Omit `--include-poi` on any training script to regenerate the 305-feature (non-PoI) trees if you need them again; their last committed snapshot also remains in git history. BibTeX references for the manuscript (`cas-refs.bib`) live in the separate, gitignored `paper/` working copy, not in this repository tree.

---

## Requirements

```
python >= 3.11
scikit-learn==1.3.4
tabpfn==0.2.8
xgboost==2.0.3
lightgbm==3.4.2
imbalanced-learn==0.11.2
pandas==2.2.2
numpy==1.25.2
matplotlib==3.8.1
joblib==1.3.2
jsonschema==4.23.0
```
See `requirements_pinned.txt` for the exact pinned set (also includes `shap` and `dice-ml` for the explainability scripts).

Install:
```bash
pip install -r requirements_pinned.txt
```

**TabPFN API key**: Required for TabPFN inference. Create a `tabpfn.key` file in the
repository root containing your Prior Labs API token (nothing else):
```bash
echo "your_token_here" > tabpfn.key
```
Scripts that use TabPFN read this file automatically and set `TABPFN_TOKEN` for you.
`tabpfn.key` is listed in `.gitignore` — never commit it.

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

Replace `mix` with `single` for the single-source dataset. Add `--include-poi` to any script to keep the PoI feature and write to `<Dataset>_Results_PoI/` instead of `<Dataset>_Results/` (see [PoI feature](#poi-feature)).

### Run multi-seed training (10 seeds, all models)
```bash
python scripts/run_multi_seed_training.py mix --include-poi
python scripts/aggregate_multi_seed_results.py --include-poi
python scripts/run_tabpfn_all_seeds.py mix --include-poi   # TabPFN-3.0 runs separately
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
