# Multi-Seed Training Guide

## Overview
All model scripts now support training with multiple random seeds to assess result stability and variance.

## What Changed

### 1. All Model Scripts Updated
All 9 model scripts now accept `--seed` parameter:
- `ExTraTrees.py`
- `gnb.py`
- `logre.py`
- `mlp.py`
- `svm.py`
- `TabPFN.py`
- `XGBoost.py`
- `lightGBM.py`
- `randomForest.py`

### 2. Results Organization
- **Seed 42 (default)**: Results stored in standard locations
  - `Mix_Results/models/`, `Mix_Results/figures/`, etc.
  
- **Other seeds**: Results stored in seed-specific subfolders
  - `Mix_Results/seed_123/models/`
  - `Mix_Results/seed_123/figures/`
  - `Mix_Results/seed_123/reports/`
  - etc.

## Usage

### Single Model, Single Seed
```bash
# Run with default seed (42)
python scripts/ExTraTrees.py mix

# Run with specific seed
python scripts/ExTraTrees.py mix --seed 123
python scripts/gnb.py single --seed 456
```

### All Models, Multiple Seeds (Recommended)
```bash
# Run all models with 10 different seeds (will take a long time!)
python scripts/run_multi_seed_training.py
```

This runs all 9 models × 2 datasets × 10 seeds = **180 training runs**.

Default seeds: `[42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]`

### Aggregate Results Across Seeds
After training completes, aggregate the results:

```bash
# Aggregate both datasets
python scripts/aggregate_multi_seed_results.py

# Aggregate specific dataset
python scripts/aggregate_multi_seed_results.py mix
python scripts/aggregate_multi_seed_results.py single
```

## Output Files

### Per-Seed Results
Each seed generates complete outputs:
- `seed_XXX/models/*.pkl` - Trained models
- `seed_XXX/figures/*.png` - Visualizations
- `seed_XXX/reports/combined_results.json` - Metrics
- `seed_XXX/reports/combined_report_all_models.html` - HTML report

### Aggregated Results
Located in `[Mix|Single]_Results/multi_seed_aggregated/`:

1. **`aggregated_results.json`**
   - Full statistics for all metrics across seeds
   - Structure: `{model: {strategy: {metric: {mean, std, min, max, n_seeds}}}}`

2. **`aggregated_results.csv`**
   - Tabular format for easy analysis
   - Columns: model, strategy, metric, mean, std, min, max, n_seeds

3. **`aggregated_results_summary.txt`**
   - Human-readable summary
   - Shows mean ± std for all metrics
   - Top 10 performers by balanced accuracy

## Interpreting Results

### Standard Deviation (Stability)
- **Low std**: Model is stable across seeds → more reliable
- **High std**: Model is sensitive to initialization → less reliable

### Example Output
```
Model: extratrees | Strategy: SMOTE
balanced_accuracy: 0.8532 ± 0.0123 (min: 0.8401, max: 0.8687, n=10)
f1_macro:         0.8456 ± 0.0145 (min: 0.8295, max: 0.8602, n=10)
```
This shows consistent performance (low std ~0.01-0.02 is good).

## Configuration

### Customize Seeds
Edit `scripts/run_multi_seed_training.py`:
```python
SEEDS = [42, 100, 200, 300, 400]  # Use fewer seeds
```

### Skip Models
Edit the `MODEL_SCRIPTS` list to skip certain models:
```python
MODEL_SCRIPTS = [
    'ExTraTrees.py',
    'XGBoost.py',
    # 'TabPFN.py',  # Skip TabPFN
]
```

## Time Estimation

Approximate times (varies by hardware):
- Single model: 2-5 minutes
- All 9 models, 1 seed: 20-45 minutes
- All 9 models, 10 seeds: **3-8 hours**

## Best Practices

1. **Test with 2-3 seeds first**
   ```bash
   python scripts/ExTraTrees.py mix --seed 42
   python scripts/ExTraTrees.py mix --seed 123
   python scripts/ExTraTrees.py mix --seed 456
   ```

2. **Run overnight** for full 10-seed training

3. **Compare stability**: Models with low std are more production-ready

4. **Report both**: 
   - Mean performance (expected value)
   - Std (confidence/reliability)

## Troubleshooting

### "Results not found for seed X"
- Training for that seed failed or hasn't run yet
- Check `multi_seed_training_log_*.txt` for details

### Memory issues
- Reduce number of seeds
- Run datasets separately
- Process models one at a time

### Disk space
- Each seed requires ~100-500 MB
- 10 seeds × 2 datasets ≈ 2-10 GB total

## Next Steps

After aggregation:
1. Review `aggregated_results_summary.txt` for top performers
2. Check std values - prefer models with low variance
3. Use mean ± std in your papers/reports
4. Select most stable model for deployment
