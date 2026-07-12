# Mixed Training Data Testing Results
## Summary

Tested all trained models on Mixed_Training_Data.csv (381 samples, 305 features)

### Test Data Distribution:
- Allele: 192 samples
- Back_Stutter: 83 samples
- Forward_Stutter: 35 samples
- Pull_Up: 71 samples

---

## Results: 62 Models Successfully Tested

## Top 10 Models by Balanced Accuracy

### 1. Mix Stacking Ensemble + Ensemble (vv1)
- **Accuracy**: 97.11%
- **Balanced Accuracy**: 97.47%
- **F1 Macro**: 97.02%
- **Log Loss**: 0.1884
- **Matthews Correlation**: 0.9564
- **Cohen Kappa**: 0.9562

### 2. Mix Extratrees + Smote (vv1)
- **Accuracy**: 97.11%
- **Balanced Accuracy**: 95.89%
- **F1 Macro**: 96.28%
- **Log Loss**: 0.1132
- **Matthews Correlation**: 0.9561
- **Cohen Kappa**: 0.9560

### 3. Mix Xgboost + Smote (vv1)
- **Accuracy**: 96.33%
- **Balanced Accuracy**: 95.74%
- **F1 Macro**: 95.96%
- **Log Loss**: 0.1267
- **Matthews Correlation**: 0.9443
- **Cohen Kappa**: 0.9441

### 4. Mix Tabpfn + Classweights (vv2)
- **Accuracy**: 97.38%
- **Balanced Accuracy**: 95.68%
- **F1 Macro**: 96.67%
- **Log Loss**: 0.1262
- **Matthews Correlation**: 0.9601
- **Cohen Kappa**: 0.9597

### 5. Mix Tabpfn + Original (vv2)
- **Accuracy**: 97.38%
- **Balanced Accuracy**: 95.68%
- **F1 Macro**: 96.67%
- **Log Loss**: 0.1262
- **Matthews Correlation**: 0.9601
- **Cohen Kappa**: 0.9597

### 6. Mix Lgbm + Smote (vv1)
- **Accuracy**: 96.33%
- **Balanced Accuracy**: 95.33%
- **F1 Macro**: 95.80%
- **Log Loss**: 0.1274
- **Matthews Correlation**: 0.9442
- **Cohen Kappa**: 0.9440

### 7. Mix Extratrees + Original (vv1)
- **Accuracy**: 97.11%
- **Balanced Accuracy**: 95.08%
- **F1 Macro**: 96.27%
- **Log Loss**: 0.1116
- **Matthews Correlation**: 0.9559
- **Cohen Kappa**: 0.9557

### 8. Mix Lgbm + Randomoversampler (vv1)
- **Accuracy**: 95.80%
- **Balanced Accuracy**: 95.07%
- **F1 Macro**: 95.33%
- **Log Loss**: 0.1396
- **Matthews Correlation**: 0.9366
- **Cohen Kappa**: 0.9362

### 9. Mix Extratrees + Randomoversampler (vv1)
- **Accuracy**: 96.85%
- **Balanced Accuracy**: 94.78%
- **F1 Macro**: 96.07%
- **Log Loss**: 0.1247
- **Matthews Correlation**: 0.9519
- **Cohen Kappa**: 0.9516

### 10. Mix Xgboost + Randomoversampler (vv1)
- **Accuracy**: 95.01%
- **Balanced Accuracy**: 94.28%
- **F1 Macro**: 94.35%
- **Log Loss**: 0.1394
- **Matthews Correlation**: 0.9247
- **Cohen Kappa**: 0.9242

---

## TabPFN v2 Results

| Dataset | Strategy | Accuracy | Balanced Acc | F1 Macro | Log Loss | MCC |
|---------|----------|----------|--------------|----------|----------|-----|
| Mix | classweights | 97.38% | 95.68% | 96.67% | 0.1262 | 0.9601 |
| Mix | original | 97.38% | 95.68% | 96.67% | 0.1262 | 0.9601 |
| Mix | smote | 96.33% | 92.55% | 94.61% | 0.1039 | 0.9440 |
| Mix | randomundersampler | 85.83% | 89.39% | 85.14% | 0.4376 | 0.7985 |
| Single | original | 75.33% | 82.77% | 75.62% | 0.7817 | 0.6802 |
| Single | classweights | 75.33% | 82.77% | 75.62% | 0.7817 | 0.6802 |
| Single | smote | 76.90% | 82.53% | 77.21% | 0.7200 | 0.6981 |
| Single | randomundersampler | 75.85% | 81.19% | 75.59% | 0.7850 | 0.6794 |

---

## All Model Results

| Dataset | Model | Strategy | Ver | Accuracy | Balanced Acc | F1 Macro | Log Loss | MCC | Kappa |
|---------|-------|----------|-----|----------|--------------|----------|----------|-----|-------|
| Mix | stacking_ensemble | ensemble | vv1 | 97.11% | 97.47% | 97.02% | 0.1884 | 0.9564 | 0.9562 |
| Mix | extratrees | smote | vv1 | 97.11% | 95.89% | 96.28% | 0.1132 | 0.9561 | 0.9560 |
| Mix | xgboost | smote | vv1 | 96.33% | 95.74% | 95.96% | 0.1267 | 0.9443 | 0.9441 |
| Mix | tabpfn | classweights | vv2 | 97.38% | 95.68% | 96.67% | 0.1262 | 0.9601 | 0.9597 |
| Mix | tabpfn | original | vv2 | 97.38% | 95.68% | 96.67% | 0.1262 | 0.9601 | 0.9597 |
| Mix | lgbm | smote | vv1 | 96.33% | 95.33% | 95.80% | 0.1274 | 0.9442 | 0.9440 |
| Mix | extratrees | original | vv1 | 97.11% | 95.08% | 96.27% | 0.1116 | 0.9559 | 0.9557 |
| Mix | lgbm | randomoversampler | vv1 | 95.80% | 95.07% | 95.33% | 0.1396 | 0.9366 | 0.9362 |
| Mix | extratrees | randomoversampler | vv1 | 96.85% | 94.78% | 96.07% | 0.1247 | 0.9519 | 0.9516 |
| Mix | xgboost | randomoversampler | vv1 | 95.01% | 94.28% | 94.35% | 0.1394 | 0.9247 | 0.9242 |
| Mix | randomforest | smote | vv1 | 95.28% | 94.19% | 94.13% | 0.2902 | 0.9280 | 0.9279 |
| Mix | lgbm | original | vv1 | 94.75% | 93.64% | 93.37% | 0.1577 | 0.9203 | 0.9201 |
| Mix | randomforest | borderline_smote | vv1 | 95.54% | 93.15% | 94.30% | 0.2478 | 0.9317 | 0.9313 |
| Mix | tabpfn | smote | vv2 | 96.33% | 92.55% | 94.61% | 0.1039 | 0.9440 | 0.9431 |
| Mix | xgboost | original | vv1 | 94.23% | 92.01% | 92.75% | 0.2025 | 0.9117 | 0.9112 |
| Mix | randomforest | svm_smote | vv1 | 94.49% | 91.71% | 93.13% | 0.2488 | 0.9156 | 0.9149 |
| Mix | tabpfn | randomundersampler | vv2 | 85.83% | 89.39% | 85.14% | 0.4376 | 0.7985 | 0.7926 |
| Mix | logistic_regression | smote | vv1 | 90.29% | 88.62% | 87.62% | 0.6241 | 0.8534 | 0.8530 |
| Mix | randomforest | no_resampling | vv1 | 93.44% | 88.41% | 90.99% | 0.3055 | 0.8996 | 0.8975 |
| Mix | logistic_regression | randomoversampler | vv1 | 90.29% | 88.03% | 87.15% | 0.6041 | 0.8535 | 0.8530 |
| Mix | randomforest | class_weights | vv1 | 87.93% | 86.76% | 85.58% | 0.5341 | 0.8204 | 0.8185 |
| Mix | mlp | smote | vv1 | 79.00% | 85.73% | 78.13% | 0.5090 | 0.7271 | 0.7054 |
| Mix | mlp | randomoversampler | vv1 | 77.69% | 85.30% | 76.71% | 0.5909 | 0.7147 | 0.6899 |
| Mix | mlp | original | vv1 | 90.03% | 84.83% | 87.21% | 0.3614 | 0.8464 | 0.8446 |
| Mix | logistic_regression | original | vv1 | 89.50% | 83.94% | 85.62% | 0.4665 | 0.8383 | 0.8366 |
| Mix | svm | randomoversampler | vv1 | 85.04% | 83.53% | 82.91% | 0.5518 | 0.7722 | 0.7719 |
| Single | tabpfn | original | vv2 | 75.33% | 82.77% | 75.62% | 0.7817 | 0.6802 | 0.6548 |
| Single | tabpfn | classweights | vv2 | 75.33% | 82.77% | 75.62% | 0.7817 | 0.6802 | 0.6548 |
| Single | logistic_regression | smote | vv1 | 81.36% | 82.75% | 78.82% | 1.1639 | 0.7326 | 0.7271 |
| Single | tabpfn | smote | vv2 | 76.90% | 82.53% | 77.21% | 0.7200 | 0.6981 | 0.6738 |
| Single | logistic_regression | randomoversampler | vv1 | 81.36% | 82.53% | 78.61% | 1.1599 | 0.7313 | 0.7267 |
| Mix | svm | smote | vv1 | 78.74% | 81.58% | 76.69% | 0.5452 | 0.6992 | 0.6915 |
| Single | logistic_regression | original | vv1 | 81.63% | 81.32% | 78.78% | 1.1187 | 0.7292 | 0.7269 |
| Single | tabpfn | randomundersampler | vv2 | 75.85% | 81.19% | 75.59% | 0.7850 | 0.6794 | 0.6583 |
| Single | extratrees | smote | vv1 | 71.39% | 80.57% | 71.93% | 1.1947 | 0.6382 | 0.6067 |
| Single | extratrees | randomoversampler | vv1 | 72.18% | 79.76% | 72.21% | 0.9525 | 0.6415 | 0.6141 |
| Single | extratrees | original | vv1 | 72.44% | 79.48% | 72.79% | 1.2389 | 0.6427 | 0.6162 |
| Mix | svm | original | vv1 | 82.94% | 76.90% | 80.48% | 0.5233 | 0.7347 | 0.7285 |
| Single | randomforest | no_resampling | vv1 | 67.19% | 76.79% | 68.13% | 4.3212 | 0.5901 | 0.5542 |
| Single | lgbm | original | vv1 | 66.14% | 76.57% | 66.53% | 1.6004 | 0.5775 | 0.5423 |
| Single | mlp | smote | vv1 | 70.87% | 75.66% | 68.86% | 1.0296 | 0.6135 | 0.5939 |
| Single | stacking_ensemble | ensemble | vv1 | 65.35% | 75.45% | 66.15% | 1.2672 | 0.5720 | 0.5325 |
| Single | lgbm | smote | vv1 | 66.40% | 75.39% | 66.30% | 2.3663 | 0.5847 | 0.5462 |
| Single | randomforest | borderline_smote | vv1 | 64.57% | 74.96% | 65.06% | 1.0633 | 0.5615 | 0.5235 |
| Single | mlp | randomoversampler | vv1 | 69.55% | 74.79% | 67.75% | 1.0401 | 0.6016 | 0.5787 |
| Mix | gaussiannb | original | vv1 | 77.17% | 74.36% | 74.62% | 8.2304 | 0.6612 | 0.6506 |
| Single | xgboost | smote | vv1 | 64.04% | 74.14% | 64.55% | 1.5387 | 0.5546 | 0.5157 |
| Single | xgboost | original | vv1 | 65.09% | 74.10% | 65.36% | 1.4009 | 0.5684 | 0.5283 |
| Single | svm | smote | vv1 | 64.57% | 74.06% | 63.94% | 0.7770 | 0.5651 | 0.5250 |
| Single | randomforest | svm_smote | vv1 | 63.52% | 73.93% | 64.02% | 1.5958 | 0.5437 | 0.5090 |
| Single | xgboost | randomoversampler | vv1 | 64.04% | 73.45% | 64.54% | 1.5258 | 0.5506 | 0.5140 |
| Single | lgbm | randomoversampler | vv1 | 64.30% | 73.37% | 64.45% | 2.6114 | 0.5557 | 0.5182 |
| Single | svm | randomoversampler | vv1 | 63.52% | 72.56% | 62.55% | 0.7824 | 0.5475 | 0.5109 |
| Single | svm | original | vv1 | 67.98% | 72.24% | 65.77% | 0.6835 | 0.5800 | 0.5552 |
| Mix | gaussiannb | randomoversampler | vv1 | 73.49% | 70.93% | 70.39% | 9.5549 | 0.6098 | 0.5964 |
| Single | randomforest | smote | vv1 | 61.68% | 70.69% | 62.64% | 1.5074 | 0.5146 | 0.4812 |
| Single | randomforest | class_weights | vv1 | 63.25% | 70.57% | 63.62% | 4.2258 | 0.5304 | 0.4985 |
| Single | gaussiannb | randomoversampler | vv1 | 75.33% | 70.37% | 71.80% | 8.8927 | 0.6165 | 0.6145 |
| Single | gaussiannb | original | vv1 | 73.75% | 70.07% | 70.58% | 9.4603 | 0.5963 | 0.5958 |
| Single | gaussiannb | smote | vv1 | 75.07% | 69.85% | 71.88% | 8.9873 | 0.6106 | 0.6075 |
| Single | mlp | original | vv1 | 68.50% | 68.67% | 64.50% | 1.0556 | 0.5672 | 0.5522 |
| Mix | gaussiannb | smote | vv1 | 76.12% | 66.14% | 68.78% | 8.6089 | 0.6336 | 0.6085 |

---

## Best Model Per Dataset

### Mix Dataset
- **Best Model**: stacking_ensemble + ensemble (vv1)
- **Accuracy**: 97.11%
- **Balanced Accuracy**: 97.47%
- **F1 Macro**: 97.02%
- **Log Loss**: 0.1884

### Single Dataset
- **Best Model**: tabpfn + original (vv2)
- **Accuracy**: 75.33%
- **Balanced Accuracy**: 82.77%
- **F1 Macro**: 75.62%
- **Log Loss**: 0.7817

---

## File Index

- `summary_results.csv` — All model metrics (sorted by balanced accuracy)
- `detailed_results.json` — Full results including per-class metrics and confusion matrices
- `detailed_per_class_results.csv` — Per-class precision/recall/F1 breakdown