import json
import numpy as np

# Load results
with open('Single_Results/reports/combined_results.json', 'r') as f:
    results = json.load(f)

lr = results['logistic_regression']['results']['Original']
cm = np.array(lr['confusion_matrix'])
classes = ['Allele', 'Back_Stutter', 'Forward_Stutter', 'Pull_Up']

print('\n' + '='*70)
print('OVERFITTING ANALYSIS - Single Dataset')
print('='*70)

print('\n1. TEST SET SIZE CONCERN:')
print(f'   Total dataset: 200 samples')
print(f'   Train/Test split: 160/40 (80/20)')
print(f'   Test set: ONLY 40 SAMPLES → Very high variance!')
print(f'   ⚠️  With such a small test set, 97.50% = 39/40 correct')
print(f'   ⚠️  Just 1 more error = 95.00%, 2 errors = 92.50%')

print('\n2. LOGISTIC REGRESSION CONFUSION MATRIX (97.50%):')
print('\n   Predicted →')
print('   True ↓          ', ' '.join(f'{c[:12]:>12}' for c in classes))
print('   ' + '-'*65)
for i, true_class in enumerate(classes):
    row = ' '.join(f'{int(cm[i][j]):12}' for j in range(4))
    print(f'   {true_class[:12]:15} {row}')

print(f'\n   Total test samples: 40')
print(f'   Correct: {int(np.trace(cm))}')
print(f'   Errors: {40 - int(np.trace(cm))} (only 1 mistake!)')

print('\n3. PER-CLASS TEST SET SIZES:')
support = lr['per_class_support']
for i, c in enumerate(classes):
    print(f'   {c}: {int(support[i])} test samples')

print('\n4. EXTERNAL VALIDATION CHECK:')
try:
    with open('Single_Results/external_validation_results/external_validation_results.json', 'r') as f:
        ext_val = json.load(f)
    
    if 'model_logistic_regression_original' in ext_val:
        ext_lr = ext_val['model_logistic_regression_original']
        print(f'   Validated on Mix dataset (426 samples):')
        print(f'   - Balanced Accuracy: {ext_lr["balanced_accuracy"]*100:.2f}%')
        print(f'   - Test samples: 86 (much larger!)')
        print(f'   ✓ This is a better reliability indicator!')
except:
    print('   External validation not available')

print('\n5. VERDICTL: IS 97.50% RELIABLE?')
print('   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')
print('   ⚠️  CAUTION: Test set too small (40 samples)')
print('   ✓ GOOD: Simple model (LogReg) less prone to overfitting')
print('   ✓ GOOD: External validation provides better estimate')
print('   ⚠️  CONCERN: Missing cross-validation scores')
print('')
print('   RECOMMENDATION:')
print('   → Report external validation score (more reliable)')
print('   → Mention small test set as limitation')
print('   → Use 5-fold CV score as primary metric')
print('   → Collect more Single dataset samples if possible')

print('\n6. COMPARISON WITH MIX DATASET:')
print('   Mix dataset:')
print(f'   - Total: 426 samples')
print(f'   - Test set: ~86 samples (2x larger)')
print(f'   - Best model: 88.10% (Stacking Ensemble)')
print(f'   → More reliable estimate due to larger test set')

print('\n' + '='*70)
print('CONCLUSION: Single results are likely REAL but UNCERTAIN')
print('The small test set (40 samples) means high variance.')
print('External validation + simple model = more trustworthy.')
print('='*70 + '\n')
