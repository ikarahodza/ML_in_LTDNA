"""
Canonical per-model colors and display names for all publication figures.
Import this module in every figure-generating script so colors stay consistent.

Usage:
    from model_colors import MODEL_COLORS, MODEL_LABELS, DISPLAY_COLORS
"""

# Internal key -> hex color  (derived from the multi-seed violin plot palette)
MODEL_COLORS = {
    'tabpfn':              '#1A237E',   # deep navy
    'extratrees':          '#283593',   # indigo blue
    'lgbm':                '#1565C0',   # royal blue
    'randomforest':        '#1976D2',   # cornflower blue
    'xgboost':             '#2E7D32',   # forest green
    'logistic_regression': '#388E3C',   # medium green
    'mlp':                 '#F57F17',   # amber
    'svm':                 '#E65100',   # deep orange
    'gnb':                 '#B71C1C',   # dark crimson
}

# Internal key -> display label
MODEL_LABELS = {
    'tabpfn':              'TabPFN',
    'extratrees':          'Extra Trees',
    'lgbm':                'LightGBM',
    'randomforest':        'Random Forest',
    'xgboost':             'XGBoost',
    'logistic_regression': 'LR',
    'mlp':                 'MLP',
    'svm':                 'SVM',
    'gnb':                 'GNB',
}

# Display label -> hex color  (convenience — same data, different key)
DISPLAY_COLORS = {MODEL_LABELS[k]: MODEL_COLORS[k] for k in MODEL_COLORS}
