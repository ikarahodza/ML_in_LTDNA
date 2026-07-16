"""Quick check that the TabPFN API key loads and the model runs."""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
key_file = PROJECT_ROOT / 'tabpfn.key'

if not key_file.exists():
    print(f"FAIL: tabpfn.key not found at {key_file}")
    sys.exit(1)

os.environ.setdefault('TABPFN_TOKEN', key_file.read_text().strip())
os.environ['TABPFN_NO_BROWSER'] = '1'

import numpy as np
from tabpfn import TabPFNClassifier

X_train = np.random.default_rng(0).random((50, 10))
y_train = (X_train[:, 0] > 0.5).astype(int)
X_test  = np.random.default_rng(1).random((10, 10))

print("Fitting TabPFN on 50 dummy samples...")
clf = TabPFNClassifier()
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
print(f"Predictions: {preds.tolist()}")
print("OK: key works and TabPFN ran successfully.")
