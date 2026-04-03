import glob
import re
import os

def extract_versions_from_pickle(filepath):
    try:
        with open(filepath, 'rb') as f:
            data = f.read()
            
        versions = {}
        
        # Look for sklearn version (usually looks like: _sklearn_version.\x051.x.x)
        sklearn_match = re.search(b'sklearn_versionq.\\x8c.([0-9]+\.[0-9]+\.[0-9a-z]+)', data)
        if not sklearn_match:
            sklearn_match = re.search(b'_sklearn_version[^0-9]*([0-9]+\.[0-9]+\.[0-9a-z]+)', data)
        if sklearn_match:
            versions['sklearn'] = sklearn_match.group(1).decode('ascii', errors='ignore')
            
        # Look for numpy version
        np_match = re.search(b'numpy[^A-Za-z0-9]*([0-9]+\.[0-9]+\.[0-9a-z]+)', data)
        if np_match:
            versions['numpy_hint'] = np_match.group(1).decode('ascii', errors='ignore')
            
        # Common classes hinting at versions
        if b'numpy.random._mt19937' in data:
            versions['numpy_generator'] = '>=1.17 (PCG64/MT19937)'
            
        if b'numpy.core.multiarray' in data:
            versions['numpy_array'] = '<2.0 style'
        if b'numpy._core.multiarray' in data:
            versions['numpy_array'] = '>=2.0 style'
            
        # XGBoost version
        xgb_match = re.search(b'xgboost[^0-9]*([0-9]+\.[0-9]+\.[0-9]+)', data)
        if xgb_match:
             versions['xgboost'] = xgb_match.group(1).decode('ascii', errors='ignore')

        # Check Python pickle protocol roughly
        protocol = data[1] if len(data) > 2 and data[0] == 128 else "Unknown/Joblib"
        versions['pickle_protocol'] = protocol
            
        return versions
    except Exception as e:
        return f"Error: {e}"

# Check a mix of models
test_files = [
    "Mix_Results/models/model_xgboost_smote.pkl",         # Works
    "Mix_Results/models/model_mlp_original.pkl",          # Failed
    "Mix_Results/models/model_randomforest_smote.pkl",    # Failed
    "Mix_Results/models_v2/model_tabpfn_smote_v2.pkl",    # Failed
    "Single_Results/models/model_logistic_regression_smote.pkl", # Failed
]

for tf in test_files:
    if os.path.exists(tf):
        print(f"\nAnalyzing: {os.path.basename(tf)}")
        creation_date = os.path.getmtime(tf)
        print(f"Mod Date:  {creation_date}")
        res = extract_versions_from_pickle(tf)
        print(f"Findings:  {res}")
    else:
        print(f"\nMissing: {tf}")
