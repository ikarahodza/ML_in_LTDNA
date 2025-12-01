"""Generate combined HTML report from `reports/combined_results.json`.

Usage:
    python scripts/generate_combined_html_runner.py

This script imports `scripts.helper_functions.generate_combined_html` and
generates `reports/combined_report_all_models.html` (fallback generator
inside the helper will be used if Jinja2/template is missing).
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_PATH = os.path.join(ROOT, 'reports', 'combined_results.json')
OUT_PATH = os.path.join(ROOT, 'reports', 'combined_report_all_models.html')

def main(json_path=JSON_PATH, out_path=OUT_PATH):
    try:
        # ensure `scripts` is importable (add its parent to sys.path)
        scripts_dir = os.path.join(ROOT, 'scripts')
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        import helper_functions as hf
    except Exception as e:
        print('Failed to import helper_functions from scripts:', e)
        sys.exit(2)

    if not os.path.exists(json_path):
        print(f'No combined JSON found at {json_path}')
        sys.exit(1)

    result = hf.generate_combined_html(json_path=json_path, output_path=out_path)
    if result:
        print(f'Generated combined HTML: {result}')
    else:
        print('generate_combined_html returned no path (see output above)')

if __name__ == '__main__':
    main()
