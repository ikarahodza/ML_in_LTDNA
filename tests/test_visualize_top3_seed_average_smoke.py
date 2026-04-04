import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "visualize_top3_seed_average.py"

spec = importlib.util.spec_from_file_location("visualize_top3_seed_average", MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load module spec from {MODULE_PATH}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TestVisualizeTop3SeedAverageSmoke(unittest.TestCase):
    def test_strategy_rows_parses_aggregated_structure(self):
        aggregated = {
            "tabpfn": {
                "ClassWeights": {
                    "balanced_accuracy": {"mean": 0.84, "std": 0.04, "n_seeds": 10},
                    "f1_macro": {"mean": 0.83, "std": 0.03},
                    "accuracy": {"mean": 0.88, "std": 0.03},
                }
            },
            "lgbm": {
                "SMOTE": {
                    "balanced_accuracy": {"mean": 0.80, "std": 0.05, "n_seeds": 10},
                    "f1_macro": {"mean": 0.79, "std": 0.04},
                    "accuracy": {"mean": 0.82, "std": 0.05},
                }
            },
        }

        rows = module.strategy_rows(aggregated)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["model"], "tabpfn")
        self.assertIn("ba_mean", rows[0])
        self.assertEqual(rows[0]["n_seeds"], 10)

    def test_pick_best_and_strict_policy_gate(self):
        rows = [
            {
                "model": "tabpfn",
                "strategy": "Original",
                "label": "TabPFN",
                "ba_mean": 0.80,
                "ba_std": 0.05,
                "f1_mean": 0.79,
                "f1_std": 0.05,
                "acc_mean": 0.82,
                "acc_std": 0.05,
                "n_seeds": 10,
            },
            {
                "model": "tabpfn",
                "strategy": "ClassWeights",
                "label": "TabPFN",
                "ba_mean": 0.85,
                "ba_std": 0.04,
                "f1_mean": 0.84,
                "f1_std": 0.04,
                "acc_mean": 0.88,
                "acc_std": 0.03,
                "n_seeds": 10,
            },
            {
                "model": "lgbm",
                "strategy": "SMOTE",
                "label": "LightGBM",
                "ba_mean": 0.81,
                "ba_std": 0.05,
                "f1_mean": 0.80,
                "f1_std": 0.05,
                "acc_mean": 0.84,
                "acc_std": 0.04,
                "n_seeds": 10,
            },
            {
                "model": "xgboost",
                "strategy": "SMOTE",
                "label": "XGBoost",
                "ba_mean": 0.79,
                "ba_std": 0.06,
                "f1_mean": 0.78,
                "f1_std": 0.06,
                "acc_mean": 0.81,
                "acc_std": 0.05,
                "n_seeds": 10,
            },
        ]

        best = module.pick_best_per_model(rows, min_seeds=10, ranking="mean", allow_fallback=False)
        self.assertEqual(len(best), 3)
        self.assertEqual(best[0]["model"], "tabpfn")
        self.assertTrue(
            module.enforce_top3_policy("Mix", best, min_seeds=10, top_n=3, top3_policy="strict")
        )

        only_two = best[:2]
        with patch("builtins.print"):
            self.assertFalse(
                module.enforce_top3_policy(
                    "Mix", only_two, min_seeds=10, top_n=3, top3_policy="strict"
                )
            )

    def test_compare_top3_models_pairwise_stats(self):
        top3 = [
            {
                "model": "tabpfn",
                "strategy": "ClassWeights",
                "label": "TabPFN",
                "ba_mean": 0.85,
                "ba_std": 0.04,
                "n_seeds": 10,
            },
            {
                "model": "lgbm",
                "strategy": "SMOTE",
                "label": "LightGBM",
                "ba_mean": 0.81,
                "ba_std": 0.05,
                "n_seeds": 10,
            },
            {
                "model": "xgboost",
                "strategy": "SMOTE",
                "label": "XGBoost",
                "ba_mean": 0.79,
                "ba_std": 0.06,
                "n_seeds": 10,
            },
        ]

        fake_seed_values = {
            ("tabpfn", "ClassWeights"): [0.85, 0.86, 0.84, 0.83, 0.87],
            ("lgbm", "SMOTE"): [0.80, 0.81, 0.79, 0.80, 0.82],
            ("xgboost", "SMOTE"): [0.78, 0.79, 0.77, 0.78, 0.80],
        }

        def fake_loader(model, strategy, dataset_name, project_root):
            return fake_seed_values[(model, strategy)]

        with patch.object(module, "load_per_seed_results", side_effect=fake_loader):
            stats = module.compare_top3_models(
                top3, dataset_name="Mix", project_root=str(PROJECT_ROOT)
            )

        self.assertEqual(stats["dataset"], "Mix")
        self.assertEqual(len(stats["pairwise_comparisons"]), 3)
        self.assertTrue(all(c["valid"] for c in stats["pairwise_comparisons"]))

    def test_generate_html_report_creates_file(self):
        summary = {
            "settings": {
                "min_seeds": 10,
                "ranking": "mean",
                "top3_policy": "strict",
                "coverage_gate": "warn",
            },
            "mix_top3": [
                {
                    "label": "TabPFN",
                    "strategy": "ClassWeights",
                    "ba_mean": 0.85,
                    "ba_std": 0.04,
                    "n_seeds": 10,
                }
            ],
            "single_top3": [
                {
                    "label": "Extra Trees",
                    "strategy": "SMOTE",
                    "ba_mean": 0.98,
                    "ba_std": 0.02,
                    "n_seeds": 10,
                }
            ],
            "statistical_comparisons": {
                "mix": {"pairwise_comparisons": []},
                "single": {"pairwise_comparisons": []},
            },
            "outputs": {
                "mix_plot": "mix_plot.png",
                "single_plot": "single_plot.png",
            },
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            out = Path(tmp_dir) / "report.html"
            module.generate_html_report(summary, str(out))

            self.assertTrue(out.exists())
            html = out.read_text(encoding="utf-8")
            self.assertIn("Top-3 Multi-Seed Performance Report", html)
            self.assertIn("Pairwise Statistical Comparisons", html)


if __name__ == "__main__":
    unittest.main()
