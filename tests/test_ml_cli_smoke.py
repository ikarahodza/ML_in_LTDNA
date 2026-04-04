import subprocess
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = PROJECT_ROOT / "scripts" / "ml_cli.py"


class TestUnifiedCliSmoke(unittest.TestCase):
    def test_cli_list_runs(self):
        result = subprocess.run(
            [sys.executable, str(CLI_PATH), "list"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("top3-seedavg", result.stdout)

    def test_cli_rejects_unknown_script(self):
        result = subprocess.run(
            [sys.executable, str(CLI_PATH), "run", "does-not-exist"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("Unknown script", result.stdout)


if __name__ == "__main__":
    unittest.main()
