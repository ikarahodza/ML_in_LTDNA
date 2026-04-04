"""Unified CLI entrypoint for running project scripts.

Examples:
  python scripts/ml_cli.py list
  python scripts/ml_cli.py run top3-seedavg -- --min-seeds 10 --ranking mean
  python scripts/ml_cli.py run visualize_top3_scientific
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

SCRIPT_ALIASES: Dict[str, str] = {
    "top3-seedavg": "visualize_top3_seed_average.py",
    "top3-scientific": "visualize_top3_scientific.py",
    "top-models-report": "generate_top_models_report.py",
    "aggregated-report": "generate_aggregated_report.py",
    "multi-seed-train": "run_multi_seed_training.py",
    "all-models": "run_all_models.py",
    "ensemble-pipeline": "run_ensemble_analysis_pipeline.py",
}


def _normalize_script_name(name: str) -> str:
    return name if name.endswith(".py") else f"{name}.py"


def resolve_script_target(target: str) -> Path:
    alias_target = SCRIPT_ALIASES.get(target, target)
    filename = _normalize_script_name(alias_target)
    script_path = SCRIPTS_DIR / filename
    if not script_path.exists():
        raise FileNotFoundError(
            f"Unknown script '{target}'. Run 'python scripts/ml_cli.py list' to view options."
        )
    return script_path


def list_scripts() -> int:
    print("Unified CLI - Available aliases")
    print("-" * 78)
    for alias, script in sorted(SCRIPT_ALIASES.items()):
        print(f"{alias:20s} -> {script}")

    print("\nAll runnable scripts in scripts/")
    print("-" * 78)
    for script_file in sorted(SCRIPTS_DIR.glob("*.py")):
        if script_file.name == Path(__file__).name:
            continue
        print(script_file.name)
    return 0


def run_script(target: str, script_args: list[str]) -> int:
    script_path = resolve_script_target(target)
    cleaned_args = script_args
    if cleaned_args and cleaned_args[0] == "--":
        cleaned_args = cleaned_args[1:]

    cmd = [sys.executable, str(script_path), *cleaned_args]
    print(f"Running: {' '.join(cmd)}")
    completed = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False)
    return int(completed.returncode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified CLI for ML_in_LTDNA scripts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List script aliases and available scripts")

    run_parser = subparsers.add_parser("run", help="Run a script by alias or filename")
    run_parser.add_argument("target", help="Alias (e.g., top3-seedavg) or script name")
    run_parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to target script. Prefix with -- for clarity.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    parsed = parser.parse_args()

    if parsed.command == "list":
        raise SystemExit(list_scripts())

    if parsed.command == "run":
        try:
            rc = run_script(parsed.target, parsed.args)
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}")
            raise SystemExit(2)
        raise SystemExit(rc)


if __name__ == "__main__":
    main()
