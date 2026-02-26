"""
Script to run TabPFN model for all 10 seeds
Created: January 18, 2026
This script runs TabPFN.py for each seed with both mix and single datasets
"""

import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path

# Define the seeds to iterate through
SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]

# Define datasets to run (comment out if you only want one)
DATASETS = ["mix", "single"]

# Get the script directory
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
TABPFN_SCRIPT = SCRIPT_DIR / "TabPFN.py"

# Create log file with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = PROJECT_ROOT / f"tabpfn_multi_seed_log_{timestamp}.txt"

# Initialize counters
total_runs = 0
successful_runs = 0
failed_runs = 0
results = []

def print_colored(text, color="white"):
    """Print colored text (basic implementation for cross-platform compatibility)"""
    colors = {
        "cyan": "\033[96m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "gray": "\033[90m",
        "white": "\033[97m",
        "reset": "\033[0m"
    }
    print(f"{colors.get(color, '')}{text}{colors['reset']}")

def write_log(message):
    """Write message to log file"""
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(message + "\n")

# Print header
print_colored("=" * 60, "cyan")
print_colored("TabPFN Multi-Seed Training Script", "cyan")
print_colored("=" * 60, "cyan")
print_colored(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "green")
print_colored(f"Seeds: {', '.join(map(str, SEEDS))}", "yellow")
print_colored(f"Datasets: {', '.join(DATASETS)}", "yellow")
print_colored(f"Log file: {log_file}", "yellow")
print_colored("=" * 60 + "\n", "cyan")

# Start logging
write_log("TABPFN MULTI-SEED TRAINING LOG")
write_log("=" * 80)
write_log("")
write_log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
write_log(f"Seeds: {', '.join(map(str, SEEDS))}")
write_log(f"Datasets: {', '.join(DATASETS)}")
write_log("")

# Loop through each seed
for seed in SEEDS:
    for dataset in DATASETS:
        total_runs += 1
        
        print_colored("-" * 60, "gray")
        print_colored(f"Running TabPFN - Seed: {seed} | Dataset: {dataset}", "white")
        print_colored(f"Run {total_runs} of {len(SEEDS) * len(DATASETS)}", "gray")
        print_colored("-" * 60, "gray")
        
        start_time = datetime.now()
        
        try:
            # Run the Python script
            result = subprocess.run(
                [sys.executable, str(TABPFN_SCRIPT), dataset, "--seed", str(seed)],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT
            )
            
            if result.returncode == 0:
                successful_runs += 1
                status = "SUCCESS"
                status_symbol = "✓"
                status_color = "green"
                print_colored(f"{status_symbol} Completed successfully", status_color)
            else:
                failed_runs += 1
                status = "FAILED"
                status_symbol = "✗"
                status_color = "red"
                print_colored(f"{status_symbol} Failed with exit code: {result.returncode}", status_color)
                if result.stderr:
                    print_colored(f"Error output: {result.stderr[:200]}", "red")
        
        except Exception as e:
            failed_runs += 1
            status = "ERROR"
            status_symbol = "✗"
            status_color = "red"
            print_colored(f"{status_symbol} Error: {str(e)}", status_color)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Record result
        result_dict = {
            "seed": seed,
            "dataset": dataset,
            "status": status,
            "duration": round(duration, 2),
            "timestamp": end_time.strftime('%Y-%m-%d %H:%M:%S')
        }
        results.append(result_dict)
        
        # Log to file
        write_log(f"{status_symbol} Seed={seed:>4} | {dataset:<6} | TabPFN.py | {status} | {duration:.2f}s")
        
        print_colored(f"Duration: {duration:.2f} seconds\n", "gray")

# Summary
end_time_total = datetime.now()

print_colored("\n" + "=" * 60, "cyan")
print_colored("SUMMARY", "cyan")
print_colored("=" * 60, "cyan")
print_colored(f"Total runs: {total_runs}", "white")
print_colored(f"Successful: {successful_runs}", "green")
print_colored(f"Failed: {failed_runs}", "red" if failed_runs > 0 else "green")
success_rate = (successful_runs / total_runs * 100) if total_runs > 0 else 0
print_colored(f"Success rate: {success_rate:.2f}%", "yellow")
print_colored(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "green")
print_colored("=" * 60 + "\n", "cyan")

# Write summary to log
write_log("")
write_log("=" * 80)
write_log("SUMMARY")
write_log("=" * 80)
write_log("")
write_log(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
write_log(f"Total runs: {total_runs}")
write_log(f"Successful: {successful_runs}")
write_log(f"Failed: {failed_runs}")
write_log(f"Success rate: {success_rate:.2f}%")
write_log("")

# Display results table
print_colored("Detailed Results:", "cyan")
print(f"{'Seed':<6} {'Dataset':<8} {'Status':<10} {'Duration':<12} {'Timestamp'}")
print("-" * 60)
for r in results:
    print(f"{r['seed']:<6} {r['dataset']:<8} {r['status']:<10} {r['duration']:<12.2f} {r['timestamp']}")

print_colored(f"\nLog file saved to: {log_file}", "yellow")
