"""
Top-3 models from multi-seed aggregates (seed-aware ranking).

Best-practice default:
- Rank by mean balanced accuracy across seeds
- Show uncertainty with std error bars
- Prefer entries with >= min_seeds (default: 10)
- If a model has no entry at min_seeds, fallback to that model's highest n_seeds

Outputs:
- top_models_report_seed_average/top3_seedavg_mix.png
- top_models_report_seed_average/top3_seedavg_single.png
- top_models_report_seed_average/top3_seedavg_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, "top_models_report_seed_average")
os.makedirs(OUT_DIR, exist_ok=True)

MIX_PATH = os.path.join(PROJECT_ROOT, "Mix_Results", "multi_seed_aggregated", "aggregated_results.json")
SINGLE_PATH = os.path.join(PROJECT_ROOT, "Single_Results", "multi_seed_aggregated", "aggregated_results.json")

MODEL_LABELS = {
    "tabpfn": "TabPFN",
    "extratrees": "Extra Trees",
    "lgbm": "LightGBM",
    "randomforest": "Random Forest",
    "xgboost": "XGBoost",
    "logistic_regression": "Logistic Regression",
    "mlp": "MLP",
    "svm": "SVM",
    "gnb": "Gaussian NB",
}

BAR_COLORS = ["#4C78A8", "#F58518", "#54A24B"]
TOP_N = 3


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def strategy_rows(data: dict) -> List[dict]:
    rows: List[dict] = []
    for model, strategies in data.items():
        for strategy, metrics in strategies.items():
            ba = metrics.get("balanced_accuracy", {})
            f1 = metrics.get("f1_macro", {})
            acc = metrics.get("accuracy", {})
            rows.append(
                {
                    "model": model,
                    "strategy": strategy,
                    "label": MODEL_LABELS.get(model, model.replace("_", " ").title()),
                    "ba_mean": float(ba.get("mean", 0.0)),
                    "ba_std": float(ba.get("std", 0.0)),
                    "f1_mean": float(f1.get("mean", 0.0)),
                    "f1_std": float(f1.get("std", 0.0)),
                    "acc_mean": float(acc.get("mean", 0.0)),
                    "acc_std": float(acc.get("std", 0.0)),
                    "n_seeds": int(ba.get("n_seeds", 0) or 0),
                }
            )
    return rows


def pick_best_per_model(
    rows: List[dict],
    min_seeds: int,
    ranking: str = "mean",
    allow_fallback: bool = False,
) -> List[dict]:
    by_model: Dict[str, List[dict]] = {}
    for row in rows:
        by_model.setdefault(row["model"], []).append(row)

    selected: List[dict] = []
    for model, candidates in by_model.items():
        eligible = [c for c in candidates if c["n_seeds"] >= min_seeds]

        if eligible:
            pool = eligible
        elif allow_fallback:
            max_seeds = max(c["n_seeds"] for c in candidates)
            pool = [c for c in candidates if c["n_seeds"] == max_seeds]
        else:
            continue

        if ranking == "lcb":
            # stability-aware: lower confidence bound proxy
            best = max(pool, key=lambda c: (c["ba_mean"] - c["ba_std"], c["ba_mean"]))
        else:
            # default and recommended: pure average performance
            best = max(pool, key=lambda c: (c["ba_mean"], -c["ba_std"]))

        selected.append(best)

    if ranking == "lcb":
        selected.sort(key=lambda c: (c["ba_mean"] - c["ba_std"], c["ba_mean"]), reverse=True)
    else:
        selected.sort(key=lambda c: (c["ba_mean"], -c["ba_std"]), reverse=True)

    return selected


def plot_top3(top3: List[dict], dataset_name: str, output_file: str, ranking: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(top3))

    means = [r["ba_mean"] for r in top3]
    stds = [r["ba_std"] for r in top3]

    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=8,
        color=BAR_COLORS[: len(top3)],
        edgecolor="black",
        linewidth=1.5,
        alpha=0.9,
        zorder=3,
    )

    for i, (bar, row) in enumerate(zip(bars, top3), start=1):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + row["ba_std"] + 0.015,
            f"{row['ba_mean']:.4f} ± {row['ba_std']:.4f}\nSeeds={row['n_seeds']}\n{row['strategy']}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    xticks = [f"#{i+1} {r['label']}" for i, r in enumerate(top3)]
    ax.set_xticks(x)
    ax.set_xticklabels(xticks, fontsize=10, fontweight="bold")

    ax.set_ylabel("Balanced Accuracy (mean ± std)", fontsize=11, fontweight="bold")
    rank_desc = "Mean BA" if ranking == "mean" else "Stability-aware (BA-STD)"
    ax.set_title(f"{dataset_name}: Top 3 Models Across Seeds ({rank_desc})", fontsize=13, fontweight="bold")

    ax.set_ylim(0, min(1.05, max(means) + max(stds) + 0.12))
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def summarize_seed_coverage(rows: List[dict]) -> Dict[str, List[int]]:
    coverage: Dict[str, List[int]] = {}
    for row in rows:
        coverage.setdefault(row["model"], set()).add(row["n_seeds"])
    return {k: sorted(list(v)) for k, v in coverage.items()}


def evaluate_seed_coverage(rows: List[dict], min_seeds: int) -> dict:
    model_max_seed: Dict[str, int] = {}
    model_label: Dict[str, str] = {}

    for row in rows:
        model = row["model"]
        model_label[model] = row["label"]
        model_max_seed[model] = max(model_max_seed.get(model, 0), row["n_seeds"])

    total_models = len(model_max_seed)
    passed_models = sorted([m for m, s in model_max_seed.items() if s >= min_seeds])
    failed_models = sorted([m for m, s in model_max_seed.items() if s < min_seeds])
    passed_count = len(passed_models)
    coverage_ratio = (passed_count / total_models) if total_models else 1.0

    return {
        "min_seeds": min_seeds,
        "total_models": total_models,
        "passed_count": passed_count,
        "failed_count": len(failed_models),
        "coverage_ratio": coverage_ratio,
        "passed_models": passed_models,
        "failed_models": failed_models,
        "failed_model_labels": [model_label[m] for m in failed_models],
        "model_max_seed": model_max_seed,
    }


def run_coverage_gate(
    dataset_name: str,
    coverage: dict,
    gate_mode: str,
    required_coverage_ratio: float,
) -> bool:
    print(
        f"[{dataset_name}] Seed coverage: "
        f"{coverage['passed_count']}/{coverage['total_models']} models meet n_seeds >= {coverage['min_seeds']} "
        f"({coverage['coverage_ratio']:.1%})"
    )

    if gate_mode == "off":
        return True

    ratio_ok = coverage["coverage_ratio"] >= required_coverage_ratio
    has_failed_models = coverage["failed_count"] > 0
    gate_ok = ratio_ok and not has_failed_models

    if gate_ok:
        return True

    failed_labels = ", ".join(coverage["failed_model_labels"]) if coverage["failed_model_labels"] else "none"
    msg = (
        f"[{dataset_name}] Coverage gate check failed: required ratio={required_coverage_ratio:.1%}, "
        f"observed={coverage['coverage_ratio']:.1%}, below-threshold models={failed_labels}."
    )

    if gate_mode == "warn":
        print(f"WARNING: {msg}")
        return True

    print(f"ERROR: {msg}")
    return False


def enforce_top3_policy(
    dataset_name: str,
    ranked_models: List[dict],
    min_seeds: int,
    top_n: int,
    top3_policy: str,
) -> bool:
    selected = ranked_models[:top_n]

    enough_models = len(ranked_models) >= top_n
    all_selected_meet_threshold = all(row["n_seeds"] >= min_seeds for row in selected)

    if top3_policy == "strict":
        if not enough_models:
            print(
                f"ERROR: [{dataset_name}] strict top-3 failed: only {len(ranked_models)} "
                f"eligible models for n_seeds >= {min_seeds} (need {top_n})."
            )
            return False
        if not all_selected_meet_threshold:
            print(
                f"ERROR: [{dataset_name}] strict top-3 failed: at least one selected top-{top_n} model "
                f"has n_seeds < {min_seeds}."
            )
            return False

    return True


def load_per_seed_results(
    model: str,
    strategy: str,
    dataset_name: str,
    project_root: str,
) -> List[float]:
    """
    Load per-seed balanced accuracy values for a model-strategy pair.
    Scans seed_* folders for model results and collects BA values.
    
    Returns list of BA values, empty if model/strategy not found.
    """
    ba_values: List[float] = []
    results_dir = os.path.join(project_root, f"{dataset_name}_Results")
    
    if not os.path.isdir(results_dir):
        return ba_values
    
    # Find all seed_* folders
    for entry in sorted(os.listdir(results_dir)):
        seed_folder = os.path.join(results_dir, entry)
        if not os.path.isdir(seed_folder) or not entry.startswith("seed_"):
            continue
        
        results_json = os.path.join(seed_folder, "reports", "combined_results.json")
        if not os.path.isfile(results_json):
            continue
        
        try:
            with open(results_json, "r", encoding="utf-8") as f:
                seed_data = json.load(f)
            
            if model in seed_data and "results" in seed_data[model]:
                model_results = seed_data[model]["results"]
                if strategy in model_results:
                    ba = model_results[strategy].get("balanced_accuracy")
                    if ba is not None:
                        ba_values.append(float(ba))
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    
    return ba_values


def compute_cohens_d(group1: List[float], group2: List[float]) -> float:
    """Compute Cohen's d effect size for two groups (assumes paired or independent)."""
    if len(group1) < 2 or len(group2) < 2:
        return float("nan")
    
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    n1, n2 = len(group1), len(group2)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return float("nan")
    
    return (mean1 - mean2) / pooled_std


def paired_t_test(model_a_ba: List[float], model_b_ba: List[float]) -> dict:
    """
    Perform paired t-test between two models.
    Requires same number of observations (aligned by seed order).
    """
    if len(model_a_ba) < 2 or len(model_b_ba) < 2:
        return {"t_statistic": float("nan"), "p_value": float("nan"), "valid": False}
    
    if len(model_a_ba) != len(model_b_ba):
        return {"t_statistic": float("nan"), "p_value": float("nan"), "valid": False, "note": "unequal_length"}
    
    t_stat, p_value = stats.ttest_rel(model_a_ba, model_b_ba)
    return {"t_statistic": float(t_stat), "p_value": float(p_value), "valid": True}


def wilcoxon_test(model_a_ba: List[float], model_b_ba: List[float]) -> dict:
    """
    Perform Wilcoxon signed-rank test (non-parametric alternative to paired t-test).
    """
    if len(model_a_ba) < 2 or len(model_b_ba) < 2:
        return {"statistic": float("nan"), "p_value": float("nan"), "valid": False}
    
    if len(model_a_ba) != len(model_b_ba):
        return {"statistic": float("nan"), "p_value": float("nan"), "valid": False, "note": "unequal_length"}
    
    try:
        w_stat, p_value = stats.wilcoxon(model_a_ba, model_b_ba)
        return {"statistic": float(w_stat), "p_value": float(p_value), "valid": True}
    except Exception:
        return {"statistic": float("nan"), "p_value": float("nan"), "valid": False}


def compute_ci_95(differences: List[float]) -> Tuple[float, float]:
    """Compute 95% confidence interval for the mean of differences."""
    if len(differences) < 2:
        return (float("nan"), float("nan"))
    
    mean_diff = np.mean(differences)
    se = stats.sem(differences)
    ci = se * stats.t.ppf(0.975, len(differences) - 1)
    
    return (float(mean_diff - ci), float(mean_diff + ci))


def compare_top3_models(
    top3: List[dict],
    dataset_name: str,
    project_root: str,
) -> dict:
    """
    Compute pairwise statistical comparisons between top-3 models.
    Loads per-seed results and computes:
    - Paired t-test
    - Wilcoxon signed-rank test
    - Cohen's d effect size
    - 95% CI for the mean difference
    
    Returns dict with comparison summary and pairwise results.
    """
    # Load per-seed BA values for each top-3 model
    model_data: Dict[str, Dict[str, List[float]]] = {}
    
    for row in top3:
        model_id = row["model"]
        strategy = row["strategy"]
        ba_values = load_per_seed_results(model_id, strategy, dataset_name, project_root)
        
        if model_id not in model_data:
            model_data[model_id] = {}
        model_data[model_id][strategy] = ba_values
    
    # Generate pairwise comparisons
    pairwise: List[dict] = []
    
    for i in range(len(top3)):
        for j in range(i + 1, len(top3)):
            model_a = top3[i]
            model_b = top3[j]
            
            model_a_id = model_a["model"]
            model_a_strategy = model_a["strategy"]
            model_b_id = model_b["model"]
            model_b_strategy = model_b["strategy"]
            
            ba_a = model_data.get(model_a_id, {}).get(model_a_strategy, [])
            ba_b = model_data.get(model_b_id, {}).get(model_b_strategy, [])
            
            # Align by minimum length
            min_len = min(len(ba_a), len(ba_b))
            if min_len < 2:
                # Not enough data for comparison
                pairwise.append({
                    "model_a": model_a["label"],
                    "model_a_id": model_a_id,
                    "model_b": model_b["label"],
                    "model_b_id": model_b_id,
                    "valid": False,
                    "reason": f"insufficient_data (n={min_len})",
                })
                continue
            
            ba_a_aligned = ba_a[:min_len]
            ba_b_aligned = ba_b[:min_len]
            
            # Compute statistics
            differences = np.array(ba_a_aligned) - np.array(ba_b_aligned)
            mean_diff = float(np.mean(differences))
            std_diff = float(np.std(differences, ddof=1)) if min_len >= 2 else float("nan")
            
            t_test = paired_t_test(ba_a_aligned, ba_b_aligned)
            wilcoxon = wilcoxon_test(ba_a_aligned, ba_b_aligned)
            cohens_d = compute_cohens_d(ba_a_aligned, ba_b_aligned)
            ci_lower, ci_upper = compute_ci_95(differences.tolist())
            
            # Determine significance (alpha = 0.05)
            t_significant = t_test["valid"] and t_test["p_value"] < 0.05
            w_significant = wilcoxon["valid"] and wilcoxon["p_value"] < 0.05
            
            pairwise.append({
                "model_a": model_a["label"],
                "model_a_id": model_a_id,
                "model_a_strategy": model_a_strategy,
                "model_a_ba_mean": float(model_a["ba_mean"]),
                "model_b": model_b["label"],
                "model_b_id": model_b_id,
                "model_b_strategy": model_b_strategy,
                "model_b_ba_mean": float(model_b["ba_mean"]),
                "n_paired_seeds": min_len,
                "mean_difference": mean_diff,
                "std_difference": std_diff,
                "paired_t_test": t_test,
                "wilcoxon_test": wilcoxon,
                "cohens_d": cohens_d if not np.isnan(cohens_d) else None,
                "ci_95_lower": ci_lower,
                "ci_95_upper": ci_upper,
                "significantly_different_ttest": t_significant,
                "significantly_different_wilcoxon": w_significant,
                "valid": True,
            })
    
    return {
        "dataset": dataset_name,
        "top3_models": [
            {
                "rank": i + 1,
                "label": row["label"],
                "model": row["model"],
                "strategy": row["strategy"],
                "ba_mean": float(row["ba_mean"]),
                "ba_std": float(row["ba_std"]),
                "n_seeds": row["n_seeds"],
            }
            for i, row in enumerate(top3)
        ],
        "pairwise_comparisons": pairwise,
        "comparison_notes": {
            "paired_tests": "Aligned by seed order; requires equal seed counts",
            "significance_threshold": 0.05,
            "cohens_d_interpretation": {
                "small": 0.2,
                "medium": 0.5,
                "large": 0.8,
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate top-3 model plots from multi-seed averages")
    parser.add_argument("--min-seeds", type=int, default=10, help="Minimum seed count to prefer (default: 10)")
    parser.add_argument(
        "--ranking",
        choices=["mean", "lcb"],
        default="mean",
        help="Ranking method: mean (recommended) or lcb (mean-std)",
    )
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="If set, include models with fewer than min-seeds when no eligible strategy exists.",
    )
    parser.add_argument(
        "--top3-policy",
        choices=["strict", "flexible"],
        default="strict",
        help="Top-3 selection policy: strict enforces min-seeds eligibility, flexible allows fallback behavior.",
    )
    parser.add_argument(
        "--coverage-gate",
        choices=["off", "warn", "fail"],
        default="warn",
        help="Seed coverage gate behavior across models: off, warn, or fail (default: warn).",
    )
    parser.add_argument(
        "--required-coverage-ratio",
        type=float,
        default=1.0,
        help="Required fraction of models meeting min-seeds for gate checks (0.0 to 1.0, default: 1.0).",
    )
    args = parser.parse_args()

    if not (0.0 <= args.required_coverage_ratio <= 1.0):
        raise ValueError("--required-coverage-ratio must be between 0.0 and 1.0")

    if args.top3_policy == "strict" and args.allow_fallback:
        print("WARNING: --allow-fallback is ignored when --top3-policy strict is active.")

    allow_fallback_effective = args.allow_fallback and args.top3_policy == "flexible"

    mix_data = load_json(MIX_PATH)
    single_data = load_json(SINGLE_PATH)

    mix_rows = strategy_rows(mix_data)
    single_rows = strategy_rows(single_data)

    mix_coverage = evaluate_seed_coverage(mix_rows, min_seeds=args.min_seeds)
    single_coverage = evaluate_seed_coverage(single_rows, min_seeds=args.min_seeds)

    print("Seed Coverage Quality Gate")
    print("-" * 78)
    mix_gate_ok = run_coverage_gate("Mix", mix_coverage, args.coverage_gate, args.required_coverage_ratio)
    single_gate_ok = run_coverage_gate("Single", single_coverage, args.coverage_gate, args.required_coverage_ratio)

    if args.coverage_gate == "fail" and not (mix_gate_ok and single_gate_ok):
        print("\nCoverage gate failed. Aborting report generation.")
        raise SystemExit(2)

    mix_best = pick_best_per_model(
        mix_rows,
        min_seeds=args.min_seeds,
        ranking=args.ranking,
        allow_fallback=allow_fallback_effective,
    )
    single_best = pick_best_per_model(
        single_rows,
        min_seeds=args.min_seeds,
        ranking=args.ranking,
        allow_fallback=allow_fallback_effective,
    )

    print("\nTop-3 Policy Check")
    print("-" * 78)
    mix_top3_ok = enforce_top3_policy("Mix", mix_best, args.min_seeds, TOP_N, args.top3_policy)
    single_top3_ok = enforce_top3_policy("Single", single_best, args.min_seeds, TOP_N, args.top3_policy)
    if args.top3_policy == "strict" and not (mix_top3_ok and single_top3_ok):
        print("\nStrict top-3 policy failed. Aborting report generation.")
        raise SystemExit(3)

    mix_top3 = mix_best[:3]
    single_top3 = single_best[:3]

    mix_png = os.path.join(OUT_DIR, "top3_seedavg_mix.png")
    single_png = os.path.join(OUT_DIR, "top3_seedavg_single.png")
    mix_png = os.path.join(OUT_DIR, f"top3_seedavg_{args.ranking}_mix.png")
    single_png = os.path.join(OUT_DIR, f"top3_seedavg_{args.ranking}_single.png")

    plot_top3(mix_top3, "Mix", mix_png, args.ranking)
    plot_top3(single_top3, "Single", single_png, args.ranking)

    # Compute statistical comparisons
    mix_stats = compare_top3_models(mix_top3, "Mix", PROJECT_ROOT)
    single_stats = compare_top3_models(single_top3, "Single", PROJECT_ROOT)

    summary = {
        "settings": {
            "min_seeds": args.min_seeds,
            "ranking": args.ranking,
            "allow_fallback": args.allow_fallback,
            "allow_fallback_effective": allow_fallback_effective,
            "top3_policy": args.top3_policy,
            "coverage_gate": args.coverage_gate,
            "required_coverage_ratio": args.required_coverage_ratio,
            "recommended_default": "mean",
            "ranking_explanation": {
                "mean": "Ranks by average balanced accuracy across seeds; best default for general model selection.",
                "lcb": "Ranks by mean-std to penalize unstable models; useful when robustness is priority.",
            },
        },
        "mix_top3": mix_top3,
        "single_top3": single_top3,
        "seed_coverage": {
            "mix": summarize_seed_coverage(mix_rows),
            "single": summarize_seed_coverage(single_rows),
        },
        "coverage_gate": {
            "mix": mix_coverage,
            "single": single_coverage,
            "gate_mode": args.coverage_gate,
            "required_coverage_ratio": args.required_coverage_ratio,
            "passed": bool(mix_gate_ok and single_gate_ok),
        },
        "top3_policy": {
            "policy": args.top3_policy,
            "required_top_n": TOP_N,
            "mix_passed": bool(mix_top3_ok),
            "single_passed": bool(single_top3_ok),
            "passed": bool(mix_top3_ok and single_top3_ok),
        },
        "statistical_comparisons": {
            "mix": mix_stats,
            "single": single_stats,
        },
        "outputs": {
            "mix_plot": mix_png,
            "single_plot": single_png,
        },
    }

    out_json = os.path.join(OUT_DIR, f"top3_seedavg_{args.ranking}_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Save separate comparison JSONs
    mix_comp_json = os.path.join(OUT_DIR, f"top3_seedavg_{args.ranking}_comparison_mix.json")
    single_comp_json = os.path.join(OUT_DIR, f"top3_seedavg_{args.ranking}_comparison_single.json")
    
    with open(mix_comp_json, "w", encoding="utf-8") as f:
        json.dump(mix_stats, f, indent=2)
    with open(single_comp_json, "w", encoding="utf-8") as f:
        json.dump(single_stats, f, indent=2)

    print("=" * 78)
    print("TOP-3 MULTI-SEED REPORT")
    print("=" * 78)
    print(f"Ranking method : {args.ranking}")
    print(f"Preferred seeds: >= {args.min_seeds}")

    print("\nMix top-3:")
    for i, row in enumerate(mix_top3, start=1):
        print(
            f"  {i}. {row['label']:20s} | BA={row['ba_mean']:.4f}±{row['ba_std']:.4f} "
            f"| Seeds={row['n_seeds']:2d} | {row['strategy']}"
        )

    print("\nSingle top-3:")
    for i, row in enumerate(single_top3, start=1):
        print(
            f"  {i}. {row['label']:20s} | BA={row['ba_mean']:.4f}±{row['ba_std']:.4f} "
            f"| Seeds={row['n_seeds']:2d} | {row['strategy']}"
        )

    # Print statistical comparison summary
    print("\n" + "=" * 78)
    print("STATISTICAL COMPARISONS (Pairwise)")
    print("=" * 78)
    
    for dataset_name, stats_data in [("Mix", mix_stats), ("Single", single_stats)]:
        print(f"\n[{dataset_name}] Statistical Comparisons:")
        print("-" * 78)
        
        for comp in stats_data["pairwise_comparisons"]:
            if not comp.get("valid"):
                print(f"  {comp['model_a']} vs {comp['model_b']}: {comp.get('reason', 'invalid')}")
                continue
            
            model_a = comp["model_a"]
            model_b = comp["model_b"]
            mean_diff = comp["mean_difference"]
            t_pval = comp["paired_t_test"].get("p_value", float("nan"))
            w_pval = comp["wilcoxon_test"].get("p_value", float("nan"))
            cohens_d = comp["cohens_d"]
            
            t_sig = "***" if comp["significantly_different_ttest"] else "ns"
            w_sig = "***" if comp["significantly_different_wilcoxon"] else "ns"
            
            if cohens_d is not None:
                print(
                    f"  {model_a:20s} vs {model_b:20s} | "
                    f"Mean_Diff={mean_diff:+.4f} | "
                    f"t-test p={t_pval:.4f} ({t_sig}) | "
                    f"Wilcoxon p={w_pval:.4f} ({w_sig}) | "
                    f"Cohen's d={cohens_d:.4f}"
                )
            else:
                print(
                    f"  {model_a:20s} vs {model_b:20s} | "
                    f"Mean_Diff={mean_diff:+.4f} | "
                    f"t-test p={t_pval:.4f} ({t_sig}) | "
                    f"Wilcoxon p={w_pval:.4f} ({w_sig})"
                )
            print(
                f"    95% CI: [{comp['ci_95_lower']:.4f}, {comp['ci_95_upper']:.4f}] | "
                f"n_paired={comp['n_paired_seeds']}"
            )

    print("\nSaved:")
    print(f"  - {mix_png}")
    print(f"  - {single_png}")
    print(f"  - {out_json}")
    print(f"  - {mix_comp_json}")
    print(f"  - {single_comp_json}")


if __name__ == "__main__":
    main()
