"""
Scientific Visualization of Top 3 Models
=========================================
Based on scientific visualization principles (Karahodža, April 2025):
  - High data-ink ratio  (remove chartjunk, background, heavy borders)
  - Categorical colors   (equally distinct, not fully saturated; ColorBrewer Set2)
  - Grouped bar charts   (discrete data; grouped by model, Mix vs Single as sub-bars)
  - Error bars           (mean ± std across 10 random seeds)
  - Direct labels        (values annotated directly on bars)
  - Full y-axis range    (0–1 for accuracy-like metrics; natural orientation)
  - Consistent encoding  (same model  color and hatch pattern throughout)
  - Works in greyscale   (hatch pattern distinguishes Mix vs Single)
  - Panel labels         (A, B) for multi-panel figures
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "top_models_report_scientific")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MIX_AGGR_PATH    = os.path.join(PROJECT_ROOT, "Mix_Results",    "multi_seed_aggregated", "aggregated_results.json")
SINGLE_AGGR_PATH = os.path.join(PROJECT_ROOT, "Single_Results", "multi_seed_aggregated", "aggregated_results.json")

# ---------------------------------------------------------------------------
# Visual design constants  (ColorBrewer Set2 – 8 colours, print-safe)
# ---------------------------------------------------------------------------
COLORBREWER_SET2 = [
    "#66c2a5",  # teal-green
    "#fc8d62",  # orange
    "#8da0cb",  # soft blue
    "#e78ac3",  # pink
    "#a6d854",  # yellow-green
    "#ffd92f",  # yellow
    "#e5c494",  # tan
    "#b3b3b3",  # grey
]

# Hatch patterns: Mix = solid, Single = dense hatching
DATASET_HATCH  = {"Mix": "", "Single": "///"}
DATASET_ALPHA  = {"Mix": 0.9, "Single": 0.75}
DATASET_LABEL  = {"Mix": "Mix dataset", "Single": "Single dataset"}

# Metrics to display (key in json, axis label, panel letter)
METRIC_CFG = [
    ("balanced_accuracy", "Balanced Accuracy", "A"),
    ("f1_macro",          "F1-Score (Macro)",  "B"),
]

TOP_N = 3   # number of models to show per dataset

# Human-readable model name mapping
MODEL_DISPLAY = {
    "tabpfn":               "TabPFN",
    "extratrees":           "Extra Trees",
    "lgbm":                 "LightGBM",
    "randomforest":         "Random Forest",
    "xgboost":              "XGBoost",
    "logistic_regression":  "Logistic Reg.",
    "mlp":                  "MLP",
    "svm":                  "SVM",
    "gnb":                  "Gaussian NB",
}

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_aggregated(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Aggregated results not found: {path}")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def best_per_model(data: dict) -> list[dict]:
    """Return list of best-strategy result per model, sorted by balanced accuracy."""
    best: dict[str, dict] = {}
    for model, strategies in data.items():
        for strat, metrics in strategies.items():
            ba = metrics.get("balanced_accuracy", {}).get("mean", 0.0)
            if model not in best or ba > best[model]["ba_mean"]:
                best[model] = {
                    "model":    model,
                    "display":  MODEL_DISPLAY.get(model, model.replace("_", " ").title()),
                    "strategy": strat,
                    "ba_mean":  metrics.get("balanced_accuracy", {}).get("mean", 0.0),
                    "ba_std":   metrics.get("balanced_accuracy", {}).get("std",  0.0),
                    "f1_mean":  metrics.get("f1_macro",          {}).get("mean", 0.0),
                    "f1_std":   metrics.get("f1_macro",          {}).get("std",  0.0),
                    "acc_mean": metrics.get("accuracy",          {}).get("mean", 0.0),
                    "acc_std":  metrics.get("accuracy",          {}).get("std",  0.0),
                }
    return sorted(best.values(), key=lambda x: x["ba_mean"], reverse=True)


# ---------------------------------------------------------------------------
# Assign a consistent color to every model seen across both datasets
# ---------------------------------------------------------------------------

def build_color_map(mix_top: list, single_top: list) -> dict[str, str]:
    seen_order: list[str] = []
    for entry in mix_top + single_top:
        m = entry["model"]
        if m not in seen_order:
            seen_order.append(m)
    return {m: COLORBREWER_SET2[i % len(COLORBREWER_SET2)] for i, m in enumerate(seen_order)}


# ---------------------------------------------------------------------------
# Core figure: per-dataset grouped bar panel
# ---------------------------------------------------------------------------

def _apply_scientific_style(ax: plt.Axes) -> None:
    """Apply data-ink-ratio style: remove top/right spines, light horizontal grid."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="both", length=3, width=0.8, labelsize=9)


def _annotate_bar(ax: plt.Axes, x: float, y: float, std: float,
                  val: float, metric_key: str) -> None:
    """Place a compact direct label at the top of a bar."""
    if metric_key == "ba_mean":
        label = f"{val:.3f}"
    elif metric_key == "f1_mean":
        label = f"{val:.3f}"
    else:
        label = f"{val:.3f}"

    offset = std + 0.012
    ax.text(
        x, y + offset,
        label,
        ha="center", va="bottom",
        fontsize=7.5, fontweight="bold", color="#333333",
        clip_on=False,
    )


def create_per_dataset_figure(
    mix_top:    list,
    single_top: list,
    color_map:  dict[str, str],
    metric_key: str,
    metric_label: str,
    panel_label:  str,
    output_path:  str,
) -> None:
    """
    Two-panel figure (Mix | Single).
    Each panel: grouped bar chart, one group per model-rank,
    hatching distinguishes Mix bars from Single bars.
    """
    fig, axes = plt.subplots(
        1, 2,
        figsize=(10, 4.5),
        sharey=True,
        gridspec_kw={"wspace": 0.08},
    )
    fig.patch.set_facecolor("white")

    for ax, (dataset_name, top_list) in zip(axes, [("Mix", mix_top), ("Single", single_top)]):
        _apply_scientific_style(ax)

        n = len(top_list)
        x = np.arange(n)
        width = 0.55

        # metric_key is the internal prefix ("ba", "f1", "acc")
        mean_key = f"{metric_key}_mean"
        std_key  = f"{metric_key}_std"

        ax.bar(
            x,
            [r[mean_key] for r in top_list],
            width,
            color=[color_map[r["model"]] for r in top_list],
            hatch=DATASET_HATCH[dataset_name],
            alpha=DATASET_ALPHA[dataset_name],
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )

        # Error bars
        ax.errorbar(
            x,
            [r[mean_key] for r in top_list],
            yerr=[r[std_key] for r in top_list],
            fmt="none",
            ecolor="#444444",
            capsize=4,
            capthick=1.2,
            elinewidth=1.2,
            zorder=4,
        )

        # Direct value labels
        for xi, row in zip(x, top_list):
            _annotate_bar(ax, xi, row[mean_key], row[std_key],
                          row[mean_key], mean_key)

        # Rank badges on x-axis ticks
        rank_symbols = ["#1", "#2", "#3"]
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{rank_symbols[i]}\n{r['display']}" for i, r in enumerate(top_list)],
            fontsize=9,
        )

        # Strategy label (small, below model name) – shows best resampling strategy
        for xi, row in zip(x, top_list):
            strat_short = (row["strategy"]
                           .replace("Borderline SMOTE", "Borderline\nSMOTE")
                           .replace("RandomOverSampler", "ROS")
                           .replace("ClassWeights", "Class\nWeights"))
            ax.text(xi, -0.11, strat_short,
                    ha="center", va="top",
                    fontsize=6.5, color="#666666",
                    transform=ax.get_xaxis_transform(), clip_on=False)

        ax.set_ylim(0, 1.12)
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))

        # Dataset label inside panel (top-left)
        ax.text(0.03, 0.97, dataset_name,
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=10, fontweight="bold", color="#333333")

    # Shared y-label on leftmost axis
    axes[0].set_ylabel(metric_label, fontsize=10, fontweight="bold")

    # Panel label (A / B / …)
    fig.text(0.01, 0.97, panel_label,
             fontsize=13, fontweight="bold", va="top", ha="left", color="#222222")

    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(output_path)}")


# ---------------------------------------------------------------------------
# Combined 2-row figure (BA + F1, side by side for Mix and Single)
# ---------------------------------------------------------------------------

def create_combined_figure(
    mix_top:    list,
    single_top: list,
    color_map:  dict[str, str],
    output_path: str,
) -> None:
    """
    Publication-style 2-row × 2-column figure.
    Row ①: Balanced Accuracy (Mix | Single)
    Row ②: F1-Score          (Mix | Single)
    """
    fig, axes = plt.subplots(
        2, 2,
        figsize=(11, 8),
        sharey="row",
        gridspec_kw={"wspace": 0.07, "hspace": 0.55},
    )
    fig.patch.set_facecolor("white")

    metric_rows = [
        ("ba_mean",  "ba_std",  "Balanced Accuracy", "A"),
        ("f1_mean",  "f1_std",  "F1-Score (Macro)",  "B"),
    ]

    for row_idx, (mean_key, std_key, metric_label, panel_ltr) in enumerate(metric_rows):
        for col_idx, (dataset_name, top_list) in enumerate([("Mix", mix_top), ("Single", single_top)]):
            ax = axes[row_idx][col_idx]
            _apply_scientific_style(ax)

            n = len(top_list)
            x = np.arange(n)
            width = 0.55

            ax.bar(
                x,
                [r[mean_key] for r in top_list],
                width,
                color=[color_map[r["model"]] for r in top_list],
                hatch=DATASET_HATCH[dataset_name],
                alpha=DATASET_ALPHA[dataset_name],
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )

            ax.errorbar(
                x,
                [r[mean_key] for r in top_list],
                yerr=[r[std_key] for r in top_list],
                fmt="none",
                ecolor="#444444",
                capsize=4,
                capthick=1.2,
                elinewidth=1.2,
                zorder=4,
            )

            for xi, row in zip(x, top_list):
                _annotate_bar(ax, xi, row[mean_key], row[std_key], row[mean_key], mean_key)

            rank_symbols = ["#1", "#2", "#3"]
            ax.set_xticks(x)
            ax.set_xticklabels(
                [f"{rank_symbols[i]}\n{r['display']}" for i, r in enumerate(top_list)],
                fontsize=8.5,
            )

            for xi, row in zip(x, top_list):
                strat_short = (row["strategy"]
                               .replace("Borderline SMOTE", "BL-SMOTE")
                               .replace("RandomOverSampler", "ROS")
                               .replace("ClassWeights", "CW"))
                ax.text(xi, -0.14, f"[{strat_short}]",
                        ha="center", va="top",
                        fontsize=6, color="#777777",
                        transform=ax.get_xaxis_transform(), clip_on=False)

            ax.set_ylim(0, 1.12)
            ax.yaxis.set_major_locator(MultipleLocator(0.2))
            ax.yaxis.set_minor_locator(MultipleLocator(0.1))

            # Dataset label (top-left of cell)
            ax.text(0.04, 0.95, dataset_name,
                    transform=ax.transAxes,
                    ha="left", va="top",
                    fontsize=9.5, fontweight="bold", color="#333333")

        # Row y-label (leftmost column only)
        axes[row_idx][0].set_ylabel(metric_label, fontsize=9.5, fontweight="bold")

        # Panel letter, left of the row
        fig.text(
            0.01,
            axes[row_idx][0].get_position().y1 + 0.01,
            panel_ltr,
            fontsize=13, fontweight="bold", va="bottom", ha="left", color="#222222",
        )

    # ---- Legend (dataset encoding) ----
    legend_handles = [
        mpatches.Patch(facecolor="#aaaaaa", hatch="",
                       edgecolor="#555555", linewidth=0.6, label="Mix dataset (solid)"),
        mpatches.Patch(facecolor="#aaaaaa", hatch="///",
                       edgecolor="#555555", linewidth=0.6, label="Single dataset (hatched)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )

    fig.suptitle(
        "Top 3 Models — Multi-Seed Performance (mean ± std, 10 seeds)",
        fontsize=12, fontweight="bold", y=1.01, color="#222222",
    )

    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(output_path)}")


# ---------------------------------------------------------------------------
# Supplementary: per-model full metric profile (BA, F1, Acc) for both datasets
# ---------------------------------------------------------------------------

def create_model_profile(
    model_key:   str,
    mix_all:     list,
    single_all:  list,
    color:       str,
    output_path: str,
) -> None:
    """
    3-metric profile for a single model, showing Mix vs Single.
    Bar style follows the same scientific guidelines.
    """
    mix_row    = next((r for r in mix_all    if r["model"] == model_key), None)
    single_row = next((r for r in single_all if r["model"] == model_key), None)

    metrics = [
        ("ba_mean",  "ba_std",  "Balanced\nAccuracy"),
        ("f1_mean",  "f1_std",  "F1\n(Macro)"),
        ("acc_mean", "acc_std", "Accuracy"),
    ]

    datasets   = [("Mix",    mix_row),    ("Single", single_row)]
    valid_data = [(ds, row) for ds, row in datasets if row is not None]

    if not valid_data:
        return

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    fig.patch.set_facecolor("white")
    _apply_scientific_style(ax)

    n_metrics  = len(metrics)
    n_datasets = len(valid_data)
    total_width = 0.65
    bar_width   = total_width / n_datasets
    offsets     = np.linspace(-total_width/2 + bar_width/2,
                              total_width/2 - bar_width/2,
                              n_datasets)
    x = np.arange(n_metrics)

    for d_idx, (ds_name, row) in enumerate(valid_data):
        means = [row[mk] for mk, _, _ in metrics]
        stds  = [row[sk] for _, sk, _ in metrics]
        ax.bar(
            x + offsets[d_idx],
            means,
            bar_width,
            color=color,
            hatch=DATASET_HATCH[ds_name],
            alpha=DATASET_ALPHA[ds_name],
            edgecolor="white",
            linewidth=0.6,
            label=DATASET_LABEL[ds_name],
            zorder=3,
        )
        ax.errorbar(
            x + offsets[d_idx],
            means,
            yerr=stds,
            fmt="none",
            ecolor="#444444",
            capsize=3,
            capthick=1.0,
            elinewidth=1.0,
            zorder=4,
        )
        for xi, mean, std in zip(x, means, stds):
            ax.text(xi + offsets[d_idx], mean + std + 0.012,
                    f"{mean:.3f}",
                    ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold", color="#333333",
                    clip_on=False)

    display_name = MODEL_DISPLAY.get(model_key, model_key.replace("_", " ").title())
    ax.set_title(f"{display_name} — metric profile", fontsize=10.5, fontweight="bold", pad=8)
    ax.set_ylabel("Score", fontsize=9.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, _, lbl in metrics], fontsize=9.5)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(output_path)}")


# ---------------------------------------------------------------------------
# Console summary table
# ---------------------------------------------------------------------------

def print_summary(mix_top: list, single_top: list) -> None:
    sep = "=" * 88
    print(f"\n{sep}")
    print(f"{'TOP 3 MODELS — Multi-Seed Aggregated Results':^88}")
    print(sep)

    for dataset_name, top_list in [("Mix",  mix_top), ("Single", single_top)]:
        print(f"\n  {'Dataset: ' + dataset_name}")
        print(f"  {'Rank':<6} {'Model':<22} {'Best Strategy':<22} "
              f"{'Bal. Acc.':>12}  {'F1 Macro':>12}  {'Accuracy':>12}")
        print(f"  {'-'*84}")
        for i, row in enumerate(top_list, 1):
            print(f"  {i:<6} {row['display']:<22} {row['strategy']:<22} "
                  f"  {row['ba_mean']:.4f}±{row['ba_std']:.4f}"
                  f"  {row['f1_mean']:.4f}±{row['f1_std']:.4f}"
                  f"  {row['acc_mean']:.4f}±{row['acc_std']:.4f}")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  Scientific Top-3 Model Visualizer")
    print("=" * 60)

    # Load data
    print("\n[1/4] Loading aggregated multi-seed results …")
    mix_data    = load_aggregated(MIX_AGGR_PATH)
    single_data = load_aggregated(SINGLE_AGGR_PATH)

    mix_all    = best_per_model(mix_data)
    single_all = best_per_model(single_data)

    mix_top3    = mix_all[:TOP_N]
    single_top3 = single_all[:TOP_N]

    print_summary(mix_top3, single_top3)

    # Assign consistent colors
    color_map = build_color_map(mix_top3 + mix_all, single_top3 + single_all)

    # ---- Per-metric panels ----
    print("[2/4] Generating per-metric panels …")
    for metric_key, metric_label, panel_ltr in METRIC_CFG:
        out = os.path.join(OUTPUT_DIR, f"top3_{metric_key}_panel.png")
        # Map JSON key → internal prefix ("ba", "f1", "acc")
        key_map  = {"balanced_accuracy": "ba", "f1_macro": "f1", "accuracy": "acc"}
        internal = key_map.get(metric_key, metric_key)
        create_per_dataset_figure(
            mix_top3, single_top3, color_map,
            metric_key=internal,
            metric_label=metric_label,
            panel_label=panel_ltr,
            output_path=out,
        )

    # ---- Combined 2×2 figure ----
    print("[3/4] Generating combined publication figure …")
    combined_out = os.path.join(OUTPUT_DIR, "top3_combined_figure.png")
    create_combined_figure(mix_top3, single_top3, color_map, combined_out)

    # ---- Per-model profiles for models appearing in either top-3 ----
    print("[4/4] Generating per-model metric profiles …")
    seen_models = {r["model"] for r in mix_top3 + single_top3}
    for model_key in seen_models:
        out = os.path.join(OUTPUT_DIR, f"profile_{model_key}.png")
        create_model_profile(
            model_key,
            mix_all, single_all,
            color=color_map[model_key],
            output_path=out,
        )

    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
