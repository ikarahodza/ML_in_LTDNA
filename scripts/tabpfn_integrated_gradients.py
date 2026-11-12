"""
Integrated Gradients for TabPFN and SHAP/IG for XGBoost.

TabPFN (transformer-based, not trained on this dataset):
  IG is approximated via finite differences along the path from baseline to
  input.  For each alpha step we batch ALL feature perturbations into a single
  predict_proba call (306 test rows per call), keeping total calls to
  N_STEPS x N_TEST.

  IG_j = (x_j - b_j) * (1/N) sum_alpha [f(x_alpha + eps*e_j) - f(x_alpha)] / eps
  where x_alpha = b + alpha * (x - b)

  The finite-difference gradient satisfies the completeness axiom to the degree
  that N_STEPS is large; 20 steps is typically sufficient for monotone paths.

XGBoost:
  SHAP TreeExplainer computes exact Shapley values, which satisfy the same
  axioms as IG (efficiency, symmetry, linearity, dummy).  Results are plotted
  side-by-side for comparison.

Outputs in ig_results/:
  ig_tabpfn_heatmap.png        IG per (test sample, feature), top-30 features
  ig_tabpfn_bar.png            Mean |IG| across test samples
  ig_xgboost_heatmap.png       SHAP values for XGBoost+SMOTE
  ig_xgboost_bar.png           Mean |SHAP| across test samples
  ig_comparison_plot.png       Side-by-side feature ranking comparison
  ig_comparison.csv            All scores for every method and feature
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────
TABPFN_MODEL_PATH  = "Mix_Results_PoI/models/model_tabpfn_original_v35.pkl"
XGB_MODEL_PATH     = "Mix_Results_PoI/models/model_xgboost_smote.pkl"
DATA_PATH          = "Input_Data-Mix-train.csv"

N_TRAIN    = 200          # training context for TabPFN
N_TEST     = 10           # test samples to explain
N_STEPS    = 20           # alpha interpolation steps for IG
EPSILON    = 1e-3         # finite-difference step size
TOP_N      = 30           # features shown in heatmaps

CLASS_NAMES = ["Allele", "Back_Stutter", "Forward_Stutter", "Pull_Up"]
OUT_DIR     = Path("ig_results")
OUT_DIR.mkdir(exist_ok=True)
# ──────────────────────────────────────────────────────────────────────────────


# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data ...")
data       = pd.read_csv(DATA_PATH)
features   = data.drop(columns=["sample", "PoI", "Classification"])
target     = data["Classification"]
feat_names = features.columns.tolist()
n_features = len(feat_names)

X_ctx     = features.iloc[:N_TRAIN].to_numpy(dtype=float)
y_ctx     = target.iloc[:N_TRAIN].to_numpy()
X_test    = features.iloc[N_TRAIN : N_TRAIN + N_TEST].to_numpy(dtype=float)
X_test_df = features.iloc[N_TRAIN : N_TRAIN + N_TEST]
y_test    = target.iloc[N_TRAIN : N_TRAIN + N_TEST].to_numpy()

baseline = X_ctx.mean(axis=0)   # mean training sample as IG baseline


# =============================================================================
# Part 1 -- TabPFN Integrated Gradients (finite differences)
# =============================================================================
print("\n-- TabPFN Integrated Gradients --------------------------------------")
tabpfn = joblib.load(TABPFN_MODEL_PATH)
tabpfn.fit(X_ctx, y_ctx)

proba_baseline = tabpfn.predict_proba(baseline[None, :])  # (1, n_classes)
proba_test     = tabpfn.predict_proba(X_test)             # (N_TEST, n_classes)
pred_classes   = [CLASS_NAMES[p] for p in proba_test.argmax(axis=1)]

print("Predictions:")
for i, (true, pred, pr) in enumerate(zip(y_test, pred_classes, proba_test)):
    status = "OK " if true == pred else "XXX"
    print(f"  [{status}] sample {N_TRAIN+i}: true={true:18s}  pred={pred:18s}  conf={pr.max():.2f}")

# IG computation: accumulate finite-difference gradients over alpha steps
grad_accum = np.zeros((N_TEST, n_features, len(CLASS_NAMES)))
alphas     = np.linspace(0, 1, N_STEPS + 1)[1:]   # exclude alpha=0

print(f"\nComputing IG ({N_STEPS} alpha steps, {n_features} features) ...")
print(f"  Each step: 1 predict_proba call with {N_TEST * (n_features + 1):,} test rows")

for step_idx, alpha in enumerate(alphas):
    rows_per_sample = n_features + 1
    X_batch = np.zeros((N_TEST * rows_per_sample, n_features))

    for i in range(N_TEST):
        x_alpha = baseline + alpha * (X_test[i] - baseline)
        start   = i * rows_per_sample
        X_batch[start] = x_alpha
        X_batch[start + 1 : start + rows_per_sample] = x_alpha
        for j in range(n_features):
            X_batch[start + 1 + j, j] += EPSILON

    proba_batch = tabpfn.predict_proba(X_batch)
    proba_batch = proba_batch.reshape(N_TEST, rows_per_sample, len(CLASS_NAMES))

    grad = (proba_batch[:, 1:, :] - proba_batch[:, 0:1, :]) / EPSILON
    grad_accum += grad

    print(f"  Step {step_idx + 1:2d}/{N_STEPS}  alpha={alpha:.2f}  "
          f"mean|grad|={np.abs(grad).mean():.5f}")

avg_grad  = grad_accum / N_STEPS
delta_x   = X_test - baseline
ig_tabpfn = delta_x[:, :, None] * avg_grad   # (N_TEST, n_features, n_classes)

ig_sum = ig_tabpfn.sum(axis=1)
completeness_err = np.abs(ig_sum - (proba_test - proba_baseline)).mean()
print(f"\n  Completeness error (should be small): {completeness_err:.4f}")

# Attribution for the predicted class per test sample
ig_pred = np.array([
    ig_tabpfn[i, :, proba_test[i].argmax()]
    for i in range(N_TEST)
])  # (N_TEST, n_features)

# Plot 1: heatmap
mean_abs_ig = np.abs(ig_pred).mean(axis=0)
top_idx     = np.argsort(mean_abs_ig)[::-1][:TOP_N]
top_names   = [feat_names[j] for j in top_idx]
top_ig      = ig_pred[:, top_idx]

ylabels = [
    f"{N_TRAIN+i}: {y_test[i][:6]}->{pred_classes[i][:6]}"
    for i in range(N_TEST)
]

fig, ax = plt.subplots(figsize=(14, 5))
vmax = np.abs(top_ig).max()
im   = ax.imshow(top_ig, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
ax.set_xticks(range(TOP_N))
ax.set_xticklabels(top_names, rotation=90, fontsize=7)
ax.set_yticks(range(N_TEST))
ax.set_yticklabels(ylabels, fontsize=8)
ax.set_title(
    f"TabPFN Integrated Gradients (top {TOP_N} features by mean |IG|)\n"
    "Red = pushes toward predicted class  |  Blue = pushes away",
    fontsize=11,
)
plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="IG score")
fig.tight_layout()
path = OUT_DIR / "ig_tabpfn_heatmap.png"
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {path}")

# Plot 2: Mean |IG| bar chart
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(range(TOP_N), mean_abs_ig[top_idx][::-1], color="steelblue")
ax.set_yticks(range(TOP_N))
ax.set_yticklabels(top_names[::-1], fontsize=8)
ax.set_xlabel("Mean |IG| across test samples", fontsize=10)
ax.set_title("TabPFN feature attribution -- Integrated Gradients", fontsize=11)
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
path = OUT_DIR / "ig_tabpfn_bar.png"
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path}")

# Save IG scores CSV
ig_rows = []
for i in range(N_TEST):
    for j, fname in enumerate(feat_names):
        ig_rows.append({
            "sample":        N_TRAIN + i,
            "true_label":    y_test[i],
            "pred_label":    pred_classes[i],
            "feature":       fname,
            "ig_allele":     ig_tabpfn[i, j, 0],
            "ig_back":       ig_tabpfn[i, j, 1],
            "ig_forward":    ig_tabpfn[i, j, 2],
            "ig_pull_up":    ig_tabpfn[i, j, 3],
            "ig_pred_class": ig_pred[i, j],
        })
ig_df    = pd.DataFrame(ig_rows)
csv_path = OUT_DIR / "ig_tabpfn_scores.csv"
ig_df.to_csv(csv_path, index=False)
print(f"Saved: {csv_path}")

print("\nTop 15 features by mean |IG| (TabPFN):")
top_feat_df = pd.DataFrame({
    "feature":     [feat_names[j] for j in np.argsort(mean_abs_ig)[::-1][:15]],
    "mean_abs_ig": np.sort(mean_abs_ig)[::-1][:15],
}).to_string(index=False)
print(top_feat_df)


# =============================================================================
# Part 2 -- XGBoost SHAP (TreeExplainer -- exact Shapley values)
# =============================================================================
print("\n\n-- XGBoost SHAP (TreeExplainer) -------------------------------------")

try:
    import shap
except ImportError:
    print("shap not installed -- skipping XGBoost section. pip install shap")
    sys.exit(0)

xgb_model = joblib.load(XGB_MODEL_PATH)


def _unwrap_model(model):
    if hasattr(model, "named_steps"):
        for step in reversed(list(model.named_steps.values())):
            if hasattr(step, "predict_proba"):
                return step
    return model


xgb_inner = _unwrap_model(xgb_model)

# SHAP TreeExplainer with training-data background
explainer  = shap.TreeExplainer(xgb_inner, data=X_ctx, feature_names=feat_names)
shap_values = explainer.shap_values(X_test)

# Normalise to (N_TEST, n_features, n_classes)
if isinstance(shap_values, list):
    shap_arr = np.stack(shap_values, axis=2)
else:
    shap_arr = shap_values

xgb_proba = xgb_inner.predict_proba(X_test_df)
xgb_preds = [CLASS_NAMES[p] for p in xgb_proba.argmax(axis=1)]

print("XGBoost predictions:")
for i, (true, pred, pr) in enumerate(zip(y_test, xgb_preds, xgb_proba)):
    status = "OK " if true == pred else "XXX"
    print(f"  [{status}] sample {N_TRAIN+i}: true={true:18s}  pred={pred:18s}  conf={pr.max():.2f}")

shap_pred = np.array([
    shap_arr[i, :, xgb_proba[i].argmax()]
    for i in range(N_TEST)
])  # (N_TEST, n_features)

mean_abs_shap = np.abs(shap_pred).mean(axis=0)
top_shap_idx  = np.argsort(mean_abs_shap)[::-1][:TOP_N]
top_shap_names = [feat_names[j] for j in top_shap_idx]

# Plot 3: XGBoost SHAP heatmap
top_shap_mat = shap_pred[:, top_shap_idx]
xgb_ylabels  = [
    f"{N_TRAIN+i}: {y_test[i][:6]}->{xgb_preds[i][:6]}"
    for i in range(N_TEST)
]

fig, ax = plt.subplots(figsize=(14, 5))
vmax = np.abs(top_shap_mat).max()
im   = ax.imshow(top_shap_mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
ax.set_xticks(range(TOP_N))
ax.set_xticklabels(top_shap_names, rotation=90, fontsize=7)
ax.set_yticks(range(N_TEST))
ax.set_yticklabels(xgb_ylabels, fontsize=8)
ax.set_title(
    f"XGBoost SHAP values (top {TOP_N} features by mean |SHAP|)\n"
    "Red = pushes toward predicted class  |  Blue = pushes away",
    fontsize=11,
)
plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="SHAP value")
fig.tight_layout()
path = OUT_DIR / "ig_xgboost_heatmap.png"
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {path}")

# Plot 4: XGBoost SHAP bar chart
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(range(TOP_N), mean_abs_shap[top_shap_idx][::-1], color="darkorange")
ax.set_yticks(range(TOP_N))
ax.set_yticklabels(top_shap_names[::-1], fontsize=8)
ax.set_xlabel("Mean |SHAP| across test samples", fontsize=10)
ax.set_title("XGBoost+SMOTE feature attribution -- SHAP TreeExplainer", fontsize=11)
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
path = OUT_DIR / "ig_xgboost_bar.png"
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path}")


# =============================================================================
# Part 3 -- Comparison: TabPFN IG vs XGBoost SHAP
# =============================================================================
print("\n-- Feature ranking comparison ----------------------------------------")

top30_tabpfn = {feat_names[j] for j in np.argsort(mean_abs_ig)[::-1][:30]}
top30_xgb    = {feat_names[j] for j in np.argsort(mean_abs_shap)[::-1][:30]}
overlap      = top30_tabpfn & top30_xgb

print(f"  Top-30 feature overlap between TabPFN-IG and XGB-SHAP: {len(overlap)}/30")
if overlap:
    print("  Shared features:")
    for f in sorted(overlap):
        ig_rank  = np.argsort(mean_abs_ig)[::-1].tolist().index(feat_names.index(f)) + 1
        shp_rank = np.argsort(mean_abs_shap)[::-1].tolist().index(feat_names.index(f)) + 1
        print(f"    {f:30s}  TabPFN rank={ig_rank:2d}  XGB rank={shp_rank:2d}")

# Side-by-side comparison plot
n_compare = 20
tabpfn_top20 = [
    (feat_names[j], float(mean_abs_ig[j]))
    for j in np.argsort(mean_abs_ig)[::-1][:n_compare]
]
xgb_top20 = [
    (feat_names[j], float(mean_abs_shap[j]))
    for j in np.argsort(mean_abs_shap)[::-1][:n_compare]
]

max_ig   = max(v for _, v in tabpfn_top20)
max_shap = max(v for _, v in xgb_top20)
tabpfn_top20_norm = [(n, v / max_ig)   for n, v in tabpfn_top20]
xgb_top20_norm    = [(n, v / max_shap) for n, v in xgb_top20]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
for ax, data, title, color in [
    (ax1, tabpfn_top20_norm, "TabPFN -- Integrated Gradients", "steelblue"),
    (ax2, xgb_top20_norm,    "XGBoost -- SHAP",                "darkorange"),
]:
    names, vals = zip(*data)
    ax.barh(range(n_compare), vals[::-1], color=color)
    ax.set_yticks(range(n_compare))
    ax.set_yticklabels(names[::-1], fontsize=8)
    ax.set_xlabel("Normalised attribution (relative to top feature)", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="x", alpha=0.3)

fig.suptitle(
    "Feature attribution comparison: TabPFN Integrated Gradients vs XGBoost SHAP\n"
    "(each model normalised to its own maximum)",
    fontsize=11,
)
fig.tight_layout()
path = OUT_DIR / "ig_comparison_plot.png"
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {path}")

# Merged CSV
comparison_rows = []
for j, fname in enumerate(feat_names):
    comparison_rows.append({
        "feature":        fname,
        "tabpfn_ig":      float(mean_abs_ig[j]),
        "xgb_shap":       float(mean_abs_shap[j]),
        "tabpfn_ig_rank": int(np.argsort(mean_abs_ig)[::-1].tolist().index(j) + 1),
        "xgb_shap_rank":  int(np.argsort(mean_abs_shap)[::-1].tolist().index(j) + 1),
    })

comp_df = pd.DataFrame(comparison_rows).sort_values("tabpfn_ig", ascending=False)
csv_path = OUT_DIR / "ig_comparison.csv"
comp_df.to_csv(csv_path, index=False)
print(f"Saved: {csv_path}")

print("\nDone.  Results in:", OUT_DIR)
