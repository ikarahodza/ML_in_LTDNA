"""
Influence functions for TabPFN, XGBoost, and LightGBM.

TabPFN -- Leave-One-Out (LOO) influence:
  TabPFN is an in-context learner: "training" is just setting the context
  window, so there are no gradient-updated weights.  This means classical
  Koh & Liang influence functions (which rely on the Hessian of training
  loss) do not apply.  Instead we use exact LOO:

    I_LOO(z_i, z_test) = f(z_test | context minus z_i) - f(z_test | context)

  For N_TRAIN=200 training samples and N_TEST=10 test samples this requires
  200 additional fit+predict_proba calls.  Each call takes ~1-3 s on CPU,
  so total runtime is roughly 5-10 minutes.  Set N_TRAIN_LOO to a smaller
  value (e.g. 50) to trade accuracy for speed.

XGBoost / LightGBM -- Leaf similarity influence:
  Each tree partitions the input space into leaves.  The influence of
  training sample z_i on test prediction z_test is measured as the fraction
  of trees where both samples end up in the same leaf:

    leaf_sim(z_i, z_test) = (1/T) sum_t 1[leaf_t(z_i) == leaf_t(z_test)]

  This naturally identifies training samples that "support" each prediction
  in the model's learned feature space -- closely related to representer-point
  methods for kernel models.

Outputs in influence_results/:
  tabpfn_loo_heatmap.png           LOO influence matrix (N_TEST x N_TRAIN)
  tabpfn_loo_top_influences.png    Top-10 influential training samples per test
  xgb_leaf_similarity.png          Leaf-sim matrix for XGBoost
  lgbm_leaf_similarity.png         Leaf-sim matrix for LightGBM
  tabpfn_loo_influence.csv         Full LOO influence matrix
  xgb_leaf_similarity.csv          Full XGBoost leaf-sim matrix
  lgbm_leaf_similarity.csv         Full LightGBM leaf-sim matrix
"""

import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
TABPFN_MODEL_PATH = "Mix_Results/models_v2/model_tabpfn_original_v2.pkl"
XGB_MODEL_PATH    = "Mix_Results/models/model_xgboost_smote.pkl"
LGBM_MODEL_PATH   = "Mix_Results/models/lgbm_model_lgbm_smote.pkl"
DATA_PATH         = "Input_Data-Mix-train.csv"

N_TRAIN         = 200     # full training context
N_TRAIN_LOO     = 50      # LOO runs on the first N_TRAIN_LOO context samples
                          # (increase to 200 for full analysis; ~10 min on CPU)
N_TEST          = 10      # test samples to explain
TOP_K           = 10      # top-K influential samples shown per test sample

CLASS_NAMES = ["Allele", "Back_Stutter", "Forward_Stutter", "Pull_Up"]
OUT_DIR     = Path("influence_results")
OUT_DIR.mkdir(exist_ok=True)


# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data ...")
data       = pd.read_csv(DATA_PATH)
features   = data.drop(columns=["sample", "PoI", "Classification"])
target     = data["Classification"]
feat_names = features.columns.tolist()

X_ctx_df  = features.iloc[:N_TRAIN]
X_test_df = features.iloc[N_TRAIN : N_TRAIN + N_TEST]

X_ctx  = X_ctx_df.to_numpy(dtype=float)
y_ctx  = target.iloc[:N_TRAIN].to_numpy()
X_test = X_test_df.to_numpy(dtype=float)
y_test = target.iloc[N_TRAIN : N_TRAIN + N_TEST].to_numpy()

ctx_sample_ids = data["sample"].iloc[:N_TRAIN].tolist()
ctx_labels     = y_ctx


# =============================================================================
# Part 1 -- TabPFN Leave-One-Out influence
# =============================================================================
print("\n-- TabPFN Leave-One-Out Influence -----------------------------------")
print(f"  Context size : {N_TRAIN}")
print(f"  LOO samples  : {N_TRAIN_LOO}  (first {N_TRAIN_LOO} context rows)")
print(f"  Test samples : {N_TEST}")
print(f"  Expected runtime: {N_TRAIN_LOO * 2:.0f}-{N_TRAIN_LOO * 5:.0f} s")

tabpfn = joblib.load(TABPFN_MODEL_PATH)

# Baseline: predict with full context
tabpfn.fit(X_ctx, y_ctx)
proba_full = tabpfn.predict_proba(X_test)   # (N_TEST, n_classes)
pred_full  = [CLASS_NAMES[p] for p in proba_full.argmax(axis=1)]

print("\nBaseline predictions (full context):")
for i, (true, pred, pr) in enumerate(zip(y_test, pred_full, proba_full)):
    status = "OK " if true == pred else "XXX"
    print(f"  [{status}] sample {N_TRAIN+i}: true={true:18s}  pred={pred:18s}  conf={pr.max():.2f}")

# LOO influence matrix: (N_TRAIN_LOO, N_TEST, n_classes)
loo_influence = np.zeros((N_TRAIN_LOO, N_TEST, len(CLASS_NAMES)))

t0 = time.time()
for i in range(N_TRAIN_LOO):
    mask   = np.arange(N_TRAIN) != i
    X_loo  = X_ctx[mask]
    y_loo  = y_ctx[mask]

    tabpfn.fit(X_loo, y_loo)
    proba_loo = tabpfn.predict_proba(X_test)

    # Influence = change in probability when training sample i is removed
    loo_influence[i] = proba_loo - proba_full

    elapsed = time.time() - t0
    eta     = elapsed / (i + 1) * (N_TRAIN_LOO - i - 1)
    print(f"  LOO {i+1:3d}/{N_TRAIN_LOO}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s", end="\r")

print(f"\n  Done in {time.time() - t0:.0f} s")

# Restore full-context fit
tabpfn.fit(X_ctx, y_ctx)

# Scalar influence per (training sample, test sample)
loo_scalar = np.abs(loo_influence).max(axis=2)   # (N_TRAIN_LOO, N_TEST)

# Save CSV
loo_df = pd.DataFrame(
    loo_scalar,
    index=[f"ctx_{i}_({ctx_labels[i][:4]})" for i in range(N_TRAIN_LOO)],
    columns=[f"test_{N_TRAIN+j}_({y_test[j][:4]})" for j in range(N_TEST)],
)
loo_df.to_csv(OUT_DIR / "tabpfn_loo_influence.csv")
print(f"Saved: {OUT_DIR / 'tabpfn_loo_influence.csv'}")

# Plot 1a: LOO influence heatmap
fig, ax = plt.subplots(figsize=(max(8, N_TEST * 0.8 + 2), max(6, N_TRAIN_LOO * 0.15 + 2)))
vmax = loo_scalar.max()
im = ax.imshow(loo_scalar, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
ax.set_xlabel("Test sample (true -> pred)", fontsize=9)
ax.set_ylabel(f"Training sample (first {N_TRAIN_LOO})", fontsize=9)
ax.set_xticks(range(N_TEST))
ax.set_xticklabels(
    [f"{N_TRAIN+j}\n{y_test[j][:4]}->{pred_full[j][:4]}" for j in range(N_TEST)],
    fontsize=7,
)
ax.set_yticks([])
ax.set_title(
    "TabPFN Leave-One-Out Influence\n"
    "max |delta_prob| when training sample is removed",
    fontsize=11,
)
plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="max |delta P|")
fig.tight_layout()
path = OUT_DIR / "tabpfn_loo_heatmap.png"
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path}")

# Plot 1b: Top-K influential training samples per test sample
fig, axes = plt.subplots(1, N_TEST, figsize=(2.5 * N_TEST, 6), sharey=False)
fig.suptitle(
    f"TabPFN: top-{TOP_K} most influential training samples per test prediction\n"
    f"(LOO over first {N_TRAIN_LOO} context samples; value = max |delta P| when removed)",
    fontsize=10,
)

for j, ax in enumerate(axes):
    influences = loo_scalar[:, j]
    top_k_idx  = np.argsort(influences)[::-1][:TOP_K]

    bar_vals   = influences[top_k_idx][::-1]
    bar_labels = [f"ctx_{i}\n({ctx_labels[i][:5]})" for i in top_k_idx[::-1]]
    bar_colors = [
        "tomato"    if ctx_labels[i] != y_test[j] else "steelblue"
        for i in top_k_idx[::-1]
    ]

    ax.barh(range(TOP_K), bar_vals, color=bar_colors)
    ax.set_yticks(range(TOP_K))
    ax.set_yticklabels(bar_labels, fontsize=6)
    ax.set_xlabel("max|dP|", fontsize=7)
    ax.set_title(
        f"Test {N_TRAIN+j}\n{y_test[j][:5]}->{pred_full[j][:5]}",
        fontsize=8,
        color="black" if y_test[j] == pred_full[j] else "crimson",
    )
    ax.grid(axis="x", alpha=0.3)
    ax.tick_params(axis="x", labelsize=6)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="steelblue", label="Same class as test"),
    Patch(facecolor="tomato",    label="Different class"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=8,
           bbox_to_anchor=(0.5, -0.03))
fig.tight_layout()
path = OUT_DIR / "tabpfn_loo_top_influences.png"
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path}")


# =============================================================================
# Part 2 -- XGBoost leaf similarity
# =============================================================================
print("\n-- XGBoost Leaf Similarity ------------------------------------------")

xgb_model = joblib.load(XGB_MODEL_PATH)


def _unwrap_model(model):
    """Unwrap sklearn Pipeline to the underlying estimator."""
    if hasattr(model, "named_steps"):
        for step in reversed(list(model.named_steps.values())):
            if hasattr(step, "apply") or hasattr(step, "predict_proba"):
                return step
    return model


xgb_inner = _unwrap_model(xgb_model)

# XGBoost apply() requires a DataFrame with feature names to match the booster
X_ctx_loo_df  = X_ctx_df.iloc[:N_TRAIN_LOO]
leaves_train_xgb = xgb_inner.apply(X_ctx_loo_df)   # (N_TRAIN_LOO, n_trees)
leaves_test_xgb  = xgb_inner.apply(X_test_df)       # (N_TEST, n_trees)

n_trees = leaves_train_xgb.shape[1]
leaf_sim_xgb = np.zeros((N_TRAIN_LOO, N_TEST))
for i in range(N_TRAIN_LOO):
    for j in range(N_TEST):
        leaf_sim_xgb[i, j] = (leaves_train_xgb[i] == leaves_test_xgb[j]).mean()

print(f"  XGBoost: {n_trees} trees, leaf-sim computed")

xgb_leaf_df = pd.DataFrame(
    leaf_sim_xgb,
    index=[f"ctx_{i}_({ctx_labels[i][:4]})" for i in range(N_TRAIN_LOO)],
    columns=[f"test_{N_TRAIN+j}_({y_test[j][:4]})" for j in range(N_TEST)],
)
xgb_leaf_df.to_csv(OUT_DIR / "xgb_leaf_similarity.csv")
print(f"Saved: {OUT_DIR / 'xgb_leaf_similarity.csv'}")


def plot_leaf_similarity(leaf_sim, model_name, ctx_lbl, test_lbl, test_preds,
                         fname_stem, top_k=TOP_K):
    """Heatmap + top-K bar chart for a leaf-similarity matrix."""
    n_train_loo, n_test_ = leaf_sim.shape

    fig, ax_h = plt.subplots(figsize=(max(8, n_test_ * 0.8 + 2), max(6, n_train_loo * 0.15 + 2)))
    im = ax_h.imshow(leaf_sim, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax_h.set_xlabel("Test sample", fontsize=9)
    ax_h.set_ylabel(f"Training sample (first {n_train_loo})", fontsize=9)
    ax_h.set_xticks(range(n_test_))
    ax_h.set_xticklabels(
        [f"{N_TRAIN+j}\n{y_test[j][:4]}->{test_preds[j][:4]}" for j in range(n_test_)],
        fontsize=7,
    )
    ax_h.set_yticks([])
    ax_h.set_title(
        f"{model_name} -- Leaf similarity matrix\n"
        "Fraction of trees where training and test sample share a leaf",
        fontsize=10,
    )
    plt.colorbar(im, ax=ax_h, fraction=0.02, pad=0.02, label="Fraction shared leaves")
    fig.tight_layout()
    path = OUT_DIR / f"{fname_stem}_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # Top-K bar chart per test sample
    fig, axes = plt.subplots(1, n_test_, figsize=(2.5 * n_test_, 6), sharey=False)
    fig.suptitle(
        f"{model_name}: top-{top_k} most similar training samples per test prediction\n"
        "(blue = same class as test, red = different class)",
        fontsize=10,
    )
    for j, ax in enumerate(axes):
        top_idx  = np.argsort(leaf_sim[:, j])[::-1][:top_k]
        vals     = leaf_sim[top_idx[::-1], j]
        labels   = [f"c{i}\n{ctx_lbl[i][:4]}" for i in top_idx[::-1]]
        colors   = [
            "steelblue" if ctx_lbl[i] == y_test[j] else "tomato"
            for i in top_idx[::-1]
        ]
        ax.barh(range(top_k), vals, color=colors)
        ax.set_yticks(range(top_k))
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_title(
            f"T{N_TRAIN+j}\n{y_test[j][:4]}->{test_preds[j][:4]}",
            fontsize=7,
            color="black" if y_test[j] == test_preds[j] else "crimson",
        )
        ax.set_xlim(0, 1)
        ax.tick_params(axis="x", labelsize=6)
        ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    path = OUT_DIR / f"{fname_stem}_top_influences.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


xgb_proba = xgb_inner.predict_proba(X_test_df)
xgb_preds = [CLASS_NAMES[p] for p in xgb_proba.argmax(axis=1)]

plot_leaf_similarity(
    leaf_sim_xgb,
    model_name="XGBoost+SMOTE",
    ctx_lbl=ctx_labels.tolist(),
    test_lbl=y_test.tolist(),
    test_preds=xgb_preds,
    fname_stem="xgb_leaf_similarity",
)


# =============================================================================
# Part 3 -- LightGBM leaf similarity
# =============================================================================
print("\n-- LightGBM Leaf Similarity -----------------------------------------")

try:
    lgbm_model = joblib.load(LGBM_MODEL_PATH)
    lgbm_inner = _unwrap_model(lgbm_model)

    # LightGBM pred_leaf=True returns (n_samples, n_trees * n_classes)
    leaves_train_lgbm = lgbm_inner.predict(
        X_ctx_loo_df, pred_leaf=True
    )  # (N_TRAIN_LOO, n_trees * n_classes)
    leaves_test_lgbm  = lgbm_inner.predict(
        X_test_df, pred_leaf=True
    )  # (N_TEST, n_trees * n_classes)

    n_leaf_cols = leaves_train_lgbm.shape[1]
    leaf_sim_lgbm = np.zeros((N_TRAIN_LOO, N_TEST))
    for i in range(N_TRAIN_LOO):
        for j in range(N_TEST):
            leaf_sim_lgbm[i, j] = (
                leaves_train_lgbm[i] == leaves_test_lgbm[j]
            ).mean()

    print(f"  LightGBM: {n_leaf_cols} leaf columns, leaf-sim computed")

    lgbm_leaf_df = pd.DataFrame(
        leaf_sim_lgbm,
        index=[f"ctx_{i}_({ctx_labels[i][:4]})" for i in range(N_TRAIN_LOO)],
        columns=[f"test_{N_TRAIN+j}_({y_test[j][:4]})" for j in range(N_TEST)],
    )
    lgbm_leaf_df.to_csv(OUT_DIR / "lgbm_leaf_similarity.csv")
    print(f"Saved: {OUT_DIR / 'lgbm_leaf_similarity.csv'}")

    lgbm_proba = lgbm_inner.predict_proba(X_test_df)
    lgbm_preds = [CLASS_NAMES[p] for p in lgbm_proba.argmax(axis=1)]

    plot_leaf_similarity(
        leaf_sim_lgbm,
        model_name="LightGBM+SMOTE",
        ctx_lbl=ctx_labels.tolist(),
        test_lbl=y_test.tolist(),
        test_preds=lgbm_preds,
        fname_stem="lgbm_leaf_similarity",
    )

except Exception as e:
    print(f"  LightGBM section skipped: {e}")


# =============================================================================
# Part 4 -- Summary: prototype identification
# =============================================================================
print("\n-- Prototype identification (cross-model) ---------------------------")

summary_rows = []
for j in range(N_TEST):
    row = {
        "test_sample":  N_TRAIN + j,
        "true_label":   y_test[j],
        "tabpfn_pred":  pred_full[j],
        "xgb_pred":     xgb_preds[j],
    }

    top_loo = int(np.argmax(loo_scalar[:, j]))
    row["loo_top_ctx_idx"]   = top_loo
    row["loo_top_label"]     = ctx_labels[top_loo]
    row["loo_top_influence"] = float(loo_scalar[top_loo, j])

    top_xgb = int(np.argmax(leaf_sim_xgb[:, j]))
    row["xgb_top_ctx_idx"]   = top_xgb
    row["xgb_top_label"]     = ctx_labels[top_xgb]
    row["xgb_top_leaf_sim"]  = float(leaf_sim_xgb[top_xgb, j])

    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_path = OUT_DIR / "influence_summary.csv"
summary_df.to_csv(summary_path, index=False)
print(f"Saved: {summary_path}")
print(summary_df.to_string(index=False))

print("\nDone.  Results in:", OUT_DIR)
