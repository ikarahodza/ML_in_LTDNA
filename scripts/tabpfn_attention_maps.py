"""
Attention map visualisation for the pre-trained TabPFN model.

TabPFN is a transformer that processes (train + test) samples together.
Each of its 24 layers has two types of attention:

  between_items    – items (samples) attending to items
                     → test→train slice shows which training examples
                       support each prediction
  between_features – features attending to features across the sequence
                     → shows which input dimensions co-vary in the model's
                       representation

Both are extracted via forward hooks that manually compute
softmax(QK^T / sqrt(d_k)) from the captured input and the layer's
weight matrices (bypassing PyTorch 2.0 flash-attention which does not
return weights).
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH  = "Mix_Results/models_v2/model_tabpfn_original_v2.pkl"
DATA_PATH   = "Input_Data-Mix-train.csv"

N_TRAIN     = 200          # context samples fed to the transformer
N_TEST      = 10           # test samples to explain
LAYERS_VIZ  = [0, 5, 11, 17, 23]   # which of the 24 layers to plot

OUT_DIR     = Path("attention_maps")
OUT_DIR.mkdir(exist_ok=True)

CLASS_NAMES = ["Allele", "Back_Stutter", "Forward_Stutter", "Pull_Up"]
# ──────────────────────────────────────────────────────────────────────────────


# ── Load data & model ─────────────────────────────────────────────────────────
print("Loading model and data …")
model = joblib.load(MODEL_PATH)

data     = pd.read_csv(DATA_PATH)
features = data.drop(columns=["sample", "PoI", "Classification"])
target   = data["Classification"]
feat_names = features.columns.tolist()   # 305 feature names

# Use first N_TRAIN rows as context, next N_TEST as queries
X_ctx  = features.iloc[:N_TRAIN].to_numpy(dtype=float)
y_ctx  = target.iloc[:N_TRAIN].to_numpy()
X_test = features.iloc[N_TRAIN : N_TRAIN + N_TEST].to_numpy(dtype=float)
y_test = target.iloc[N_TRAIN : N_TRAIN + N_TEST].to_numpy()


# ── Hook storage ──────────────────────────────────────────────────────────────
# Each entry: layer_idx → np.ndarray of shape (n_items, n_items) [item]
#                                           or (n_feat_blocks, n_feat_blocks) [feat]
# Each layer may be called multiple times (TabPFN batches train data internally).
# We accumulate all calls and average, using the last N_TEST rows as test items.
item_attn_accum: dict[int, list[np.ndarray]] = {}
feat_attn_accum: dict[int, list[np.ndarray]] = {}
item_attn: dict[int, np.ndarray] = {}
feat_attn: dict[int, np.ndarray] = {}


def _attn_weights(module: torch.nn.Module, x: torch.Tensor) -> np.ndarray | None:
    """
    Compute softmax attention weights from an input tensor and a
    MultiHeadAttention module's weight matrices.

    x shape: (flat_batch, seq_len, d_model)
    Returns: (flat_batch, n_heads, seq_len, seq_len)
    """
    if not hasattr(module, "_w_qkv") or module._w_qkv is None:
        return None

    with torch.no_grad():
        # _w_qkv: (3=qkv, n_heads, d_k, d_model)
        w_q  = module._w_qkv[0]        # (n_heads, d_k, d_model)
        w_kv = module._w_qkv[1:]       # (2, n_heads, d_k, d_model)

        # q, k: (flat_batch, seq_len, n_heads, d_k)
        q  = torch.einsum("b n s, h d s -> b n h d", x, w_q)
        kv = torch.einsum("b n s, j h d s -> b n j h d", x, w_kv)
        k  = kv[:, :, 0]               # (flat_batch, seq_len, n_heads, d_k)

        d_k    = q.shape[-1]
        scores = torch.einsum("b q h d, b k h d -> b h q k", q, k) / (d_k ** 0.5)
        return torch.softmax(scores, dim=-1).cpu().numpy()
        # → (flat_batch, n_heads, seq_len, seq_len)


def make_item_hook(layer_idx: int):
    def hook(module, inp, _out):
        x = inp[0]                              # (batch, n_items, n_feat_blocks, d_model)
        batch, n_items, n_feats, d_model = x.shape
        # Only process calls where test items are present (n_items > N_TEST).
        # Test items sit at the END of the item sequence; training items come first.
        if n_items <= N_TEST:
            return
        x2 = x.permute(0, 2, 1, 3).reshape(batch * n_feats, n_items, d_model)
        w = _attn_weights(module, x2)
        if w is not None:
            # w: (batch*n_feats, n_heads, n_items, n_items)
            avg = w.mean(axis=(0, 1))           # → (n_items, n_items)
            # Keep only the test→train block: last N_TEST rows, all-but-last-N_TEST cols
            block = avg[-N_TEST:, :-N_TEST]     # (N_TEST, n_train_in_batch)
            if layer_idx not in item_attn_accum:
                item_attn_accum[layer_idx] = []
            item_attn_accum[layer_idx].append(block)
    return hook


def make_feat_hook(layer_idx: int):
    def hook(module, inp, _out):
        x = inp[0]                              # (batch, n_items, n_feat_blocks, d_model)
        batch, n_items, n_feats, d_model = x.shape
        x2 = x.reshape(batch * n_items, n_feats, d_model)
        w = _attn_weights(module, x2)
        if w is not None:
            avg = w.mean(axis=(0, 1))           # → (n_feats, n_feats)
            if layer_idx not in feat_attn_accum:
                feat_attn_accum[layer_idx] = []
            feat_attn_accum[layer_idx].append(avg)
    return hook


# ── Fit first, then register hooks on the live model object ──────────────────
# fit() may rebuild the executor; hooks must be registered after fit() so they
# point to the model object that iter_outputs() will actually call.
print(f"Running predict_proba (context={N_TRAIN}, test={N_TEST}) …")
model.fit(X_ctx, y_ctx)

inner = model.model_
hooks = []
for i, layer in enumerate(inner.transformer_encoder.layers):
    hooks.append(
        layer.self_attn_between_items.register_forward_hook(make_item_hook(i))
    )
    if layer.self_attn_between_features is not None:
        hooks.append(
            layer.self_attn_between_features.register_forward_hook(make_feat_hook(i))
        )

proba = model.predict_proba(X_test)

for h in hooks:
    h.remove()

# Collapse accumulated lists into single arrays by averaging
for layer_idx, blocks in item_attn_accum.items():
    # blocks is a list of (N_TEST, n_train_in_batch) arrays with varying widths;
    # average over calls (mean across sub-batches, then normalise rows)
    summed = np.zeros((N_TEST, 1))
    for b in blocks:
        summed = np.hstack([summed, b]) if summed.shape[1] > 1 else b
    # Instead, just use the widest block (most training context captured)
    widest = max(blocks, key=lambda b: b.shape[1])
    item_attn[layer_idx] = widest

for layer_idx, mats in feat_attn_accum.items():
    # Average feature attention across all sub-batch calls
    # mats may have different sizes; use the largest
    largest = max(mats, key=lambda m: m.shape[0])
    feat_attn[layer_idx] = largest

print(f"  Captured item-attn from {len(item_attn)} layers")
print(f"  Captured feat-attn from {len(feat_attn)} layers")

pred_classes = [CLASS_NAMES[p] for p in proba.argmax(axis=1)]
print("\nPredictions:")
for i, (true, pred, prob) in enumerate(zip(y_test, pred_classes, proba)):
    print(f"  sample {N_TRAIN+i}: true={true}, pred={pred}, conf={prob.max():.2f}")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 1 – Test→Train attention across layers (one plot per layer in LAYERS_VIZ)
# ═══════════════════════════════════════════════════════════════════════════════
print("\nPlotting item attention …")

fig, axes = plt.subplots(
    1, len(LAYERS_VIZ),
    figsize=(5 * len(LAYERS_VIZ), max(4, N_TEST * 0.6 + 2)),
)
fig.suptitle(
    "Item attention: test samples → training context\n"
    "(higher = test sample attends more to that training example)",
    fontsize=11
)

for ax, layer_idx in zip(axes, LAYERS_VIZ):
    if layer_idx not in item_attn:
        ax.set_visible(False)
        continue
    # item_attn[layer_idx] is already the (N_TEST, n_train_in_batch) block
    block = item_attn[layer_idx]
    n_test_rows, n_train_cols = block.shape

    # TabPFN sub-samples training data internally, so we can't map columns back
    # to original class labels — plot raw attention as-is.
    im = ax.imshow(block, aspect="auto", cmap="viridis", vmin=0)
    ax.set_title(f"Layer {layer_idx}", fontsize=9)
    ax.set_xlabel(f"Training sub-batch (n={n_train_cols})", fontsize=7)
    if ax is axes[0]:
        ax.set_ylabel("Test sample", fontsize=8)
        ax.set_yticks(range(n_test_rows))
        ax.set_yticklabels(
            [f"{N_TRAIN+i}: {y_test[i][:4] if i < len(y_test) else '?'}"
             for i in range(n_test_rows)],
            fontsize=7
        )
    else:
        ax.set_yticks([])

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

fig.tight_layout()
out_path = OUT_DIR / "item_attention_test_to_train.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 2 – Feature attention: top-N attended features, averaged over all layers
# ═══════════════════════════════════════════════════════════════════════════════
print("Plotting feature attention …")

TOP_N = 30   # show top-N features by total attention received

if feat_attn:
    # Average across selected layers
    stacked = np.stack(
        [feat_attn[l] for l in LAYERS_VIZ if l in feat_attn], axis=0
    )
    avg_feat = stacked.mean(axis=0)     # (n_feat_blocks, n_feat_blocks)

    # n_feat_blocks = n_features + 1 (last block = target encoding)
    n_blocks = avg_feat.shape[0]
    block_names = feat_names + ["[target]"] if n_blocks > len(feat_names) else feat_names
    block_names = block_names[:n_blocks]

    # Rank features by total attention received (column sum = how much others attend to it)
    col_sums = avg_feat.sum(axis=0)
    top_idx  = np.argsort(col_sums)[::-1][:TOP_N]
    top_mat  = avg_feat[np.ix_(top_idx, top_idx)]
    top_names = [block_names[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(top_mat, aspect="auto", cmap="plasma")
    ax.set_xticks(range(TOP_N))
    ax.set_xticklabels(top_names, rotation=90, fontsize=7)
    ax.set_yticks(range(TOP_N))
    ax.set_yticklabels(top_names, fontsize=7)
    ax.set_title(
        f"Feature co-attention (top {TOP_N} by attention received)\n"
        f"Averaged over layers {LAYERS_VIZ}",
        fontsize=11
    )
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    fig.tight_layout()
    out_path = OUT_DIR / "feature_attention_top30.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

    # Also save ranked feature list
    ranked = pd.DataFrame({
        "feature":           [block_names[i] for i in np.argsort(col_sums)[::-1]],
        "attention_received": col_sums[np.argsort(col_sums)[::-1]],
    })
    csv_path = OUT_DIR / "feature_attention_ranking.csv"
    ranked.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    print("\nTop 10 most attended features:")
    print(ranked.head(10).to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 3 – Per-layer evolution of item attention entropy
#           (low entropy = model is confident about which samples to attend to)
# ═══════════════════════════════════════════════════════════════════════════════
def entropy(w: np.ndarray, axis: int = -1) -> np.ndarray:
    w = np.clip(w, 1e-12, None)
    return -(w * np.log(w)).sum(axis=axis)

if item_attn:
    layers_sorted = sorted(item_attn.keys())
    # For each layer, take test→train block and compute mean entropy per test sample
    entropies = []
    for l in layers_sorted:
        # item_attn[l] is already the (N_TEST, n_train_in_batch) block
        block = item_attn[l]
        block = block / block.sum(axis=1, keepdims=True).clip(1e-12)
        entropies.append(entropy(block, axis=1).mean())

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(layers_sorted, entropies, marker="o", color="steelblue")
    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel("Mean attention entropy (test→train)", fontsize=10)
    ax.set_title(
        "Attention entropy across layers\n"
        "Low = model strongly focuses on specific training examples",
        fontsize=11
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = OUT_DIR / "item_attention_entropy.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {out_path}")

print("\nDone. Results in:", OUT_DIR)
