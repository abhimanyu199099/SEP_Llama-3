"""Generate poster figures for the SEP + Lookback Gate paper.

Saves to output/figures/:
  fig1_layer_auroc.png
  fig2_ood_heatmap.png
  fig3_sep_distribution.png
  fig4_gating_accuracy.png
  fig5_causal_validation.png

Usage:
    python generate_figures.py
"""
import os, pickle, warnings
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import make_interp_spline
from scipy.stats import gaussian_kde
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore", category=UserWarning)

# ── Style ─────────────────────────────────────────────────────────────────── #
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "legend.fontsize":   10,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.08,
})

DS_COLOR  = {"xsum": "#1976D2", "halueval_qa": "#E64A19"}
DS_LABEL  = {"xsum": "XSum",    "halueval_qa": "HaluEval QA"}
DATASETS  = ["xsum", "halueval_qa"]
SEP_TRIG  = 0.5          # inference trigger threshold (args.sep_threshold default)
OUT_DIR   = "output/figures"
os.makedirs(OUT_DIR, exist_ok=True)


# ── Inline accuracy helpers (avoids importing the LLM inference module) ───── #
def _lcs(x, y):
    m, n = len(x), len(y)
    prev = [0] * (n + 1)
    for i in range(m):
        curr = [0] * (n + 1)
        for j in range(n):
            curr[j+1] = prev[j] + 1 if x[i] == y[j] else max(prev[j+1], curr[j])
        prev = curr
    return prev[n]

def rouge_l(pred, ref):
    p, r = pred.lower().split(), ref.lower().split()
    if not p or not r: return 0.0
    lcs = _lcs(p, r)
    if lcs == 0: return 0.0
    return 2 * lcs / (len(p) + len(r))

def token_f1(pred, ref):
    from collections import Counter
    p, r = pred.lower().split(), ref.lower().split()
    if not p or not r: return 0.0
    common = sum((Counter(p) & Counter(r)).values())
    if common == 0: return 0.0
    return 2 * common / (len(p) + len(r))

def is_correct(answer, refs, use_rouge):
    fn, thresh = (rouge_l, 0.2) if use_rouge else (token_f1, 0.5)
    return any(fn(answer, r) >= thresh for r in refs)


# ── Data loaders ─────────────────────────────────────────────────────────── #
def load_features(ds):
    d = torch.load(f"output/{ds}/all_layers.pt", map_location="cpu")
    return d["X_tbg"].numpy(), d["entropy"].numpy()

def load_se_threshold(ds):
    with open(f"output/{ds}/sep_probe_TBG.pkl", "rb") as f:
        return float(pickle.load(f)["threshold"])

def load_gated(ds):
    with open(f"output/{ds}/gated_results.pkl", "rb") as f:
        return pickle.load(f)

def load_causal(ds):
    with open(f"output/{ds}/causal_validation.pkl", "rb") as f:
        return pickle.load(f)

def smooth(x, y, n=300):
    xs = np.linspace(x[0], x[-1], n)
    return xs, make_interp_spline(x, y, k=min(3, len(x)-1))(xs)


# ═══════════════════════════════════════════════════════════════════════════ #
# Fig 1 — Layer-wise Probe AUROC                                              #
# ═══════════════════════════════════════════════════════════════════════════ #
def fig1_layer_auroc():
    print("Fig 1: layer-wise AUROC …")
    best = {"xsum": (4, 9), "halueval_qa": (23, 28)}
    fig, ax = plt.subplots(figsize=(6.5, 3.8))

    for ds in DATASETS:
        X, entropy = load_features(ds)
        thresh = load_se_threshold(ds)
        labels = (entropy > thresh).astype(int)

        N, L, H = X.shape
        aurocs = []
        for l in range(L):
            feat = StandardScaler().fit_transform(X[:, l, :])
            clf  = LogisticRegression(max_iter=300, C=1.0, solver="lbfgs")
            clf.fit(feat, labels)
            aurocs.append(roc_auc_score(labels, clf.predict_proba(feat)[:, 1]))
        aurocs = np.array(aurocs)
        layers = np.arange(L)
        xs, ys = smooth(layers.astype(float), aurocs)

        ax.plot(xs, ys, color=DS_COLOR[ds], lw=2.2, label=DS_LABEL[ds])
        ax.scatter(layers[::2], aurocs[::2], color=DS_COLOR[ds], s=22, zorder=5)

        lo, hi = best[ds]
        ax.axvspan(lo, hi, alpha=0.10, color=DS_COLOR[ds])
        peak_y = aurocs[lo:hi+1].max()
        ax.annotate(f"Best [{lo},{hi})\nAUROC={aurocs[lo:hi].mean():.3f}",
                    xy=((lo+hi)/2, peak_y),
                    xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=8, color=DS_COLOR[ds],
                    arrowprops=dict(arrowstyle="-", color=DS_COLOR[ds], alpha=0.4))

    ax.axhline(0.5, color="grey", lw=1, ls="--", alpha=0.6, label="Random (0.5)")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("AUROC (TBG probe, single-layer)")
    ax.set_title("Layer-wise Hallucination Probe Performance")
    ax.set_xlim(0, L - 1)
    ax.set_ylim(0.44, 0.84)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.yaxis.grid(True, alpha=0.25, ls="--")
    fig.savefig(f"{OUT_DIR}/fig1_layer_auroc.png")
    plt.close(fig)
    print(f"  saved → {OUT_DIR}/fig1_layer_auroc.png")


# ═══════════════════════════════════════════════════════════════════════════ #
# Fig 2 — OOD Cross-dataset Heatmap                                           #
# ═══════════════════════════════════════════════════════════════════════════ #
def fig2_ood_heatmap():
    print("Fig 2: OOD heatmap …")
    # From probe_matrix.log — TBG token, rows=train, cols=test
    matrix = np.array([
        [0.6445, 0.6326],   # train=XSum       → test [XSum, HaluEval]
        [0.5583, 0.7452],   # train=HaluEval   → test [XSum, HaluEval]
    ])
    labels = [DS_LABEL[d] for d in DATASETS]

    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    cmap = LinearSegmentedColormap.from_list("sep", ["#F5F5F5", "#0D47A1"])
    im   = ax.imshow(matrix, cmap=cmap, vmin=0.50, vmax=0.78)

    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_xlabel("Test dataset")
    ax.set_ylabel("Train dataset")
    ax.set_title("Cross-dataset AUROC Matrix (TBG probe)")

    for i in range(2):
        for j in range(2):
            v    = matrix[i, j]
            diag = (i == j)
            txt  = f"{v:.3f}" + ("\n(ID)" if diag else "\n(OOD)")
            col  = "white" if v > 0.64 else "#111111"
            ax.text(j, i, txt, ha="center", va="center", color=col,
                    fontsize=12, fontweight="bold" if diag else "normal")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("AUROC")

    id_mean  = np.mean([matrix[0,0], matrix[1,1]])
    ood_mean = np.mean([matrix[0,1], matrix[1,0]])
    ax.set_xlabel(f"Test dataset\n\nID mean: {id_mean:.3f} | OOD mean: {ood_mean:.3f} | ratio: {ood_mean/id_mean:.2f}")

    fig.savefig(f"{OUT_DIR}/fig2_ood_heatmap.png")
    plt.close(fig)
    print(f"  saved → {OUT_DIR}/fig2_ood_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════════ #
# Fig 3 — SEP Score Distribution                                              #
# ═══════════════════════════════════════════════════════════════════════════ #
def fig3_sep_distribution():
    print("Fig 3: SEP distributions …")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    for ax, ds in zip(axes, DATASETS):
        res      = load_gated(ds)
        use_rg   = (ds == "xsum")

        correct, wrong = [], []
        for r in res:
            score = r.get("sep_score")
            if score is None: continue
            ans = r.get("most_likely_answer", "")
            refs = r.get("answers", [])
            if not refs: continue
            (correct if is_correct(ans, refs, use_rg) else wrong).append(score)

        xs = np.linspace(0, 1, 400)
        for scores, label, color in [
            (correct, f"Correct  (n={len(correct)})",      "#2E7D32"),
            (wrong,   f"Hallucinated (n={len(wrong)})", "#C62828"),
        ]:
            if len(scores) < 10: continue
            ys = gaussian_kde(scores, bw_method=0.10)(xs)
            ax.plot(xs, ys, color=color, lw=2, label=label)
            ax.fill_between(xs, ys, alpha=0.12, color=color)

        # SEP trigger threshold (0.5 = default args.sep_threshold, in [0,1])
        ax.axvline(SEP_TRIG, color="black", lw=1.4, ls="--", alpha=0.75)
        ymax = ax.get_ylim()[1]
        ax.text(SEP_TRIG + 0.02, ymax * 0.88, f"trigger\nθ={SEP_TRIG}",
                fontsize=8.5, va="top")

        ax.set_xlim(0, 1)
        ax.set_xlabel("SEP uncertainty score")
        ax.set_ylabel("Density")
        ax.set_title(DS_LABEL[ds])
        ax.legend(loc="upper left", fontsize=9)
        ax.yaxis.grid(True, alpha=0.25, ls="--")

    fig.suptitle("SEP Score: Correct vs Hallucinated Responses", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig3_sep_distribution.png")
    plt.close(fig)
    print(f"  saved → {OUT_DIR}/fig3_sep_distribution.png")


# ═══════════════════════════════════════════════════════════════════════════ #
# Fig 4 — Before / After Gating Accuracy                                      #
# ═══════════════════════════════════════════════════════════════════════════ #
def fig4_gating_accuracy():
    print("Fig 4: gating accuracy …")

    # Hard numbers from inference logs (fix4 run — most recent)
    # Overall accuracy across all samples
    orig  = {"xsum": 0.5141, "halueval_qa": 0.8070}
    gated = {"xsum": 0.5060, "halueval_qa": 0.8050}

    COND_COLORS = ["#546E7A", "#1565C0"]   # original=steel, gated=blue
    COND_LABELS = ["Original", "After Gating"]

    x     = np.arange(len(DATASETS))
    width = 0.32
    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    for ci, (vals, color, label) in enumerate(zip(
            [[orig[d]  for d in DATASETS], [gated[d] for d in DATASETS]],
            COND_COLORS, COND_LABELS)):
        offset = (ci - 0.5) * width
        bars   = ax.bar(x + offset, vals, width, color=color, label=label,
                        edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    # Delta arrows between paired bars
    for i, ds in enumerate(DATASETS):
        o, g  = orig[ds], gated[ds]
        delta = g - o
        sign  = "+" if delta >= 0 else ""
        col   = "#2E7D32" if delta >= 0 else "#C62828"
        ax.annotate("", xy=(x[i]+width/2, g), xytext=(x[i]-width/2, o),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.5))
        mid_x = x[i]
        mid_y = max(o, g) + 0.025
        ax.text(mid_x, mid_y, f"{sign}{delta*100:.1f}%",
                ha="center", color=col, fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([DS_LABEL[d] for d in DATASETS])
    ax.set_ylabel("Overall Accuracy")
    ax.set_title("Effect of SEP-Triggered Lookback Gating")
    ax.set_ylim(0, 1.0)
    ax.yaxis.grid(True, alpha=0.25, ls="--")
    ax.legend(loc="upper left")
    fig.savefig(f"{OUT_DIR}/fig4_gating_accuracy.png")
    plt.close(fig)
    print(f"  saved → {OUT_DIR}/fig4_gating_accuracy.png")


# ═══════════════════════════════════════════════════════════════════════════ #
# Fig 5 — Causal Validation                                                   #
# ═══════════════════════════════════════════════════════════════════════════ #
def fig5_causal_validation():
    print("Fig 5: causal validation …")

    # Load aggregate stats
    ko_stats, bl_stats = {}, {}
    for ds in DATASETS:
        cv  = load_causal(ds)
        ko  = cv["knockout_results"]
        bl  = cv["blindness_results"]
        ko_stats[ds] = (
            np.mean([r["acc_original"] for r in ko if r["acc_original"] is not None]),
            np.mean([r["acc_gated"]    for r in ko if r["acc_gated"]    is not None]),
            np.mean([r["acc_knockout"] for r in ko if r["acc_knockout"] is not None]),
        )
        bl_stats[ds] = (
            np.mean([r["acc_original"]  for r in bl if r["acc_original"]  is not None]),
            np.mean([r["acc_blindness"] for r in bl if r["acc_blindness"] is not None]),
        )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    # ── T1: condition-grouped, dataset-colored bars ── #
    COND_KO     = ["Original", "Soft Gate", "Hard\nKnockout"]
    COND_COLORS = ["#78909C", "#1565C0", "#B71C1C"]   # grey, blue, dark-red

    n_cond = len(COND_KO)
    n_ds   = len(DATASETS)
    width  = 0.28
    group_gap = 0.15
    group_width = n_ds * width
    xs = np.arange(n_cond) * (group_width + group_gap)

    for di, ds in enumerate(DATASETS):
        vals   = ko_stats[ds]
        offset = (di - (n_ds-1)/2) * width
        bars   = ax1.bar(xs + offset, vals, width,
                         color=DS_COLOR[ds], label=DS_LABEL[ds],
                         edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax1.text(bar.get_x()+bar.get_width()/2, v + 0.008,
                     f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax1.set_xticks(xs)
    ax1.set_xticklabels(COND_KO)
    ax1.set_ylabel("Accuracy (triggered samples, N=100)")
    ax1.set_title("Test 1 — Knockout\n(SEP-triggered samples)")
    ax1.set_ylim(0, 0.80)
    ax1.yaxis.grid(True, alpha=0.25, ls="--")
    ax1.legend(loc="upper right")

    # ── T2: same layout, 2 conditions ── #
    COND_BL = ["Original", "Grounding\nHeads Blinded"]
    xs2 = np.arange(2) * (group_width + group_gap)

    for di, ds in enumerate(DATASETS):
        vals   = bl_stats[ds]
        offset = (di - (n_ds-1)/2) * width
        bars   = ax2.bar(xs2 + offset, vals, width,
                         color=DS_COLOR[ds], label=DS_LABEL[ds],
                         edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax2.text(bar.get_x()+bar.get_width()/2, v + 0.008,
                     f"{v:.2f}", ha="center", va="bottom", fontsize=8)

        # Delta annotation above "blinded" bar — offset high enough to clear bar label
        delta = vals[1] - vals[0]
        sign  = "+" if delta >= 0 else ""
        col   = "#2E7D32" if delta >= 0 else "#C62828"
        bx    = xs2[1] + offset
        ax2.text(bx, vals[1] + 0.07, f"{sign}{delta*100:.1f}%",
                 ha="center", color=col, fontsize=8.5, fontweight="bold")

    ax2.set_xticks(xs2)
    ax2.set_xticklabels(COND_BL)
    ax2.set_ylabel("Accuracy (passthrough samples, N=100)")
    ax2.set_title("Test 2 — Blindness\n(high-LR grounding heads zeroed)")
    ax2.set_ylim(0, 1.10)
    ax2.yaxis.grid(True, alpha=0.25, ls="--")
    ax2.legend(loc="upper right")

    fig.suptitle("Causal Validation of Attention Head Roles", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig5_causal_validation.png")
    plt.close(fig)
    print(f"  saved → {OUT_DIR}/fig5_causal_validation.png")


# ═══════════════════════════════════════════════════════════════════════════ #
if __name__ == "__main__":
    fig2_ood_heatmap()
    fig3_sep_distribution()
    fig4_gating_accuracy()
    fig5_causal_validation()
    print(f"\nAll figures saved to {OUT_DIR}/")
