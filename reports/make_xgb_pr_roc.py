"""XGBoost on the external test set: PR curve (AUPRC) + ROC curve (AUROC).

Two-panel figure, no insets. Best-practice choices for a strongly imbalanced
test set (64 positives / 4,438 negatives):

  * PR curve is drawn with **step interpolation** and summarised by **average
    precision** (step area), not the trapezoidal `auc(recall, precision)`
    (sklearn docs; scikit-learn issue #7372; Saito & Rehmsmeier, PLOS ONE 2015).
  * Iso-F1 reference curves and the chance level (prevalence) are drawn directly,
    so the trade-off and the operating point can be read without a legend.
  * 95 % bootstrap confidence bands quantify the small positive class.
  * The full-range ROC is a right angle near the origin at AUROC = 0.9998, so the
    FPR axis is **log-scaled**: the region where the model actually moves
    (FPR ~ 2e-4 to 6e-3) becomes visible without an inset.
  * The operating point is annotated with TP/FP/FN, linking the
    curves to the confusion matrix. The operating threshold is tau_op = 0.50,
    the value that reproduces the manuscript's reported confusion matrix
    (TP 63 / FP 4 / FN 1 / TN 4,434, F1 = 0.9618); the previous
    convention (tau_op = 0.40) gave 5 FP and F1 = 0.9545.

Inputs
  /tmp/apr18_cress_emb.npy            4,502 x 768 CReSS embeddings (external test)
  viz_tmp/test_labels.npy             4,502 labels (64 positives)
  models/xgboost/model_xgb_cress_ex_train_seed42.pkl

Outputs
  reports/fig_xgb_pr_roc.pdf / .svg / .png
  reports/xgb_pr_roc_metrics.json
  reports/xgb_pr_roc_curves.npz
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    roc_curve, roc_auc_score, auc, confusion_matrix,
)
import joblib

B = os.environ.get("NMR_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
B = B.rstrip("/") + "/"

BLUE = "#2166AC"
RED = "#D55E00"
GRAY = "#999999"
TEXT = "#262626"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 10.5,
    "axes.labelcolor": TEXT,
    "text.color": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "axes.linewidth": 0.9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


# ----------------------------------------------------------------- helpers ----
def pr_arrays(y_true, y_score):
    """precision_recall_curve with recall forced to ascending order."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if len(recall) > 1 and recall[0] > recall[-1]:
        precision = precision[::-1]
        recall = recall[::-1]
        thresholds = thresholds[::-1]
    return precision, recall, thresholds


def bootstrap_bands(y_true, y_score, n_boot=1000, seed=42):
    """95 % percentile bootstrap bands on a fixed recall / log-FPR grid."""
    rng = np.random.default_rng(seed)
    recall_grid = np.linspace(0.0, 1.0, 401)
    fpr_grid = np.logspace(np.log10(2e-4), 0, 241)
    pr_curves, roc_curves, aps, aucs = [], [], [], []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, pb = y_true[idx], y_score[idx]
        if yb.sum() == 0 or yb.sum() == n:
            continue
        pr, rc, _ = pr_arrays(yb, pb)
        j = np.clip(np.searchsorted(rc, recall_grid, side="right") - 1,
                    0, len(pr) - 1)
        pr_curves.append(pr[j])
        aps.append(average_precision_score(yb, pb))
        fpr, tpr, _ = roc_curve(yb, pb)
        k = np.clip(np.searchsorted(fpr, fpr_grid, side="right") - 1,
                    0, len(tpr) - 1)
        roc_curves.append(tpr[k])
        aucs.append(roc_auc_score(yb, pb))

    pr_curves = np.asarray(pr_curves)
    roc_curves = np.asarray(roc_curves)
    aps = np.asarray(aps)
    aucs = np.asarray(aucs)
    return dict(
        recall_grid=recall_grid,
        fpr_grid=fpr_grid,
        pr_lo=np.percentile(pr_curves, 2.5, axis=0),
        pr_hi=np.percentile(pr_curves, 97.5, axis=0),
        roc_lo=np.percentile(roc_curves, 2.5, axis=0),
        roc_hi=np.percentile(roc_curves, 97.5, axis=0),
        ap_ci=np.percentile(aps, [2.5, 97.5]),
        auroc_ci=np.percentile(aucs, [2.5, 97.5]),
        n_boot=len(aps),
    )


def style_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(color="#EAEAEA", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(TEXT)
    ax.tick_params(direction="out", length=3.5, width=0.8)


# ------------------------------------------------------------------ figure ----
def plot_pr_roc(y_true, y_score, threshold=0.50, output="pr_roc",
                n_boot=1000):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    precision, recall, _ = pr_arrays(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score, drop_intermediate=False)

    ap = average_precision_score(y_true, y_score)
    pr_auc_trapz = auc(recall, precision)          # for comparison only
    roc_auc = roc_auc_score(y_true, y_score)
    prevalence = float(np.mean(y_true))

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_score >= threshold, labels=[0, 1]
    ).ravel()
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    fpr_op = fp / (fp + tn) if fp + tn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0

    bands = bootstrap_bands(y_true, y_score, n_boot=n_boot)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.155,
                        top=0.905, wspace=0.22)

    # ---------------------------------------------------------- A: PR ------
    ax = axes[0]
    ax.set_xlim(-0.015, 1.02)
    ax.set_ylim(-0.015, 1.035)
    style_axes(ax)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))

    ax.fill_between(bands["recall_grid"], bands["pr_lo"], bands["pr_hi"],
                    step="post", color=BLUE, alpha=0.14, linewidth=0, zorder=2)
    ax.step(recall, precision, where="post", color=BLUE, linewidth=2.1,
            zorder=4)

    # iso-F1 reference curves (directly labelled, no legend)
    for f in (0.80, 0.90, 0.95):
        # iso-F1 curve exists (P <= 1) only for R >= f/(2-f)
        rr = np.linspace(f / (2.0 - f), 1.0, 240)
        pp = f * rr / (2.0 * rr - f)
        ax.plot(rr, pp, color=GRAY, lw=0.8, ls=(0, (4, 2)),
                alpha=0.85, zorder=1)
        ax.text(0.925, f * 0.925 / (2 * 0.925 - f) + 0.012, f"F1 {f:.2f}",
                fontsize=7.2, color="#888888", ha="right", va="bottom",
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.6))

    ax.axhline(prevalence, color=GRAY, linestyle=(0, (3, 3)), linewidth=1.1,
               zorder=1)
    ax.text(0.03, prevalence + 0.035,
            f"chance (prevalence = {prevalence:.3f})",
            transform=ax.transAxes, color="#777777", fontsize=8.2,
            ha="left", va="bottom")

    ax.scatter([r], [p], s=60, color=RED, edgecolor="white", linewidth=1.3,
               zorder=6)
    ax.annotate(
        f"threshold = {threshold:g}\n"
        f"TP {tp}  FP {fp}  FN {fn}\n"
        f"precision {p:.3f}  recall {r:.3f}",
        xy=(r, p), xycoords="data",
        xytext=(0.90, 0.33), textcoords="axes fraction",
        fontsize=8.6, color=TEXT, linespacing=1.35, ha="right",
        arrowprops=dict(arrowstyle="-", color=RED, lw=1.1,
                        shrinkA=6, shrinkB=7),
    )

    ax.set(xlabel="Recall", ylabel="Precision")
    ax.set_title("A. AUPRC", loc="left", pad=10, fontweight="bold")
    ax.text(1.0, 1.02,
            f"{ap:.4f}  (95% CI {bands['ap_ci'][0]:.3f}\u2013{bands['ap_ci'][1]:.3f})",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.8, color=BLUE, fontweight="bold")

    # --------------------------------------------------------- B: ROC ------
    ax = axes[1]
    style_axes(ax)
    ax.set_yscale("linear")
    ax.set_ylim(-0.015, 1.035)
    ax.set_xscale("log")
    ax.set_xlim(1.8e-4, 1.0)
    ax.set_xticks([1e-3, 1e-2, 1e-1, 1.0])
    ax.set_xticklabels(["$10^{-3}$", "$10^{-2}$", "$10^{-1}$", "1"])
    ax.minorticks_off()
    ax.set_yticks(np.linspace(0, 1, 6))

    ax.fill_between(bands["fpr_grid"], bands["roc_lo"], bands["roc_hi"],
                    color=BLUE, alpha=0.12, linewidth=0, zorder=1)
    ax.plot(fpr, tpr, color=BLUE, linewidth=2.1, zorder=4)
    chance = np.logspace(np.log10(1.8e-4), 0, 60)
    ax.plot(chance, chance, color=GRAY, linestyle=(0, (3, 3)), linewidth=1.1,
            zorder=1)
    ax.scatter([fpr_op], [r], s=60, color=RED, edgecolor="white", linewidth=1.3,
               zorder=6)
    ax.annotate(
        f"threshold = {threshold:g}\nTPR {r:.3f}  FPR {fpr_op:.4f}",
        xy=(fpr_op, r), xycoords="data",
        xytext=(0.30, 0.72), textcoords="axes fraction",
        fontsize=8.6, color=TEXT, linespacing=1.35,
        arrowprops=dict(arrowstyle="-", color=RED, lw=1.1,
                        shrinkA=6, shrinkB=7),
    )

    ax.set(xlabel="False positive rate (log scale)",
           ylabel="True positive rate")
    ax.set_title("B. AUROC", loc="left", pad=10, fontweight="bold")
    ax.text(1.0, 1.02,
            f"{roc_auc:.4f}  (95% CI {bands['auroc_ci'][0]:.4f}\u2013{bands['auroc_ci'][1]:.4f})",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.8, color=BLUE, fontweight="bold")

    for ext in ("pdf", "svg", "png"):
        fig.savefig(f"{output}.{ext}", dpi=600, facecolor="white")

    np.savez_compressed(
        output + "_curves.npz",
        recall=recall, precision=precision, fpr=fpr, tpr=tpr,
        recall_grid=bands["recall_grid"], pr_lo=bands["pr_lo"],
        pr_hi=bands["pr_hi"], fpr_grid=bands["fpr_grid"],
        roc_lo=bands["roc_lo"], roc_hi=bands["roc_hi"],
    )

    return fig, axes, dict(
        auprc=float(ap), ap=float(ap),
        ap_ci=[float(v) for v in bands["ap_ci"]],
        pr_auc_trapezoid=float(pr_auc_trapz),
        auroc=float(roc_auc),
        auroc_ci=[float(v) for v in bands["auroc_ci"]],
        prevalence=prevalence,
        n_test=int(len(y_true)), n_pos=int(y_true.sum()),
        n_neg=int((1 - y_true).sum()),
        operating_point=dict(
            threshold=float(threshold), precision=float(p), recall=float(r),
            f1=float(f1), tpr=float(r), fpr=float(fpr_op),
            tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn)),
        n_bootstrap=int(bands["n_boot"]),
    )


if __name__ == "__main__":
    emb_te = np.load(B + "reports/calib_data/apr18_cress_emb.npy").astype(np.float32)
    y_te = np.load(B + "viz_tmp/test_labels.npy").astype(int)
    model = joblib.load(B + "models/xgboost/model_xgb_cress_ex_train_seed42.pkl")
    p_te = model.predict_proba(emb_te)[:, 1]

    fig, axes, m = plot_pr_roc(y_te, p_te, threshold=0.50,
                               output=B + "reports/fig_xgb_pr_roc")
    with open(B + "reports/xgb_pr_roc_metrics.json", "w") as fjson:
        json.dump(m, fjson, indent=2)
    op = m["operating_point"]
    print(f"AP {m['ap']:.4f} (95% CI {m['ap_ci'][0]:.4f}-{m['ap_ci'][1]:.4f}), "
          f"trapezoid {m['pr_auc_trapezoid']:.4f}")
    print(f"AUROC {m['auroc']:.4f} (95% CI {m['auroc_ci'][0]:.4f}-"
          f"{m['auroc_ci'][1]:.4f})")
    print(f"@tau_op={m['operating_point']['threshold']:g}: TP {op['tp']} FP {op['fp']} FN {op['fn']} TN {op['tn']} | "
          f"P {op['precision']:.3f} R {op['recall']:.3f} F1 {op['f1']:.4f}")
