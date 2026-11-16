#!/usr/bin/env python3
"""One combined model-analysis figure at tau = 0.50.

Panels
  A  precision-recall curve (AUPRC)
  B  ROC curve (log FPR)
  C  reliability diagram (external test, raw vs Platt + prior correction)
  D  uncertainty-guided review outcome (external test)
  E  MC-Dropout: quinolone-core pass frequency vs structural dispersion
  F  accuracy per ring system
"""
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

B = "/mnt/data2/NMR_Core_Mol_Detection/"
BLUE, ORANGE, RED, GREEN, GREY, LIGHT = ("#0072B2", "#E69F00", "#D55E00",
                                         "#009E73", "#8C8C8C", "#C9CCD4")
TEXT = "#262626"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 7.0, "axes.labelsize": 7.6, "axes.titlesize": 8.2,
    "axes.labelcolor": TEXT, "text.color": TEXT, "xtick.color": TEXT,
    "ytick.color": TEXT, "axes.linewidth": 0.7, "xtick.major.width": 0.55,
    "ytick.major.width": 0.55, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

m = json.load(open(B + "reports/xgb_pr_roc_metrics.json"))
cal = json.load(open(B + "reports/calibration_review_metrics.json"))
mc = json.load(open(B + "reports/mc_dropout_generator.json"))
nh = json.load(open(B + "reports/nhetero_systems.json"))
z = np.load(B + "reports/fig_xgb_pr_roc_curves.npz")

fig = plt.figure(figsize=(7.10, 7.35))
gs = GridSpec(2, 3, hspace=0.40, wspace=0.40, left=0.085, right=0.985,
              top=0.935, bottom=0.095, width_ratios=[1.18, 1.0, 1.0])

# ------------------------------------------------------------------ A PR --
ax = fig.add_subplot(gs[0, 0])
ax.fill_between(z["recall_grid"], z["pr_lo"], z["pr_hi"], color=BLUE,
                alpha=0.14, lw=0, zorder=1)
ax.step(z["recall"], z["precision"], where="post", color=BLUE, lw=1.5,
        zorder=3)
ax.axhline(m["prevalence"], color=GREY, ls="--", lw=0.8, zorder=1)
op = m["operating_point"]
ax.plot([op["recall"]], [op["precision"]], "o", ms=5, color=RED,
        mec="white", mew=0.6, zorder=4)
ax.set_xlim(-0.01, 1.01)
ax.set_ylim(-0.01, 1.04)
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("A  AUPRC", loc="left", fontweight="bold")
ax.text(0.04, 0.06, f"AP = {m['ap']:.4f}\n95% CI {m['ap_ci'][0]:.4f}–{m['ap_ci'][1]:.4f}",
        fontsize=6.0, color=TEXT, va="bottom", linespacing=1.3)
ax.grid(True, lw=0.3, color="#EEEEEE")

# ----------------------------------------------------------------- B ROC --
ax = fig.add_subplot(gs[0, 1])
ax.fill_between(z["fpr_grid"], z["roc_lo"], z["roc_hi"], color=BLUE,
                alpha=0.14, lw=0, zorder=1)
ax.plot(z["fpr"], z["tpr"], color=BLUE, lw=1.5, zorder=3)
ch = np.logspace(np.log10(1.5e-4), 0, 80)
ax.plot(ch, ch, ls="--", lw=0.8, color=GREY, zorder=1)
ax.plot([op["fpr"]], [op["tpr"]], "o", ms=5, color=RED, mec="white",
        mew=0.6, zorder=4)
ax.set_xscale("log")
ax.set_xlim(1.5e-4, 1.02)
ax.set_ylim(-0.01, 1.04)
ax.set_xlabel("False positive rate (log scale)")
ax.set_ylabel("True positive rate")
ax.set_title("B  AUROC", loc="left", fontweight="bold")
ax.text(0.04, 0.06, f"AUROC = {m['auroc']:.4f}\n95% CI {m['auroc_ci'][0]:.4f}–{m['auroc_ci'][1]:.4f}",
        fontsize=6.0, color=TEXT, va="bottom", linespacing=1.3)
ax.grid(True, lw=0.3, color="#EEEEEE")

# ------------------------------------------------------- C reliability ----
ax = fig.add_subplot(gs[0, 2])
for key, col, mk, lab in (("raw", GREY, "o", "Raw"),
                          ("platt_prior", BLUE, "s", "Platt + prior")):
    rel = cal["sets"]["external_apr18peak"]["reliability"][key]
    rr = [x for x in rel if x["n"] >= 5 and x["pred"] > 0]
    x = np.array([q["pred"] for q in rr])
    y = np.array([q["obs"] for q in rr])
    lo = np.clip(np.array([q["obs_lo"] for q in rr]), 1e-6, 1)
    hi = np.clip(np.array([q["obs_hi"] for q in rr]), 1e-6, 1)
    ax.plot(x, y, marker=mk, ms=2.6, lw=0.9, color=col, label=lab, zorder=3)
    ax.fill_between(x, lo, hi, color=col, alpha=0.13, lw=0, zorder=1)
ax.plot([1e-5, 1], [1e-5, 1], ls="--", lw=0.7, color="#666666", zorder=0)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1e-5, 1)
ax.set_ylim(1e-5, 1)
ax.set_xlabel("Predicted probability")
ax.set_ylabel("Observed frequency")
ax.set_title("C  Calibration (external)", loc="left", fontweight="bold")
ax.legend(loc="upper left", fontsize=5.6, handlelength=1.1)
ax.grid(True, lw=0.3, color="#EEEEEE")

# ------------------------------------------------------- D review rule ----
ax = fig.add_subplot(gs[1, 0])
q = cal["sets"]["external_apr18peak"]["review"]
n_neg_auto = 4438 - q["auto_positive"]["fp"] - q["referred"]["neg"]
ax.axis("off")
ax.set_title("D  Review workflow", loc="left", fontweight="bold", fontsize=7.8,
             pad=4)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.text(0.02, 0.76, "Positive (n = 64)", fontsize=6.6, fontweight="bold",
        color=BLUE, transform=ax.transAxes)
ax.text(0.02, 0.62, f"{q['auto_positive']['tp']} auto-positive  ·  "
        f"{q['referred']['pos']} referred  ·  0 missed", fontsize=6.2,
        color=TEXT, transform=ax.transAxes)
ax.text(0.02, 0.38, "Negative (n = 4,438)", fontsize=6.6, fontweight="bold",
        color=GREY, transform=ax.transAxes)
ax.text(0.02, 0.24, f"{q['auto_positive']['fp']} auto-positive  ·  "
        f"{q['referred']['neg']} referred  ·  {n_neg_auto:,} auto-negative",
        fontsize=6.2, color=TEXT, transform=ax.transAxes)
ax.text(0.02, 0.05, "0 silent misses; referred spectra require MS confirmation,\n"
        "not a negative call.", fontsize=5.4, color=GREY, style="italic",
        transform=ax.transAxes, linespacing=1.3)

# ------------------------------------------------------- E MC-Dropout -----
ax = fig.add_subplot(gs[1, 1])
groups = {}
for row in mc["spectra"]:
    groups.setdefault(row["group"], []).append(row)
gorder = ["confident positive", "borderline positive", "FN #12 (p=0.012)",
          "borderline negative (FP)", "confident negative"]
gcol = {"confident positive": BLUE, "borderline positive": GREEN,
        "FN #12 (p=0.012)": RED, "borderline negative (FP)": ORANGE,
        "confident negative": GREY}
glab = {"confident positive": "Conf. pos.", "borderline positive": "Border. pos.",
        "FN #12 (p=0.012)": "FN #12", "borderline negative (FP)": "FP",
        "confident negative": "Conf. neg."}
for g in gorder:
    sub = groups[g]
    x = np.mean([r["core_pass_freq"] for r in sub])
    y = np.mean([r["cress_cos_mean"] for r in sub])
    ax.scatter([x], [y], s=40 + 12 * len(sub), color=gcol[g],
               edgecolor="white", linewidth=0.6, zorder=3)
    off = {"confident positive": (-8, 8), "borderline positive": (6, -10),
           "FN #12 (p=0.012)": (0, 14), "borderline negative (FP)": (12, -10),
           "confident negative": (6, -10)}[g]
    ax.annotate(glab[g], (x, y), xytext=off, textcoords="offset points",
                fontsize=5.6, color=gcol[g])
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(0.30, 0.90)
ax.set_xlabel("Core-pass frequency (MC-Dropout)")
ax.set_ylabel("Mean CReSS cosine of candidates")
ax.set_title("E  Generator variability", loc="left", fontweight="bold")
ax.grid(True, lw=0.3, color="#EEEEEE")

# ------------------------------------------------------- F ring accuracy --
ax = fig.add_subplot(gs[1, 2])
ORDER = ["quinolone core", "quinolinone-like", "fused N-heteroarene",
         "pyridine", "diazine/triazine", "azole", "saturated N-ring",
         "carbocycle", "acyclic"]
NICE = {"quinolone core": "Quinolone", "quinolinone-like": "Quinolinone",
        "fused N-heteroarene": "Fused N-het.",
        "pyridine": "Pyridine", "diazine/triazine": "Diazine",
        "azole": "Azole", "saturated N-ring": "Sat. N-ring",
        "carbocycle": "Carbocycle", "acyclic": "Acyclic"}
COL = {"quinolone core": BLUE, "quinolinone-like": RED,
       "fused N-heteroarene": "#CC79A7", "pyridine": ORANGE,
       "diazine/triazine": "#56B4E9", "azole": GREEN,
       "saturated N-ring": "#8C6D31", "carbocycle": "#6A3D9A",
       "acyclic": "#9E9E9E"}
BASE = "#BFD0DB"
yy = np.arange(len(ORDER))[::-1]
for y, c in zip(yy, ORDER):
    s = nh["classes"][c]
    err = (s["n"] - s["fp_at_tau"]) if c == "quinolone core" else s["fp_at_tau"]
    acc = 100 * (1 - err / s["n"])
    col = COL[c] if c in ("quinolone core", "quinolinone-like") else BASE
    ax.barh(y, acc - 90, left=90, height=0.55, color=col, zorder=2)
ax.set_yticks(yy, [NICE[c] for c in ORDER], fontsize=5.7)
ax.get_yticklabels()[0].set_fontweight("bold")
ax.get_yticklabels()[0].set_color(BLUE)
ax.get_yticklabels()[1].set_fontweight("bold")
ax.get_yticklabels()[1].set_color(RED)
ax.set_xlim(90, 100.4)
ax.set_xticks([90, 95, 100], ["90", "95", "100"], fontsize=5.9)
ax.set_xlabel("Accuracy (%) at $\\tau$ = 0.50")
ax.set_title("F  Ring-system accuracy", loc="left", fontweight="bold")
ax.axvline(100, color="#CCCCCC", lw=0.6, ls=":", zorder=1)
ax.grid(True, axis="x", lw=0.3, color="#EFEFEF", zorder=0)

for ext in ("pdf", "png"):
    fig.savefig(B + f"reports/fig_model_analysis.{ext}",
                dpi=600 if ext == "png" else None)
print("saved fig_model_analysis")
