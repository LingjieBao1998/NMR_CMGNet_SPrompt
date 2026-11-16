#!/usr/bin/env python3
"""Figure for the MC-Dropout generator variability analysis (reports/mc_dropout_generator.py)."""
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os as _os
B = _os.environ.get('NMR_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
B = B.rstrip('/') + '/'
BLUE, GREEN, RED, ORANGE, GREY = "#0072B2", "#009E73", "#D55E00", "#E69F00", "#999999"
GROUP_ORDER = ["FN #12 (p=0.012)", "borderline positive", "confident positive",
               "borderline negative (FP)", "confident negative"]
COL = {"FN #12 (p=0.012)": RED, "borderline positive": GREEN,
       "confident positive": BLUE, "borderline negative (FP)": ORANGE,
       "confident negative": GREY}
SHORT = {"FN #12 (p=0.012)": "FN #12", "borderline positive": "Border.\npos.",
         "confident positive": "Conf.\npos.",
         "borderline negative (FP)": "FP", "confident negative": "Conf.\nneg."}
TEXT = "#262626"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 7.5, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "axes.labelcolor": TEXT, "text.color": TEXT, "xtick.color": TEXT,
    "ytick.color": TEXT, "axes.linewidth": 0.7, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

d = json.load(open(B + "reports/mc_dropout_generator.json"))
rows = d["spectra"]


def jitter(ax, x, vals, col, label):
    rng = np.random.default_rng(0)
    xx = np.full(len(vals), x, dtype=float) + rng.uniform(-0.09, 0.09, len(vals))
    ax.plot(xx, vals, "o", ms=3.2, mfc=col, mec="white", mew=0.5,
            label=label, alpha=0.95, zorder=3)


fig, axes = plt.subplots(1, 3, figsize=(7.10, 2.65))
fig.subplots_adjust(left=0.115, right=0.985, bottom=0.20, top=0.86, wspace=0.42)

# A: core-hit frequency vs classifier score
ax = axes[0]
for r in rows:
    ax.plot(max(r["p_xgb"], 3e-3), r["core_pass_freq"], "o", ms=4,
            mfc=COL[r["group"]], mec="white", mew=0.5, alpha=0.95, zorder=3)
ax.annotate("FN #12", (0.012, 0.0), xytext=(0.02, 0.22),
            textcoords="data", color=RED, fontsize=6.0,
            arrowprops=dict(arrowstyle="-", lw=0.6, color=RED))
ax.set_xscale("log")
ax.set_xlabel("XGBoost score")
ax.set_ylabel("Passes with quinolone core\n(in top-5, /20)")
ax.set_ylim(-0.05, 1.12)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xlim(2.5e-3, 1.15)
ax.set_title("A  Core-hit frequency across passes", loc="left", fontweight="bold")
ax.grid(True, lw=0.35, color="#E5E5E5")

# B: modal top-1 frequency
ax = axes[1]
for i, g in enumerate(GROUP_ORDER):
    sub = [r for r in rows if r["group"] == g]
    jitter(ax, i, [r["top1_mode_freq"] for r in sub], COL[g], g)
    med = np.median([r["top1_mode_freq"] for r in sub])
    ax.plot([i - 0.28, i + 0.28], [med, med], "_", ms=10, color=TEXT, lw=1.2, zorder=4)
ax.set_xticks(range(5), [SHORT[g] for g in GROUP_ORDER], fontsize=6.2)
ax.set_ylabel("Top-1 modal frequency (/20)")
ax.set_ylim(-0.05, 1.18)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_title("B  Reproducibility of the top-1", loc="left", fontweight="bold")
ax.grid(True, axis="y", lw=0.35, color="#E5E5E5")

# C: CReSS cosine dispersion
ax = axes[2]
for i, g in enumerate(GROUP_ORDER):
    sub = [r for r in rows if r["group"] == g]
    jitter(ax, i, [r["cress_cos_mean"] for r in sub], COL[g], g)
    med = np.median([r["cress_cos_mean"] for r in sub])
    ax.plot([i - 0.28, i + 0.28], [med, med], "_", ms=10, color=TEXT, lw=1.2, zorder=4)
ax.set_xticks(range(5), [SHORT[g] for g in GROUP_ORDER], fontsize=6.2)
ax.set_ylabel("Mean pairwise CReSS cosine\nof generated SMILES")
ax.set_title("C  Structural dispersion", loc="left", fontweight="bold")
ax.grid(True, axis="y", lw=0.35, color="#E5E5E5")

handles = [plt.Line2D([], [], color=COL[g], marker="o", ls="", ms=4,
                      mfc=COL[g], mec=COL[g], label=SHORT[g].replace("\n", " "))
           for g in GROUP_ORDER]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.53, 0.985),
           ncol=5, fontsize=6.0, frameon=False, handletextpad=0.3,
           columnspacing=1.1)

for ext in ("pdf", "png"):
    fig.savefig(B + f"reports/fig_mc_dropout.{ext}", dpi=600 if ext == "png" else None)
print("saved fig_mc_dropout")
