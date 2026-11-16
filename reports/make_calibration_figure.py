#!/usr/bin/env python3
"""Standalone calibration figure (Fig. Sy main version): reliability diagrams
only. Reads reports/calibration_review_metrics.json (produced by
reports/calibrate_and_review.py), so no re-fitting happens here."""
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = "/mnt/data2/NMR_Core_Mol_Detection/"
r = json.load(open(B + "reports/calibration_review_metrics.json"))
SETS = [("validation", "A  Validation (reserved)"),
        ("internal_test", "B  Internal test (natural prevalence)"),
        ("external_apr18peak", "C  External test (enriched)")]
BLUE, GREEN, ORANGE, GREY = "#0072B2", "#009E73", "#E69F00", "#8C8C8C"
TEXT = "#262626"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 7.5, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "axes.labelcolor": TEXT, "text.color": TEXT, "xtick.color": TEXT,
    "ytick.color": TEXT, "axes.linewidth": 0.7, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

fig, axes = plt.subplots(1, 3, figsize=(7.10, 2.75))
fig.subplots_adjust(left=0.075, right=0.985, bottom=0.17, top=0.80, wspace=0.36)

series_all = [("raw", GREY, "o", "Raw"),
              ("platt", BLUE, "s", "Platt"),
              ("isotonic", ORANGE, "D", "Isotonic"),
              ("platt_prior", GREEN, "^", "Platt + prior\n(deployment prev.)")]

for ax, (s, title) in zip(axes, SETS):
    for key, col, mk, lab in series_all:
        rows = [x for x in r["sets"][s]["reliability"][key]
                if x["n"] >= 5 and x["pred"] > 0]
        if not rows:
            continue
        if s != "external_apr18peak" and key == "platt_prior":
            # not a planned curve for natural-prevalence panels
            pass
        x = np.array([q["pred"] for q in rows])
        y = np.array([q["obs"] for q in rows])
        lo = np.clip(np.array([q["obs_lo"] for q in rows]), 1e-6, 1)
        hi = np.clip(np.array([q["obs_hi"] for q in rows]), 1e-6, 1)
        ax.plot(x, y, marker=mk, ms=2.8, lw=0.9, color=col, label=lab, zorder=3)
        if key in ("raw", "platt"):
            ax.fill_between(x, lo, hi, color=col, alpha=0.13, lw=0, zorder=1)
    ax.plot([1e-5, 1], [1e-5, 1], ls="--", lw=0.8, color="#666666", zorder=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-5, 1.0)
    ax.set_ylim(1e-5, 1.0)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(True, which="major", lw=0.35, color="#E5E5E5", zorder=0)

handles = [plt.Line2D([], [], color=col, marker=mk, ls="", ms=3.2, label=lab)
           for key, col, mk, lab in series_all]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.53, 0.985),
           ncol=4, fontsize=6.0, frameon=False, handletextpad=0.35,
           columnspacing=1.2)

for ext in ("pdf", "png"):
    fig.savefig(B + f"reports/fig_calibration_reliability.{ext}",
                dpi=600 if ext == "png" else None)
print("saved fig_calibration_reliability")
