#!/usr/bin/env python3
"""Visualise the 90% bootstrap CI of the calibrated XGBoost-CReSS probability.

A. External test set in the CReSS-UMAP, every point coloured by the width of
   its 90% CI (log colour scale); FP (diamond) and FN (star) follow the
   convention of the main paper figure (FP #D55E00 diamond, FN #0072B2 star).
B. Vertical forest plot of the 64 positives and the 4 FP: calibrated
   probability (y, log) with its 90% CI, molecules sorted along x. The
   decision threshold tau = 0.50 on the raw model score maps to 0.0235 on the
   calibrated scale.

The interval is a calibration-map bootstrap: the Platt map is refitted on 200
stratified bootstrap resamples of the validation split and re-applied to every
evaluation sample (5th-95th percentile).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

import os as _os
B = _os.environ.get('NMR_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
B = B.rstrip('/') + '/'
OUT = B + "reports/"

TEXT = "#262626"
POS = "#E69F00"     # positives  (triangle)
FP_C = "#D55E00"    # false positive (diamond)
FP_E = "#8a3b00"
FN_C = "#0072B2"    # false negative (star)
FN_E = "#003f57"
TAU_CAL = 0.023500  # calibrated value of the tau = 0.50 model score

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 7.0, "axes.labelsize": 7.2, "axes.titlesize": 8.6,
    "axes.labelcolor": TEXT, "text.color": TEXT, "xtick.color": TEXT,
    "ytick.color": TEXT, "axes.linewidth": 0.6, "xtick.major.width": 0.5,
    "ytick.major.width": 0.5, "xtick.major.size": 2.2,
    "ytick.major.size": 2.2, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

# ---------------------------------------------------------------- data ----
E = np.load(OUT + "calib_data/external_apr18peak.npz", allow_pickle=True)
I = np.load(OUT + "calib_data/external_apr18peak_p_interval.npz")
y = E["y"].astype(int)
p_raw = E["p_raw"].astype(float)
p = I["p"].astype(float)
lo = I["lo"].astype(float)
hi = I["hi"].astype(float)
w = hi - lo
pred = (p_raw >= 0.5).astype(int)

C_all = np.load(B + "viz_tmp/coords_fit.npy")
nfit = len(np.load(B + "viz_tmp/fit_idx.npy"))
XY = C_all[nfit:]                       # 4502 external-test UMAP coords

tp = np.where((y == 1) & (pred == 1))[0]
fn = np.where((y == 1) & (pred == 0))[0]
fp = np.where((y == 0) & (pred == 1))[0]
tn = np.where((y == 0) & (pred == 0))[0]

# ---------------------------------------------------------------- figure ----
fig = plt.figure(figsize=(7.10, 3.55))
axA = fig.add_axes([0.045, 0.165, 0.400, 0.745])
axB = fig.add_axes([0.575, 0.150, 0.400, 0.755])

# ---- A: UMAP coloured by CI width
# ramp starts at the grey used for "negative" in the embedding figure and ends
# at the orange used for "positive", so the panel keeps the printed palette
CMAP = LinearSegmentedColormap.from_list(
    "ci_grey_orange", ["#d7d9de", "#f6d79a", "#f0a04b", "#d55e00", "#8a3b00"])
norm = LogNorm(vmin=1e-9, vmax=0.25)
sc = axA.scatter(XY[tn, 0], XY[tn, 1], c=np.clip(w[tn], 1e-9, None), s=2.0,
                 cmap=CMAP, norm=norm, alpha=0.60, linewidths=0,
                 rasterized=True, zorder=1)
axA.scatter(XY[tp, 0], XY[tp, 1], c=np.clip(w[tp], 1e-9, None), s=6.5,
            cmap=CMAP, norm=norm, marker="o", edgecolors="white",
            linewidths=0.25, zorder=2)
halo = [pe.withStroke(linewidth=1.5, foreground="white")]
axA.scatter(XY[fp, 0], XY[fp, 1], s=42, marker="D", facecolors=FP_C,
            edgecolors=FP_E, linewidths=0.8, zorder=4, path_effects=halo)
axA.scatter(XY[fn, 0], XY[fn, 1], s=140, marker="*", facecolors=FN_C,
            edgecolors=FN_E, linewidths=0.7, zorder=5, path_effects=halo)

LBL = {
    3460: ("FP p = 0.99", (-0.10, 0.30), "left", "bottom"),
    3898: ("FP p = 0.93", (0.18, 0.02), "left", "center"),
    4201: ("FP p = 0.54", (0.18, 0.00), "left", "center"),
    763:  ("FP p = 0.59", (0.28, -0.05), "left", "top"),
    12:   ("FN p = 0.01", (0.30, -0.10), "left", "top"),
}
for i, (txt, (dx, dy), ha, va) in LBL.items():
    col = FN_C if i == 12 else FP_C
    axA.annotate(txt, (XY[i, 0] + dx, XY[i, 1] + dy), fontsize=6.0, color=col,
                 fontweight="bold", ha=ha, va=va, zorder=7,
                 bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none",
                           alpha=0.80))

axA.set_aspect("equal")
axA.set_xticks([]); axA.set_yticks([])
for sp in axA.spines.values():
    sp.set_visible(False)
axA.set_title("A  Test set in CReSS-UMAP, coloured by 90% CI width",
              loc="left", fontweight="bold", pad=13)
axA.text(0.0, 1.006, "each circle = one spectrum (n = 4,502, 64 positives); colour = 90% CI width",
         transform=axA.transAxes, fontsize=5.8, color="#666666", va="bottom")

cax = fig.add_axes([0.055, 0.105, 0.380, 0.011])
cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
cb.set_ticks([1e-9, 1e-7, 1e-5, 1e-3, 1e-1])
cb.ax.set_xticklabels(["$10^{-9}$", "$10^{-7}$", "$10^{-5}$",
                       "$10^{-3}$", "$10^{-1}$"])
cb.outline.set_linewidth(0.4)
cb.ax.tick_params(labelsize=5.6, length=2, pad=1.2)
cb.set_label("90% CI width of calibrated p", fontsize=6.2, labelpad=1.2)

# ---- B: vertical forest plot
sel = np.concatenate([tp, fn, fp])
order = sel[np.argsort(p[sel])]
xx = np.arange(1, len(order) + 1)
grp = np.where(y[order] == 0, "fp", np.where(pred[order] == 0, "fn", "tp"))
col = {"tp": POS, "fn": FN_C, "fp": FP_C}

axB.axhline(TAU_CAL, color="#555555", lw=0.9, ls=(0, (3, 2)), zorder=1)
axB.text(len(order) * 0.62, TAU_CAL * 1.18, r"$\tau$ = 0.50", fontsize=5.8,
         color="#555555", ha="left", va="bottom",
         bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none",
                   alpha=0.85), zorder=6)

for r, i, g in zip(xx, order, grp):
    axB.plot([r, r], [max(lo[i], 3e-5), hi[i]], color=col[g], lw=0.9,
             alpha=0.85, solid_capstyle="butt", zorder=2)

m = grp == "tp"
axB.scatter(xx[m], p[order][m], s=10, marker="o", color=POS,
            edgecolors="white", linewidths=0.35, zorder=3)
m = grp == "fp"
axB.scatter(xx[m], p[order][m], s=22, marker="D", color=FP_C,
            edgecolors=FP_E, linewidths=0.5, zorder=4)
m = grp == "fn"
axB.scatter(xx[m], p[order][m], s=48, marker="*", color=FN_C,
            edgecolors=FN_E, linewidths=0.5, zorder=5)

i_fn = order[pred[order] == 0][0]
r_fn = int(xx[np.where(order == i_fn)[0][0]])
axB.annotate(f"FN #{i_fn}", (r_fn, p[i_fn]), xytext=(3, -2),
             textcoords="offset points", fontsize=5.6, color=FN_C,
             fontweight="bold", va="top", ha="left")

axB.set_yscale("log")
axB.set_ylim(3e-5, 3.0)
axB.set_xlim(0.0, len(order) + 1.0)
axB.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1])
axB.set_yticklabels(["$10^{-4}$", "$10^{-3}$", "$10^{-2}$",
                     "$10^{-1}$", "1"], fontsize=6.0)
axB.set_ylabel("Calibrated probability (90% CI)", fontsize=7.0)
axB.set_xlabel("Positives and FP, sorted by calibrated p", fontsize=6.6)
axB.set_xticks([])
axB.tick_params(axis="y", length=2.2, pad=1.5)
for sp in ("top", "right", "bottom"):
    axB.spines[sp].set_visible(False)
axB.set_title("B  Per-molecule 90% CI", loc="left", fontweight="bold", pad=13)
axB.text(0.0, 1.006, "64 positives + 4 FP", transform=axB.transAxes,
         fontsize=5.8, color="#666666", va="bottom")
axB.grid(True, axis="y", lw=0.3, color="#EFEFEF", zorder=0)

handles = [
    Line2D([0], [0], marker="o", color="none", markerfacecolor=POS,
           markeredgecolor="white", markeredgewidth=0.4, markersize=4.2,
           label="TP (63)"),
    Line2D([0], [0], marker="*", color="none", markerfacecolor=FN_C,
           markeredgecolor=FN_E, markeredgewidth=0.4, markersize=7.5,
           label="FN (1)"),
    Line2D([0], [0], marker="D", color="none", markerfacecolor=FP_C,
           markeredgecolor=FP_E, markeredgewidth=0.4, markersize=3.8,
           label="FP (4)"),
]
axB.legend(handles=handles, loc="upper left", fontsize=6.0,
           handletextpad=0.35, borderaxespad=0.5, labelspacing=0.35)

for e in ("png", "pdf"):
    fig.savefig(OUT + f"fig_ci_external.{e}", dpi=600 if e == "png" else None)
print("saved fig_ci_external")
