#!/usr/bin/env python3
"""R2.3 -- spectral-resolution / C-F splitting sensitivity.

Two paired 48-spectrum sets of the same fluorinated quinolones:
  merged : C-F split components merged into single peak positions
           (with_f_split; median 18.5 peaks)
  split  : C-F split components kept separately
           (without_f_split; median 23 peaks)

Outputs
  reports/resolution_split_metrics.json
  reports/fig_resolution_split.pdf / .png
"""
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os as _os
B = _os.environ.get('NMR_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
B = B.rstrip('/') + '/'
BLUE, RED, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#8C8C8C"
TEXT = "#262626"
TAU = 0.50
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 7.5, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "axes.labelcolor": TEXT, "text.color": TEXT, "xtick.color": TEXT,
    "ytick.color": TEXT, "axes.linewidth": 0.7, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

z = np.load(B + "viz_tmp/f48_pred.npz")
pm, ps = z["p_merged"], z["p_split"]
d1 = json.load(open(B + "data/new_dataset/f_split/filtered_with_f_split_fixed_stereo.json"))
d2 = json.load(open(B + "data/new_dataset/f_without_split/filtered_without_f_split_fixed_stereo.json"))
nm = np.array([len(x["C_nmr"]) for x in d1])
ns = np.array([len(x["C_nmr"]) for x in d2])

rec_m = float((pm >= TAU).mean())
rec_s = float((ps >= TAU).mean())
flip = (pm >= TAU) != (ps >= TAU)
lost = int(((pm >= TAU) & (ps < TAU)).sum())
gained = int(((pm < TAU) & (ps >= TAU)).sum())
summary = {
    "tau": TAU, "n": int(len(pm)),
    "recall_merged": rec_m, "recall_split": rec_s,
    "detected_merged": int((pm >= TAU).sum()),
    "detected_split": int((ps >= TAU).sum()),
    "flips": int(flip.sum()), "lost_by_splitting": lost,
    "gained_by_splitting": gained,
    "median_abs_dp": float(np.median(np.abs(pm - ps))),
    "max_abs_dp": float(np.max(np.abs(pm - ps))),
    "median_peaks_merged": float(np.median(nm)),
    "median_peaks_split": float(np.median(ns)),
    "per_molecule": [
        dict(idx=i, p_merged=float(pm[i]), p_split=float(ps[i]),
             peaks_merged=int(nm[i]), peaks_split=int(ns[i]),
             flip=bool(flip[i]))
        for i in range(len(pm))],
}
json.dump(summary, open(B + "reports/resolution_split_metrics.json", "w"),
          indent=1)

fig, axes = plt.subplots(1, 2, figsize=(7.10, 2.85))
fig.subplots_adjust(left=0.085, right=0.98, bottom=0.20, top=0.85, wspace=0.38)

# A paired probabilities
ax = axes[0]
xs = np.clip(pm, 1e-5, 1)
ys = np.clip(ps, 1e-5, 1)
ax.plot([1e-5, 1], [1e-5, 1], ls="--", lw=0.8, color="#666666")
ax.scatter(xs[~flip], ys[~flip], s=14, color=GREY, alpha=0.85,
           edgecolor="white", linewidth=0.4, label="same call")
ax.scatter(xs[flip], ys[flip], s=22, color=RED, alpha=0.95,
           edgecolor="white", linewidth=0.4, label="call flips")
ax.axhline(TAU, color="#BBBBBB", lw=0.6, ls=":")
ax.axvline(TAU, color="#BBBBBB", lw=0.6, ls=":")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1e-5, 1)
ax.set_ylim(1e-5, 1)
ax.set_xlabel("Merged C–F peaks: predicted probability")
ax.set_ylabel("Split C–F peaks: predicted probability")
ax.set_title("A  Paired predictions", loc="left", fontweight="bold")
ax.legend(loc="lower right", fontsize=6.0, handlelength=1.0)
ax.grid(True, lw=0.3, color="#EEEEEE")

# B recall
ax = axes[1]
vals = [100 * rec_m, 100 * rec_s]
labs = [f"Merged peaks\n(n={len(pm)}, median {np.median(nm):.0f} peaks)",
        f"Split peaks\n(n={len(ps)}, median {np.median(ns):.0f} peaks)"]
cols = [BLUE, RED]
bars = ax.bar([0, 1], vals, width=0.55, color=cols, alpha=0.92)
for x, v, det in zip([0, 1], vals, [int((pm >= TAU).sum()), int((ps >= TAU).sum())]):
    ax.text(x, v + 1.0, f"{v:.1f}%  ({det}/{len(pm)})", ha="center",
            fontsize=6.4, color=TEXT)
ax.set_xticks([0, 1], labs, fontsize=6.2)
ax.set_ylim(0, 108)
ax.set_ylabel("Recall at τ = 0.50 (%)")
ax.set_title("B  Effect of C–F peak splitting", loc="left", fontweight="bold")
ax.grid(True, axis="y", lw=0.3, color="#EEEEEE")
ax.text(0.5, 0.06, f"median |Δp| = {np.median(np.abs(pm-ps)):.1e}   "
        f"max |Δp| = {np.max(np.abs(pm-ps)):.2f}",
        transform=ax.transAxes, ha="center", fontsize=6.0, color=GREY)

for ext in ("pdf", "png"):
    fig.savefig(B + f"reports/fig_resolution_split.{ext}",
                dpi=600 if ext == "png" else None)
print(json.dumps({k: v for k, v in summary.items() if k != "per_molecule"},
                 indent=1))
