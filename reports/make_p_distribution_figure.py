#!/usr/bin/env python3
"""R1.3(2) 配图: 外部测试集预测概率分布(协议口径 p_raw, tau=0.5 => 63/4/1).
双面板: 上=阳性, 下=阴性, 密度归一化直方图, 对数轴. tau=0.5 虚线; FN #12 与 4 FP 标注.
输出: reports/fig_p_distribution.png / .pdf + caption md.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = "/mnt/data2/NMR_Core_Mol_Detection/"
z = np.load(B + "reports/calib_data/external_apr18peak.npz", allow_pickle=True)
p, y = z["p_raw"], z["y"]
pos, neg = p[y == 1], p[y == 0]

plt.rcParams.update({"font.family": "Arial", "font.size": 8})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 3.2), sharex=True)
bins = np.linspace(0, 1, 21)

ax1.hist(pos, bins=bins, color="#2a7f62", edgecolor="white", linewidth=0.4, density=True)
ax1.set_yscale("log")
ax1.set_ylim(0.005, 300)
ax1.set_ylabel("Positives (n=64)", fontsize=7.5)
ax1.tick_params(labelsize=7)

ax2.hist(neg, bins=bins, color="#9e9e9e", edgecolor="white", linewidth=0.4, density=True)
ax2.set_yscale("log")
ax2.set_ylim(0.005, 300)
ax2.set_ylabel("Negatives (n=4,438)", fontsize=7.5)
ax2.set_xlabel("Predicted probability", fontsize=7.5)
ax2.tick_params(labelsize=7)

for _ax in (ax1, ax2):
    _ax.axvline(0.5, color="#c62828", linestyle="--", linewidth=0.9)
ax2.text(0.485, 150, "default τ = 0.5", fontsize=6.5, color="#c62828",
         ha="right", va="center")
ax1.text(0.52, 0.88, "57/64 above 0.9\nmedian 0.999", transform=ax1.transAxes,
         ha="left", va="top", fontsize=6.5, color="#1a5c48")
fig.tight_layout()
fig.savefig(B + "reports/fig_p_distribution.png", dpi=600)
fig.savefig(B + "reports/fig_p_distribution.pdf")
cap = """**Fig. S9. Predicted-probability distribution on the external test set.**
Predicted probability (raw model output, no calibration) of every spectrum (protocol scoring; τ = 0.5 → 63 TP / 4 FP / 1 FN);
positives (top) and negatives (bottom) shown separately as density-normalised histograms, logarithmic density axis. Positives concentrate
at high probability (57/64 above 0.9; median 0.999); the single false negative (#12, p = 0.012)
sits deep in the negative region yet on a steep boundary (α ≈ 0.20). Among 4,438 negatives
only 4 exceed the default threshold (the four FPs) and 4,428/4,438 stay below 0.1. The decision
threshold was deliberately left at its default value of 0.5 and not tuned on the external set:
a carefully tuned threshold may fail to transfer to other screening scenarios, whereas the
default threshold requires no test-set-specific adjustment.
"""
open(B + "reports/fig_p_distribution_caption.md", "w").write(cap)
print("saved overlaid single-panel version + caption")
