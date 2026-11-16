#!/usr/bin/env python3
"""Hit rate vs exact reconstruction (64 external positives, tau = 0.50).

Dumbbell chart: filled dot = top-10 hit rate (any candidate with the quinolone
core counts), open dot = exact-match-at-10 rate. Structures are left to SI.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from pathlib import Path

ROOT = Path(os.environ.get("NMR_ROOT", str(Path(__file__).resolve().parent.parent)))

TEXT = "#262626"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 7.6, "axes.labelsize": 8.0, "axes.titlesize": 8.8,
    "axes.labelcolor": TEXT, "text.color": TEXT, "xtick.color": TEXT,
    "ytick.color": TEXT, "axes.linewidth": 0.7, "xtick.major.width": 0.55,
    "ytick.major.width": 0.55, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

# model, hit%, hit sd, exact%, family
rows = [
    ("CMGNet-SPrompt", 99.0, 0.9, 29.2, "sprompt"),
    ("XGBoost",        98.4, 0.0, None, "xgb"),
    ("CMGNet-Frag",    97.9, 0.9, 0.0, "cmgnet"),
    ("CMGNet-SMILES",  97.4, 0.9, 30.2, "cmgnet"),
    ("CMGNet 0910",    81.2, 0.0, 9.4, "cmgnet"),
    ("NMRMind",        75.0, 0.0, 1.6, "other"),
    ("NMR-Solver",     64.1, 0.0, 25.0, "other"),
]
col = {"sprompt": "#D55E00", "cmgnet": "#0072B2", "xgb": "#009E73",
       "other": "#8C8C8C"}

fig, ax = plt.subplots(figsize=(7.10, 3.55))
fig.subplots_adjust(left=0.20, right=0.97, top=0.84, bottom=0.16)

yy = np.arange(len(rows))[::-1]
for y, (name, hit, sd, exact, fam) in zip(yy, rows):
    c = col[fam]
    if exact is not None:
        ax.plot([exact, hit], [y, y], color=c, lw=1.6, alpha=0.30, zorder=1)
        ax.scatter([exact], [y], s=42, facecolors="white", edgecolors=c,
                   linewidths=1.2, zorder=3)
        if exact < 8:
            ax.text(exact + 1.6, y, f"{exact:.1f}%", ha="left", va="center",
                    fontsize=5.9, color=c)
        else:
            ax.text(exact - 1.4, y, f"{exact:.1f}%", ha="right", va="center",
                    fontsize=5.9, color=c)
    ax.errorbar(hit, y, xerr=sd, fmt="none", ecolor="#333333", elinewidth=0.8,
                capsize=2.0, capthick=0.7, zorder=2)
    ax.scatter([hit], [y], s=50, color=c, edgecolors="white", linewidths=0.5,
               zorder=4)
    ax.text(hit + 1.6, y, f"{hit:.1f}%", ha="left", va="center",
            fontsize=5.9, color=c)

ax.set_yticks(yy, [r[0] for r in rows], fontsize=6.6)
ax.get_yticklabels()[0].set_fontweight("bold")
ax.get_yticklabels()[0].set_color(col["sprompt"])
ax.set_xlim(-2, 108)
ax.set_xticks([0, 20, 40, 60, 80, 100])
ax.tick_params(axis="x", labelsize=6.4)
ax.set_xlabel("Rate among 64 external positives (%)")
ax.set_title("Hit rate vs exact reconstruction", loc="left", fontweight="bold",
             fontsize=9.0, pad=14)
ax.text(0.0, 1.035, "Filled: any one of top-10 candidates contains the "
        "quinolone core. Open: exact match in top-10. XGBoost does not generate.",
        transform=ax.transAxes, fontsize=5.8, color="#555555", ha="left")
ax.grid(True, axis="x", lw=0.35, color="#EEEEEE", zorder=0)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

# legend
from matplotlib.lines import Line2D
h = [Line2D([0], [0], marker="o", color="none", markerfacecolor="#555555",
            markeredgecolor="white", markersize=5, label="top-10 hit (≥1)"),
     Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
            markeredgecolor="#555555", markersize=5, label="exact@10")]
ax.legend(handles=h, loc="lower right", fontsize=6.2, handletextpad=0.4,
          borderaxespad=0.4)

for e in ("pdf", "png"):
    fig.savefig(str(ROOT / "reports") + f"/fig_hit_rate_summary.{e}",
                dpi=600 if e == "png" else None)
print("saved dumbbell hit-rate figure")
