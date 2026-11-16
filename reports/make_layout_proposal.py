#!/usr/bin/env python3
"""Wireframe proposals for merging the embedding figure with the model-analysis
panels. No data; layout only."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

TEXT = "#262626"
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "pdf.fonttype": 42})

fig, axes = plt.subplots(1, 3, figsize=(10.2, 4.6))
for ax in axes:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def box(ax, x, y, w, h, label, fc="#EAF1F6", ec="#7A9BB5", fs=5.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.006,rounding_size=0.012",
                                fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fs, color=TEXT, fontweight="bold" if bold else "normal")


def title(ax, t):
    ax.text(0.02, 0.97, t, ha="left", va="top", fontsize=8.0,
            fontweight="bold", color=TEXT)


# ---- option 1: two figures (recommended) --------------------------------
ax = axes[0]
title(ax, "Option 1 (recommended)\nTwo figures")
ax.add_patch(Rectangle((0.02, 0.08), 0.44, 0.82, fill=False, ec="#BBBBBB",
                       lw=0.7, ls="--"))
ax.text(0.04, 0.87, "Figure 1", fontsize=6.4, fontweight="bold")
box(ax, 0.04, 0.72, 0.40, 0.11, "A  global UMAP", fc="#DCEBF5", bold=True, fs=6.0)
box(ax, 0.04, 0.55, 0.40, 0.11, "B  FN#12 nbh.")
box(ax, 0.04, 0.38, 0.40, 0.11, "C  neighbour structures")
box(ax, 0.04, 0.21, 0.40, 0.11, "D  interpolation")
ax.add_patch(Rectangle((0.50, 0.08), 0.48, 0.82, fill=False, ec="#BBBBBB",
                       lw=0.7, ls="--"))
ax.text(0.52, 0.87, "Figure 2", fontsize=6.4, fontweight="bold")
labs = [("E  PR", 0.52, 0.71), ("F  ROC", 0.675, 0.71),
        ("G  Cal.", 0.83, 0.71), ("H  Review", 0.52, 0.52),
        ("I  MC-Drop", 0.675, 0.52), ("J  Accuracy", 0.83, 0.52)]
for lab, xx, yy in labs:
    box(ax, xx, yy, 0.14, 0.11, lab, fc="#F5EEE6" if lab[0] in "EFG" else "#E9F3EE",
        fs=5.7)
ax.text(0.5, 0.15, "each figure ≤ 180 mm wide; fonts ≥ 6 pt",
        ha="center", fontsize=5.6, color="#666666")

# ---- option 2: one full-page figure -------------------------------------
ax = axes[1]
title(ax, "Option 2\nOne full-page figure (10 panels)")
ax.add_patch(Rectangle((0.02, 0.03), 0.96, 0.86, fill=False, ec="#BBBBBB",
                       lw=0.7, ls="--"))
box(ax, 0.04, 0.74, 0.92, 0.12, "A  global UMAP", fc="#DCEBF5", bold=True)
for i, (lab, xx) in enumerate([("B  FN#12\nneighbourhood", 0.04),
                               ("C  neighbours", 0.355),
                               ("D  interpolation", 0.67)]):
    box(ax, xx, 0.60, 0.29, 0.11, lab, fc="#EAF1F6")
for i, (lab, xx) in enumerate([("E  PR", 0.04), ("F  ROC", 0.355),
                               ("G  calibration", 0.67)]):
    box(ax, xx, 0.46, 0.29, 0.10, lab, fc="#F5EEE6")
for i, (lab, xx) in enumerate([("H  review", 0.04),
                               ("I  MC-Dropout", 0.355),
                               ("J  ring accuracy", 0.67)]):
    box(ax, xx, 0.32, 0.29, 0.10, lab, fc="#E9F3EE")
ax.text(0.5, 0.20, "one figure = ~180 × 260 mm (full page)\n"
        "fonts 5.5–6.5 pt; structures from N-heterocycle\ngallery go to SI",
        ha="center", fontsize=5.8, color="#666666")

# ---- option 3: compact 8-panel ------------------------------------------
ax = axes[2]
title(ax, "Option 3\nOne compact figure (8 panels)")
ax.add_patch(Rectangle((0.02, 0.03), 0.96, 0.86, fill=False, ec="#BBBBBB",
                       lw=0.7, ls="--"))
box(ax, 0.04, 0.76, 0.92, 0.10, "A  global UMAP", fc="#DCEBF5", bold=True)
box(ax, 0.04, 0.63, 0.29, 0.10, "B  FN#12\nneighbourhood")
box(ax, 0.355, 0.63, 0.29, 0.10, "C  neighbours")
box(ax, 0.67, 0.63, 0.29, 0.10, "D  interpolation")
box(ax, 0.04, 0.49, 0.22, 0.10, "E  PR")
box(ax, 0.28, 0.49, 0.22, 0.10, "F  ROC")
box(ax, 0.52, 0.49, 0.22, 0.10, "G  calibration")
box(ax, 0.76, 0.49, 0.20, 0.10, "H  accuracy")
ax.text(0.5, 0.35, "review workflow, MC-Dropout and ring-system\n"
        "structures stay in SI; one ~180 × 180 mm figure",
        ha="center", fontsize=5.8, color="#666666")

for ext in ("pdf", "png"):
    fig.savefig(f"/mnt/data2/NMR_Core_Mol_Detection/reports/fig_layout_proposal.{ext}",
                dpi=600 if ext == "png" else None)
print("saved layout proposal")
