#!/usr/bin/env python3
"""Do misclassified external-test molecules have wider uncertainty intervals?

Two per-molecule interval estimates are available without retraining:
  * range of raw probabilities across the three trained seeds (0 / 42 / 2006)
  * 90% bootstrap interval of the calibrated probability (Platt refit on
    validation resamples)
Errors (4 FP + 1 FN at tau = 0.50) are compared with correct calls
(Mann-Whitney one-sided).

Outputs
  reports/error_uncertainty_metrics.json
  reports/fig_error_uncertainty.pdf / .png
"""
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

B = "/mnt/data2/NMR_Core_Mol_Detection/"
RED, GREY, BLUE = "#D55E00", "#A6A6A6", "#0072B2"
TEXT = "#262626"
rng = np.random.default_rng(0)
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 7.5, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "axes.labelcolor": TEXT, "text.color": TEXT, "xtick.color": TEXT,
    "ytick.color": TEXT, "axes.linewidth": 0.7, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

sp = np.load(B + "reports/calib_data/seed_probs.npz")["external_apr18peak_p"]
y = np.load(B + "reports/calib_data/external_apr18peak.npz",
            allow_pickle=True)["y"]
p = sp.mean(0)
pred = (p >= 0.5).astype(int)
err = pred != y
w_seed = sp.max(0) - sp.min(0)
iv = np.load(B + "reports/calib_data/external_apr18peak_p_interval.npz")
w_cal = iv["hi"] - iv["lo"]

metrics = {}
for name, w in (("seed_range", w_seed), ("calibrated_ci_width", w_cal)):
    a, b = w[err], w[~err]
    metrics[name] = {
        "error_median": float(np.median(a)),
        "error_mean": float(np.mean(a)),
        "correct_median": float(np.median(b)),
        "correct_mean": float(np.mean(b)),
        "mannwhitney_p_greater": float(
            mannwhitneyu(a, b, alternative="greater").pvalue),
        "n_error": int(len(a)), "n_correct": int(len(b)),
    }
metrics["n_errors"] = int(err.sum())
metrics["n_fp"] = int(((pred == 1) & (y == 0)).sum())
metrics["n_fn"] = int(((pred == 0) & (y == 1)).sum())
metrics["errors"] = [
    dict(idx=int(i), label=int(y[i]), p=float(p[i]),
         seed_range=float(w_seed[i]), cal_ci_width=float(w_cal[i]))
    for i in np.where(err)[0]]
json.dump(metrics, open(B + "reports/error_uncertainty_metrics.json", "w"),
          indent=1)


def panel(ax, w, title, ylabel):
    data = [np.log10(np.clip(w[~err], 1e-12, None)),
            np.log10(np.clip(w[err], 1e-12, None))]
    bp = ax.boxplot(data, positions=[0, 1], widths=0.45, patch_artist=True,
                    showfliers=False, medianprops=dict(color=TEXT, lw=1.0),
                    whiskerprops=dict(color="#555555", lw=0.8),
                    capprops=dict(color="#555555", lw=0.8),
                    boxprops=dict(lw=0.8))
    for patch, col in zip(bp["boxes"], [GREY, RED]):
        patch.set_facecolor(col)
        patch.set_alpha(0.55)
    for i, vals in enumerate([w[~err], w[err]]):
        x = i + rng.uniform(-0.11, 0.11, len(vals))
        ax.scatter(x, np.log10(np.clip(vals, 1e-12, None)), s=3,
                   color=GREY if i == 0 else RED, alpha=0.35, lw=0)
    ax.set_xticks([0, 1], [f"Correct\n(n={int((~err).sum()):,})",
                           f"Error\n(n={int(err.sum())})"], fontsize=6.4)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(True, axis="y", lw=0.3, color="#EEEEEE")
    pval = metrics["seed_range" if "seed" in ylabel.lower() else
                   "calibrated_ci_width"]["mannwhitney_p_greater"]
    ax.text(0.5, 0.97, f"Mann–Whitney p = {pval:.1e}", transform=ax.transAxes,
            ha="center", va="top", fontsize=6.0, color=TEXT)


fig, axes = plt.subplots(1, 2, figsize=(7.10, 2.90))
fig.subplots_adjust(left=0.09, right=0.98, bottom=0.22, top=0.84, wspace=0.36)
panel(axes[0], w_seed, "A  Seed-ensemble range",
      "log$_{10}$(max − min p across 3 seeds)")
panel(axes[1], w_cal, "B  Calibrated-probability CI width",
      "log$_{10}$(90% CI width)")

for ext in ("pdf", "png"):
    fig.savefig(B + f"reports/fig_error_uncertainty.{ext}",
                dpi=600 if ext == "png" else None)
print(json.dumps(metrics, indent=1))
