#!/usr/bin/env python3
"""Figure for the XGBoost confidence-interval analysis (reports/uncertainty_ci.py).

A. composition of the 90% class-conditional conformal prediction sets
B. realized class coverage vs the nominal 90% (Wilson 95% CIs)
C. 3-seed model disagreement versus score (internal test)
"""
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os as _os
B = _os.environ.get('NMR_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
B = B.rstrip('/') + '/'
BLUE, RED, ORANGE, GREY, LIGHT = "#0072B2", "#D55E00", "#E69F00", "#8C8C8C", "#C9CCD4"
TEXT = "#262626"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 7.5, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "axes.labelcolor": TEXT, "text.color": TEXT, "xtick.color": TEXT,
    "ytick.color": TEXT, "axes.linewidth": 0.7, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

r = json.load(open(B + "reports/xgb_uncertainty_metrics.json"))
conf = r["conformal"]["coverage_90"]
t_pos = conf["thresholds"]["include_positive_if_p_geq"]
t_neg = conf["thresholds"]["include_negative_if_p_leq"]
PLATT = [(pl["a"], pl["b"]) for pl in
         [dict(a=r["protocol"]["platt_params_per_seed"][0][0],
               b=r["protocol"]["platt_params_per_seed"][0][1]),
          dict(a=r["protocol"]["platt_params_per_seed"][1][0],
               b=r["protocol"]["platt_params_per_seed"][1][1]),
          dict(a=r["protocol"]["platt_params_per_seed"][2][0],
               b=r["protocol"]["platt_params_per_seed"][2][1])]]


def calibrate(P):
    """P: (n_seeds, n) raw probabilities -> per-seed Platt-calibrated mean."""
    out = []
    for i, p_i in enumerate(P):
        lp = np.log(np.clip(p_i, 1e-9, 1 - 1e-9) /
                    (1 - np.clip(p_i, 1e-9, 1 - 1e-9)))
        out.append(1.0 / (1.0 + np.exp(-(lp * PLATT[i][0] + PLATT[i][1]))))
    return np.mean(np.stack(out), axis=0)

sp = np.load(B + "reports/calib_data/seed_probs.npz")
CONFIG = [("External test", "external_apr18peak"), ("Internal test", "internal_test")]
VALIDATION_NAME = "Validation\n(cross-fitted)"

rows = []
for name, key in CONFIG:
    y = np.load(B + f"reports/calib_data/{key}.npz", allow_pickle=True)["y"]
    p = calibrate(sp[f"{key}_p"])
    has0, has1 = p <= t_neg, p >= t_pos
    for cls, cls_name, col in ((1, "Positive", BLUE), (0, "Negative", GREY)):
        m = y == cls
        n = int(m.sum())
        seg = {
            "pos": int((has1 & ~has0 & m).sum()),
            "unc": int((((has1 & has0) | (~has1 & ~has0)) & m).sum()),
            "neg": int((has0 & ~has1 & m).sum()),
        }
        rows.append((name, cls_name, col, seg, n))

fig = plt.figure(figsize=(7.10, 2.80))
gs = fig.add_gridspec(1, 3, left=0.135, right=0.985, bottom=0.22, top=0.82,
                      wspace=0.46)

# ---------------- A: set composition -----------------------------------
ax = fig.add_subplot(gs[0, 0])
cols3 = [BLUE, ORANGE, LIGHT]
labs3 = ["{positive}", "uncertain (review)", "{negative}"]
ypos = [1.0, 0.62, 0.22, -0.16]
order = [(0, 1), (1, 0), (2, 1), (3, 0)]
for k, (row_i, cls_i) in enumerate(order):
    name, cls_name, col, seg, n = rows[row_i]
    vals = [seg["pos"], seg["unc"], seg["neg"]]
    left = 0
    for v, c in zip(vals, cols3):
        frac = v / n
        ax.barh(ypos[k], frac, left=left, height=0.48, color=c,
                edgecolor="white", lw=0.4)
        if frac > 0.12:
            ax.text(left + frac / 2, ypos[k], f"{v}", ha="center", va="center",
                    fontsize=6.0, color="white" if c == BLUE else TEXT)
        left += frac
    ax.text(-0.02, ypos[k], f"{cls_name} (n={n:,})", ha="right", va="center",
            fontsize=6.0, color=BLUE if cls_i == 1 else GREY)
ax.text(0.01, 1.30, "External test", ha="left", va="center", fontsize=6.4,
        fontweight="bold", color=TEXT)
ax.text(0.01, 0.53, "Internal test", ha="left", va="center", fontsize=6.4,
        fontweight="bold", color=TEXT)
ax.set_yticks([])
ax.set_xlim(0, 1.0)
ax.set_ylim(-0.42, 1.40)
ax.set_xlabel("Fraction of spectra")
ax.set_title("A  90% conformal prediction sets", loc="left", fontweight="bold")
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cols3]
ax.legend(handles, labs3, loc="upper center", bbox_to_anchor=(0.5, 1.34),
          ncol=3, fontsize=5.6, handlelength=0.9, columnspacing=0.9,
          handletextpad=0.4, frameon=False)
ax.grid(True, axis="x", lw=0.35, color="#E5E5E5")

# ---------------- B: realized coverage ---------------------------------
ax = fig.add_subplot(gs[0, 1])
splits = ["validation", "internal_test", "external_apr18peak"]
names = [VALIDATION_NAME, "Internal test", "External test"]
for j, (cls, col, lab) in enumerate(((1, BLUE, "Positive class"),
                                     (0, GREY, "Negative class"))):
    for i, s in enumerate(splits):
        v = conf["sets"][s]
        cov = v["coverage_positive"] if cls == 1 else v["coverage_negative"]
        ci = (v["coverage_positive_ci"] if cls == 1
              else v["coverage_negative_ci"])
        xc = i + (0.17 if cls == 1 else -0.17)
        ax.plot([xc], [cov], "o", ms=4.5, color=col, mec="white", mew=0.5,
                zorder=3, label=lab if i == 0 else None)
        ax.plot([xc, xc], [ci[0], ci[1]], color=col, lw=1.0, zorder=2)
        for cc in (ci[0], ci[1]):
            ax.plot([xc - 0.055, xc + 0.055], [cc, cc], color=col, lw=1.0,
                    zorder=2)
ax.axhline(0.90, ls="--", lw=0.9, color=RED, zorder=1)
ax.text(2.44, 0.885, "nominal 90%", fontsize=5.8, color=RED, ha="right",
        va="top")
ax.set_xticks(range(3), names, fontsize=6.0)
ax.set_ylim(0.62, 1.02)
ax.set_ylabel("Empirical coverage\nof the 90% set", fontsize=7.2)
ax.set_title("B  Coverage check", loc="left", fontweight="bold")
ax.legend(loc="lower left", fontsize=5.8, handlelength=0.9)
ax.grid(True, axis="y", lw=0.35, color="#E5E5E5")

# ---------------- C: seed disagreement vs score -----------------------
ax = fig.add_subplot(gs[0, 2])
P = sp["internal_test_p"]
mean_p = P.mean(0)
rng_ = P.max(0) - P.min(0)
bins = np.logspace(-4, 0, 19)
mids, med, q95 = [], [], []
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (mean_p >= lo) & (mean_p < hi)
    if m.sum() < 30:
        continue
    mids.append(np.sqrt(lo * hi))
    med.append(np.median(rng_[m]))
    q95.append(np.quantile(rng_[m], 0.95))
mids, med, q95 = np.array(mids), np.array(med), np.array(q95)
ax.plot(mids, med, color=BLUE, lw=1.3, label="median")
ax.plot(mids, q95, color=BLUE, lw=1.0, ls="--", label="95th pct.")
ax.fill_between(mids, np.maximum(med / 4, 1e-7),
                np.minimum(med * 4, 1.0), color=BLUE, alpha=0.13, lw=0)
ax.axvspan(0.182, 0.818, color=ORANGE, alpha=0.13, lw=0)
ax.text(0.20, 6e-3, "review\nband", color=ORANGE, fontsize=5.8, ha="center",
        rotation=90)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1e-4, 1.0)
ax.set_ylim(1e-7, 1.0)
ax.set_xlabel("Mean score (3 seeds)")
ax.set_ylabel("Seed disagreement\n|max p − min p|")
ax.set_title("C  Model instability by score", loc="left", fontweight="bold")
ax.legend(loc="lower right", fontsize=5.8, handlelength=1.1)
ax.grid(True, which="major", lw=0.35, color="#E5E5E5")

for ext in ("pdf", "png"):
    fig.savefig(B + f"reports/fig_xgb_uncertainty.{ext}",
                dpi=600 if ext == "png" else None)
print("saved fig_xgb_uncertainty")
print("conformal thresholds: {+} iff p>=%.4g ; {-} iff p<=%.3g (calibrated)"
      % (t_pos, t_neg))
for name, cls_name, col, seg, n in rows:
    print(name, cls_name, seg, n)
