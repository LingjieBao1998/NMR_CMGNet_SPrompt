#!/usr/bin/env python3
"""Comment R2.4 -- probability calibration and uncertainty-guided manual review.

Inputs (created by reports/encode_eval_sets.py; external set = april18 peak list,
the one used for the reported external-test metrics):
  reports/calib_data/validation.npz            40,667 (37 pos)  reserved dev set
  reports/calib_data/internal_test.npz         40,642 (37 pos)  held-out internal test
  reports/calib_data/external_apr18peak.npz     4,502 (64 pos)  external test
  reports/calib_data/maxcos_trainpos.npz        max cosine to training positives

Protocol
--------
* Calibrators (Platt, isotonic) are fitted ONLY on the reserved validation split.
* The method is selected by 5-fold cross-fitted NLL/Brier on validation, never on
  external-test labels.
* External test is deliberately enriched (1.42% vs 0.091% prevalence); for the
  deployment-calibrated curve we apply the standard prior-probability shift using
  the known sampling prevalences (no labels involved).
* Review rule: two-sided grey band around the reported operating threshold
  (tau_op = 0.50, default threshold; TP 63 / FP 4 / FN 1 / F1 0.9618)
  plus a novelty/discordance trigger: a confidently negative spectrum whose
  nearest-training-positive cosine is >= 0.85 (the lower bound observed for
  validation positives) is referred, not called negative.

Outputs
  reports/calibration_review_metrics.json
  reports/fig_calibration_review.pdf / .png
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold

B = "/mnt/data2/NMR_Core_Mol_Detection/"
OUT = B + "reports/calib_data/"
SETS = ["validation", "internal_test", "external_apr18peak"]
LABELS = {"validation": "Validation (reserved)", "internal_test": "Internal test",
          "external_apr18peak": "External test"}
TAU_OP = 0.50          # reproduces the manuscript confusion matrix
DELTA = 1.5            # grey band = +/- 1.5 in logit units around tau_op
NOVELTY_COS = 0.85     # lower bound of validation-positive nearest-positive cosine
N_BOOT = 1000
RNG = np.random.default_rng(0)

# Okabe-Ito
BLUE, ORANGE, GREEN = "#0072B2", "#E69F00", "#009E73"
RED, GREY, LIGHT = "#D55E00", "#8C8C8C", "#C9CCD4"
TEXT = "#262626"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 7.5,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "axes.labelcolor": TEXT,
    "text.color": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def load():
    d = {s: dict(np.load(OUT + s + ".npz", allow_pickle=True)) for s in SETS}
    mx = np.load(OUT + "maxcos_trainpos.npz")
    for s in SETS:
        d[s]["maxcos"] = mx[s + "_max"]
    return d


def clip_logit(p):
    p = np.asarray(p, dtype=np.float64)
    return logit(np.clip(p, 1e-9, 1 - 1e-9))


def cal_ece(y, p, nb=10):
    """Equal-count (quantile) bins; returns ECE and MCE."""
    edges = np.unique(np.quantile(p, np.linspace(0, 1, nb + 1)))
    idx = np.digitize(p, edges[1:-1])
    ece = mce = 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        gap = abs(p[m].mean() - y[m].mean())
        ece += m.mean() * gap
        mce = max(mce, gap)
    return float(ece), float(mce)


def cal_slope(y, p):
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(clip_logit(p).reshape(-1, 1), y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def metrics(y, p):
    ece, mce = cal_ece(y, p)
    slope, intercept = cal_slope(y, p)
    prev = float(y.mean())
    brier = float(brier_score_loss(y, p))
    return {
        "n": int(len(y)), "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
        "prevalence": prev,
        "ece": ece, "mce": mce,
        "brier": brier,
        "brier_ref": float(prev * (1 - prev)),
        "brier_skill": float(1 - brier / (prev * (1 - prev) + 1e-30)),
        "nll": float(log_loss(y, p, labels=[0, 1])),
        "slope": slope, "intercept": intercept,
        "mean_pred": float(np.mean(p)),
    }


def stratified_idx(y, rng):
    out = []
    for c in (0, 1):
        idx = np.where(y == c)[0]
        out.append(rng.choice(idx, size=len(idx), replace=True))
    return np.concatenate(out)


def bootstrap(y, p, n=N_BOOT):
    vals = {"ece": [], "brier": [], "nll": []}
    for _ in range(n):
        i = stratified_idx(y, RNG)
        e, _ = cal_ece(y[i], p[i])
        vals["ece"].append(e)
        vals["brier"].append(brier_score_loss(y[i], p[i]))
        vals["nll"].append(log_loss(y[i], p[i], labels=[0, 1]))
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for k, v in vals.items()}


def reliability_bins(y, p, edges):
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        k, n = int(y[m].sum()), int(m.sum())
        obs = k / n
        # Wilson 95% interval
        z = 1.96
        den = 1 + z * z / n
        ctr = (obs + z * z / (2 * n)) / den
        half = z * np.sqrt(obs * (1 - obs) / n + z * z / (4 * n * n)) / den
        rows.append(dict(lo=lo, hi=hi, n=n, k=k, pred=float(p[m].mean()),
                         obs=obs, obs_lo=max(0, ctr - half), obs_hi=min(1, ctr + half)))
    return rows


def review_rule(y, p, mx, tau_op=TAU_OP, delta=DELTA, c=NOVELTY_COS):
    lo = float(expit(logit(tau_op) - delta))
    hi = float(expit(logit(tau_op) + delta))
    auto_pos = p >= hi
    grey = (p >= lo) & (p < hi)
    novelty = (p < lo) & (mx >= c)
    review = grey | novelty
    auto_neg = ~auto_pos & ~review
    tp = int((auto_pos & (y == 1)).sum())
    fp = int((auto_pos & (y == 0)).sum())
    rev_pos = int((review & (y == 1)).sum())
    rev_neg = int((review & (y == 0)).sum())
    miss_pos = int((auto_neg & (y == 1)).sum())
    return {
        "tau_op": tau_op, "tau_lo": lo, "tau_hi": hi, "novelty_cos": c,
        "auto_positive": {"tp": tp, "fp": fp, "precision": tp / max(1, tp + fp)},
        "referred": {"pos": rev_pos, "neg": rev_neg,
                     "total": rev_pos + rev_neg,
                     "fraction": (rev_pos + rev_neg) / len(y),
                     "grey_neg": int((grey & (y == 0)).sum()),
                     "novelty_neg": int((novelty & (y == 0)).sum())},
        "auto_negative": int(auto_neg.sum()),
        "silent_miss_pos": miss_pos,
        "sensitivity_before_review": tp / max(1, int(y.sum())),
        "sensitivity_after_review": (tp + rev_pos) / max(1, int(y.sum())),
    }


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(max(0, ctr - half)), float(min(1, ctr + half))


def main():
    d = load()
    yv, pv = d["validation"]["y"], d["validation"]["p_raw"]

    # ---------------- calibrators fitted on validation only ----------------
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    p_pl_cv = np.zeros_like(pv)
    p_iso_cv = np.zeros_like(pv)
    for tr, te in skf.split(pv, yv):
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        lr.fit(clip_logit(pv[tr]).reshape(-1, 1), yv[tr])
        p_pl_cv[te] = lr.predict_proba(clip_logit(pv[te]).reshape(-1, 1))[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        iso.fit(pv[tr], yv[tr])
        p_iso_cv[te] = iso.predict(pv[te])
    cv = {k: metrics(yv, p) for k, p in
          [("raw", pv), ("platt", p_pl_cv), ("isotonic", p_iso_cv)]}
    chosen = min(("platt", "isotonic"), key=lambda k: cv[k]["nll"])

    platt = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    platt.fit(clip_logit(pv).reshape(-1, 1), yv)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(pv, yv)

    pi_val = float(yv.mean())
    pi_ext = float(d["external_apr18peak"]["y"].mean())

    results = {"protocol": {
        "calibrator_selected": chosen,
        "selection_metric": "5-fold cross-fitted NLL on validation",
        "platt_a": float(platt.coef_[0, 0]), "platt_b": float(platt.intercept_),
        "validation_prevalence": pi_val, "external_prevalence": pi_ext,
        "tau_op": TAU_OP, "grey_delta_logit": DELTA, "novelty_cos": NOVELTY_COS,
        "n_bootstrap": N_BOOT,
    }, "cross_fitted_validation": cv, "sets": {}}

    edge_sets = {
        "validation": [0, 1e-6, 1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
        "internal_test": [0, 1e-6, 1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
        "external_apr18peak": [0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.25, 0.5, 0.75, 0.9, 0.97, 1.0],
    }

    for s in SETS:
        y, p, mx = d[s]["y"], d[s]["p_raw"], d[s]["maxcos"]
        p_pl = platt.predict_proba(clip_logit(p).reshape(-1, 1))[:, 1]
        p_is = iso.predict(p)
        curves = {"raw": p, "platt": p_pl, "isotonic": p_is}
        if s == "external_apr18peak":
            # prior correction: validation prevalence -> external sampling prevalence
            shift = np.log(pi_ext / (1 - pi_ext)) - np.log(pi_val / (1 - pi_val))
            curves["platt_prior"] = expit(clip_logit(p_pl) + shift)
        else:
            # same natural prevalence; report the prior-corrected curve too for reference
            shift = np.log(pi_ext / (1 - pi_ext)) - np.log(pi_val / (1 - pi_val))
            curves["platt_prior"] = expit(clip_logit(p_pl) + shift)

        set_res = {"metrics": {}, "reliability": {}, "bootstrap": {},
                   "review": review_rule(y, p, mx)}
        for k, q in curves.items():
            set_res["metrics"][k] = metrics(y, q)
            set_res["reliability"][k] = reliability_bins(y, q, edge_sets[s])
            if k in ("raw", "platt", "platt_prior"):
                set_res["bootstrap"][k] = bootstrap(y, q)
        results["sets"][s] = set_res

    # rule CIs on external
    ext = d["external_apr18peak"]
    r = results["sets"]["external_apr18peak"]["review"]
    se_lo, se_hi = wilson(r["auto_positive"]["tp"] + r["referred"]["pos"],
                          int(ext["y"].sum()))
    ref_lo, ref_hi = wilson(r["referred"]["total"], len(ext["y"]))
    r["sensitivity_after_review_ci"] = [se_lo, se_hi]
    r["referral_fraction_ci"] = [ref_lo, ref_hi]

    json.dump(results, open(B + "reports/calibration_review_metrics.json", "w"),
              indent=2)

    # ------------------------------------------------------------- figure --
    fig = plt.figure(figsize=(7.10, 5.30))
    gs = GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.34,
                  left=0.075, right=0.985, top=0.88, bottom=0.09)

    def rel_panel(ax, s, title, show_iso=True):
        rel = results["sets"][s]["reliability"]
        series = [("raw", GREY, "o", "Raw"), ("platt", BLUE, "s", "Platt"),
                  ("platt_prior", GREEN, "^", "Platt+prior")]
        if show_iso:
            series.append(("isotonic", ORANGE, "D", "Isotonic"))
        for key, col, mk, lab in series:
            rows = [x for x in rel[key] if x["n"] >= 5 and x["pred"] > 0]
            if not rows:
                continue
            x = np.array([r["pred"] for r in rows])
            y = np.array([r["obs"] for r in rows])
            lo = np.clip(np.array([r["obs_lo"] for r in rows]), 1e-6, 1)
            hi = np.clip(np.array([r["obs_hi"] for r in rows]), 1e-6, 1)
            ax.plot(x, y, marker=mk, ms=2.6, lw=0.9, color=col, label=lab,
                    zorder=3)
            if key in ("platt", "raw"):
                ax.fill_between(x, lo, hi, color=col, alpha=0.13, lw=0, zorder=1)
        ax.plot([1e-6, 1], [1e-6, 1], ls="--", lw=0.8, color="#666666", zorder=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1e-5, 1.0)
        ax.set_ylim(1e-5, 1.0)
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.set_title(title, loc="left", fontweight="bold")
        m = results["sets"][s]["metrics"]
        txt = (f"raw  ECE {m['raw']['ece']:.1e}\n"
               f"Platt ECE {m['platt']['ece']:.1e}")
        if s == "external_apr18peak":
            txt = (f"raw  ECE {m['raw']['ece']:.1e}\n"
                   f"prior ECE {m['platt_prior']['ece']:.1e}")
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left",
                fontsize=6.3, color=TEXT,
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.85))
        ax.grid(True, which="major", lw=0.35, color="#E5E5E5", zorder=0)

    for j, (s, title) in enumerate([
            ("validation", "A  Validation (reserved)"),
            ("internal_test", "B  Internal test"),
            ("external_apr18peak", "C  External test")]):
        ax = fig.add_subplot(gs[0, j])
        rel_panel(ax, s, title, show_iso=(s != "external_apr18peak"))
        if j == 0:
            pass

    # D: external score CCDF + review band
    ax = fig.add_subplot(gs[1, 0])
    y, p = ext["y"], ext["p_raw"]
    for cls, col, lab in [(0, GREY, "Negative (4,438)"), (1, BLUE, "Positive (64)")]:
        x = np.clip(np.sort(p[y == cls]), 1e-5, 1.0)
        ccdf = 1.0 - np.arange(len(x)) / len(x)
        ax.plot(x, ccdf, color=col, lw=1.15, label=lab)
    ax.axvspan(r["tau_lo"], r["tau_hi"], color=ORANGE, alpha=0.13, lw=0)
    ax.axvline(TAU_OP, color=RED, ls="--", lw=1.0)
    ax.text(r["tau_lo"] * 1.25, 0.22, "review\nband",
            color=ORANGE, fontsize=5.6, ha="center", va="bottom", rotation=90)
    ax.text(TAU_OP * 1.12, 0.86, r"$\tau_{op}$", color=RED, fontsize=6.0,
            ha="left", va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-5, 1.0)
    ax.set_ylim(8e-5, 1.5)
    ax.set_xlabel("Raw score")
    ax.set_ylabel("Fraction with score $\\geq x$")
    ax.set_title("D  External score distribution", loc="left", fontweight="bold")
    ax.legend(loc="lower left", fontsize=5.6, handlelength=1.0,
              borderaxespad=0.3)
    ax.grid(True, lw=0.35, color="#E5E5E5")

    # E: review outcome, external (table)
    ax = fig.add_subplot(gs[1, 1])
    rr = results["sets"]["external_apr18peak"]["review"]
    pos_parts = [rr["auto_positive"]["tp"], rr["referred"]["pos"],
                 rr["silent_miss_pos"]]
    neg_auto = int(((p < rr["tau_lo"]) & (ext["maxcos"] < NOVELTY_COS) &
                    (y == 0)).sum())
    neg_parts = [rr["auto_positive"]["fp"], rr["referred"]["neg"], neg_auto]
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("E  Review outcome (external)", loc="left", fontweight="bold",
                 x=0.0)
    col_labs = ["Auto-\npositive", "Referred", "Auto-\nnegative"]
    col_cols = [BLUE, ORANGE, GREY]
    for j, (lab, col) in enumerate(zip(col_labs, col_cols)):
        ax.text(0.30 + 0.235 * j, 0.86, lab, ha="center", va="center",
                fontsize=6.0, color=col, fontweight="bold")
    rows = [("Positive\n(n = 64)", pos_parts, BLUE),
            ("Negative\n(n = 4,438)", neg_parts, GREY)]
    for i, (rlab, parts, rcol) in enumerate(rows):
        yy = 0.58 - 0.32 * i
        ax.text(0.02, yy, rlab, ha="left", va="center", fontsize=6.2,
                color=rcol, fontweight="bold")
        tot = sum(parts)
        for j, v in enumerate(parts):
            frac = 100.0 * v / tot
            txt = f"{v}" if frac >= 0.1 else f"{v}"
            ax.text(0.30 + 0.235 * j, yy + 0.035, txt, ha="center", va="center",
                    fontsize=7.4, color=TEXT)
            ax.text(0.30 + 0.235 * j, yy - 0.065, f"({frac:.2f}%)",
                    ha="center", va="center", fontsize=5.6, color="#555555")
    ax.plot([0.02, 0.98], [0.72, 0.72], color="#BBBBBB", lw=0.6)
    ax.plot([0.02, 0.98], [0.40, 0.40], color="#BBBBBB", lw=0.6)
    ax.text(0.02, 0.03, f"silent misses (positive called negative): "
            f"{rr['silent_miss_pos']}", ha="left", va="bottom", fontsize=6.0,
            color=RED if rr["silent_miss_pos"] else TEXT)

    # F: ECE comparison
    ax = fig.add_subplot(gs[1, 2])
    names = ["Validation", "Internal", "External"]
    keys = ["validation", "internal_test", "external_apr18peak"]
    raw_e = [results["sets"][k]["metrics"]["raw"]["ece"] for k in keys]
    cal_key = ["platt", "platt", "platt_prior"]
    cal_e = [results["sets"][k]["metrics"][cal_key[i]]["ece"]
             for i, k in enumerate(keys)]
    xx = np.arange(3)
    ax.bar(xx - 0.19, raw_e, 0.36, color=GREY, label="Raw")
    ax.bar(xx + 0.19, cal_e, 0.36, color=BLUE, label="Calibrated")
    for i in range(3):
        ax.text(xx[i] - 0.19, raw_e[i] * 1.35, f"{raw_e[i]:.1e}",
                ha="center", va="bottom", fontsize=5.3)
        ax.text(xx[i] + 0.19, cal_e[i] * 1.35, f"{cal_e[i]:.1e}",
                ha="center", va="bottom", fontsize=5.3)
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1e-1)
    ax.set_xticks(xx, names)
    ax.set_ylabel("ECE (log scale)")
    ax.set_title("F  Calibration error", loc="left", fontweight="bold")
    ax.legend(loc="upper right", fontsize=5.8, handlelength=1.0)
    ax.grid(True, axis="y", lw=0.35, color="#E5E5E5")

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=GREY, marker="o", ms=3.0, lw=0.9, label="Raw"),
        Line2D([0], [0], color=BLUE, marker="s", ms=3.0, lw=0.9, label="Platt"),
        Line2D([0], [0], color=GREEN, marker="^", ms=3.0, lw=0.9,
               label="Platt + prior correction"),
        Line2D([0], [0], color=ORANGE, marker="D", ms=3.0, lw=0.9,
               label="Isotonic"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.53, 0.965),
               ncol=4, fontsize=6.0, frameon=False, handlelength=1.2,
               columnspacing=1.6, handletextpad=0.5)

    for ext_ in ("pdf", "png"):
        fig.savefig(B + f"reports/fig_calibration_review.{ext_}",
                    dpi=600 if ext_ == "png" else None)
    print(json.dumps({
        "selected": chosen,
        "cv_nll": {k: v["nll"] for k, v in cv.items()},
        "external_review": r,
        "external_metrics": {k: {kk: results["sets"]["external_apr18peak"]["metrics"][k][kk]
                                 for kk in ("ece", "brier", "nll", "slope", "intercept")}
                             for k in ("raw", "platt", "platt_prior")},
    }, indent=2))


if __name__ == "__main__":
    main()
