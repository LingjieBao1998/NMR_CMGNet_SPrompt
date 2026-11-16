#!/usr/bin/env python3
"""Comment R2.4 (follow-up) -- confidence intervals for XGBoost predictions.

XGBoost itself yields a point probability only (no native CI), so three
resampling-based interval types are computed here:

  (1) Bootstrap CI of the *calibrated* class probability for an individual
      sample: the Platt map is refitted on 200 stratified bootstrap resamples
      of the validation split and applied to each eval sample.
  (2) Conformal prediction sets (Mondrian, class-conditional, coverage 90%):
      the calibration scores come from 5-fold cross-fitted Platt probabilities
      on validation, so no sample is calibrated with a model that saw it.
  (3) Model instability: probabilities from the three trained seeds
      (0 / 42 / 2006, identical configuration and data).

Outputs
  reports/calib_data/seed_probs.npz
  reports/xgb_uncertainty_metrics.json
  reports/fig_xgb_uncertainty.pdf / .png
"""
import json
import os
import sys
import time

import numpy as np
import joblib
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

import os as _os
B = _os.environ.get('NMR_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
B = B.rstrip('/') + '/'
sys.path.insert(0, B)
OUT = B + "reports/"
D = OUT + "calib_data/"
SETS = ["validation", "internal_test", "external_apr18peak"]
MODELS = [
    "models/xgboost/model_xgb_cress_ex_train_seed0.pkl",
    "models/xgboost/model_xgb_cress_ex_train_seed42.pkl",
    "models/xgboost/model_xgb_cress_ex_train_seed2006.pkl",
]
N_BOOT_CAL = 200
ALPHA = 0.10
RNG = np.random.default_rng(0)
BLUE, RED, ORANGE, GREY, GREEN, LIGHT = ("#0072B2", "#D55E00", "#E69F00",
                                         "#8C8C8C", "#009E73", "#C9CCD4")
TEXT = "#262626"

plt = None  # filled at import time below to keep matplotlib import optional


def clip_logit(p):
    p = np.asarray(p, dtype=np.float64)
    return logit(np.clip(p, 1e-9, 1 - 1e-9))


def fit_platt(logit_p, y):
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(logit_p.reshape(-1, 1), y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def apply_platt(params, lp):
    a, b = params
    return expit(lp * a + b)


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(p), float(max(0, ctr - half)), float(min(1, ctr + half))


def main():
    data = {s: dict(np.load(D + s + ".npz", allow_pickle=True)) for s in SETS}
    print("loading models ...", flush=True)
    clfs = [joblib.load(B + m) for m in MODELS]
    probs = {}
    for s in SETS:
        X = np.load(D + f"{s}_emb.npy")
        t0 = time.time()
        P = np.stack([c.predict_proba(X)[:, 1] for c in clfs]).astype(np.float32)
        probs[s] = P
        print(f"  {s}: {P.shape} ({time.time()-t0:.1f}s) "
              f"pos>0.5 per seed {[int((P[i][data[s]['y']==1]>0.5).sum()) for i in range(3)]}",
              flush=True)
    np.savez(D + "seed_probs.npz", **{f"{s}_p": probs[s] for s in SETS})

    yv = data["validation"]["y"]
    lp_all = {s: clip_logit(probs[s]) for s in SETS}

    # ---- calibrator per seed, fitted on validation (each seed's own scores) --
    platt = [fit_platt(lp_all["validation"][i], yv) for i in range(3)]
    pcal = {s: np.stack([apply_platt(platt[i], lp_all[s][i]) for i in range(3)])
            for s in SETS}

    # ---- (3) model instability -------------------------------------------
    spread = {}
    for s in SETS:
        p = pcal[s]
        spread[s] = {"sd_mean": float(p.std(0).mean()),
                     "range_mean": float((p.max(0) - p.min(0)).mean()),
                     "range_median": float(np.median(p.max(0) - p.min(0))),
                     "range_p95": float(np.quantile(p.max(0) - p.min(0), 0.95))}

    # ---- (2) conformal: cross-fitted Platt on validation ------------------
    pcv = np.zeros((3, len(yv)), dtype=np.float32)
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    for tr, te in skf.split(np.zeros(len(yv)), yv):
        for i in range(3):
            a, b = fit_platt(lp_all["validation"][i][tr], yv[tr])
            pcv[i, te] = apply_platt((a, b), lp_all["validation"][i][te])
    p_cv_mean = pcv.mean(0)

    conf = {}
    for cov in (0.90,):
        n1 = int(yv.sum())
        n0 = int((1 - yv).sum())
        # scores: class k set membership uses s_k = 1 - p_hat_k
        q1 = np.quantile(1.0 - p_cv_mean[yv == 1],
                         min(1.0, (np.ceil((n1 + 1) * cov)) / n1), method="lower")
        q0 = np.quantile(p_cv_mean[yv == 0],
                         min(1.0, (np.ceil((n0 + 1) * cov)) / n0), method="lower")
        t_pos = 1.0 - float(q1)     # include 1 iff p >= t_pos
        t_neg = float(q0)           # include 0 iff p <= t_neg
        level_pos = int(np.ceil((n1 + 1) * cov))
        level_neg = int(np.ceil((n0 + 1) * cov))
        conf[f"coverage_{int(cov*100)}"] = {
            "levels": {"pos": level_pos, "neg": level_neg, "n_pos": n1, "n_neg": n0},
            "thresholds": {"include_positive_if_p_geq": t_pos,
                           "include_negative_if_p_leq": t_neg},
            "sets": {},
        }
        for s in SETS:
            y = data[s]["y"]
            p = pcal[s].mean(0)
            has1 = p >= t_pos
            has0 = p <= t_neg
            single0 = has0 & ~has1
            single1 = has1 & ~has0
            double = has0 & has1
            empty = ~has0 & ~has1
            cov_pos = float((has1[y == 1]).mean())
            cov_neg = float((has0[y == 0]).mean())
            cw_p = wilson(int((has1[y == 1]).sum()), int(y.sum()))
            cw_n = wilson(int((has0[y == 0]).sum()), int((1 - y).sum()))
            conf[f"coverage_{int(cov*100)}"]["sets"][s] = {
                "single_negative": int(single0.sum()), "doubleton": int(double.sum()),
                "single_positive": int(single1.sum()), "empty": int(empty.sum()),
                "doubleton_among_pos": int(double[y == 1].sum()),
                "doubleton_among_neg": int(double[y == 0].sum()),
                "coverage_positive": cov_pos, "coverage_positive_ci": [cw_p[1], cw_p[2]],
                "coverage_negative": cov_neg, "coverage_negative_ci": [cw_n[1], cw_n[2]],
            }

    # ---- (1) calibration-map bootstrap intervals --------------------------
    boot = {}
    for s in SETS:
        y = data[s]["y"]
        lp = lp_all[s].mean(0)          # ensemble logit (seeds agree closely)
        p_cal = pcal[s].mean(0)         # calibrated ensemble probability
        lows, ups = [], []
        for _ in range(N_BOOT_CAL):
            idx = []
            for c in (0, 1):
                sub = np.where(yv == c)[0]
                idx.append(RNG.choice(sub, len(sub), replace=True))
            idx = np.concatenate(idx)
            a, b = fit_platt(lp_all["validation"].mean(0)[idx], yv[idx])
            q = apply_platt((a, b), lp)
            lows.append(q)
            ups.append(q)
        L = np.percentile(np.stack(lows), 5, axis=0)
        U = np.percentile(np.stack(ups), 95, axis=0)
        boot[s] = {"interval_width_median": float(np.median(U - L)),
                   "interval_width_mean": float(np.mean(U - L))}
        np.savez(D + f"{s}_p_interval.npz", lo=L, hi=U, p=p_cal)
        print(f"  boot {s}: median 90% width on calibrated p = "
              f"{boot[s]['interval_width_median']:.4f}", flush=True)

    res = {
        "protocol": {"n_seeds": 3, "models": MODELS, "platt_params_per_seed": platt,
                     "conformal_alpha": 0.10, "n_boot_cal": N_BOOT_CAL},
        "spread": spread, "conformal": conf, "calibration_bootstrap": boot,
    }
    json.dump(res, open(OUT + "xgb_uncertainty_metrics.json", "w"), indent=2)
    print(json.dumps({"spread": spread, "conformal": conf, "boot": boot}, indent=2))


if __name__ == "__main__":
    main()
