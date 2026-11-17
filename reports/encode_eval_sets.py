#!/usr/bin/env python3
"""Encode the reserved development (validation) and evaluation sets with the
frozen CReSS encoder, and store raw XGBoost probabilities.

The calibration / review analysis in
reports/legacy_calibration/calibrate_and_review.py only needs these cached
arrays, so the expensive CReSS forward pass is done once.

Outputs (reports/calib_data/):
  <split>_emb.npy   float32 (n, 768), L2-normalised CReSS embeddings
  <split>.npz       y (int8), p_raw (float32), smiles (object)
  meta.json         counts / timing
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = os.environ.get('NMR_ROOT', str(Path(__file__).resolve().parent.parent))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import joblib  # noqa: E402
import torch  # noqa: E402

from infer import ModelInference  # noqa: E402
from train_xgb_cress_new_dataset import load_json  # noqa: E402
from train_xgb_cress_nmrmind import build_cress_config, CRESS_WEIGHTS  # noqa: E402

OUT = os.path.join(ROOT, "reports", "calib_data")
os.makedirs(OUT, exist_ok=True)

SPLITS = {
    "validation": "data/validation/validation.json",
    # 40,642-record held-out internal split from the source distribution
    # (not included in this release)
    "internal_test": "data/new_dataset/test.json",
    "external_test": "data/external_test/external_test.json",
}
MODEL = "models/xgboost/model_xgb_cress_ex_train_seed42.pkl"
BATCH = 512


def log(msg):
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def encode(mi, nmrs):
    X = np.empty((len(nmrs), 768), dtype=np.float32)
    t0 = time.time()
    for i in range(0, len(nmrs), BATCH):
        e = mi.nmr_encode(nmrs[i:i + BATCH])
        X[i:i + len(e)] = e.detach().cpu().numpy().astype(np.float32, copy=False)
        if i and (i // BATCH) % 20 == 0:
            log(f"  {i:,}/{len(nmrs):,} ({time.time() - t0:.0f}s)")
    log(f"encoded {len(nmrs):,} in {time.time() - t0:.1f}s")
    return X


def main():
    torch.set_num_threads(32)
    mi = ModelInference(build_cress_config(), CRESS_WEIGHTS, "cuda")
    clf = joblib.load(MODEL)
    meta = {}

    for name, path in SPLITS.items():
        log(f"=== {name}: {path}")
        nmrs, y, smiles = load_json(path)
        X = encode(mi, nmrs)
        p = clf.predict_proba(X)[:, 1].astype(np.float32)
        np.save(os.path.join(OUT, f"{name}_emb.npy"), X)
        np.savez(os.path.join(OUT, f"{name}.npz"), y=y, p_raw=p,
                 smiles=np.asarray(smiles, dtype=object))
        meta[name] = {"n": int(len(y)), "pos": int(y.sum()),
                      "neg": int((y == 0).sum()), "source": path}
        log(f"{name}: n={len(y):,} pos={int(y.sum())} "
            f"p_raw range {p.min():.3e}-{p.max():.6f}")

    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=2)
    log("done")


if __name__ == "__main__":
    main()
