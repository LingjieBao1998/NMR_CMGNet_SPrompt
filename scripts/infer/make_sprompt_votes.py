#!/usr/bin/env python
"""Compute the XGBoost-CReSS gate used by CMGNet-SPrompt.

For every spectrum in `--test_json` the 768-d CReSS embedding is extracted and
scored by the XGBoost core classifier.  The resulting 0/1 list (threshold 0.5)
is written as JSON and can be fed to `main_cmgnet.py --sprompt_votes`.

Usage:
  python scripts/infer/make_sprompt_votes.py \
      --test_json data/external_test/external_test.json \
      --out_json  output/sprompt_votes_external.json \
      --batch_size 256 --threshold 0.5
"""
import argparse
import json
import os
import sys
import time

import joblib
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from nmr_core_system.core_detection.infer import ModelInference  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_json", default="data/external_test/external_test.json")
    ap.add_argument("--out_json", default="output/sprompt_votes_external.json")
    ap.add_argument("--cress_config", default="checkpoints/cress/2_5_w_model/8.json")
    ap.add_argument("--cress_weights", default="checkpoints/cress/2_5_w_model/8.pth")
    ap.add_argument("--tokenizer", default="checkpoints/cress/tokenizer-smiles-roberta-1e")
    ap.add_argument("--xgb_model", default="models/xgboost/model_xgb_cress_1220.pkl")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    data = json.load(open(args.test_json, encoding="utf-8"))
    nmr = [r.get("NMR", r.get("C_nmr")) for r in data]
    assert all(x is not None for x in nmr), "records need an NMR/C_nmr field"

    cfg = json.load(open(args.cress_config))
    cfg["model_config"]["roberta_tokenizer_path"] = os.path.abspath(args.tokenizer)
    cfg["dataset_config"]["smiles_tokenizer"] = os.path.abspath(args.tokenizer)
    tmp_cfg = os.path.join(os.path.dirname(os.path.abspath(args.out_json)), "_cress_config.json")
    os.makedirs(os.path.dirname(tmp_cfg), exist_ok=True)
    json.dump(cfg, open(tmp_cfg, "w"))

    cress = ModelInference(tmp_cfg, args.cress_weights, device=args.device)
    xgb = joblib.load(args.xgb_model)

    probs = []
    t0 = time.time()
    for i in range(0, len(nmr), args.batch_size):
        batch = nmr[i:i + args.batch_size]
        with torch.no_grad():
            emb = cress.nmr_encode(batch).cpu().numpy()
        probs.extend(xgb.predict_proba(emb)[:, 1].tolist())
        print(f"[{min(i + args.batch_size, len(nmr))}/{len(nmr)}] "
              f"{time.time() - t0:.1f}s", flush=True)

    votes = [int(p >= args.threshold) for p in probs]
    with open(args.out_json, "w") as f:
        json.dump(votes, f)
    with open(args.out_json.replace(".json", "_probs.json"), "w") as f:
        json.dump(probs, f)
    print(f"wrote {args.out_json}: {len(votes)} votes, {sum(votes)} positive "
          f"(threshold {args.threshold})")


if __name__ == "__main__":
    main()
