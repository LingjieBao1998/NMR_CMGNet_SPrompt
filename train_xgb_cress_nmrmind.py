#!/usr/bin/env python3
"""Train XGBoost on frozen CReSS embeddings of NMRMind 13C NMR.

Protocol
--------
train : data/nmrmind_cmgnet_1004/train.lmdb
valid : data/nmrmind_cmgnet_1004/validation.json  (early stopping, AUC-PR)
test  : data/nmrmind_cmgnet_1004/test.json        (in-domain)
external : data/nmrmind_cmgnet_1004/external_test.json (4,438 neg + 64 core-pos)

Molecules appearing in ANY eval set are removed from training (exact canonical
SMILES), so both in-domain and external numbers are leakage-free.

CReSS encoder is frozen; only the XGBoost head is trained.
Seeds: 0, 2006, 42.

Usage (from repo root, on 4090):
    CUDA_VISIBLE_DEVICES=0 python -u train_xgb_cress_nmrmind.py
"""
import gc
import json
import os
import pickle
import sys
import time

import joblib
import lmdb
import numpy as np
import torch
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from infer import ModelInference  # noqa: E402

TRAIN_LMDB = "data/nmrmind_cmgnet_1004/train.lmdb"
VALID_JSON = "data/nmrmind_cmgnet_1004/validation.json"
TEST_JSON = "data/nmrmind_cmgnet_1004/test.json"
EXT_JSON = "data/nmrmind_cmgnet_1004/external_test.json"
CRESS_CONFIG = "models/2_5_w_model/8.json"
CRESS_WEIGHTS = "models/2_5_w_model/8.pth"
OUT_DIR = "output/xgb_cress_nmrmind"
SEEDS = [0, 2006, 42]
EMB_BATCH = 512
XGB_DEVICE = "cuda:0"   # 4090 when CUDA_VISIBLE_DEVICES=0


def log(msg):
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def label_of(x):
    lab = x.get("label")
    if isinstance(lab, (list, tuple)):
        return 1 if len(lab) > 0 and int(lab[0]) == 1 else 0
    return 1 if int(lab) == 1 else 0


def load_eval_json(path):
    data = json.load(open(path))
    nmrs, y, smiles = [], [], []
    for x in data:
        raw = x.get("NMR")
        if isinstance(raw, str):
            try:
                nmr = json.loads(raw)
            except Exception:
                nmr = []
        else:
            nmr = raw or []
        smiles.append(x.get("cano_smiles", ""))
        nmrs.append(nmr)
        y.append(label_of(x))
    return nmrs, np.asarray(y, dtype=np.int8), smiles


def load_train_lmdb():
    smiles, nmrs, labels = [], [], []
    env = lmdb.open(TRAIN_LMDB, readonly=True, lock=False, subdir=True)
    t0 = time.time()
    n = 0
    with env.begin() as txn:
        for _k, v in txn.cursor():
            n += 1
            o = pickle.loads(v)
            s = o.get("canonical_smiles")
            nmr = o.get("C_nmr")
            if not s or nmr is None or len(nmr) == 0:
                continue
            smiles.append(s)
            nmrs.append(nmr)
            labels.append(label_of(o))
            if n % 500000 == 0:
                log(f"  loaded {n:,} train records ({time.time()-t0:.0f}s)")
    env.close()
    log(f"loaded {n:,} records, kept {len(smiles):,}, {time.time()-t0:.1f}s")
    return smiles, nmrs, np.asarray(labels, dtype=np.int8)


def encode(mi, nmrs, batch=EMB_BATCH, tag=""):
    X = np.empty((len(nmrs), 768), dtype=np.float32)
    t0 = time.time()
    for i in range(0, len(nmrs), batch):
        e = mi.nmr_encode(nmrs[i:i + batch])
        e = e.detach().cpu().numpy().astype(np.float32, copy=False)
        X[i:i + len(e)] = e
        if tag and (i // batch) % 500 == 0 and i:
            log(f"  embed {tag}: {i:,}/{len(nmrs):,} ({time.time()-t0:.0f}s)")
    log(f"embedded {tag or 'data'}: {X.shape} in {time.time()-t0:.1f}s")
    return X


def choose_threshold(y, p):
    prec, rec, ths = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    if len(ths) == 0:
        return 0.5
    return float(ths[int(np.argmax(f1[:-1]))])


def metrics(y, p, th):
    pred = (p >= th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    k = int(y.sum())
    topk = {}
    if k > 0:
        idx = np.argsort(-p)[:k]
        tp_k = int(y[idx].sum())
        topk = {
            "k": k,
            "tp": tp_k,
            "precision": tp_k / k,
            "recall": tp_k / k,
        }
    return {
        "n": int(len(y)),
        "pos": k,
        "auprc": float(average_precision_score(y, p)) if k else 0.0,
        "roc_auc": float(roc_auc_score(y, p)) if k and k < len(y) else 0.0,
        "threshold": float(th),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "accuracy": float((tp + tn) / len(y)),
        "topK": topk,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    log("=" * 80)
    log("XGBoost-CReSS on NMRMind | seeds 0, 2006, 42")
    log("=" * 80)

    # ---- eval sets ----
    val_nmrs, y_val, val_smiles = load_eval_json(VALID_JSON)
    test_nmrs, y_test, test_smiles = load_eval_json(TEST_JSON)
    ext_nmrs, y_ext, ext_smiles = load_eval_json(EXT_JSON)
    log(f"valid {len(y_val):,} (pos {int(y_val.sum())}), test {len(y_test):,} (pos {int(y_test.sum())}), "
        f"external {len(y_ext):,} (pos {int(y_ext.sum())})")

    # ---- train set + leakage removal ----
    smiles, nmrs, y = load_train_lmdb()
    ext_set, test_set, val_set = set(ext_smiles), set(test_smiles), set(val_smiles)
    blacklist = val_set | test_set | ext_set
    n_before = len(smiles)
    pos_before = int(y.sum())
    overlaps = {
        "external": sum(1 for s in smiles if s in ext_set),
        "test": sum(1 for s in smiles if s in test_set),
        "valid": sum(1 for s in smiles if s in val_set),
    }
    keep = np.fromiter((s not in blacklist for s in smiles), dtype=bool, count=len(smiles))
    smiles = [s for s, k in zip(smiles, keep) if k]
    nmrs = [n for n, k in zip(nmrs, keep) if k]
    y = y[keep]
    log(f"removed train molecules present in eval sets: {n_before-len(smiles):,} records "
        f"(pos {pos_before-int(y.sum())}); train now {len(y):,} (pos {int(y.sum())})")
    log(f"raw train overlaps (records): {overlaps}")

    # ---- embeddings (CReSS frozen, CPU is fast) ----
    torch.set_num_threads(32)
    mi = ModelInference(config_path=CRESS_CONFIG, pretrain_model_path=CRESS_WEIGHTS, device="cpu")
    X_train = encode(mi, nmrs, tag="train")
    del nmrs
    gc.collect()
    X_val = encode(mi, val_nmrs, tag="valid")
    X_test = encode(mi, test_nmrs, tag="nmrmind-test")
    X_ext = encode(mi, ext_nmrs, tag="external")

    neg = int((y == 0).sum())
    pos = int((y == 1).sum())
    spw = float(neg / max(1, pos))
    log(f"train shape {X_train.shape}, neg={neg:,}, pos={pos:,}, scale_pos_weight={spw:.2f}")

    summary = {
        "created": time.strftime("%F %T"),
        "data": {
            "train_records": int(len(y)),
            "train_pos": pos,
            "train_neg": neg,
            "removed_records": int(n_before - len(y)),
            "removed_pos": int(pos_before - pos),
            "raw_overlap_records": overlaps,
            "valid": {"n": int(len(y_val)), "pos": int(y_val.sum())},
            "nmrmind_test": {"n": int(len(y_test)), "pos": int(y_test.sum())},
            "external": {"n": int(len(y_ext)), "pos": int(y_ext.sum())},
        },
        "config": {
            "n_estimators": 2000,
            "max_depth": 5,
            "learning_rate": 0.1,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "gamma": 0.1,
            "reg_alpha": 0.5,
            "reg_lambda": 0.5,
            "scale_pos_weight": spw,
            "eval_metric": ["logloss", "auc", "aucpr"],
            "early_stopping_rounds": 100,
            "tree_method": "hist",
            "device": XGB_DEVICE,
            "cress": CRESS_WEIGHTS,
            "seeds": SEEDS,
        },
        "seeds": {},
    }

    ext_ps, test_ps = [], []
    for seed in SEEDS:
        log("-" * 80)
        log(f"seed {seed}: training")
        clf = xgb.XGBClassifier(
            n_estimators=2000,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            gamma=0.1,
            reg_alpha=0.5,
            reg_lambda=0.5,
            scale_pos_weight=spw,
            objective="binary:logistic",
            eval_metric=["logloss", "auc", "aucpr"],
            tree_method="hist",
            device=XGB_DEVICE,
            n_jobs=32,
            random_state=seed,
            early_stopping_rounds=100,
        )
        t0 = time.time()
        clf.fit(X_train, y, eval_set=[(X_val, y_val)], verbose=100)
        log(f"seed {seed}: fit done in {time.time()-t0:.1f}s, best_iteration={clf.best_iteration}")

        # Keep only trees up to the early-stopping best iteration and export a
        # CPU-safe model.  Do NOT call Booster.set_param here: with a list
        # eval_metric it corrupts the booster config in xgboost 3.3.
        n_trees = (int(clf.best_iteration) + 1) if clf.best_iteration is not None \
            else clf.get_booster().num_boosted_rounds()
        bst = clf.get_booster()[:n_trees]
        model_json = os.path.join(OUT_DIR, f"model_xgb_cress_nmrmind_seed{seed}.json")
        bst.save_model(model_json)
        clf_cpu = xgb.XGBClassifier()
        clf_cpu.load_model(model_json)
        if not hasattr(clf_cpu, "classes_") or clf_cpu.classes_ is None:
            clf_cpu.classes_ = np.array([0, 1])
        if not hasattr(clf_cpu, "n_features_in_"):
            clf_cpu.n_features_in_ = X_train.shape[1]

        p_val = clf_cpu.predict_proba(X_val)[:, 1]
        p_ext = clf_cpu.predict_proba(X_ext)[:, 1]
        p_test = clf_cpu.predict_proba(X_test)[:, 1]
        ext_ps.append(p_ext)
        test_ps.append(p_test)
        th = choose_threshold(y_val, p_val)

        res = {
            "best_iteration": int(clf.best_iteration),
            "n_trees": int(n_trees),
            "val_auprc": float(average_precision_score(y_val, p_val)),
            "val_threshold_f1max": th,
            "external@0.5": metrics(y_ext, p_ext, 0.5),
            "external@val_th": metrics(y_ext, p_ext, th),
            "nmrmind_test@0.5": metrics(y_test, p_test, 0.5),
            "nmrmind_test@val_th": metrics(y_test, p_test, th),
        }
        summary["seeds"][str(seed)] = res
        log(f"seed {seed}: val AUPRC={res['val_auprc']:.4f} th={th:.4f} | "
            f"ext AUPRC={res['external@0.5']['auprc']:.4f} F1@0.5={res['external@0.5']['f1']:.4f} "
            f"F1@th={res['external@val_th']['f1']:.4f} | "
            f"test AUPRC={res['nmrmind_test@0.5']['auprc']:.4f} F1@th={res['nmrmind_test@val_th']['f1']:.4f}")

        model_path = os.path.join(OUT_DIR, f"model_xgb_cress_nmrmind_seed{seed}.pkl")
        joblib.dump(clf_cpu, model_path)
        log(f"seed {seed}: saved {model_path} ({n_trees} trees)")

    # ---- seed ensemble ----
    if ext_ps:
        p_ext_ens = np.mean(ext_ps, axis=0)
        p_test_ens = np.mean(test_ps, axis=0)
        # threshold for ensemble: mean of the per-seed validation F1-optimal thresholds
        th_ens = float(np.mean([summary["seeds"][str(s)]["val_threshold_f1max"] for s in SEEDS]))
        summary["ensemble"] = {
            "threshold": th_ens,
            "external@0.5": metrics(y_ext, p_ext_ens, 0.5),
            "external@val_th_mean": metrics(y_ext, p_ext_ens, th_ens),
            "nmrmind_test@0.5": metrics(y_test, p_test_ens, 0.5),
            "nmrmind_test@val_th_mean": metrics(y_test, p_test_ens, th_ens),
        }
        log(f"ensemble: ext AUPRC={summary['ensemble']['external@0.5']['auprc']:.4f} "
            f"F1@th={summary['ensemble']['external@val_th_mean']['f1']:.4f}; "
            f"test AUPRC={summary['ensemble']['nmrmind_test@0.5']['auprc']:.4f}")

    # ---- save ----
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    lines = [
        "# XGBoost-CReSS on NMRMind (frozen CReSS embeddings)",
        "",
        f"train {len(y):,} (pos {pos:,}, neg {neg:,}) | valid {len(y_val):,} (pos {int(y_val.sum())}) "
        f"| NMRMind test {len(y_test):,} (pos {int(y_test.sum())}) | external {len(y_ext):,} (pos {int(y_ext.sum())})",
        f"scale_pos_weight={spw:.2f}; leaked train records removed={n_before-len(y):,}",
        "",
        "| seed | best_iter | val AUPRC | ext AUPRC | ext ROC-AUC | ext P@0.5 | ext R@0.5 | ext F1@0.5 | ext F1@th | ext topK P/R | test AUPRC | test F1@th |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for seed in SEEDS:
        r = summary["seeds"][str(seed)]
        e05 = r["external@0.5"]
        eth = r["external@val_th"]
        tth = r["nmrmind_test@val_th"]
        lines.append(
            f"| {seed} | {r['best_iteration']} | {r['val_auprc']:.4f} | {e05['auprc']:.4f} | "
            f"{e05['roc_auc']:.4f} | {e05['precision']:.4f} | {e05['recall']:.4f} | "
            f"{e05['f1']:.4f} | {eth['f1']:.4f} | {e05['topK'].get('precision',0):.4f}/{e05['topK'].get('recall',0):.4f} | "
            f"{tth['auprc']:.4f} | {tth['f1']:.4f} |"
        )
    if "ensemble" in summary:
        ens = summary["ensemble"]
        e05 = ens["external@0.5"]
        eth = ens["external@val_th_mean"]
        tth = ens["nmrmind_test@val_th_mean"]
        lines.append(
            f"| ensemble | - | - | {e05['auprc']:.4f} | {e05['roc_auc']:.4f} | "
            f"{e05['precision']:.4f} | {e05['recall']:.4f} | {e05['f1']:.4f} | {eth['f1']:.4f} | "
            f"{e05['topK'].get('precision',0):.4f}/{e05['topK'].get('recall',0):.4f} | "
            f"{tth['auprc']:.4f} | {tth['f1']:.4f} |"
        )
    with open(os.path.join(OUT_DIR, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    log("done")
    log("\n".join(lines))


if __name__ == "__main__":
    main()
