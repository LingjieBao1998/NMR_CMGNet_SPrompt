#!/usr/bin/env python3
"""XGBoost-CReSS on the organized new_dataset.

train : data/new_dataset/train.lmdb  (new_db, 3,196,326)
extra : optional --extra_train_lmdb, e.g. data/new_data_1209.filtered.lmdb
        (77,328 positives; eval-overlap molecules removed)
        optional --extra_train_json, e.g.
        data/extra_training/ex_train_add_label_filtered.evalclean.json
        (24,520 nmrshiftdb records: 24,509 neg + 11 pos; eval overlap 0)
valid : data/new_dataset/validation.json (40,667 / 37 pos)
test  : data/new_dataset/test.json (40,642 / 37 pos)
external : data/new_dataset/external_test.json (4,438 neg + 64 pos)
f_split / f_without_split : 48 each (all positives)

Seeds: 0, 2006.
CReSS is frozen; only the XGBoost head is trained.
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

# new_db was pickled with numpy 2.x; py312/numpy 1.26 needs the _core aliases
sys.modules["numpy._core"] = np.core
for _sub in ["numeric", "multiarray", "umath", "_multiarray_umath", "function_base", "shape_base"]:
    _mod = getattr(np.core, _sub, None)
    if _mod is not None:
        sys.modules["numpy._core." + _sub] = _mod

import lmdb  # noqa: E402
import xgboost as xgb  # noqa: E402
import joblib  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

# reuse the metric/encoding helpers from the NMRMind XGBoost script
import train_xgb_cress_nmrmind as base  # noqa: E402


def label_of(rec):
    lab = rec.get("label")
    if isinstance(lab, np.ndarray):
        return int(lab.reshape(-1)[0])
    if isinstance(lab, (list, tuple)):
        return 1 if len(lab) > 0 and int(lab[0]) == 1 else 0
    return 1 if int(lab) == 1 else 0


def parse_nmr(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        vals = value
    elif isinstance(value, str):
        s = value.strip()
        try:
            parsed = json.loads(s)
            vals = parsed if isinstance(parsed, list) else s.replace("[", "").replace("]", "").replace(",", " ").split()
        except Exception:
            vals = s.replace("[", "").replace("]", "").replace(",", " ").split()
    else:
        return []
    out = []
    for v in vals:
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            continue
    return out


def load_json(path):
    data = json.load(open(path))
    nmrs, y, smiles = [], [], []
    for x in data:
        nmrs.append(parse_nmr(x.get("C_nmr", x.get("NMR"))))
        y.append(label_of(x))
        smiles.append(x.get("cano_smiles", x.get("canonical_smiles", "")))
    return nmrs, np.asarray(y, dtype=np.int8), smiles


def load_new_db(path):
    env = lmdb.open(path, readonly=True, lock=False, subdir=True)
    smiles, nmrs, labels = [], [], []
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
                base.log(f"  loaded {n:,} ({time.time()-t0:.0f}s)")
    env.close()
    base.log(f"new_db: loaded {n:,}, kept {len(smiles):,}, pos {int(np.sum(labels)):,}, {time.time()-t0:.1f}s")
    return smiles, nmrs, np.asarray(labels, dtype=np.int8)


def load_extra_lmdb(path):
    """Load an additional positive LMDB (e.g. new_data_1209) for training.

    Handles both `cano_smiles` and `canonical_smiles`; drops records with no
    C_nmr and exact duplicates of (smiles, C_nmr) within this file.
    """
    env = lmdb.open(path, readonly=True, lock=False, subdir=True)
    smiles, nmrs, labels = [], [], []
    seen = set()
    total = dup = empty = 0
    t0 = time.time()
    with env.begin() as txn:
        for _k, v in txn.cursor():
            total += 1
            o = pickle.loads(v)
            s = o.get("cano_smiles", o.get("canonical_smiles"))
            nmr = o.get("C_nmr")
            if not s or nmr is None or len(nmr) == 0:
                empty += 1
                continue
            # Some LMDBs store C_nmr as 1-D numpy arrays (e.g. float16);
            # CReSS's batching checks isinstance(nmr_list[0], list), so
            # normalize to plain Python floats here.
            try:
                nmr = [float(x) for x in nmr]
            except (TypeError, ValueError):
                empty += 1
                continue
            key = (s, tuple(nmr))
            if key in seen:
                dup += 1
                continue
            seen.add(key)
            smiles.append(s)
            nmrs.append(nmr)
            labels.append(label_of(o))
    env.close()
    stats = {
        "path": path, "records": total, "kept": len(smiles),
        "pos": int(np.sum(labels)), "empty_nmr": empty,
        "exact_duplicate_within_file": dup,
        "elapsed_s": round(time.time() - t0, 1),
    }
    base.log(f"extra {path}: loaded {total:,}, kept {len(smiles):,}, pos {stats['pos']:,}, "
             f"empty_nmr={empty}, exact_dup={dup}, {stats['elapsed_s']}s")
    return smiles, nmrs, np.asarray(labels, dtype=np.int8), stats


def load_extra_json(path):
    """Load an additional JSON training set (e.g. ex_train) for training.

    Handles `cano_smiles`/`canonical_smiles` and `C_nmr`/`NMR`, normalizes
    C_nmr to plain float lists, and drops exact (smiles, C_nmr) duplicates
    within this file.
    """
    data = json.load(open(path))
    smiles, nmrs, labels = [], [], []
    seen = set()
    total = dup = empty = 0
    t0 = time.time()
    for rec in data:
        total += 1
        s = rec.get("cano_smiles") or rec.get("canonical_smiles")
        nmr = rec.get("C_nmr", rec.get("NMR"))
        if not s or nmr is None or len(nmr) == 0:
            empty += 1
            continue
        try:
            nmr = [float(x) for x in nmr]
        except (TypeError, ValueError):
            empty += 1
            continue
        key = (s, tuple(nmr))
        if key in seen:
            dup += 1
            continue
        seen.add(key)
        smiles.append(s)
        nmrs.append(nmr)
        labels.append(label_of(rec))
    stats = {
        "path": path, "records": total, "kept": len(smiles),
        "pos": int(np.sum(labels)), "empty_nmr": empty,
        "exact_duplicate_within_file": dup,
        "elapsed_s": round(time.time() - t0, 1),
    }
    base.log(f"extra_json {path}: loaded {total:,}, kept {len(smiles):,}, "
             f"pos {stats['pos']:,}, empty_nmr={empty}, exact_dup={dup}, "
             f"{stats['elapsed_s']}s")
    return smiles, nmrs, np.asarray(labels, dtype=np.int8), stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_lmdb", default="data/new_dataset/train.lmdb")
    ap.add_argument("--extra_train_lmdb", nargs="*", default=[],
                    help="additional positive LMDB(s) appended to the training set")
    ap.add_argument("--extra_train_json", nargs="*", default=[],
                    help="additional JSON training set(s) appended to the training set")
    ap.add_argument("--valid_json", default="data/new_dataset/validation.json")
    ap.add_argument("--test_json", default="data/new_dataset/test.json")
    ap.add_argument("--external_json", default="data/new_dataset/external_test.json")
    ap.add_argument("--f_split_json", default="data/new_dataset/f_split/filtered_with_f_split_fixed_stereo.json")
    ap.add_argument("--f_without_json", default="data/new_dataset/f_without_split/filtered_without_f_split_fixed_stereo.json")
    ap.add_argument("--out_dir", default="output/xgb_cress_new_dataset")
    ap.add_argument("--seeds", default="0,2006")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    os.makedirs(args.out_dir, exist_ok=True)

    base.log("=" * 80)
    base.log(f"XGBoost-CReSS on new_dataset | seeds {seeds}")
    base.log("=" * 80)

    val_nmrs, y_val, val_smiles = load_json(args.valid_json)
    test_nmrs, y_test, test_smiles = load_json(args.test_json)
    ext_nmrs, y_ext, ext_smiles = load_json(args.external_json)
    fs_nmrs, y_fs, fs_smiles = load_json(args.f_split_json)
    fw_nmrs, y_fw, fw_smiles = load_json(args.f_without_json)
    base.log(f"valid {len(y_val):,} (pos {int(y_val.sum())}), test {len(y_test):,} (pos {int(y_test.sum())}), "
             f"external {len(y_ext):,} (pos {int(y_ext.sum())}), f_split {len(y_fs)}/{int(y_fs.sum())}, "
             f"f_without {len(y_fw)}/{int(y_fw.sum())}")

    smiles, nmrs, y = load_new_db(args.train_lmdb)
    blacklist = set(val_smiles) | set(test_smiles) | set(ext_smiles) | set(fs_smiles) | set(fw_smiles)
    keep = np.fromiter((s not in blacklist for s in smiles), dtype=bool, count=len(smiles))
    removed = int((~keep).sum())
    if removed:
        smiles = [s for s, k in zip(smiles, keep) if k]
        nmrs = [n for n, k in zip(nmrs, keep) if k]
        y = y[keep]
    base.log(f"removed {removed:,} records matching eval sets; train now {len(y):,} (pos {int(y.sum()):,})")
    n_base = len(y)

    extra_stats = []
    for extra_path in args.extra_train_lmdb:
        ex_smiles, ex_nmrs, ex_y, st = load_extra_lmdb(extra_path)
        keep_e = np.fromiter((s not in blacklist for s in ex_smiles), dtype=bool, count=len(ex_smiles))
        removed_e = int((~keep_e).sum())
        if removed_e:
            ex_smiles = [s for s, k in zip(ex_smiles, keep_e) if k]
            ex_nmrs = [n for n, k in zip(ex_nmrs, keep_e) if k]
            ex_y = ex_y[keep_e]
        st["removed_eval_overlaps"] = removed_e
        st["added"] = int(len(ex_y))
        st["added_pos"] = int(ex_y.sum())
        extra_stats.append(st)
        smiles += ex_smiles
        nmrs += ex_nmrs
        y = np.concatenate([y, ex_y])
        base.log(f"extra {extra_path}: removed {removed_e} eval overlaps; "
                 f"added {len(ex_y):,} (pos {int(ex_y.sum()):,}); train now {len(y):,}")

    for extra_path in args.extra_train_json:
        ex_smiles, ex_nmrs, ex_y, st = load_extra_json(extra_path)
        keep_e = np.fromiter((s not in blacklist for s in ex_smiles), dtype=bool, count=len(ex_smiles))
        removed_e = int((~keep_e).sum())
        if removed_e:
            ex_smiles = [s for s, k in zip(ex_smiles, keep_e) if k]
            ex_nmrs = [n for n, k in zip(ex_nmrs, keep_e) if k]
            ex_y = ex_y[keep_e]
        st["removed_eval_overlaps"] = removed_e
        st["added"] = int(len(ex_y))
        st["added_pos"] = int(ex_y.sum())
        extra_stats.append(st)
        smiles += ex_smiles
        nmrs += ex_nmrs
        y = np.concatenate([y, ex_y])
        base.log(f"extra_json {extra_path}: removed {removed_e} eval overlaps; "
                 f"added {len(ex_y):,} (pos {int(ex_y.sum()):,}); train now {len(y):,}")

    torch.set_num_threads(32)
    mi = base.ModelInference(config_path="models/2_5_w_model/8.json",
                             pretrain_model_path="models/2_5_w_model/8.pth", device="cpu")
    X_train = base.encode(mi, nmrs, tag="train")
    del nmrs
    X_val = base.encode(mi, val_nmrs, tag="valid")
    X_test = base.encode(mi, test_nmrs, tag="test")
    X_ext = base.encode(mi, ext_nmrs, tag="external")
    X_fs = base.encode(mi, fs_nmrs, tag="f_split")
    X_fw = base.encode(mi, fw_nmrs, tag="f_without")

    neg = int((y == 0).sum()); pos = int((y == 1).sum())
    spw = float(neg / max(1, pos))
    base.log(f"train shape {X_train.shape}, neg={neg:,}, pos={pos:,}, scale_pos_weight={spw:.2f}")

    summary = {
        "data": {"train": int(len(y)), "train_pos": pos, "train_neg": neg,
                 "train_base": n_base, "extra_train": extra_stats,
                 "removed_eval_overlaps": removed,
                 "valid": {"n": int(len(y_val)), "pos": int(y_val.sum())},
                 "test": {"n": int(len(y_test)), "pos": int(y_test.sum())},
                 "external": {"n": int(len(y_ext)), "pos": int(y_ext.sum())},
                 "f_split": {"n": int(len(y_fs)), "pos": int(y_fs.sum())},
                 "f_without": {"n": int(len(y_fw)), "pos": int(y_fw.sum())}},
        "config": {"seeds": seeds, "scale_pos_weight": spw, "cress": "models/2_5_w_model/8.pth",
                   "n_estimators": 2000, "max_depth": 5, "learning_rate": 0.1,
                   "subsample": 0.9, "colsample_bytree": 0.9, "gamma": 0.1,
                   "reg_alpha": 0.5, "reg_lambda": 0.5,
                   "eval_metric": ["logloss", "auc", "aucpr"], "early_stopping_rounds": 100},
        "seeds": {},
    }
    ext_ps, test_ps = [], []

    for seed in seeds:
        base.log("-" * 80)
        base.log(f"seed {seed}: training")
        clf = xgb.XGBClassifier(
            n_estimators=2000, max_depth=5, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, gamma=0.1,
            reg_alpha=0.5, reg_lambda=0.5, scale_pos_weight=spw,
            objective="binary:logistic", eval_metric=["logloss", "auc", "aucpr"],
            tree_method="hist", device=base.XGB_DEVICE, n_jobs=32,
            random_state=seed, early_stopping_rounds=100,
        )
        t0 = time.time()
        clf.fit(X_train, y, eval_set=[(X_val, y_val)], verbose=100)
        base.log(f"seed {seed}: fit {time.time()-t0:.1f}s best_iter={clf.best_iteration}")

        n_trees = int(clf.best_iteration) + 1 if clf.best_iteration is not None else clf.get_booster().num_boosted_rounds()
        bst = clf.get_booster()[:n_trees]
        model_json = os.path.join(args.out_dir, f"model_xgb_cress_new_dataset_seed{seed}.json")
        bst.save_model(model_json)
        clf_cpu = xgb.XGBClassifier()
        clf_cpu.load_model(model_json)
        if not hasattr(clf_cpu, "classes_") or clf_cpu.classes_ is None:
            clf_cpu.classes_ = np.array([0, 1])
        if not hasattr(clf_cpu, "n_features_in_"):
            clf_cpu.n_features_in_ = X_train.shape[1]

        p_val = clf_cpu.predict_proba(X_val)[:, 1]
        p_test = clf_cpu.predict_proba(X_test)[:, 1]
        p_ext = clf_cpu.predict_proba(X_ext)[:, 1]
        p_fs = clf_cpu.predict_proba(X_fs)[:, 1]
        p_fw = clf_cpu.predict_proba(X_fw)[:, 1]
        ext_ps.append(p_ext); test_ps.append(p_test)
        th = base.choose_threshold(y_val, p_val)

        res = {
            "best_iteration": int(clf.best_iteration), "n_trees": int(n_trees),
            "val_auprc": float(base.average_precision_score(y_val, p_val)),
            "val_threshold_f1max": th,
            "test@0.5": base.metrics(y_test, p_test, 0.5),
            "test@val_th": base.metrics(y_test, p_test, th),
            "external@0.5": base.metrics(y_ext, p_ext, 0.5),
            "external@val_th": base.metrics(y_ext, p_ext, th),
            "f_split@0.5": base.metrics(y_fs, p_fs, 0.5),
            "f_split@val_th": base.metrics(y_fs, p_fs, th),
            "f_without@0.5": base.metrics(y_fw, p_fw, 0.5),
            "f_without@val_th": base.metrics(y_fw, p_fw, th),
        }
        summary["seeds"][str(seed)] = res
        model_path = os.path.join(args.out_dir, f"model_xgb_cress_new_dataset_seed{seed}.pkl")
        joblib.dump(clf_cpu, model_path)
        base.log(f"seed {seed}: val AUPRC={res['val_auprc']:.4f} th={th:.4f} | "
                 f"test AUPRC={res['test@0.5']['auprc']:.4f} F1@0.5={res['test@0.5']['f1']:.4f} | "
                 f"ext AUPRC={res['external@0.5']['auprc']:.4f} F1@0.5={res['external@0.5']['f1']:.4f} "
                 f"F1@th={res['external@val_th']['f1']:.4f}")
        base.log(f"seed {seed}: saved {model_path}")

    if ext_ps:
        p_ext_ens = np.mean(ext_ps, axis=0)
        p_test_ens = np.mean(test_ps, axis=0)
        th_ens = float(np.mean([summary["seeds"][str(s)]["val_threshold_f1max"] for s in seeds]))
        summary["ensemble"] = {
            "threshold": th_ens,
            "test@0.5": base.metrics(y_test, p_test_ens, 0.5),
            "test@val_th_mean": base.metrics(y_test, p_test_ens, th_ens),
            "external@0.5": base.metrics(y_ext, p_ext_ens, 0.5),
            "external@val_th_mean": base.metrics(y_ext, p_ext_ens, th_ens),
        }
        base.log(f"ensemble: test AUPRC={summary['ensemble']['test@0.5']['auprc']:.4f} "
                 f"ext AUPRC={summary['ensemble']['external@0.5']['auprc']:.4f}")

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    extra_added = sum(e["added"] for e in extra_stats)
    lines = [
        "# XGBoost-CReSS on new_dataset (new_db train + extra positives; clean val/test)",
        "",
        f"train {len(y):,} (pos {pos:,}, neg {neg:,}) | valid {len(y_val):,} (pos {int(y_val.sum())}) "
        f"| test {len(y_test):,} (pos {int(y_test.sum())}) | external {len(y_ext):,} (pos {int(y_ext.sum())})",
        f"scale_pos_weight={spw:.2f}; removed eval overlaps={removed:,}; extra added={extra_added:,}",
        "",
        "| seed | best_iter | val AUPRC | test AUPRC | test F1@0.5 | test F1@th | ext AUPRC | ext P@0.5 | ext R@0.5 | ext F1@0.5 | ext F1@th | ext topK P/R |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for seed in seeds:
        r = summary["seeds"][str(seed)]
        t05, tth = r["test@0.5"], r["test@val_th"]
        e05, eth = r["external@0.5"], r["external@val_th"]
        lines.append(
            f"| {seed} | {r['best_iteration']} | {r['val_auprc']:.4f} | {t05['auprc']:.4f} | "
            f"{t05['f1']:.4f} | {tth['f1']:.4f} | {e05['auprc']:.4f} | {e05['precision']:.4f} | "
            f"{e05['recall']:.4f} | {e05['f1']:.4f} | {eth['f1']:.4f} | "
            f"{e05['topK'].get('precision',0):.4f}/{e05['topK'].get('recall',0):.4f} |"
        )
    if "ensemble" in summary:
        e = summary["ensemble"]
        lines.append(
            f"| ensemble | - | - | {e['test@0.5']['auprc']:.4f} | {e['test@0.5']['f1']:.4f} | "
            f"{e['test@val_th_mean']['f1']:.4f} | {e['external@0.5']['auprc']:.4f} | "
            f"{e['external@0.5']['precision']:.4f} | {e['external@0.5']['recall']:.4f} | "
            f"{e['external@0.5']['f1']:.4f} | {e['external@val_th_mean']['f1']:.4f} | "
            f"{e['external@0.5']['topK'].get('precision',0):.4f}/{e['external@0.5']['topK'].get('recall',0):.4f} |"
        )
    with open(os.path.join(args.out_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    base.log("done")
    base.log("\n".join(lines))


if __name__ == "__main__":
    main()
