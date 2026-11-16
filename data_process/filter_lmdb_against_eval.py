#!/usr/bin/env python3
"""Filter a training LMDB against evaluation JSON sets.

Removes training records whose canonical SMILES exactly matches, or whose
stereo-flat 2D structure matches, any molecule in the given eval JSONs.
The source LMDB is left untouched; kept records are copied with their
original keys and raw pickle bytes (no re-pickling).
"""
import argparse
import json
import os
import pickle
import time

import lmdb
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


def flat(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def extract_smiles(value):
    try:
        obj = pickle.loads(value)
        if isinstance(obj, dict):
            s = obj.get("cano_smiles") or obj.get("canonical_smiles")
            if isinstance(s, str):
                return s
    except Exception:
        pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--eval_json", nargs="+", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--map_size_gb", type=int, default=10)
    args = ap.parse_args()

    if os.path.exists(args.dst):
        raise SystemExit(f"destination already exists: {args.dst}")

    exact, flat_set, per_eval = set(), set(), {}
    for path in args.eval_json:
        data = json.load(open(path))
        es, fs = set(), set()
        for rec in data:
            s = rec.get("cano_smiles") or rec.get("canonical_smiles")
            if not s:
                continue
            es.add(s)
            f = flat(s)
            if f:
                fs.add(f)
        exact |= es
        flat_set |= fs
        per_eval[path] = {"n": len(data), "unique_smiles": len(es), "unique_flat": len(fs)}
        print(f"eval {path}: n={len(data):,} unique_smiles={len(es):,} unique_flat={len(fs):,}", flush=True)
    print(f"blacklist: exact={len(exact):,} flat={len(flat_set):,}", flush=True)

    src = lmdb.open(args.src, readonly=True, lock=False, subdir=True)
    dst = lmdb.open(args.dst, map_size=args.map_size_gb * 1024**3, subdir=True)

    t0 = time.time()
    total = kept = removed_exact = removed_flat = missing_smiles = 0
    examples = []
    txn = dst.begin(write=True)
    with src.begin() as stxn:
        for key, value in stxn.cursor():
            total += 1
            s = extract_smiles(value)
            if not s:
                missing_smiles += 1
                txn.put(key, value)
                kept += 1
                continue
            if s in exact:
                removed_exact += 1
                if len(examples) < 20:
                    examples.append({"key": key.decode(), "smiles": s, "how": "exact"})
                continue
            f = flat(s)
            if f and f in flat_set:
                removed_flat += 1
                if len(examples) < 20:
                    examples.append({"key": key.decode(), "smiles": s, "flat": f, "how": "stereo-flat"})
                continue
            txn.put(key, value)
            kept += 1
            if kept % 20000 == 0:
                txn.commit()
                txn = dst.begin(write=True)
                print(f"  scanned={total:,} kept={kept:,} removed_exact={removed_exact} "
                      f"removed_flat={removed_flat} ({time.time()-t0:.0f}s)", flush=True)

    txn.commit()
    dst.sync(True)
    dst.close()
    src.close()

    removed = removed_exact + removed_flat
    stats = {
        "source_lmdb": args.src,
        "output_lmdb": args.dst,
        "eval_json": per_eval,
        "matching": "exact canonical + stereo-flat 2D structure",
        "total": total,
        "kept": kept,
        "removed": removed,
        "removed_exact": removed_exact,
        "removed_stereo_flat": removed_flat,
        "missing_smiles_kept": missing_smiles,
        "examples": examples,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    with open(args.report, "w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"DONE total={total:,} kept={kept:,} removed={removed:,} "
          f"(exact={removed_exact}, flat={removed_flat}) missing_smiles={missing_smiles} "
          f"elapsed={time.time()-t0:.1f}s", flush=True)
    print(f"report: {args.report}")
    if kept + removed != total:
        raise SystemExit("UNEXPECTED COUNTS")


if __name__ == "__main__":
    main()
