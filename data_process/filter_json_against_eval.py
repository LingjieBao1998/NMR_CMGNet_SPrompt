#!/usr/bin/env python3
"""Filter a JSON training set against evaluation JSON sets.

Removes records whose canonical SMILES exactly matches, or whose stereo-flat
2D structure matches, any molecule in the given eval JSONs. The source file is
left untouched; a filtered copy and a JSON report are written.
"""
import argparse
import json
import time

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


def flat(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--eval_json", nargs="+", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

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

    data = json.load(open(args.src))
    t0 = time.time()
    kept, removed_exact, removed_flat, missing_smiles = [], 0, 0, 0
    examples = []
    for rec in data:
        s = rec.get("cano_smiles") or rec.get("canonical_smiles")
        if not s:
            missing_smiles += 1
            kept.append(rec)
            continue
        if s in exact:
            removed_exact += 1
            if len(examples) < 20:
                examples.append({"cano_smiles": s, "how": "exact"})
            continue
        f = flat(s)
        if f and f in flat_set:
            removed_flat += 1
            if len(examples) < 20:
                examples.append({"cano_smiles": s, "flat": f, "how": "stereo-flat"})
            continue
        kept.append(rec)

    with open(args.dst, "w") as fh:
        json.dump(kept, fh, ensure_ascii=False)

    removed = removed_exact + removed_flat
    stats = {
        "source_json": args.src,
        "output_json": args.dst,
        "eval_json": per_eval,
        "matching": "exact canonical + stereo-flat 2D structure",
        "total": len(data),
        "kept": len(kept),
        "removed": removed,
        "removed_exact": removed_exact,
        "removed_stereo_flat": removed_flat,
        "missing_smiles_kept": missing_smiles,
        "examples": examples,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    with open(args.report, "w") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)

    print(f"DONE total={len(data):,} kept={len(kept):,} removed={removed:,} "
          f"(exact={removed_exact}, flat={removed_flat}) missing_smiles={missing_smiles} "
          f"elapsed={time.time()-t0:.1f}s", flush=True)
    print(f"report: {args.report}")


if __name__ == "__main__":
    main()
