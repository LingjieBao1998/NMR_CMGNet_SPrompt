#!/usr/bin/env python
"""Split the 3.20 M-record `new_db.lmdb` into the two semantic corpora used here.

Category mapping (by the `source` field stored in each record):
  experimental : NMR BANK / c13_naproc / bmrb / Spektraris_NMR / GlycoNMR / hmdb   (~50 k)
  synthetic    : multi / napacd / acd_lotusv9 / pnmrnp2                            (~1.49 M)
  excluded     : NMRium (15,495,052-entry upload corpus, provenance unclear)      (~1.66 M)

Values are copied as raw pickle bytes, so every field (C_nmr, ring_system,
ecfp, maccs, label, ...) is preserved byte-for-byte.  Keys are renumbered
0..N-1 in source order.

Usage:
  python scripts/build_lmdb/split_newdb_by_source.py \
      --src data/_source/new_db.lmdb \
      --exp_out data/experimental/exp_lit.lmdb \
      --synth_out data/synthetic/synth_predicted.lmdb
"""
import argparse
import os
import pickle
import shutil
import sys
import time

import lmdb
import numpy as _np
import numpy.core as _nc

# pickles written with numpy 2.x; the runtime env may still be numpy 1.x
sys.modules.setdefault("numpy._core", _nc)
sys.modules.setdefault("numpy._core.numeric", _nc.numeric)
sys.modules.setdefault("numpy._core.multiarray", _nc.multiarray)

EXPERIMENTAL = {"NMR BANK", "c13_naproc", "bmrb", "Spektraris_NMR", "GlycoNMR", "hmdb"}
SYNTHETIC = {"multi", "napacd", "acd_lotusv9", "pnmrnp2"}
EXCLUDED = {"NMRium"}


def open_out(path, map_gb, overwrite):
    if os.path.exists(os.path.join(path, "data.mdb")):
        if not overwrite:
            raise SystemExit(
                f"{path} already exists; pass --overwrite to replace it"
            )
        shutil.rmtree(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return lmdb.open(path, map_size=map_gb * (1024 ** 3), subdir=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/_source/new_db.lmdb")
    ap.add_argument("--exp_out", default="data/experimental/exp_lit.lmdb")
    ap.add_argument("--synth_out", default="data/synthetic/synth_predicted.lmdb")
    ap.add_argument("--exp_map_gb", type=int, default=4)
    ap.add_argument("--synth_map_gb", type=int, default=64)
    ap.add_argument("--overwrite", action="store_true",
                    help="replace existing output LMDBs")
    args = ap.parse_args()

    src = lmdb.open(args.src, readonly=True, lock=False, subdir=True,
                    readahead=False, meminit=False)
    exp_env = open_out(args.exp_out, args.exp_map_gb, args.overwrite)
    syn_env = open_out(args.synth_out, args.synth_map_gb, args.overwrite)

    counts, kept_exp, kept_syn, skipped = {}, 0, 0, 0
    t0 = time.time()

    with src.begin() as txn:
        total = txn.stat()["entries"]
        print(f"source entries: {total}", flush=True)
        exp_txn = exp_env.begin(write=True)
        syn_txn = syn_env.begin(write=True)
        for k, v in txn.cursor():
            d = pickle.loads(v)
            s = d.get("source", "<none>") if isinstance(d, dict) else "<notdict>"
            counts[s] = counts.get(s, 0) + 1
            if s in EXPERIMENTAL:
                exp_txn.put(str(kept_exp).encode(), v)
                kept_exp += 1
            elif s in SYNTHETIC:
                syn_txn.put(str(kept_syn).encode(), v)
                kept_syn += 1
            else:
                skipped += 1
            done = sum(counts.values())
            if done % 200_000 == 0:
                exp_txn.commit(); exp_txn = exp_env.begin(write=True)
                syn_txn.commit(); syn_txn = syn_env.begin(write=True)
                rate = done / (time.time() - t0)
                print(f"[{done}/{total}] exp={kept_exp} synth={kept_syn} "
                      f"{rate:.0f} rec/s ETA {(total-done)/max(rate,1)/60:.1f} min", flush=True)
        exp_txn.commit()
        syn_txn.commit()

    print("\n===== source counts =====", flush=True)
    for s, c in sorted(counts.items(), key=lambda x: -x[1]):
        tag = "experimental" if s in EXPERIMENTAL else ("synthetic" if s in SYNTHETIC else "excluded")
        print(f"  {s}: {c} ({c/total:.2%})  [{tag}]", flush=True)
    print(f"\nexperimental={kept_exp}  synthetic={kept_syn}  excluded={skipped}", flush=True)

    for env, path, expect in ((exp_env, args.exp_out, kept_exp),
                              (syn_env, args.synth_out, kept_syn)):
        got = env.begin().stat()["entries"]
        print(f"verify {path}: {got} (expect {expect})", flush=True)
        if got != expect:
            raise SystemExit(f"entry count mismatch for {path}")

    src.close(); exp_env.close(); syn_env.close()
    print(f"done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
