#!/usr/bin/env python
"""Copy a subset of an LMDB corpus selected by its `source` field.

Values are copied as raw pickle bytes, so spectra and features are preserved
byte-for-byte and keys are renumbered 0..N-1.  Useful for building a
redistribution-safe subset (for example the CC BY 4.0 ACD/Lotus records).

Usage:
  # count only
  python scripts/build_lmdb/filter_lmdb_by_source.py \
      --src data/synthetic/synth_predicted.lmdb --list

  # write a subset
  python scripts/build_lmdb/filter_lmdb_by_source.py \
      --src data/synthetic/synth_predicted.lmdb \
      --sources acd_lotusv9 \
      --out data/synthetic/synth_acd_lotusv9.lmdb
"""
import argparse
import os
import pickle
import sys
import time

import lmdb
import numpy as _np
import numpy.core as _nc

# pickles written with numpy 2.x; the runtime env may still be numpy 1.x
sys.modules.setdefault("numpy._core", _nc)
sys.modules.setdefault("numpy._core.numeric", _nc.numeric)
sys.modules.setdefault("numpy._core.multiarray", _nc.multiarray)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--sources", default="", help="comma-separated source names")
    ap.add_argument("--out", default="", help="output LMDB (omit for count only)")
    ap.add_argument("--list", action="store_true", help="print per-source counts")
    ap.add_argument("--map_gb", type=int, default=64)
    args = ap.parse_args()

    wanted = {s.strip() for s in args.sources.split(",") if s.strip()}
    if not wanted and not args.list:
        raise SystemExit("give --sources, or --list to print counts")

    src = lmdb.open(args.src, readonly=True, lock=False, subdir=True,
                    readahead=False, meminit=False)
    out_env = None
    if args.out:
        if not wanted:
            raise SystemExit("--out requires --sources")
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        out_env = lmdb.open(args.out, map_size=args.map_gb * (1024 ** 3),
                            subdir=True)

    counts = {}
    kept = 0
    t0 = time.time()

    with src.begin() as txn:
        total = txn.stat()["entries"]
        out_txn = out_env.begin(write=True) if out_env else None
        for k, v in txn.cursor():
            record = pickle.loads(v)
            source = record.get("source", "<none>") if isinstance(record, dict) else "<notdict>"
            counts[source] = counts.get(source, 0) + 1
            if out_txn is not None and source in wanted:
                out_txn.put(str(kept).encode(), v)
                kept += 1
                if kept % 200_000 == 0:
                    out_txn.commit()
                    out_txn = out_env.begin(write=True)
                    rate = kept / max(time.time() - t0, 1)
                    print(f"  {kept} kept ({rate:.0f} rec/s)", flush=True)
        if out_txn is not None:
            out_txn.commit()

    src.close()
    if out_env:
        got = out_env.begin().stat()["entries"]
        out_env.close()
        print(f"wrote {got} records to {args.out}")
        if got != kept:
            raise SystemExit(f"entry count mismatch: wrote {got}, expected {kept}")

    print(f"source records: {total}")
    for source, count in sorted(counts.items(), key=lambda x: -x[1]):
        mark = "  <-- kept" if source in wanted else ""
        print(f"  {source or '<none>'}: {count}{mark}")

    if wanted:
        missing = wanted - set(counts)
        if missing:
            print(f"warning: sources not found: {', '.join(sorted(missing))}")


if __name__ == "__main__":
    main()
