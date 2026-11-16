#!/usr/bin/env python
"""Rewrite the `source` field of selected LMDB records into a new LMDB.

Examples:

  # extract napacd records and relabel them as NMRium
  python scripts/build_lmdb/relabel_source.py \
      --src data/synthetic/synth_predicted.lmdb \
      --from napacd --to NMRium --only-matches \
      --out data/nmrium/napm_from_napacd.lmdb

  # full copy with the source field rewritten
  python scripts/build_lmdb/relabel_source.py \
      --src data/synthetic/synth_predicted.lmdb \
      --from napacd --to NMRium \
      --out data/synthetic/synth_predicted_relabeled.lmdb

`--only-matches` writes only the rewritten records and renumbers the keys from 0;
otherwise every record is copied and the original keys are kept.
"""
import argparse
import os
import pickle
import sys
import time

import lmdb
import numpy as _np
import numpy.core as _nc

sys.modules.setdefault("numpy._core", _nc)
sys.modules.setdefault("numpy._core.numeric", _nc.numeric)
sys.modules.setdefault("numpy._core.multiarray", _nc.multiarray)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--from", dest="from_sources", required=True,
                    help="comma-separated source values to rewrite")
    ap.add_argument("--to", dest="to_source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only-matches", action="store_true")
    ap.add_argument("--map_gb", type=int, default=64)
    args = ap.parse_args()

    wanted = {s.strip() for s in args.from_sources.split(",") if s.strip()}
    if not wanted:
        raise SystemExit("--from is empty")
    if os.path.abspath(args.src) == os.path.abspath(args.out):
        raise SystemExit("--src and --out must differ")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    src = lmdb.open(args.src, readonly=True, lock=False, subdir=True,
                    readahead=False, meminit=False)
    out = lmdb.open(args.out, map_size=args.map_gb * (1024 ** 3), subdir=True)

    counts = {}
    changed = 0
    written = 0
    t0 = time.time()

    with src.begin() as txn:
        total = txn.stat()["entries"]
        wtxn = out.begin(write=True)
        for k, v in txn.cursor():
            record = pickle.loads(v)
            source = record.get("source", "") if isinstance(record, dict) else ""
            counts[source] = counts.get(source, 0) + 1
            match = source in wanted
            if match:
                record["source"] = args.to_source
                v = pickle.dumps(record, protocol=4)
            if args.only_matches and not match:
                continue
            out_key = str(written).encode() if args.only_matches else k
            wtxn.put(out_key, v)
            written += 1
            if match:
                changed += 1
            if written % 200_000 == 0:
                wtxn.commit()
                wtxn = out.begin(write=True)
                print(f"  {written} written ({written/max(time.time()-t0,1):.0f} rec/s)",
                      flush=True)
        wtxn.commit()

    src.close()
    out.close()
    check = lmdb.open(args.out, readonly=True, lock=False, subdir=True)
    got = check.begin().stat()["entries"]
    check.close()
    print(f"scanned {total}, wrote {got}, rewrote {changed} to {args.to_source}")
    for source, count in sorted(counts.items(), key=lambda x: -x[1]):
        mark = "  <-- rewritten" if source in wanted else ""
        print(f"  {source or '<none>'}: {count}{mark}")


if __name__ == "__main__":
    main()
