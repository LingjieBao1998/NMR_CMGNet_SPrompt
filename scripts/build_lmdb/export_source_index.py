#!/usr/bin/env python
"""Export a redistribution-safe index of an LMDB corpus.

Only identifiers and provenance are written: key, source, canonical SMILES and
the binary label.  Spectra (`C_nmr`), fingerprints and ring systems are
deliberately omitted, so the output can be published for corpora whose original
licence does not allow redistribution of the measurements.

Usage:
  python scripts/build_lmdb/export_source_index.py \
      --src data/synthetic/synth_predicted.lmdb \
      --out data/_index/synth_predicted_index.csv
"""
import argparse
import csv
import os
import pickle
import sys

import lmdb
import numpy as _np
import numpy.core as _nc

# pickles written with numpy 2.x; the runtime env may still be numpy 1.x
sys.modules.setdefault("numpy._core", _nc)
sys.modules.setdefault("numpy._core.numeric", _nc.numeric)
sys.modules.setdefault("numpy._core.multiarray", _nc.multiarray)


def label_of(record):
    value = record.get("label", 0)
    try:
        return int(_np.asarray(value).ravel()[0])
    except Exception:
        return int(value)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="input LMDB directory")
    ap.add_argument("--out", required=True, help="output .csv / .tsv path")
    ap.add_argument("--delimiter", default=None,
                    help="default: comma for .csv, tab for .tsv")
    args = ap.parse_args()

    delimiter = args.delimiter
    if delimiter is None:
        delimiter = "\t" if args.out.endswith(".tsv") else ","

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    env = lmdb.open(args.src, readonly=True, lock=False, subdir=True,
                    readahead=False, meminit=False)
    counts = {}

    with env.begin() as txn, open(args.out, "w", newline="") as fh:
        writer = csv.writer(fh, delimiter=delimiter)
        writer.writerow(["key", "source", "canonical_smiles", "label"])
        for k, v in txn.cursor():
            record = pickle.loads(v)
            source = record.get("source", "")
            counts[source] = counts.get(source, 0) + 1
            writer.writerow([
                k.decode("utf-8", "replace"),
                source,
                record.get("canonical_smiles", record.get("cano_smiles", "")),
                label_of(record),
            ])
    env.close()

    print(f"wrote {sum(counts.values())} rows to {args.out}")
    for source, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {source or '<none>'}: {count}")


if __name__ == "__main__":
    main()
