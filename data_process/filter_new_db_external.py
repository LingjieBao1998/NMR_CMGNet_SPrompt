#!/usr/bin/env python3
"""Filter new_db.lmdb.

Removes:
  1. records whose canonical SMILES appear in the 4,502-record external test set
  2. records whose C_nmr list is empty / missing / None

Copies raw pickle bytes (no re-pickling), preserves keys, and reports the
source-field distribution of the kept records.
"""
import collections
import json
import os
import time

import lmdb

BASE = "data/NMR_other_data_1020_add_ring_system_1029"
SRC = os.path.join(BASE, "new_db.lmdb")
DST = os.path.join(BASE, "new_db.filtered.tmp.lmdb")
EXT = "data/nmrmind_cmgnet_1004/external_test.json"
STATS = os.path.join("data_process", "filter_new_db_external_stats.json")

CANON_MARKER = b"\x8c\x10canonical_smiles\x94"
CNMR_MARKER = b"\x8c\x05C_nmr\x94"
SOURCE_MARKER = b"\x8c\x06source\x94"


def _read_string(value: bytes, marker: bytes):
    i = value.find(marker)
    if i < 0:
        return None
    j = i + len(marker)
    tag = value[j]
    if tag == 0x8C:  # SHORT_BINUNICODE
        n = value[j + 1]
        return value[j + 2 : j + 2 + n].decode()
    if tag == 0x8D:  # BINUNICODE8
        n = int.from_bytes(value[j + 1 : j + 9], "little")
        return value[j + 9 : j + 9 + n].decode()
    if tag == 0x58:  # BINUNICODE
        n = int.from_bytes(value[j + 1 : j + 5], "little")
        return value[j + 5 : j + 5 + n].decode()
    return None


def extract_smiles(value: bytes):
    return _read_string(value, CANON_MARKER)


def extract_source(value: bytes):
    return _read_string(value, SOURCE_MARKER)


def cnmr_is_empty(value: bytes):
    """True if C_nmr is [], None, or missing. Empty/None is followed by the next dict key."""
    i = value.find(CNMR_MARKER)
    if i < 0:
        return True
    j = i + len(CNMR_MARKER)
    # next dict key marker (SHORT_BINUNICODE/BINUNICODE8/BINUNICODE) or None
    return value[j] in (0x8C, 0x8D, 0x58, 0x4E)


def main():
    if os.path.exists(DST):
        raise SystemExit(f"destination already exists: {DST}")

    ext = json.load(open(EXT))
    blacklist = {r["cano_smiles"] for r in ext}
    print(f"external records={len(ext)} unique smiles={len(blacklist)}", flush=True)

    src = lmdb.open(SRC, readonly=True, lock=False, subdir=True)
    dst = lmdb.open(DST, map_size=70 * 1024**3, subdir=True, max_dbs=1)

    t0 = time.time()
    total = 0
    kept = 0
    removed_ext = 0
    removed_empty = 0
    removed_ext_examples = []
    removed_empty_examples = []
    removed_empty_by_source = collections.Counter()
    kept_by_source = collections.Counter()
    missing_source = 0

    txn = dst.begin(write=True)
    with src.begin() as stxn:
        for key, value in stxn.cursor():
            total += 1
            smiles = extract_smiles(value)
            source = extract_source(value)

            if smiles in blacklist:
                removed_ext += 1
                if len(removed_ext_examples) < 30:
                    removed_ext_examples.append(
                        {"key": key.decode(), "smiles": smiles, "source": source}
                    )
                continue

            if cnmr_is_empty(value):
                removed_empty += 1
                removed_empty_by_source[source or "<missing>"] += 1
                if len(removed_empty_examples) < 30:
                    removed_empty_examples.append(
                        {"key": key.decode(), "smiles": smiles, "source": source}
                    )
                continue

            txn.put(key, value)
            kept += 1
            if source is None:
                missing_source += 1
                kept_by_source["<missing>"] += 1
            else:
                kept_by_source[source] += 1

            if kept % 100000 == 0:
                txn.commit()
                txn = dst.begin(write=True)
                print(
                    f"  scanned={total:,} kept={kept:,} "
                    f"removed_ext={removed_ext} removed_empty_nmr={removed_empty} "
                    f"elapsed={time.time() - t0:.0f}s",
                    flush=True,
                )

    txn.commit()
    dst.sync(True)
    dst.close()
    src.close()

    stats = {
        "source_file": SRC,
        "output_file": DST,
        "total": total,
        "kept": kept,
        "removed_external_overlap": removed_ext,
        "removed_empty_nmr": removed_empty,
        "removed_empty_nmr_by_source": dict(removed_empty_by_source),
        "kept_by_source": dict(kept_by_source.most_common()),
        "kept_missing_source": missing_source,
        "removed_external_examples": removed_ext_examples,
        "removed_empty_nmr_examples": removed_empty_examples,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    with open(STATS, "w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(
        f"DONE total={total:,} kept={kept:,} removed_external={removed_ext} "
        f"removed_empty_nmr={removed_empty} missing_source={missing_source} "
        f"elapsed={time.time() - t0:.1f}s",
        flush=True,
    )
    print("kept source distribution:", flush=True)
    for s, c in kept_by_source.most_common():
        print(f"  {s}: {c:,}", flush=True)
    print(f"stats written to {STATS}", flush=True)

    expected = total - removed_ext - removed_empty
    if kept != expected or removed_ext != 18:
        raise SystemExit(
            f"UNEXPECTED COUNTS: total={total} kept={kept} "
            f"removed_ext={removed_ext} removed_empty={removed_empty} expected={expected}; "
            "do NOT delete source until investigated"
        )


if __name__ == "__main__":
    main()
