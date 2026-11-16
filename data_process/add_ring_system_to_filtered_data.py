#!/usr/bin/env python3
"""Precompute `ring_system` for the filtered new positives and ex_train.

Why:
  * CMGNet's LMDB dataset assumes keys are exactly 0..entries-1. The filtered
    new-positive LMDB preserved original keys and has 90 gaps, so it must be
    re-keyed contiguously for `main_cmgnet.py`.
  * With `--use_ring_system`, CMGNet otherwise computes ring systems on the
    fly for every sample on every epoch. Precomputing once avoids that cost.

Both outputs use the same RingSystemFinder logic as main_cmgnet.py
(get_ring_system with proba=0, i.e. deterministic SMILES).
"""
import argparse
import json
import os
import pickle
import sys
import time
from multiprocessing import Pool

import lmdb
from rdkit import Chem
from rdkit import RDLogger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from new_RingSystemFinder import RingSystemFinder  # noqa: E402

RDLogger.DisableLog("rdApp.*")

_FINDER = None


def _init_worker():
    global _FINDER
    _FINDER = RingSystemFinder()


def remove_specific_terminal_carbons(mol):
    rwmol = Chem.RWMol(mol)
    ring_info = mol.GetRingInfo()
    ring_atoms = set()
    for ring in ring_info.AtomRings():
        ring_atoms.update(ring)
    atoms_to_remove = []
    for atom in rwmol.GetAtoms():
        if atom.GetDegree() == 1 and atom.GetAtomicNum() == 6:
            neighbor = atom.GetNeighbors()[0]
            bond = atom.GetBonds()[0]
            if neighbor.GetIdx() in ring_atoms and bond.GetBondType() != Chem.BondType.SINGLE:
                atoms_to_remove.append(atom.GetIdx())
    for atom_idx in sorted(atoms_to_remove, reverse=True):
        rwmol.RemoveAtom(atom_idx)
    return rwmol.GetMol()


def ring_of(smiles):
    if not smiles:
        return []
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    try:
        rings = _FINDER.find_ring_systems(mol, as_mols=True)
        return [Chem.MolToSmiles(remove_specific_terminal_carbons(r)) for r in rings]
    except Exception:
        return []


def load_lmdb(path):
    env = lmdb.open(path, readonly=True, lock=False, subdir=True)
    records = []
    with env.begin() as txn:
        for key, value in txn.cursor():
            obj = pickle.loads(value)
            smiles = obj.get("cano_smiles") or obj.get("canonical_smiles")
            records.append((key, value, smiles))
    env.close()
    return records


def write_lmdb(src_path, dst_path, records, rings, map_size_gb=8):
    if os.path.exists(dst_path):
        raise SystemExit(f"destination exists: {dst_path}")
    env = lmdb.open(dst_path, map_size=map_size_gb * 1024**3, subdir=True)
    txn = env.begin(write=True)
    t0 = time.time()
    for i, ((_old_key, value, _smiles), rs) in enumerate(zip(records, rings)):
        obj = pickle.loads(value)
        obj["ring_system"] = rs
        txn.put(str(i).encode("ascii"), pickle.dumps(obj))
        if (i + 1) % 10000 == 0:
            txn.commit()
            txn = env.begin(write=True)
            print(f"  wrote {i+1:,}/{len(records):,} ({time.time()-t0:.0f}s)", flush=True)
    txn.commit()
    env.sync(True)
    env.close()
    print(f"LMDB written: {dst_path} entries={len(records):,} ({time.time()-t0:.0f}s)", flush=True)


def write_json(src_path, dst_path, records, rings):
    out = []
    for rec, rs in zip(records, rings):
        rec = dict(rec)
        rec["ring_system"] = rs
        out.append(rec)
    with open(dst_path, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"JSON written: {dst_path} entries={len(out):,}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb_src", default=None)
    ap.add_argument("--lmdb_dst", default=None)
    ap.add_argument("--json_src", default=None)
    ap.add_argument("--json_dst", default=None)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--chunksize", type=int, default=64)
    args = ap.parse_args()

    if args.lmdb_src:
        print(f"loading {args.lmdb_src} ...", flush=True)
        records = load_lmdb(args.lmdb_src)
        smiles = [s for _k, _v, s in records]
        t0 = time.time()
        with Pool(args.workers, initializer=_init_worker) as pool:
            rings = list(pool.imap(ring_of, smiles, chunksize=args.chunksize))
        print(f"computed ring_system for {len(rings):,} LMDB records in {time.time()-t0:.0f}s", flush=True)
        write_lmdb(args.lmdb_src, args.lmdb_dst, records, rings)

    if args.json_src:
        print(f"loading {args.json_src} ...", flush=True)
        records = json.load(open(args.json_src))
        smiles = [r.get("cano_smiles") or r.get("canonical_smiles") for r in records]
        t0 = time.time()
        with Pool(args.workers, initializer=_init_worker) as pool:
            rings = list(pool.imap(ring_of, smiles, chunksize=args.chunksize))
        print(f"computed ring_system for {len(rings):,} JSON records in {time.time()-t0:.0f}s", flush=True)
        write_json(args.json_src, args.json_dst, records, rings)


if __name__ == "__main__":
    main()
