#!/usr/bin/env python
"""Build a featurised LMDB from a JSON split (validation / external test / ex_train).

Every record gets the same schema as the legacy `new_db.lmdb` so that
`main_cmgnet.py` and the XGBoost-CReSS trainer can consume it unchanged:

  key (int) | canonical_smiles | C_nmr | source | split | label
  ecfp (2048) | maccs (167) | element_list (12) | function_list (92)
  ring_system | extra_ring_system

`function_list` is filled with zeros because the optional `thermo` package is
not required by CMGNet or by the CReSS/XGBoost head; the field is kept only for
schema compatibility.

Usage:
  python scripts/build_lmdb/json_to_lmdb.py \
      --json data/validation/validation.json \
      --out  data/validation/validation.lmdb \
      --split valid
"""
import argparse
import ast
import json
import os
import pickle
import sys
import time

import lmdb
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, MACCSkeys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from new_RingSystemFinder import RingSystemFinder  # noqa: E402

RDLogger.DisableLog("rdApp.*")

CORE_SMARTS = [
    "O=C1C(C(O)=O)=CNC2=CC=CC=C12",
    "O=C1C(C(O)=O)=CNC2=NC=CC=C12",
]
ELEMENTS = ["C", "H", "O", "N", "P", "S", "Cl", "Br", "I", "B", "F", "Si"]
FUNCTION_DIM = 92

RSF = RingSystemFinder()


def remove_terminal_carbons(mol):
    rwmol = Chem.RWMol(mol)
    ring_atoms = set()
    for ring in mol.GetRingInfo().AtomRings():
        ring_atoms.update(ring)
    remove = []
    for atom in rwmol.GetAtoms():
        if atom.GetDegree() == 1 and atom.GetAtomicNum() == 6:
            bond = atom.GetBonds()[0]
            if atom.GetNeighbors()[0].GetIdx() in ring_atoms and bond.GetBondType() != Chem.BondType.SINGLE:
                remove.append(atom.GetIdx())
    for idx in sorted(remove, reverse=True):
        rwmol.RemoveAtom(idx)
    return rwmol.GetMol()


def ring_systems(mol, existing=None):
    if existing:
        if isinstance(existing, str):
            try:
                existing = ast.literal_eval(existing)
            except Exception:
                existing = [existing]
        existing = [s for s in existing if isinstance(s, str) and s]
        if existing:
            return existing
    mols = RSF.find_ring_systems(mol, as_mols=True)
    out = []
    for m in mols:
        try:
            out.append(Chem.MolToSmiles(remove_terminal_carbons(m)))
        except Exception:
            continue
    return out


def extra_ring_systems(mol):
    out = []
    for smarts in CORE_SMARTS:
        if mol.HasSubstructMatch(Chem.MolFromSmiles(smarts)):
            out.append(Chem.MolToSmiles(Chem.MolFromSmiles(smarts)))
    return out


def element_counts(mol):
    mh = Chem.AddHs(Chem.Mol(mol))
    counts = {e: 0 for e in ELEMENTS}
    for atom in mh.GetAtoms():
        if atom.GetSymbol() in counts:
            counts[atom.GetSymbol()] += 1
    return np.array([counts[e] for e in ELEMENTS], dtype=np.int16)


def pick(rec, *names):
    for n in names:
        if n in rec and rec[n] is not None:
            return rec[n]
    return None


def build_record(rec, idx):
    smiles = pick(rec, "canonical_smiles", "cano_smiles", "smiles", "SMILES")
    nmr = pick(rec, "C_nmr", "NMR", "nmr", "shifts")
    if not smiles or not nmr:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    label = pick(rec, "label")
    if isinstance(label, (list, tuple, np.ndarray)):
        label = int(label[0])
    elif label is None:
        label = int(bool(extra_ring_systems(mol)))
    else:
        label = int(label)

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    ecfp = np.zeros((2048,), dtype=np.int8)
    AllChem.DataStructs.ConvertToNumpyArray(fp, ecfp)

    maccs_fp = MACCSkeys.GenMACCSKeys(mol)
    maccs = np.zeros((167,), dtype=np.int8)
    AllChem.DataStructs.ConvertToNumpyArray(maccs_fp, maccs)

    return dict(
        key=idx,
        canonical_smiles=Chem.MolToSmiles(mol),
        cano_smiles=Chem.MolToSmiles(mol),
        C_nmr=[float(x) for x in nmr],
        source=pick(rec, "source") or "unknown",
        split=pick(rec, "split") or "unknown",
        ecfp=ecfp,
        maccs=maccs,
        function_list=np.zeros((FUNCTION_DIM,), dtype=np.int8),
        label=np.array([label]),
        element_list=element_counts(mol),
        extra_ring_system=extra_ring_systems(mol),
        ring_system=ring_systems(mol, pick(rec, "ring_system")),
        ori_key=str(pick(rec, "index", "key", "ori_key", "")),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="")
    ap.add_argument("--map_gb", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    data = json.load(open(args.json, encoding="utf-8"))
    if args.limit:
        data = data[: args.limit]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    env = lmdb.open(args.out, map_size=args.map_gb * (1024 ** 3), subdir=True)
    txn = env.begin(write=True)

    kept, skipped = 0, 0
    t0 = time.time()
    for i, rec in enumerate(data):
        try:
            out = build_record(rec, kept)
        except Exception:
            out = None
        if out is None:
            skipped += 1
            continue
        if args.split:
            out["split"] = args.split
        txn.put(str(kept).encode(), pickle.dumps(out, protocol=4))
        kept += 1
        if kept and kept % 5000 == 0:
            txn.commit(); txn = env.begin(write=True)
            print(f"[{i+1}/{len(data)}] kept={kept} skipped={skipped}", flush=True)
    txn.commit()

    got = env.begin().stat()["entries"]
    print(f"wrote {args.out}: {got} records (skipped {skipped}) "
          f"in {time.time()-t0:.1f}s", flush=True)
    if got != kept:
        raise SystemExit("entry count mismatch")
    env.close()


if __name__ == "__main__":
    main()
