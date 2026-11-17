#!/usr/bin/env python3
"""Predict a 13C NMR peak list from SMILES with the nmrshiftdb2 PredictionTool.

This is the cleaned-up version of the legacy `smiles2nmr` snippets that were
copied from ChemTSv2 (`reward/NMR_reward.py`).  It writes a temporary MDL mol
file, runs

    java -cp lib/predictorc.jar:lib/cdk-2.3.jar Test <mol> <solvent>

and parses the `min. / mean / max.` table printed by `Test.class`.

Examples
--------
  python scripts/synthesize/predict_c_nmr.py --smiles "CCO"
  python scripts/synthesize/predict_c_nmr.py --input smiles.txt --out preds.jsonl
  python scripts/synthesize/predict_c_nmr.py --smiles "..." --column max

The default is the **mean** column (parts[2] of a data row).  The legacy
scrapers used parts[3], which is actually the **max** column; pass
`--column max` if you need to reproduce those files.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

from rdkit import Chem

SOLVENTS = {
    "chloroform": "Chloroform-D1 (CDCl3)",
    "methanol": "Methanol-D4 (CD3OD)",
    "dmso": "Dimethylsulphoxide-D6 (DMSO-D6, C2D6SO)",
}
ROW = re.compile(r"^\s*(\d+):\s+(.*)$")


def parse_table(text, column="mean"):
    """Return the sorted unique shifts from one Test.class output."""
    header = None
    values = []
    for line in text.splitlines():
        stripped = line.strip()
        if "min." in stripped and "mean" in stripped and "max." in stripped:
            header = stripped.split()
            continue
        m = ROW.match(line)
        if not m or header is None:
            continue
        parts = m.group(2).split()
        try:
            idx = header.index(column) - 1  # header starts with "C"
            values.append(float(parts[idx]))
        except (ValueError, IndexError):
            continue
    if not values:
        raise RuntimeError("no prediction rows found; check the solvent and Java output")
    return sorted(set(values))


def predict_one(args):
    smiles, lib_dir, solvent, column, java, timeout = args
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"smiles": smiles, "error": "invalid SMILES"}
    mol = Chem.AddHs(mol)
    predictor = os.path.join(lib_dir, "predictorc.jar")
    cdk = os.path.join(lib_dir, "cdk-2.3.jar")
    with tempfile.TemporaryDirectory(prefix="nmr_pred_") as tmp:
        mol_path = os.path.join(tmp, "mol.mol")
        Chem.MolToMolFile(mol, mol_path)
        cmd = [java, "-cp", f"{predictor}:{cdk}", "Test", mol_path, solvent]
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"smiles": smiles, "error": "java timeout"}
    text = proc.stdout.decode("utf-8", "replace")
    if proc.returncode != 0 and "solvent must be" in text:
        return {"smiles": smiles, "error": text.strip()}
    try:
        shifts = parse_table(text, column)
    except Exception as exc:
        return {"smiles": smiles, "error": str(exc), "stderr": proc.stderr.decode("utf-8", "replace")[:500]}
    return {
        "smiles": smiles,
        "canonical_smiles": Chem.MolToSmiles(Chem.MolFromSmiles(smiles)),
        "solvent": solvent,
        "column": column,
        "C_nmr": shifts,
        "n_peaks": len(shifts),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smiles", action="append", default=[], help="repeatable")
    ap.add_argument("--input", help="text file, one SMILES per line")
    ap.add_argument("--out", help="JSONL output (default: stdout)")
    ap.add_argument("--lib-dir", default="lib",
                    help="directory containing predictorc.jar and cdk-2.3.jar")
    ap.add_argument("--java", default="java")
    ap.add_argument("--solvent", default="chloroform",
                    choices=sorted(SOLVENTS), help="solvent key")
    ap.add_argument("--column", default="mean", choices=["min", "mean", "max"],
                    help="table column to extract (legacy scripts used max)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args()

    smiles = list(args.smiles)
    if args.input:
        with open(args.input) as fh:
            smiles += [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]
    if not smiles:
        ap.error("give --smiles or --input")

    predictor = os.path.join(args.lib_dir, "predictorc.jar")
    cdk = os.path.join(args.lib_dir, "cdk-2.3.jar")
    missing = [p for p in (predictor, cdk) if not os.path.exists(p)]
    if missing:
        sys.exit(
            "missing predictor jars:\n  " + "\n  ".join(missing) +
            "\nRun scripts/fetch_predictor.sh or pass --lib-dir."
        )

    solvent = SOLVENTS[args.solvent]
    tasks = [(s, args.lib_dir, solvent, args.column, args.java, args.timeout) for s in smiles]
    out = open(args.out, "w") if args.out else sys.stdout

    try:
        if args.jobs > 1:
            with ProcessPoolExecutor(max_workers=args.jobs) as pool:
                futures = [pool.submit(predict_one, t) for t in tasks]
                for i, fut in enumerate(as_completed(futures), 1):
                    rec = fut.result()
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out.flush()
                    print(f"[{i}/{len(tasks)}] {rec.get('smiles')} -> "
                          f"{rec.get('n_peaks', rec.get('error'))}", file=sys.stderr)
        else:
            for i, task in enumerate(tasks, 1):
                rec = predict_one(task)
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                print(f"[{i}/{len(tasks)}] {rec.get('smiles')} -> "
                      f"{rec.get('n_peaks', rec.get('error'))}", file=sys.stderr)
    finally:
        if args.out:
            out.close()


if __name__ == "__main__":
    main()
