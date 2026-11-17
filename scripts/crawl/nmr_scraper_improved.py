#!/usr/bin/env python3
"""Batch-predict 13C NMR spectra through the Zakodium carbon API.

Inputs are a CSV with a SMILES column and/or a JSON list with `cano_smiles`
(optionally `key`).  Output is a JSON list with one record per molecule:

    {"source": "CSV"|"JSON", "index": int, "smiles": str,
     "nmr_result": {...}, "status": "success"|"failed"}

The script resumes from an existing output file: only `status == "success"`
records are skipped, so failed requests are retried.

Examples:
  python scripts/crawl/nmr_scraper_improved.py \
      --csv data/test_core_mol.csv --out predictions.json
  python scripts/crawl/nmr_scraper_improved.py \
      --json data/external_test/external_test.json --out predictions.json
"""
import argparse
import json
import os
import random
import time

import pandas as pd
import requests

DEFAULT_API = "https://nmr-prediction.service.zakodium.com/v1/predict/carbon"


def predict(smiles, args):
    payload = {
        "molfile": None,
        "smiles": smiles,
        "includeJDX": False,
        "frequency": 100,
        "nbPoints": 65536,
        "lineWidth": 1,
        "from": -10,
        "to": 230,
        "maxSphereSize": 5,
    }
    headers = {"accept": "*/*", "Content-Type": "application/json"}

    for attempt in range(1, args.retries + 1):
        try:
            response = requests.post(args.api, headers=headers, json=payload,
                                     timeout=args.timeout)
        except requests.RequestException as exc:
            print(f"  retry {attempt}/{args.retries}: {exc}")
            time.sleep(args.retry_wait)
            continue

        if response.status_code == 200:
            try:
                return response.json()
            except ValueError:
                print(f"  retry {attempt}/{args.retries}: invalid JSON")
        elif response.status_code == 429:
            wait = args.rate_limit_wait + random.uniform(0, 1)
            print(f"  429 rate limited, waiting {wait:.1f}s")
            time.sleep(wait)
            continue
        else:
            print(f"  retry {attempt}/{args.retries}: HTTP {response.status_code}")
            print(f"  {response.text[:300]}")
        time.sleep(args.retry_wait)
    return None


def load_success(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as fh:
            records = json.load(fh)
    except (ValueError, OSError):
        return {}
    done = {}
    for rec in records:
        if rec.get("status") == "success":
            done[(rec.get("source"), rec.get("index"))] = rec
    print(f"resume: {len(done)} successful records in {path}")
    return done


def save(path, records):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(records, fh, indent=2)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="CSV with a SMILES column")
    ap.add_argument("--csv-column", default="SMILES")
    ap.add_argument("--json", help="JSON list with cano_smiles")
    ap.add_argument("--json-key", default="cano_smiles")
    ap.add_argument("--out", default="nmr_predictions_combined.json")
    ap.add_argument("--api", default=DEFAULT_API)
    ap.add_argument("--timeout", type=float, default=60)
    ap.add_argument("--retries", type=int, default=10)
    ap.add_argument("--interval", type=float, default=2.0,
                    help="base seconds between requests (0-1 s jitter added)")
    ap.add_argument("--rate-limit-wait", type=float, default=5.0)
    ap.add_argument("--retry-wait", type=float, default=2.0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if not args.csv and not args.json:
        ap.error("give --csv, --json or both")

    tasks = []
    if args.csv:
        df = pd.read_csv(args.csv)
        if args.csv_column not in df.columns:
            ap.error(f"{args.csv} has no column {args.csv_column!r}")
        for idx, row in df.iterrows():
            tasks.append(("CSV", idx, str(row[args.csv_column]), {}))
        print(f"CSV: {len(tasks)} molecules")

    if args.json:
        with open(args.json) as fh:
            data = json.load(fh)
        start = len(tasks)
        for idx, item in enumerate(data):
            smiles = item.get(args.json_key) or item.get("smiles") or item.get("cano_smiles")
            if smiles:
                tasks.append(("JSON", idx, smiles, {"original_key": item.get("key", idx)}))
        print(f"JSON: {len(tasks) - start} molecules")

    if args.limit:
        tasks = tasks[:args.limit]

    results = load_success(args.out)
    total = len(tasks)
    new = 0
    for i, (source, idx, smiles, extra) in enumerate(tasks, 1):
        key = (source, idx)
        if key in results:
            continue
        print(f"[{i}/{total}] {source} {idx}: {smiles}")
        nmr = predict(smiles, args)
        rec = {
            "source": source,
            "index": idx,
            "smiles": smiles,
            **extra,
            "nmr_result": nmr,
            "status": "success" if nmr is not None else "failed",
        }
        results[key] = rec
        new += 1
        if new % 10 == 0:
            save(args.out, list(results.values()))
            print(f"  saved {len(results)} records")
        time.sleep(args.interval + random.uniform(0, 1))

    save(args.out, list(results.values()))
    ok = sum(1 for r in results.values() if r["status"] == "success")
    print(f"done: {len(results)} records, {ok} successful, written to {args.out}")


if __name__ == "__main__":
    main()
