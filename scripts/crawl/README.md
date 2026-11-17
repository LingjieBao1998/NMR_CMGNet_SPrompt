# scripts/crawl — Zakodium ¹³C prediction API

Batch wrapper for `https://nmr-prediction.service.zakodium.com/v1/predict/carbon`.

Inputs:

* CSV with a SMILES column (`--csv`, `--csv-column SMILES`)
* JSON list with `cano_smiles` (and optional `key`): `--json`, `--json-key`

Output is a JSON list with `source`, `index`, `smiles`, `nmr_result`, `status`.
Only `status == "success"` records are skipped on rerun, so failures are retried.

```bash
python scripts/crawl/nmr_scraper_improved.py \
    --csv data/test_core_mol.csv --out output/zakodium_predictions.json

python scripts/crawl/nmr_scraper_improved.py \
    --json data/external_test/external_test.json \
    --out output/zakodium_predictions.json --interval 2
```

Defaults: 60 s request timeout, 10 retries, 2 s base interval (+0–1 s jitter),
5 s wait on HTTP 429.  Use `--limit N` for a smoke test.

This is the predecessor of `scripts/synthesize/predict_c_nmr.py`: the API is
hosted and rate-limited, whereas `lib/` runs the nmrshiftdb2 PredictionTool
locally.  Neither is used by the released models.
