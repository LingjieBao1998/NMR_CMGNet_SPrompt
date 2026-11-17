# scripts/synthesize — SMILES → predicted 13C NMR

Clean wrapper around the optional nmrshiftdb2 `PredictionTool` + CDK 2.3 jars
(see [`lib/README.md`](../../lib/README.md)).

```bash
# one molecule
python scripts/synthesize/predict_c_nmr.py --smiles "CCO"

# batch, 4 Java workers
python scripts/synthesize/predict_c_nmr.py \
    --input smiles.txt --out predictions.jsonl --jobs 4

# reproduce the legacy scrapers (they used the max column)
python scripts/synthesize/predict_c_nmr.py \
    --input smiles.txt --out legacy.jsonl --column max
```

Input `--input` is one SMILES per line; blank lines and `#` comments are
ignored.  Output is JSONL:

```json
{"smiles": "CCO", "canonical_smiles": "CCO", "solvent": "Chloroform-D1 (CDCl3)",
 "column": "mean", "C_nmr": [17.34, 57.97], "n_peaks": 2}
```

Invalid molecules and Java failures produce `{"smiles": ..., "error": ...}`.

## Column selection

`Test.class` prints `atom: min. mean max.`.  The wrapper defaults to `mean`
(the scientifically appropriate value).  The legacy scraper/BART code used
`parts[3]`, which is the **max** column, despite comments calling it the mean;
use `--column max` only when reproducing those files.

## Requirements

Java 8+ (tested with OpenJDK 17), RDKit, and the three jars in `lib/`:

```bash
bash scripts/fetch_predictor.sh
```

The jars are not tracked (AGPL-3.0 + nmrshiftdb2 Database License).  Solvents:
`chloroform`, `methanol`, `dmso`.

## Related

* Original Python calling pattern: ChemTSv2 `reward/NMR_reward.py` (MIT).
* Legacy callers in the old repo: `scrapers/new_nmr_scraper_1104.py`,
  `overlap_check/methylate_and_predict.py`, `overlap_check/analog_expand.py`,
  `third_party/bart_infer/infer*.py`.
* Dataset construction is separate: `data/synthetic/synth_predicted.lmdb` is a
  union of `multi`, `napacd`, `acd_lotusv9` and `pnmrnp2`, not the output of
  this wrapper.  See `data/README.md`.
