---
name: nmr-synthesis
description: Generate predicted 13C NMR peak lists from SMILES with the optional nmrshiftdb2 PredictionTool + CDK 2.3 jars, and reproduce the legacy scraper / analog-expansion synthetic-data workflows.
---

# nmr-synthesis

Use this skill when the task is to **generate synthetic 13C NMR data from
SMILES**, extend a library with analogs, or reproduce the legacy `lib/`
workflow.  It is separate from model training and paper-figure generation.

## What is in `lib/`

| File | Origin | Licence |
|---|---|---|
| `lib/predictorc.jar` | nmrshiftdb2 `PredictionTool` + embedded `nmrshiftdb.csv` | **AGPL-3.0** + **nmrshiftdb2 Database License** |
| `lib/predictorh.jar` | nmrshiftdb2 proton predictor (unused) | same |
| `lib/cdk-2.3.jar` | Chemistry Development Kit 2.3 | LGPL-2.1 (bundle also has Apache-2.0 `xml-apis`) |

The jars are **not in git** (`lib/*.jar`).  Obtain them from the legacy checkout:

```bash
bash scripts/fetch_predictor.sh
# or
LEGACY_LIB=/path/to/NMR_Core_Mol_Detection/lib bash scripts/fetch_predictor.sh
```

The Python calling pattern came from ChemTSv2 `reward/NMR_reward.py` (MIT,
Copyright (c) 2021 Shoichi Ishida).

## Quick start

```bash
python scripts/synthesize/predict_c_nmr.py --smiles "CCO"
python scripts/synthesize/predict_c_nmr.py --input smiles.txt --out preds.jsonl --jobs 4
```

Output JSONL: `smiles`, `canonical_smiles`, `solvent`, `column`, `C_nmr`,
`n_peaks`.  Failures return `error`.

Solvent keys: `chloroform` (default), `methanol`, `dmso`.

## Critical parsing detail

`Test.class` prints:

```
  C    min.    mean    max.
 ---------------------------
  1:   16.40   17.34   18.60
```

The wrapper defaults to **mean** (`parts[2]`).  The legacy scrapers used
`parts[3]`, which is **max**, despite their comments saying mean.  Use
`--column max` only to reproduce the legacy files.

Raw Java call (the solvent argument has **no** quotes, even though the error
message says otherwise):

```bash
java -cp lib/predictorc.jar:lib/cdk-2.3.jar Test mol.mol "Chloroform-D1 (CDCl3)"
```

## Legacy call sites

These live in the old repository, not in this release:

* `scrapers/new_nmr_scraper_1104.py::smiles2nmr`
* `overlap_check/methylate_and_predict.py::cdk_predict`
* `overlap_check/analog_expand.py`
* `third_party/bart_infer/infer*.py`

Workflow: parse SMILES with RDKit → `Chem.AddHs` → write MDL mol → run
`java ... Test` → parse the table → deduplicate shifts → merge into the target
corpus.

## Analog expansion

The legacy `overlap_check/methylate_and_predict.py` applies one of six
RDKit reactions (N-H→N-CH3, aromatic C-H→C-CH3, aliphatic CH/CH2/CH3 edits),
filters by the quinolone core and by absence from evaluation sets, then
predicts `C_nmr` with the same jars.  Reuse the reaction definitions and filters
if extending the training set; do not add molecules that overlap the external
test, validation or test splits.

## Data and licensing

* `acd_lotusv9` is the CC BY 4.0 Zenodo dataset
  [10.5281/zenodo.8175939](https://doi.org/10.5281/zenodo.8175939) (LOTUS v9
  structures + ACD/Labs-predicted shifts).  Redistribution requires attribution.
* `multi`, `napacd` and `pnmrnp2` in `data/synthetic/` have no documented
  licence in the legacy tree.  Do not publish their spectra; use
  `scripts/build_lmdb/export_source_index.py` for SMILES/record indices.
* The nmrshiftdb2 database inside `predictorc.jar` is share-alike.  If the jars
  are redistributed, include the AGPL-3.0 and nmrshiftdb2 Database License texts
  and honour the AGPL source offer.
* `data/synthetic/synth_predicted.lmdb` was **not** produced by this wrapper;
  it is a four-source union.  Do not claim otherwise in the paper.

## Verification

A quick sanity check on ethanol should return two mean shifts:

```
C_nmr: [17.34, 57.97]
```

Java warnings about missing stereo fields on stderr are normal.
