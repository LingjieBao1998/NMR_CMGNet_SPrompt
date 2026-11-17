# Third-party notices

The original code in this repository is licensed under the Apache License 2.0
(see `LICENSE`).  Third-party components and datasets keep their own licences;
the Apache-2.0 grant does **not** override them.

## Software

| Component | Version | Licence | Used for |
|---|---|---|---|
| RDKit | >= 2023.9 | BSD-3-Clause | SMILES parsing, fingerprints, substructure matching |
| PyTorch | 2.4–2.9 | BSD-3-Clause | CMGNet / CReSS |
| PyTorch Lightning | 2.1–2.9 | Apache-2.0 | training loops |
| Transformers (Hugging Face) | 4.35–5.x | Apache-2.0 | BART backbone |
| BART base (`facebook/bart-base`) | 0910 checkpoint | MIT | encoder–decoder initialisation |
| XGBoost | >= 2.0 | Apache-2.0 | CReSS classifier head |
| scikit-learn | >= 1.3 | BSD-3-Clause | Platt / isotonic calibration, metrics |
| NumPy | >= 1.24 | BSD-3-Clause | arrays |
| pandas | >= 2.0 | BSD-3-Clause | tabular I/O |
| SciPy | >= 1.11 | BSD-3-Clause | Wasserstein distance, statistics |
| matplotlib | >= 3.8 | PSF-based (matplotlib licence) | figures |
| LMDB | >= 1.4 | OpenLDAP Public License | dataset storage |
| Django | 4.2 | BSD-3-Clause | web service |
| Celery / Redis client | 5.3 / 5.0 | BSD-3-Clause | web-service queue |
| CReSS | upstream checkpoint | see upstream repository | chemical encoder |

## Optional NMR predictor (`lib/`)

`lib/` is **not** part of the Apache-2.0 work and is not required to run the
released models.

| File | Origin | Licence |
|---|---|---|
| `predictorc.jar`, `predictorh.jar` | nmrshiftdb2 `PredictionTool` (Stefan Kuhn and NMRShiftDB contributors) | **AGPL-3.0** + **nmrshiftdb2 Database License** |
| `Test.class`, `TestSdf.class` | local command-line wrapper around the nmrshiftdb2 tool | inherits the AGPL-3.0 terms of the jar |
| `cdk-2.3.jar` | Chemistry Development Kit 2.3 | LGPL-2.1; the bundle also contains Apache-2.0 `xml-apis` |

The Python calling pattern (`smiles2nmr`, mean-column extraction, Wasserstein
reward) was adapted from **ChemTSv2** (`reward/NMR_reward.py`,
https://github.com/molecule-generator-collection/ChemTSv2), MIT licence,
Copyright (c) 2021 Shoichi Ishida.

If the jars are redistributed with a release, the AGPL-3.0 and nmrshiftdb2
Database License texts must accompany them, and the corresponding source offer
must be honoured.  The `nmrshiftdb2` database embedded in `predictorc.jar` is
itself covered by the nmrshiftdb2 Database License (share-alike).

## Data sources

| `source` field | Used in | Original terms | Redistributed here |
|---|---|---|---|
| nmrshiftdb2 | `exp_lit`, `ex_train`, validation | nmrshiftdb2 Database License (share-alike) | derived records only, with attribution |
| BMRB | `exp_lit` | BMRB data-use terms (citation required) | see note |
| HMDB | `exp_lit` | HMDB academic-use terms | no |
| NMR BANK | `exp_lit` | source terms | no |
| c13_naproc | `exp_lit` | source terms | no |
| Spektraris_NMR | `exp_lit` | source terms | no |
| GlycoNMR | `exp_lit` | source terms | no |
| PubChem | `pubchem_1209` | PubChem terms (public domain) | yes |
| **ACD/Lotus (`acd_lotusv9`)** | `synthetic` | **CC BY 4.0** (Zenodo 10.5281/zenodo.8175939; shifts generated with ACD/Labs) | yes, with attribution |
| NPACD / `napacd` | `synthetic` | no licence recorded in the legacy tree | confirm before release |
| `multi` | `synthetic` | no licence recorded in the legacy tree | confirm before release |
| `pnmrnp2` | `synthetic` | no licence recorded in the legacy tree | confirm before release |
| NMRMind | `nmrmind/*` | upstream release terms | no |
| NMRium | `validation` (40,667) | provenance recorded; used only for calibration/validation, excluded from training | validation split only |

The `acd_lotusv9` component (194,550 records) is derived from LOTUS version 9
structures with ACD/Labs-predicted 13C shifts, published by Nuzillard & Kuhn on
Zenodo under **CC BY 4.0** (DOI 10.5281/zenodo.8175939).  It can be redistributed
with attribution, even though the ACD/Labs prediction software itself is
commercial.

The remaining synthetic components have no licence recorded in the legacy tree:
`multi` (789,028 records), `napacd` (313,343) and `pnmrnp2` (191,511).  Confirm
their terms before publishing the full 1,488,432-record
`data/synthetic/synth_predicted.lmdb`; otherwise publish the cleared subset plus
per-source record indices.

The external test set (`data/external_test/`) is released with per-record
provenance so that each record can be traced to its original source.
