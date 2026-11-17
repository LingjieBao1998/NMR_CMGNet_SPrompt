# lib — optional nmrshiftdb2 13C/1H predictor

This directory holds the **optional** Java predictor used to generate synthetic
13C spectra from SMILES.  It is not needed for training, inference or figure
regeneration; those use the shipped `data/` corpora and checkpoints.

The jars are **not tracked by git** (`lib/*.jar` is ignored) because they are
third-party code with share-alike terms.  Run `scripts/fetch_predictor.sh` to
copy them from a local legacy checkout.

| File | Origin | Licence |
|---|---|---|
| `predictorc.jar` | nmrshiftdb2 `PredictionTool` (Stefan Kuhn / NMRShiftDB contributors) + embedded `nmrshiftdb.csv` | **AGPL-3.0** + **nmrshiftdb2 Database License** |
| `predictorh.jar` | nmrshiftdb2 proton predictor | same; currently unused |
| `cdk-2.3.jar` | Chemistry Development Kit 2.3 | LGPL-2.1; bundle also contains Apache-2.0 `xml-apis` |

The command-line wrapper `Test.class` / `TestSdf.class` is a thin CDK CLI around
`PredictionTool`.  The Python calling pattern was adapted from ChemTSv2
(`reward/NMR_reward.py`, MIT, Copyright (c) 2021 Shoichi Ishida).

## Obtain the jars

```bash
# from the legacy working tree (default path)
bash scripts/fetch_predictor.sh

# or from another checkout
LEGACY_LIB=/path/to/NMR_Core_Mol_Detection/lib bash scripts/fetch_predictor.sh
```

Expected SHA-256:

```
bf60002d40b88136f185a96b1f38135e240733360c6b684cfc2bfd64972eb04a  cdk-2.3.jar
95211fcaf148600057743b657afdaced6cc352cdb3dbf242992b0cd2ff425419  predictorc.jar
5e29217c7792b0b3651e5520d076e61df7dec3d29f1ec6fdfbf13f48e0cb3156  predictorh.jar
```

If the jars are redistributed (GitHub release, Zenodo, Docker image), the
AGPL-3.0 and nmrshiftdb2 Database License texts must accompany them, and the
AGPL source offer must be honoured.  The nmrshiftdb2 database embedded in
`predictorc.jar` is share-alike.

## Use

```bash
python scripts/synthesize/predict_c_nmr.py --smiles "CCO"
python scripts/synthesize/predict_c_nmr.py --input smiles.txt --out preds.jsonl --jobs 4
```

Raw Java call (note: **no quotes** around the solvent, despite the error text):

```bash
java -cp lib/predictorc.jar:lib/cdk-2.3.jar Test mol.mol "Chloroform-D1 (CDCl3)"
```

Supported solvents:

* `Chloroform-D1 (CDCl3)`
* `Methanol-D4 (CD3OD)`
* `Dimethylsulphoxide-D6 (DMSO-D6, C2D6SO)`

Output rows are `atom: min. mean max.`.  The wrapper defaults to the **mean**
column; the legacy scraper code used `parts[3]`, which is the **max** column.

## Relationship to `data/synthetic/`

`data/synthetic/synth_predicted.lmdb` (1,488,432 records) was **not** generated
by this wrapper.  It is a union of `multi`, `napacd`, `acd_lotusv9` and
`pnmrnp2`; the `acd_lotusv9` part comes from the CC BY 4.0 Zenodo dataset
[10.5281/zenodo.8175939](https://doi.org/10.5281/zenodo.8175939) (LOTUS v9
structures with ACD/Labs-predicted shifts).  The wrapper is useful for new
SMILES, analog expansion and reproducing the legacy scraper/BART workflows.
