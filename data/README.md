# Data

All splits are rebuilt into LMDB/JSON pairs so that every category is self-contained
and directly consumable by `main_cmgnet.py` (CMGNet) and
`train_xgb_cress_new_dataset.py` (CReSS + XGBoost).

```
data/
├── README.md
├── _source/new_db.lmdb -> <original 3.20 M-record corpus>   (optional, for rebuilds)
├── experimental/
│   ├── exp_lit.lmdb                 # literature experimental spectra (49,984)
│   ├── ex_train.json                # nmrshiftdb extra train, 24,520
│   ├── ex_train.ring.json           # same, with ring-system field
│   └── ex_train.lmdb                # LMDB built from the ring JSON
├── synthetic/
│   └── synth_predicted.lmdb         # predicted spectra (1,488,432; 313,343
│                                     # carry source=NMRium, formerly napacd)
├── pubchem/
│   └── pubchem_1209.lmdb            # PubChem positives, filtered (77,328)
├── nmrmind/
│   ├── train.lmdb / validation.lmdb / test.lmdb
│   └── validation.json / test.json
├── validation/
│   ├── validation.json              # clean validation split (40,667 / 37 pos)
│   └── validation.lmdb
├── external_test/
│   ├── external_test.json           # 4,502 (4,438 neg + 64 pos)
│   ├── external_test.lmdb
│   └── external_test_pos_only.json  # 64 positives only
└── f_split/
    ├── with_f_split.json            # 48 positives, C–F splitting retained
    └── without_f_split.json         # 48 positives, splitting merged
```

## Category provenance

| Category | Source | Records | Positives |
|---|---|---:|---:|
| `experimental/exp_lit.lmdb` | `new_db.lmdb` records with `source ∈ {NMR BANK, c13_naproc, bmrb, Spektraris_NMR, GlycoNMR, hmdb}` | 49,984 | 55 |
| `synthetic/synth_predicted.lmdb` | `new_db.lmdb` records with `source ∈ {multi, acd_lotusv9, pnmrnp2}` plus 313,343 records whose source was rewritten from `napacd` to `NMRium` | 1,488,432 | 2,461 |
| `pubchem/pubchem_1209.lmdb` | PubChem positive corpus `new_data_1209.filtered` after removal of 90 evaluation overlaps | 77,328 | 77,328 |
| `nmrmind/*` | original NMRMind splits (unchanged) | 2,137,771 / 55,167 / 55,150 | 3,288 / 81 / 87 |
| `validation/*` | clean `new_dataset` validation split | 40,667 | 37 |
| `external_test/*` | April-18 external test set | 4,502 | 64 |
| `f_split/*` | positive subset re-processed with/without C–F splitting | 48 + 48 | 48 |
| `experimental/ex_train.*` | `nmrshiftdb` extra training split (0 overlap with evaluation sets) | 24,520 | 11 |

The original corpus contains 3,196,326 records. 1,657,910 of them (51.9 %) come from the
`NMRium` upload collection; **that subset is deliberately excluded** because its
provenance could not be verified. Total retained: 49,984 experimental + 1,488,432 synthetic.

Inside `synthetic/synth_predicted.lmdb`, the 313,343 records whose original
`source` was `napacd` have been rewritten to `source = "NMRium"` because their
provenance follows the NMRium upload collection.  The records (spectra, labels,
fingerprints) are unchanged and remain in the file, so the entry count is still
1,488,432; only the provenance field changed.  `synth_predicted.pre_napm.lmdb`
is the pre-rewrite copy kept locally for comparison and is not part of the release.

## Licensing and availability

| `source` | Records | Terms |
|---|---:|---|
| `multi` | 789,028 | not documented in the legacy tree; confirm before redistribution |
| `NMRium` (313,343 formerly `napacd`) | 313,343 | unverified provenance; not redistributed |
| `acd_lotusv9` | 194,550 | **CC BY 4.0** — LOTUS v9 structures + ACD/Labs-predicted ¹³C shifts, Zenodo [10.5281/zenodo.8175939](https://doi.org/10.5281/zenodo.8175939) |
| `pnmrnp2` | 191,511 | not documented in the legacy tree; confirm before redistribution |
| `NMR BANK` | 28,406 | source terms |
| `c13_naproc` | 20,340 | source terms |
| `bmrb` | 894 | BMRB data-use terms (cite) |
| `Spektraris_NMR` | 159 | source terms |
| `GlycoNMR` | 119 | source terms |
| `hmdb` | 66 | HMDB academic-use terms |

The ACD/Labs **software** is commercial, but the `acd_lotusv9` **dataset** was
published under CC BY 4.0 by Nuzillard & Kuhn; redistribution is allowed with
attribution.  The other synthetic sources have no licence recorded; use
`scripts/build_lmdb/filter_lmdb_by_source.py` to build a cleared subset and
`scripts/build_lmdb/export_source_index.py` to publish SMILES/record indices
without spectra.  `data/_source/new_db.lmdb` is a legacy absolute symlink and is
not part of the release.

## LMDB schema

Each value is a pickled dict:

```
key               int
canonical_smiles  str        (cano_smiles kept as an alias)
C_nmr             list[float]  13C shifts in ppm
source            str
split             str
label             np.ndarray([0|1])   quinolone core present
ecfp              np.ndarray(2048,)   Morgan FP, r=2
maccs             np.ndarray(167,)
element_list      np.ndarray(12,)     C,H,O,N,P,S,Cl,Br,I,B,F,Si counts
function_list     np.ndarray(92,)     reserved (zeros; `thermo` optional)
ring_system       list[str]           ring systems with terminal carbons removed
extra_ring_system list[str]           matched quinolone-core SMARTS
ori_key           str                 original index (JSON-built splits)
```

`function_list` is zero-filled because the functional-group enumeration (`thermo`)
is not required by CMGNet or the CReSS/XGBoost head. Re-enable it by installing
`thermo` and filling the field in `scripts/build_lmdb/json_to_lmdb.py`.

## Rebuild

```bash
# 1. split the 3.20 M source corpus by source category (~10 min)
#    (add --overwrite to replace existing output LMDBs)
python scripts/build_lmdb/split_newdb_by_source.py \
    --src data/_source/new_db.lmdb \
    --exp_out data/experimental/exp_lit.lmdb \
    --synth_out data/synthetic/synth_predicted_release.lmdb \
    --overwrite

# NOTE: the current classification treats napacd as NMRium-derived and
# excludes it, so this yields 1,175,089 synthetic records.  The shipped
# synth_predicted.lmdb was built before that change and still has 1,488,432
# records (313,343 of them relabelled NMRium); it is the corpus used by the
# reported models.  To rebuild that training file, temporarily add "napacd"
# to SYNTHETIC in this script, then run:
#   python scripts/build_lmdb/split_newdb_by_source.py \
#       --exp_out data/experimental/exp_lit_napacd.lmdb \
#       --synth_out data/synthetic/synth_predicted_napacd.lmdb
#   python scripts/build_lmdb/relabel_source.py \
#       --src data/synthetic/synth_predicted_napacd.lmdb \
#       --from napacd --to NMRium \
#       --out data/synthetic/synth_predicted.lmdb
# To build the release corpus, keep the split script as-is and package
# synth_predicted_release.lmdb.

# 2. JSON -> featurised LMDB (validation / external test / ex_train)
python scripts/build_lmdb/json_to_lmdb.py \
    --json data/validation/validation.json \
    --out  data/validation/validation.lmdb --split valid

python scripts/build_lmdb/json_to_lmdb.py \
    --json data/external_test/external_test.json \
    --out  data/external_test/external_test.lmdb --split external_test

python scripts/build_lmdb/json_to_lmdb.py \
    --json data/experimental/ex_train.ring.json \
    --out  data/experimental/ex_train.lmdb --split train
```

`pubchem/pubchem_1209.lmdb` and the NMRMind LMDBs are copied from the upstream
release (`new_data_1209.filtered.ring.lmdb`, `nmrmind_cmgnet_1004/*.lmdb`); their
records already contain ring systems, ECFP/MACCS and labels.

## Leakage control

* External test, validation and test are exact-canonical-SMILES and stereo-stripped
  disjoint from all training corpora (0 overlapping molecules).
* `pubchem_1209` was filtered by the same rule (77,418 → 77,328; 90 removed).
* The external positives are never used for threshold selection; τ = 0.50 was fixed
  a priori on the raw score.
* External record #57 was corrected from the 3-ethyl ester to the 3-carboxylic acid
  (15 → 14 peaks); the correction is reflected in the JSON and all LMDBs.
