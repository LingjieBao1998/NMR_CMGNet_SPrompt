# data_process — legacy source parsers

These scripts document how each raw source was converted into the JSON records
that were later merged into `new_db.lmdb`.  They are kept for provenance and
are **not** part of the maintained, repo-portable pipeline.

> **Warning:** most files still contain absolute paths from the original
> working tree (`/home/baolingjie/...`, `/mnt/data2/...`).  They will not run
> as-is on another machine.  Search for `/home/` and `/mnt/` and replace the
> paths with your local copies before using them.

| Script | Source | Notes |
|---|---|---|
| `process_acd_lotusv9.py` | ACD/Lotus v9 SDF | reads `NMREDATA_ASSIGNMENT`; dataset is CC BY 4.0 (Zenodo 10.5281/zenodo.8175939) |
| `process_pnmrnp2.py` | `pnmrnp2` SDF | predicted spectra |
| `process_bmrb.py` | BMRB | experimental |
| `process_c13_naproc.py` | c13.materia-medica.net | experimental |
| `process_GlycoNMR.py` | GlycoNMR | experimental |
| `process_hmdb.py` | HMDB | experimental |
| `process_Spektraris_NMR.py` | Spektraris NMR | experimental |
| `process_NMRium_1014.py` | NMRium upload collection | excluded from the release |
| `process_nmrshiftdb_v2.py` | nmrshiftdb2 | experimental + extra training |
| `process_vliad_and_test_1020.py` | evaluation splits | test/validation |
| `merge_all_other_data_*.py` | merge step | legacy, several revisions |
| `extract_experimental_newdb_1024.py` | experimental/synthetic split | superseded by `scripts/build_lmdb/split_newdb_by_source.py` |

For the maintained pipeline use:

```bash
python scripts/build_lmdb/split_newdb_by_source.py   # source LMDB -> exp / synth
python scripts/build_lmdb/json_to_lmdb.py             # JSON -> featurised LMDB
python scripts/build_lmdb/filter_lmdb_by_source.py    # source subsets
python scripts/build_lmdb/export_source_index.py      # redistribution-safe index
```
