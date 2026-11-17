# legacy — archived material

Nothing in this folder is required to train, infer or reproduce the released
figures.  It is kept so that the original experiment history remains auditable.

| Folder | Contents |
|---|---|
| `tests/` | ad-hoc `test_*.py` scripts; not a pytest suite |
| `data_scripts/` | old scrapers/parsers moved out of the repository root |
| `plot/` | old enumeration notebooks and exploratory plots |

Most scripts contain absolute paths from the original working tree
(`/home/...`, `/mnt/data2/...`) and are not runnable as-is.  The maintained
pipeline is in `scripts/`, `analysis/` and `reports/`.
