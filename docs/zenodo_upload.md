# Zenodo upload plan

Status: draft for the v1.0.0 submission. Sizes are from the release tree on
2026-11-16.

Zenodo allows **50 GB per record**.  The release tree is ~37.5 GB once the
Docker image and runtime outputs are excluded, so the upload is split into
records with different licences and different citation roles:

| Record | Contents | Size | Role |
|---|---|---:|---|
| **1. Software** | code, configs, docs, result tables, figures, figure caches, prediction JSON | ~400 MB | `[Software]` citation |
| **2. Data** | processed LMDB corpora + JSON splits + manifest | 1.6–33 GB (see below) | `[Data]` citation |
| **3. Weights** | CMGNet / BART / CReSS checkpoints + XGBoost heads | ~4.1 GB | can be merged into Record 1 if a single DOI is preferred |

`docker/my-nmr-service_v1.0-web.tar.gz` (6.8 GB) is **not** uploaded to Zenodo;
publish it on GHCR / Docker Hub and link the image digest in the README.

---

## Record 1 — Software

Build with:

```bash
bash scripts/package_zenodo.sh          # writes dist/zenodo/*.tar.gz + SHA256SUMS
```

Contents (everything tracked by git):

| Path | Size | Note |
|---|---:|---|
| `main_cmgnet.py`, `infer.py`, `train_xgb_cress_new_dataset.py`, `summarize_core_smiles.py` | ~200 KB | entry points |
| `scripts/` | 44 KB | train / infer / plot / build_lmdb |
| `data_process/` | 188 KB | raw-source parsers |
| `model/`, `utils/`, `plot/` | ~4 MB | model + helpers |
| `nmr_core_system/` | 20 MB | Django service (includes RDKit wasm) |
| `results/` | 82 MB | tables, figures, shipped predictions |
| `reports/` | 120 MB | figure scripts, captions, calibration caches |
| `viz_tmp/` (8 tracked arrays) | 111 MB | required to regenerate figures |
| `data/**/*.json`, `data/manifest.json`, `data/README.md` | 47 MB | split definitions and schemas |
| `notebook/`, `docs/`, `README.md` | 1 MB | analysis notebooks and documentation |
| `requirements.txt`, `Dockerfile` | small | environment |

Excluded by `.gitignore` and therefore absent from the archive:

```
data/**/*.lmdb/        (33 GB)
checkpoints/*.ckpt     (4.1 GB)
docker/*.tar.gz        (6.8 GB)
output/                (2.8 GB)
logs/  .env  __pycache__/
```

## Record 2 — Data

Upload the LMDB directories and the JSON splits, preserving the layout.

### Licensing status of the synthetic corpus

The 1,488,432-record `synth_predicted.lmdb` is a union of three documented
sources plus 313,343 records relabelled `NMRium` (the former `napacd` source):

| `source` | Records | Licence status |
|---|---:|---|
| `multi` | 789,028 | **no licence recorded** in the legacy tree |
| `NMRium` (313,343 formerly `napacd`) | 313,343 | **unverified provenance**; excluded |
| `acd_lotusv9` | 194,550 | **CC BY 4.0** — Zenodo 10.5281/zenodo.8175939 |
| `pnmrnp2` | 191,511 | **no licence recorded** |

`acd_lotusv9` is the LOTUS v9 structure database with ACD/Labs-predicted 13C
shifts (Nuzillard & Kuhn, 2023).  The ACD/Labs **software** is commercial, but
the **dataset** is published under CC BY 4.0, so it can be redistributed with
attribution.  The 313,343 `NMRium`-labelled records are NMRium-derived and are
excluded from release; `multi` and `pnmrnp2` have no documented licence.

### Scenario A — all remaining sources cleared (33 GB)

The full file is published only after `multi` and `pnmrnp2` are cleared **and**
the 313,343 `NMRium`-labelled records are filtered out.  Rebuild with:

```bash
python scripts/build_lmdb/filter_lmdb_by_source.py \
    --src data/synthetic/synth_predicted.lmdb \
    --sources multi,acd_lotusv9,pnmrnp2 \
    --out data/synthetic/synth_predicted_release.lmdb
```

The release file then has 1,175,089 records (~23 GB); the training file
`synth_predicted.lmdb` keeps 1,488,432 records for reproducibility of the
reported models.

| Path | Size | Records |
|---|---:|---:|
| `data/experimental/exp_lit.lmdb` | 928 MB | 49,984 |
| `data/experimental/ex_train.lmdb` + JSON | 107 MB | 24,520 |
| `data/synthetic/synth_predicted_release.lmdb` | ~23 GB | 1,175,089 |
| `data/pubchem/pubchem_1209.lmdb` | 1.5 GB | 77,328 |
| `data/validation/validation.lmdb` + `.json` | 171 MB | 40,667 |
| `data/external_test/external_test.lmdb` + JSON | 19 MB | 4,502 |
| `data/nmrmind/*.lmdb` + JSON | 1.4 GB | 2.2 M |
| `data/f_split/*.json` | 40 KB | 48 + 48 |
| `data/manifest.json`, `data/README.md` | 8 KB | — |

Check `exp_lit` (HMDB / scraped sources), `validation` and `nmrmind` separately;
they are not covered by the ACD/Lotus CC BY 4.0 licence.

### Scenario B — only ACD/Lotus cleared (recommended default, ~5.5 GB)

Keep the public evaluation data and the cleared ACD/Lotus subset; publish
record indices for the rest.

| Path | Size | Records |
|---|---:|---:|
| `data/synthetic/synth_acd_lotusv9.lmdb` (built with `filter_lmdb_by_source.py`) | ~3.8 GB | 194,550 |
| `data/pubchem/pubchem_1209.lmdb` | 1.5 GB | 77,328 |
| `data/external_test/external_test.lmdb` + JSON | 19 MB | 4,502 |
| `data/experimental/ex_train.lmdb` + JSON (nmrshiftdb2, share-alike) | 107 MB | 24,520 |
| `data/f_split/*.json` | 40 KB | 48 + 48 |
| `data/manifest.json`, `data/README.md` | 8 KB | — |
| per-source index CSV (no spectra) for `multi` / `pnmrnp2` / `NMRium` | ~120 MB | 1.29 M |

### Scenario C — minimum (~1.6 GB)

Only unambiguously redistributable data:

* `data/external_test/*` (19 MB)
* `data/pubchem/pubchem_1209.lmdb` (1.5 GB)
* `data/f_split/*.json` (40 KB)
* `data/manifest.json`, `data/README.md`
* per-source index CSV for every restricted corpus

Commands:

```bash
# build a source-specific subset without touching the spectra
python scripts/build_lmdb/filter_lmdb_by_source.py \
    --src data/synthetic/synth_predicted.lmdb \
    --sources acd_lotusv9 \
    --out data/synthetic/synth_acd_lotusv9.lmdb

# redistribution-safe index (no C_nmr, no fingerprints)
python scripts/build_lmdb/export_source_index.py \
    --src data/synthetic/synth_predicted.lmdb \
    --out data/_index/synth_predicted_index.csv
```

`data/_source/new_db.lmdb` is a symlink to the legacy working tree
(`/mnt/data2/NMR_Core_Mol_Detection/...`).  **Do not upload the symlink**; it is
not a self-contained dataset.  Either document the upstream accession list or
rebuild it from the per-source files.

## Record 3 — Weights

| Path | Size |
|---|---:|
| `checkpoints/sprompt_ex1209_1ep_seed0.ckpt` | 413 MB |
| `checkpoints/sprompt_ex1209_1ep_seed2006.ckpt` | 413 MB |
| `checkpoints/sprompt_ex1209_5ep_seed42.ckpt` | 413 MB |
| `checkpoints/train_cmgnet_ring_system_3.2_1029_v2_best.ckpt` | 1.24 GB |
| `checkpoints/1220_last.ckpt` | 1.24 GB |
| `checkpoints/bart_base_0910/` | 395 MB |
| `checkpoints/cress/` | 178 MB |
| `models/xgboost/*.pkl` (4 files) | 17 MB |

`checkpoints/1220_last.ckpt` is only needed for the legacy baseline; it can be
dropped to save 1.24 GB.

---

## Do **not** upload

| Path | Reason |
|---|---|
| `docker/*.tar.gz` | 6.8 GB; use a container registry |
| `output/` | 2.8 GB runtime outputs; published predictions are in `results/predictions/` |
| `logs/`, `__pycache__/` | runtime |
| `nmr_core_system/.env` | live SMTP credential (also purge from git history) |
| `data/_source/new_db.lmdb` | broken absolute symlink |
| `data/nmrmind/*` (if licence unclear) | third-party redistribution |
| `data/synthetic/*` (only the cleared subset) | see Scenarios A–C |

---

## Pre-upload checklist

- [x] Add `LICENSE` (Apache-2.0) and `THIRD_PARTY_NOTICES.md`.
- [x] Add `CITATION.cff` and `.zenodo.json` (author list and metadata).
- [x] Relabel the 313,343 `napacd` records to `source = "NMRium"` in
      `data/synthetic/synth_predicted.lmdb` (spectra and labels unchanged).
- [ ] Confirm `multi` / `pnmrnp2` terms; choose Scenario A or B.
- [ ] Purge `nmr_core_system/.env` from git history and rotate the SMTP password.
- [ ] Tag the release: `git tag -a v1.0.0 -m "..." && git push origin v1.0.0`.
- [ ] Generate `SHA256SUMS` for every archive; attach it to the record.
- [ ] Reserve the DOI before submitting the manuscript; put it in the Data
      Availability Statement and in `README.md`.
- [ ] Add the GitHub URL, Zenodo DOI and Docker image digest to `README.md`.
- [ ] Verify that a fresh clone can run the CPU smoke test and `make_all_figures.sh`.
- [ ] Check that every figure in `results/figures/` is regenerated from the
      shipped caches (no path into `/mnt/data2/...`).

## Data Availability Statement (template)

> The source code, trained weights and processed data splits for this study are
> available at https://github.com/[TODO]/[TODO] (version v1.0.0) and archived at
> https://doi.org/[TODO].  The experimental spectra are derived from public
> resources (nmrshiftdb2, BMRB, HMDB, PubChem, NMR BANK and the sources listed in
> `data/README.md`).  The ACD/Lotus predicted-shift subset is redistributed
> under CC BY 4.0 (Nuzillard & Kuhn, Zenodo 10.5281/zenodo.8175939); the
> nmrshiftdb2-derived splits are released under the nmrshiftdb2 Database
> License.  The 313,343 records originally labelled `napacd` are relabelled
> `NMRium` and are not redistributed.  Predicted records whose original terms do
> not permit redistribution are provided as SMILES/record indices together with
> the rebuild scripts, and the full records are available from the corresponding
> author on reasonable request.  The Docker image for the web service is
> available at [registry/image:tag].
