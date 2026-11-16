# Zenodo upload plan

Status: draft for the v1.0.0 submission. Sizes are from the release tree on
2026-11-16.

Zenodo allows **50 GB per record**.  The release tree is ~37.5 GB once the
Docker image and runtime outputs are excluded, so the upload is split into two
records with different licences and different citation roles:

| Record | Contents | Size | Role |
|---|---|---:|---|
| **1. Software** | code, configs, docs, result tables, figures, figure caches, prediction JSON | ~400 MB | `[Software]` citation |
| **2. Data** | processed LMDB corpora + JSON splits + manifest | ~33 GB | `[Data]` citation |
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

Upload the LMDB directories and the JSON splits, preserving the layout:

| Path | Size | Records | Redistributable? |
|---|---:|---:|---|
| `data/experimental/exp_lit.lmdb` | 928 MB | 49,984 | nmrshiftdb2 Database License (share-alike) |
| `data/experimental/ex_train.lmdb` | 97 MB | 24,520 | same |
| `data/experimental/ex_train.json`, `ex_train.ring.json` | 10 MB | 24,520 | same |
| `data/synthetic/synth_predicted.lmdb` | 29 GB | 1,488,432 | **check ACD/Lotus and pnmrnp2 terms first** |
| `data/pubchem/pubchem_1209.lmdb` | 1.5 GB | 77,328 | PubChem terms (public domain) |
| `data/validation/validation.lmdb` + `.json` | 171 MB | 40,667 | derived; nmrshiftdb2 share-alike |
| `data/external_test/external_test.lmdb` + `.json` + `_pos_only.json` | 19 MB | 4,502 | check each original source |
| `data/nmrmind/*.lmdb` + `validation.json`, `test.json` | 1.4 GB | 2.2 M | check NMRMind upstream licence |
| `data/f_split/*.json` | 40 KB | 48 + 48 | derived |
| `data/manifest.json`, `data/README.md` | 8 KB | — | — |

`data/_source/new_db.lmdb` is a symlink to the legacy working tree
(`/mnt/data2/NMR_Core_Mol_Detection/...`).  **Do not upload the symlink**; it is
not a self-contained dataset.  Either document the upstream accession list or
rebuild it from the per-source files.

If the ACD/Lotus or NMRMind terms forbid redistribution, publish only:

* the SMILES / record-key list per source,
* the `source` field mapping,
* `scripts/build_lmdb/*.py` and a download/rebuild note,

and state in the paper that the restricted spectra are available from the
corresponding author under a data-use agreement.

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
| `data/synthetic/*` (if ACD/Lotus restricted) | third-party redistribution |

---

## Pre-upload checklist

- [ ] Choose and add `LICENSE` (code) and a data licence statement.
- [ ] Add `THIRD_PARTY_NOTICES.md` (nmrshiftdb2, CDK, ChemTSv2, each data source).
- [ ] Add `CITATION.cff` with authors, ORCID, affiliation, version.
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
> `data/README.md`); the derived corpora are released under the nmrshiftdb2
> Database License.  The predicted spectra in `data/synthetic/` include
> third-party predictions that cannot be redistributed; the corresponding
> SMILES lists and rebuild scripts are provided, and the full records are
> available from the corresponding author on reasonable request.  The Docker
> image for the web service is available at [registry/image:tag].
