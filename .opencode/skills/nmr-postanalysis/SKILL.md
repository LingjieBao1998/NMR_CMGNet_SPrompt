---
name: nmr-postanalysis
description: Run post-hoc analysis and figures for the CMGNet-SPrompt NMR repository — top-N metrics, error cases, embeddings/retrieval, confidence intervals and paper figure regeneration.
---

# NMR post-analysis

Use this skill when the task is to analyse CMGNet-SPrompt results or regenerate
paper/SI figures.  Training entry points are `main_cmgnet.py` and
`train_xgb_cress_new_dataset.py`; they are not needed for these analyses.

## Repository map

| Path | Purpose |
|---|---|
| `analysis/metrics/` | top-N / ExactMatch, seed aggregation, model comparison |
| `analysis/errors/` | FP/FN cases, core matching, attention heads |
| `analysis/embedding/` | CReSS embeddings, UMAP, cosine / Wasserstein |
| `analysis/inference/` | forward-pass examples and reruns |
| `reports/` | paper/SI figures, CI analysis, cached calibration arrays |
| `scripts/plot/make_all_figures.sh` | one-command figure regeneration |
| `data/` | LMDB/JSON splits (see `data/README.md`) |
| `results/` | shipped tables, figures and predictions |
| `legacy/` | archived launchers/tests; do not use for release runs |

## Common commands

```bash
# all default figures (CPU, cached inputs, ~5 min)
bash scripts/plot/make_all_figures.sh

# confidence-interval figures (calibration-map bootstrap)
python reports/make_ci_external_figure.py
python reports/make_uncertainty_figure.py

# top-N / ExactMatch summary for a prediction JSON
PYTHONPATH=. python -m analysis.metrics.summarize_core_smiles \
    output/summary.md "CMGNet-SPrompt" output/test_sprompt_seed0/*.json
```

Run analysis modules with `PYTHONPATH=.` from the repository root so that
`infer`, `utils`, `model` and `new_RingSystemFinder` resolve.

## Calibration policy

Calibration/reliability analysis is archived in
`reports/legacy_calibration/`; the default pipeline reports **confidence
intervals only**.  The CI is still a calibration-map bootstrap internally (the
Platt map is refitted on validation resamples), so do not remove the calibration
map from the CI calculation — only the standalone calibration figure/metrics.

## Data and licensing

- ACD/Lotus-derived records: CC BY 4.0, cite Zenodo 10.5281/zenodo.8175939.
- `multi`, `napacd`, `pnmrnp2`: no documented licence; do not publish raw
  records.  Use `scripts/build_lmdb/export_source_index.py` for indices.
- Never upload `data/_source/new_db.lmdb` (absolute symlink) or `.env`.

## Outputs

Figures are written to `reports/`; shipped copies live in `results/figures/`.
Keep `FP = #D55E00` diamond, `FN = #0072B2` star, and the τ = 0.50 raw-score
convention when editing figures.
