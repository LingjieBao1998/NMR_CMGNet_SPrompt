# analysis — post-hoc analysis

Everything here runs **after** training/inference.  The training entry points
(`main_cmgnet.py`, `train_xgb_cress_new_dataset.py`) do not import this package.

| Folder | Contents |
|---|---|
| `metrics/` | top-N / ExactMatch summaries, seed aggregation, model-comparison tables, XGBoost reports, `summarize_core_smiles.py` |
| `errors/` | FP/FN case studies, core-matching checks, attention-head selection |
| `embedding/` | CReSS embeddings, UMAP/retrieval, cosine and Wasserstein comparisons |
| `inference/` | forward-pass examples, single-spectrum inference, one-off reruns |
| `legacy/` | superseded model variants and prototypes kept for provenance |

## Running

Most scripts are CLI tools.  Run them as modules from the repository root so
that `infer`, `utils`, `model` and `new_RingSystemFinder` stay importable:

```bash
PYTHONPATH=. python -m analysis.metrics.summarize_core_smiles \
    output/summary.md "CMGNet-SPrompt" output/test_sprompt_seed0/*.json

PYTHONPATH=. python -m analysis.metrics.aggregate_cmgnet_topn_stats
PYTHONPATH=. python -m analysis.errors.analyze_core_confusion
PYTHONPATH=. python -m analysis.embedding.nmr_distance_comparison_100
```

Some scripts still read `output/` or absolute paths from the original working
tree; check the header before running.  The maintained, repo-portable paths are
the ones in `scripts/` and `reports/`.

## Figures

Paper/SI figures are in `reports/` (see `reports/README.md`):

```bash
bash scripts/plot/make_all_figures.sh    # all default figures (CPU, ~5 min)
python reports/make_ci_external_figure.py
python reports/make_uncertainty_figure.py
```

Calibration/reliability analysis is archived in
`reports/legacy_calibration/`; the default pipeline keeps the CI analysis.

## Calibration note

The per-molecule 90 % interval is a **calibration-map bootstrap**: the Platt
map is refitted on validation resamples.  The CI figure therefore still uses the
calibration map internally, but no standalone calibration figure or reliability
metrics are produced by the default pipeline.
