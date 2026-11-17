# Figures and statistics

Every figure in the paper/SI can be regenerated from the artefacts shipped in
this directory.  No GPU is needed: the CReSS embeddings, classifier
probabilities and UMAP coordinates are cached.

```bash
bash scripts/plot/make_all_figures.sh        # all figures -> reports/
PY=... bash scripts/plot/make_all_figures.sh # override interpreter
```

Individual scripts resolve paths relative to the repository root (override with
`NMR_ROOT=/path/to/repo`).

## Figure inventory

| Script | Output | Content |
|---|---|---|
| `make_figure_model_analysis_v5.py` | `fig_model_analysis_v5.png/.pdf` | **Figure 2**: (A) global CReSS embedding (training negatives/positives + 4,502 external spectra, FP diamonds, FN star); (B) local UMAP within R = 0.78 of FN #12; (C) FN #12 and its nearest neighbours; (D) probability along the embedding path (boundary at α ≈ 0.20); (E) test set coloured by 90 % CI width; (F) per-molecule 90 % intervals for the 64 positives and 4 FP; (G) accuracy per ring system; (H) representative ring systems |
| `make_ci_external_figure.py` | `fig_ci_external.png/.pdf` | probability with its 90 % bootstrap interval for every external-test molecule (forest plot, log scale) |
| `make_xgb_pr_roc.py` | `fig_xgb_pr_roc.png/.pdf/.svg`, `xgb_pr_roc_metrics.json`, `fig_xgb_pr_roc_curves.npz` | PR and ROC curves with 95 % bootstrap bands; operating point τ = 0.50 (TP 63 / FP 4 / FN 1 / TN 4,434, F1 = 0.9618) |
| `make_hit_rate_figure.py` | `fig_hit_rate_summary.png/.pdf` | top-10 core-hit rate and exact@10 for all models (dumbbell) |
| `make_3d_hit_chart.py` | `fig_hit_rate_3d.png/.pdf` | 3-D view of top-10 hit rate |
| `make_nhetero_systems_figure.py` | `fig_nhetero_systems.png/.pdf` | accuracy per N-hetero ring system |
| `make_mc_dropout_figure.py` | `fig_mc_dropout.png/.pdf` | MC-dropout uncertainty of the CReSS encoder |
| `make_p_distribution_figure.py` | `fig_p_distribution.png/.pdf` | per-molecule probability distribution |
| `make_uncertainty_figure.py` | `fig_xgb_uncertainty.png/.pdf` | 3-seed spread + 90 % conformal prediction sets |
| `make_fp_structures_figure.py` | `fig_fp_structures.png`, `fp_struct_*.png` | structures of the four false positives |
| `make_fn12_case_figure.py` | `fig_fn12_case.png` | FN #12 case study |
| `make_fn12_local_figure.py` | `fn12_local_boundary.png` | local decision boundary around FN #12 |
| `make_paper_figure_fn12.py` | `fig_paper_fn12_composite.png/.pdf` | FN #12 composite in paper layout |
| `make_r2_qualitative_figure.py` | `fig_r2_qualitative.png` | qualitative comparison of six cross-substituent families |
| `make_model_analysis_figure.py`, `make_figure_model_analysis_v3/v4.py` | `fig_model_analysis*.png/.pdf` | earlier layouts of Figure 2 (v5 is the published one) |
| `make_layout_proposal.py` | `fig_layout_proposal.png/.pdf` | wireframe layout proposal (no data) |

The standalone calibration/reliability figure and the calibration-review
metrics are **not part of the default pipeline**; they are archived in
`reports/legacy_calibration/`.  The CI figure still uses the Platt map internally
(the interval is a calibration-map bootstrap), but no separate calibration
figure is produced.
| `make_fn12_report_assets.py` | `fn12_peak_ablation.png` | peak-ablation panel for the FN #12 report |

## Cached artefacts

| Path | Producer | Content |
|---|---|---|
| `reports/calib_data/<split>_emb.npy` | `encode_eval_sets.py` | L2-normalised 768-d CReSS embeddings |
| `reports/calib_data/<split>.npz` | `encode_eval_sets.py` | `y`, `p_raw`, `smiles` |
| `reports/calib_data/<split>_p_interval.npz` | `uncertainty_ci.py` | `lo`, `hi`, `p` — 90 % calibration-bootstrap interval |
| `reports/calib_data/seed_probs.npz` | `uncertainty_ci.py` | per-seed calibrated probabilities |
| `reports/calib_data/apr18_cress_emb.npy` | `encode_eval_sets.py` | external-test embeddings used by the panels |
| `reports/assets/global_2x.png` | exported from the interactive embedding map | background image of panel A |
| `viz_tmp/coords_*.npy`, `viz_tmp/*_idx.npy`, `viz_tmp/*_labels.npy` | `prepare_tmap_cress.py` | UMAP coordinates / labels of the 3.30 M training set |
| `viz_tmp/train_emb_subset.npz` | subset of `train_emb.npy` | 41,961 embeddings in the FN #12 neighbourhood (59 MB instead of 4.8 GB) |
| `results/predictions/*.json` | CMGNet / NMR-Solver runs | prediction dumps used by the qualitative figures |
| `models/xgboost/model_xgb_cress_ex_train_seed42.pkl` | `train_xgb_cress_new_dataset.py` | classifier used by the panels |
| `models/xgboost/model_xgb_cress_ex_train_seed{0,2006}.pkl` | `train_xgb_cress_new_dataset.py` | the other two seeds, used for the 3-seed spread in `uncertainty_ci.py` |

## How the confidence intervals are computed

XGBoost returns a point probability only, so all intervals are resampling-based.

**1. Raw score.** `p_raw` = XGBoost `predict_proba[:, 1]` on the frozen CReSS
embedding (`encode_eval_sets.py`, `models/xgboost/model_xgb_cress_ex_train_seed42.pkl`).

**2. Calibration (fitted on the reserved validation split only).**
`logit(p_raw) → Platt` (`LogisticRegression(C=1e6)`), fitted on the 40,667
validation spectra (37 positives); Platt vs. isotonic is chosen by 5-fold
cross-fitted NLL on validation, never on external labels.  The fitted map is
`p_cal = σ(0.9542·logit(p_raw) − 4.716)`, so raw τ = 0.50 corresponds to
`p_cal ≈ 0.0089`.  The dashed reference in the figures is drawn at **0.0235**
instead: any calibrated threshold in the gap (0.0049, 0.0239] selects exactly
the same 67 external spectra (TP 63 / FP 4), and 0.0235 is the round value just
below the lowest-scoring false positive (0.0239).  Both numbers therefore encode
the same operating point (`calibration_review_metrics.json`).

**3. Per-molecule 90 % interval (Figures 2E–2F, `fig_ci_external`).**
`uncertainty_ci.py` draws `N_BOOT_CAL = 200` **stratified** bootstrap resamples
of the validation split (positives and negatives resampled separately, so the
class ratio is preserved), refits the Platt map on each resample, applies it to
the ensemble logit of the target spectrum, and reports the 5th/95th
percentiles:

```
for b in 1..200:  (a_b, b_b) = Platt(logit(p_val[idx_b]), y_val[idx_b])
lo = percentile_5 ({ sigmoid(a_b·logit(p) + b_b) })
hi = percentile_95({ sigmoid(a_b·logit(p) + b_b) })
```

This interval therefore quantifies **uncertainty of the calibration mapping**
(finite validation set, 37 positives), not measurement noise.  For confident
negatives the width is ~1e-9 (log scale grey in panel E); for positives the
median width is ~0.065.  Two of the four FP (#763, #4201) have intervals that
straddle 0.0235, which is why they are flagged for manual review.

**4. Model-instability spread.** The same embedding is scored by the three
independently trained seeds (0 / 42 / 2006); `xgb_uncertainty_metrics.json`
reports the mean s.d. and the 95th percentile of the per-sample range.

**5. Conformal prediction sets (90 %, Mondrian/class-conditional).**
Calibration scores come from 5-fold cross-fitted Platt probabilities on
validation, so no sample is scored by a map fitted on itself.  With
`n1` positives and `n0` negatives the quantile levels are
`ceil((n1+1)·0.9)/n1` and `ceil((n0+1)·0.9)/n0`; empirical coverage is reported
with a Wilson 95 % interval.

**6. Curve-level intervals.**
`make_xgb_pr_roc.py` uses 1,000 percentile bootstrap resamples of the external
test set for the PR/ROC bands and the AUPRC/AUROC intervals (AUPRC 0.9891,
95 % CI 0.9713–0.9996).  `legacy_calibration/calibrate_and_review.py` uses 1,000
stratified bootstrap resamples (2.5/97.5 percentiles) for ECE / Brier / NLL and
the reliability curve; that analysis is archived and not part of the default
pipeline.

**Regenerating the figures and the intervals**

The figures themselves are fully reproducible from the cached files shipped here:

```bash
bash scripts/plot/make_all_figures.sh
python reports/make_ci_external_figure.py
python reports/make_uncertainty_figure.py
```

The cached arrays were produced by (for provenance; these steps need the GPU
CReSS encoder and the 768-d embedding caches for the 40,667 validation and
40,642 internal-test spectra, which are too large to ship):

```bash
python reports/encode_eval_sets.py     # embeddings + raw probabilities
python reports/uncertainty_ci.py       # per-sample intervals, conformal, seeds
```

`reports/calib_data/*_emb.npy` and `<split>_emb.npy` are therefore inputs, not
outputs, of the shipped figure scripts.
