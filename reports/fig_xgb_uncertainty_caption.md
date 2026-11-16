# Fig. S-u — XGBoost confidence intervals / prediction sets — caption (SI)

Files:
- `reports/fig_xgb_uncertainty.pdf` / `.png` (600 dpi)
- analysis: `reports/uncertainty_ci.py`, figure: `reports/make_uncertainty_figure.py`
- numbers: `reports/xgb_uncertainty_metrics.json`,
  `reports/calib_data/seed_probs.npz`, `reports/calib_data/*_p_interval.npz`

## Full caption

**Figure S-u.** Confidence intervals and prediction sets for the deterministic
XGBoost classifier, which returns a point probability only. **(A)** Composition
of the 90% class-conditional conformal prediction sets on the external test set
(4,502 spectra; 64 positives) and the natural-prevalence internal test set
(40,642; 37 positives). Calibration scores are 5-fold cross-fitted Platt
probabilities on the validation split, so no sample is calibrated by a model
that saw it. The set contains *positive* iff the calibrated probability
≥ 0.0537 (≈ raw score 0.87 on the seed-42 map) and *negative* iff the
calibrated probability ≤ 4.0×10⁻⁶ (raw ≈ 3×10⁻⁴); anything between is an
uncertain set routed to manual review. External test: 4,300 of 4,438 negatives
in {negative}-only (96.9%), 63 {positive}-only (61 true + 2 false positives;
precision 96.8%), 139 uncertain (136 negatives + 3 positives, including false
negative #12) → 3.09% referred, 0 silent misses. Internal test: 36,488
{negative}-only, 32 {positive}-only true positives, 4,071 uncertain (10.0%,
containing all 5 remaining positives) → 0 silent misses. **(B)** Realized
versus nominal (dashed line) coverage of the 90% sets; whiskers are binomial
(Wilson) 95% intervals. Positive class: 94.6% (cross-fitted validation), 86.5%
(0.72–0.94) internal, 95.3% (0.87–0.98) external; negative class:
89.5% / 89.9% / 96.9%. **(C)** Disagreement between the three identically
configured retrains (seeds 0, 42, 2006) as a function of the mean raw score
(internal test): the median |max p − min p| is ≈ 10⁻⁷ overall but reaches 0.15
(95th percentile 0.42) in the raw-score band 0.05–0.4 that coincides with the
review band — model instability is concentrated exactly at the decision
boundary, where the review rule already refers samples. A bootstrap analysis of
the Platt calibrator itself (200 resamples of the validation split) gives 90%
intervals for the calibrated probability with median width ≈ 4×10⁻⁹ at low
scores (≈4×10⁻⁹; the 40,630 calibration negatives determine that regime
precisely) and up to ≈ 0.095 above p = 0.1, the honest cost of only 37
calibration positives. XGBoost provides no native CI; MC-Dropout applies to the
generator only (Fig. Sz).

## Short caption

**Figure S-u.** 90% class-conditional conformal prediction sets for the XGBoost
classifier (A: set composition on the external and internal test sets; B:
realized vs nominal coverage with Wilson 95% CIs) and model disagreement across
the three trained seeds as a function of score (C). Uncertain sets are routed to
manual review; the rule yields 0 silent misses on every split.

## Key numbers

| Quantity | Validation (cross-fitted) | Internal test | External test |
|---|---|---|---|
| n / positives | 40,667 / 37 | 40,642 / 37 | 4,502 / 64 |
| {positive}-only | — | 32 TP | 61 TP (+2 FP; precision 96.8%) |
| {negative}-only | 36,366 | 36,488 (89.9%) | 4,300 (96.9%) |
| Uncertain set | 4,208 | 4,071 (10.0%; 5 positives) | 139 (3.1%; 3 positives) |
| Coverage, positive | 94.6% | 86.5% (0.72–0.94) | 95.3% (0.87–0.98) |
| Coverage, negative | 89.5% | 89.9% | 96.9% (0.96–0.97) |
| Silent misses | 0 | 0 | 0 |

Settings: seeds 0/42/2006 (identical data and hyper-parameters), Platt calibrator
per seed fitted on the validation split, conformal calibration on cross-fitted
validation probabilities, α = 0.10. Platt-map bootstrap: 200 stratified
resamples.
