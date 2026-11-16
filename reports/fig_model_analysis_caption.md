# Figure 2 — combined model analysis (τ = 0.50) — caption

File: `reports/fig_model_analysis.pdf` / `.png` (600 dpi)
Generator: `reports/make_model_analysis_figure.py`

## Full caption

**Figure 2.** Model analysis at the default decision threshold τ = 0.50.
**(A)** Precision–recall curve on the external test set (step-interpolated;
4,502 spectra, 64 positives; prevalence 1.42%); average precision
AUPRC = 0.9891 (95% bootstrap CI 0.9713–0.9996). The red point is the operating
point. **(B)** ROC curve on a logarithmic false-positive-rate axis;
AUROC = 0.9998 (95% CI 0.9995–1.0000); shaded bands in A and B are 95%
bootstrap intervals (1,000 resamples). **(C)** Reliability diagram of the
external scores (log–log axes; Wilson 95% intervals per bin) for raw scores and
for Platt scaling fitted on the validation split and prior-corrected to the
external sampling prevalence; the diagonal is perfect calibration.
**(D)** Uncertainty-guided review at τ = 0.50: of 64 positives, 59 are
auto-positive, 5 are referred and 0 are missed; of 4,438 negatives, 2 are
auto-positive, 16 are referred and 4,420 are auto-negative. Referral rate
0.47% (95% CI 0.31–0.71%), silent misses 0; referred spectra require MS
confirmation and are never reported as negative. **(E)** MC-Dropout generator
variability (T = 20 passes): quinolone-core pass frequency versus mean pairwise
CReSS cosine of the generated candidates, grouped by classifier confidence.
Confident positives show consensus (core in every pass, cosine 0.80), whereas
borderline positives and false positives are dispersed (cosine 0.49–0.51); the
false negative #12 never produces a quinolone core (0/20 passes).
**(F)** Accuracy per ring system at τ = 0.50 (axis 90–100%). Accuracy is
≥98.44% for every system; the lowest values are the quinolone core (1 FN/64)
and quinolinone-like negatives (1 FP/69), while pyridine, diazine/triazine,
saturated N-ring and acyclic systems have 0 errors.

## Short caption

**Figure 2.** Model analysis at τ = 0.50: **(A)** PR and **(B)** ROC curves
(AUPRC 0.9891, AUROC 0.9998; 95% bootstrap bands); **(C)** calibration of the
external scores (raw vs Platt + prior); **(D)** review outcome (59 auto-positive
and 5 referred positives; 2 auto-positive and 16 referred negatives; 0 silent
misses); **(E)** MC-Dropout generator variability (core-pass frequency vs CReSS
dispersion); **(F)** accuracy per ring system (all ≥98.44%).
