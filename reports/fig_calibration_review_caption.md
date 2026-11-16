# Fig. Sy — calibration + uncertainty-guided review — caption (SI)

Files:
- `reports/fig_calibration_review.pdf` / `.png` (600 dpi)
- generator: `reports/calibrate_and_review.py`
- numbers: `reports/calibration_review_metrics.json`

## Full caption

**Figure Sy.** Probability calibration and the uncertainty-guided review rule of
the XGBoost classifier at the default threshold τ = 0.50. **(A–C)** Reliability
diagrams on log–log axes (Wilson 95% intervals per probability bin). **(A)**
Validation split (n = 40,667; 37 positives); **(B)** internal test at natural
prevalence (40,642; 37 / 40,605); **(C)** enriched external test (4,502;
64 / 4,438; 1.42% by design). Raw scores are overconfident at natural prevalence
(ECE 4.7×10⁻³ in B); Platt scaling fitted on validation corrects this (ECE
1.7×10⁻⁵); on the enriched external set the validation calibrator is
prior-corrected with the known sampling prevalences (ECE 1.1×10⁻³). **(D)**
Empirical survival curves of the external scores with τ_op = 0.50 and the
two-sided review band [0.182, 0.818] (±1.5 logit). **(E)** Review outcome:
59 true positives and 2 false positives are auto-positive (precision 96.7%);
5 positives and 16 negatives are referred (0.47%, 95% CI 0.31–0.71%);
4,420 negatives are auto-negative; silent misses = 0. **(F)** Calibration error
(raw vs calibrated) on the three sets.

## Short caption

**Figure Sy.** Reliability diagrams (A–C), external score distributions with the
review band at τ_op = 0.50 (D), review outcome (E) and calibration error (F).
Platt scaling is fitted on validation; prior correction is applied to the
enriched external set. Referral rate 0.47% (5 positives, 16 negatives); silent
misses 0; auto-positive precision 96.7%.

## Key numbers

| Quantity | Internal test | External test |
|---|---|---|
| n / pos / neg | 40,642 / 37 / 40,605 | 4,502 / 64 / 4,438 |
| ECE raw | 4.71×10⁻³ | 6.19×10⁻⁴ |
| ECE calibrated | 1.74×10⁻⁵ (Platt) | 1.11×10⁻³ (Platt + prior) |
| Brier raw → calibrated | 2.71×10⁻³ → 3.59×10⁻⁴ | 1.01×10⁻³ → 1.27×10⁻³ |
| Review referral | 5.52% (2,242; 4 pos) | 0.47% (21; 5 pos) |
| Auto-positive TP/FP | 32 / 59 | 59 / 2 |
| Silent misses | 1 | 0 |
