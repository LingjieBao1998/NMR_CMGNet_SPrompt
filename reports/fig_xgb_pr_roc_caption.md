# Figure Sx — caption text (SI)

## Full caption

**Figure Sx.** Precision–recall (AUPRC) and receiver operating characteristic
(AUROC) curves of the XGBoost scaffold classifier on the external test set.
**(A)** Precision–recall curve (step-interpolated) for the 4,502-spectrum
external test set (64 positives, 4,438 negatives; prevalence 1.42%), summarised
by average precision (AUPRC = 0.9891; 95% bootstrap CI 0.9713–0.9996). The grey
dashed line is the chance level (0.0142); grey dash-dotted curves are iso-F1
contours (0.80, 0.90, 0.95). **(B)** ROC curve on a logarithmic false-positive-
rate axis; AUROC = 0.9998 (95% bootstrap CI 0.9995–1.0000). Shaded bands are
95% percentile bootstrap intervals (1,000 resamples). The red marker is the
operating point at the default threshold τ_op = 0.50: TP 63, FP 4, FN 1,
TN 4,434 (precision 0.940, recall 0.984, F1 = 0.9618). At τ_op = 0.40 one
additional borderline negative (p = 0.418) enters, giving 5 FP and F1 = 0.9545;
both conventions are stated in the response.

## Short caption

**Figure Sx.** AUPRC (A) and AUROC (B) of the XGBoost classifier on the external
test set (n = 4,502; 64 positives). AUPRC = 0.9891 (95% CI 0.9713–0.9996);
AUROC = 0.9998 (95% CI 0.9995–1.0000). The red marker is τ_op = 0.50
(TP 63 / FP 4 / FN 1 / TN 4,434; F1 = 0.9618).

## Numbers used (τ = 0.50)

| Quantity | Value |
|---|---|
| Test spectra | 4,502 (64 positive, 4,438 negative) |
| Prevalence | 0.0142 |
| AUPRC (average precision) | 0.9891 (95% CI 0.9713–0.9996) |
| AUROC | 0.9998 (95% CI 0.9995–1.0000) |
| Operating threshold τ_op | 0.50 |
| τ_op = 0.50 | TP 63 / FP 4 / FN 1 / TN 4,434 |
| Precision / recall / F1 | 0.940 / 0.984 / 0.9618 |
| τ_op = 0.40 (previous convention) | TP 63 / FP 5 / FN 1 / TN 4,433; F1 = 0.9545 |
