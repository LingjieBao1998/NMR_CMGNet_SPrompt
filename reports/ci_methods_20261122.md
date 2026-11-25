# Confidence intervals — methods text (English)

Implementation: `analysis/metrics/ci_bootstrap_4502.py`; output:
`results/ci_bootstrap_4502.md`. Parameters below are the ones actually used.

---

## Main-text version (~150 words)

All intervals on the 4,502-spectrum external set are 95% percentile bootstrap
intervals obtained by resampling spectra, not model fits. Because the set is
strongly imbalanced (64 positives, 4,438 negatives), resampling is stratified:
each replicate draws 64 positives with replacement and 4,438 negatives with
replacement, so the class sizes — and hence the prevalence of 1.42% on which
precision depends — are held fixed. For each replicate the contingency table is
rebuilt from the per-spectrum hit indicators and Precision, Recall, F1,
Specificity, Accuracy and top-k SMILES accuracy are recomputed; the reported
interval is the 2.5th–97.5th percentile of 10,000 such replicates (fixed seed).
For models trained with three seeds the metric is computed for each seed on the
same replicate and then averaged, so the interval describes the mean-over-seeds
estimator, while seed-to-seed scatter is reported separately as ±s.d. Intervals
therefore quantify sampling uncertainty of the test set only; they exclude
training variability and structure-assignment ambiguity.

## SI / methods-section version (~330 words)

Confidence intervals on the external test set (n = 4,502; 64 positives,
4,438 negatives) were obtained by nonparametric bootstrap over spectra. A
percentile bootstrap was preferred to a normal (Wald) approximation because
several reported quantities lie close to the boundary — Specificity ≈ 0.999 and
Accuracy ≈ 0.998 — where Wald limits exceed unity and are markedly too narrow,
and because Precision and F1 have a random denominator (TP + FP) for which no
closed-form variance exists.

Resampling was stratified by class: in each of B = 10,000 replicates, 64
positives were drawn with replacement from the 64 positive spectra and 4,438
negatives with replacement from the 4,438 negative spectra, keeping the class
sizes — and therefore the 1.42% prevalence that enters Precision — fixed.
Implementation-wise each replicate is a pair of multinomial weight vectors over
the per-spectrum hit indicators, which is algebraically equivalent to drawing
with replacement. For every replicate the contingency table was rebuilt and
Precision, Recall, F1, Specificity and Accuracy recomputed from it; top-k SMILES
accuracy was recomputed as the fraction of positives for which the canonical
reference SMILES was reproduced exactly among the k candidates. The 95%
interval is the 2.5th–97.5th percentile of the B replicate values, without
bias correction (no BCa); the random seed was fixed (20261122) so the intervals
are exactly reproducible.

Models trained with three seeds (SCAFFOLD, SMILES, SPrompt) were handled by
computing each seed's metric on the same replicate and then averaging over the
three seeds, matching the mean-over-seeds estimator reported in the tables. The
interval thus reflects test-set sampling variability of that averaged metric;
run-to-run variability is reported separately as the standard deviation over
seeds (ddof = 1), and the two sources are deliberately not pooled. The
single-run model (CMGNet-0910) yields an interval that reflects test-set
sampling only.

As a cross-check, single-proportion quantities (Recall, Specificity, Precision,
Accuracy) were also given Wilson score intervals (z = 1.96), which are
closed-form and do not require resampling; bootstrap and Wilson limits agreed
to within two percentage points at every reported k (differences are largest
for Recall, where only 64 positives are available). For AUPRC and AUROC, where the
resampling unit is the ranked score rather than the contingency table,
intervals come from 2,000 unstratified replicates of all 4,502 rows using the
mean of the three seed probabilities; replicates containing a single class were
discarded.

Finally, these intervals are conditional on the composition of the evaluation
set: they quantify how much the reported metric would move under a different
draw of 4,502 spectra from the same population, holding the models and the
core-structure criterion (τ = 0.50; a hit requires the full quinolone core in at
least one of the top-k candidates) fixed. They do not represent uncertainty in
the trained weights, in the reference annotations, or in the choice of
criterion.

---

## Parameter sheet (for reproducibility statements)

| Item | Value |
|---|---|
| Resampling unit | spectrum (molecule-level hit indicator) |
| Scheme | stratified by class (positives and negatives resampled separately) |
| Replicates B | 10,000 (AUPRC/AUROC: 2,000, unstratified) |
| Random seed | 20261122 (`numpy.random.default_rng`) |
| Interval type | percentile, 2.5th–97.5th, no BCa |
| Multi-seed handling | metric per seed per replicate, then mean over 3 seeds |
| Cross-check | Wilson score interval, z = 1.96 |
| Not included | training/seed variability (±s.d. reported separately), annotation error, criterion sensitivity |
