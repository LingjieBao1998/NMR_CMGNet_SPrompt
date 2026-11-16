# Fig. Sz — MC-Dropout generator variability — caption (SI)

Files:
- `reports/fig_mc_dropout.pdf` / `.png` (600 dpi)
- generator: `reports/mc_dropout_generator.py`, `reports/make_mc_dropout_figure.py`
- numbers: `reports/mc_dropout_generator.json`

## Full caption

**Figure Sz.** MC-Dropout variability of the CMGNet generator, computed on 21
external-test spectra grouped by classifier confidence (8 confident positives,
4 borderline positives, the false negative #12, the 4 false positives, 4
confident negatives). For each spectrum, T = 20 stochastic forward passes
(dropout p = 0.1 enabled) were run with beam search (beam 5, 5 returned
sequences per pass; 100 candidate generations per spectrum). **(A)** Frequency
of passes in which at least one of the top-5 candidates contains the quinolone
core, as a function of the XGBoost score; for every confident positive the core
is recovered in all 20 passes, whereas the false negative #12 (XGBoost score
0.012) never yields a quinolone-core candidate. **(B)** Modal top-1 frequency
(fraction of passes whose top-1 prediction equals the modal top-1 molecule).
**(C)** Structural dispersion, quantified as the mean pairwise cosine similarity
of CReSS SMILES embeddings (no fingerprint similarity) of the distinct valid
candidates generated per spectrum. Confident positives produce a compact,
reproducible candidate set (≈8–10 distinct Murcko scaffolds, mean pairwise
cosine 0.81), whereas borderline positives and false positives are dispersed
(≈15–34 scaffolds; 0.49–0.51). The generator's failure on #12 is complete and
noise-independent: across 20 stochastic passes (52 unique canonical SMILES,
30 scaffolds) no quinolone scaffold is generated, corroborating that the
spectrum of #12 is represented as a non-quinolone. The per-pass top-1 sequence
log-likelihood (SD = 0.01 to 0.04 across passes) is reported as a generator score,
not as a calibrated probability that the entire structure is correct.

## Short caption

**Figure Sz.** MC-Dropout variability of the CMGNet generator over T = 20
passes for 21 grouped external-test spectra: (A) quinolone-core pass frequency
versus XGBoost score; (B) modal top-1 frequency; (C) mean pairwise CReSS-cosine
dispersion of generated candidates. Confident positives are reproducible (core in
20/20 passes) and compact; borderline/false-positive spectra are dispersed; the
false negative #12 never yields a quinolone core.

## Key numbers (per-group mean / median where useful)

| Group (n) | Core in top-5 (fraction of passes) | Top-1 modal frequency (median) | Scaffolds per spectrum | CReSS cosine |
|---|---|---|---|---|
| Confident positives (8) | 1.00 | 0.40 | 8–10 | 0.81 |
| Borderline positives (4) | 0.63 | 0.23 | 32 (median 34) | 0.49 |
| False positives (4) | 0.03 | 0.28 | 15 | 0.51 |
| FN #12 | 0.00 | 0.50 | 30 | 0.51 |
| Confident negatives (4) | 0.00 | 0.48 | 16 | 0.44 |

Settings: checkpoint `train_cmgnet_ring_system_3.2_1029_v2_best.ckpt`
(dropout p = 0.1), T = 20, beam = 5, 5 returned sequences per pass, max length 128.
MC-Dropout is applied to the generator only; the XGBoost classifier contains no
dropout.
