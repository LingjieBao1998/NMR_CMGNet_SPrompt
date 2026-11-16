# Figure 2 — combined model analysis (τ = 0.50) — caption

File: `reports/fig_model_analysis_v5.png` / `.pdf` (600 dpi, 180 mm print width)
Generator: `reports/make_figure_model_analysis_v5.py`

## Full caption

**Figure 2.** Error analysis and model behaviour at the decision threshold τ = 0.50.
**(A)** CReSS embedding of the full training set (grey = negatives, orange = positives) with the 4,502 external-test spectra overlaid; the four false positives (diamonds) and the single false negative #12 (star) are marked. **(B)** Local UMAP of the 41,961 training molecules within R = 0.78 of FN #12 (2,732 positives). **(C)** FN #12 and its three nearest neighbours: the two closest positives (P1, para-F analogue, cosine 0.819; P2, cosine 0.857) and the nearest negative (N1, cosine 0.840). **(D)** Classifier probability along the embedding path from FN #12 to P1; the decision boundary is crossed at α ≈ 0.20. **(E)** Every external-test spectrum is one circle in the same embedding, coloured by the width of the 90% bootstrap confidence interval of its calibrated probability (logarithmic scale from the grey of confident negatives to the dark orange of uncertain predictions); diamonds mark the four false positives and the star the false negative. **(F)** The same 90% intervals for the 64 positives and the 4 FP, with molecules sorted along x and the calibrated probability on the y axis (log scale); the dashed line is τ = 0.50, which corresponds to 0.0235 on the calibrated scale because the calibrator is fitted on the strongly imbalanced validation split. Two of the four FP (#763, #4201) have intervals straddling the threshold, whereas FN #12 is confidently wrong. **(G)** Accuracy per ring system at τ = 0.50; accuracy is ≥98.44% for every system, with the lowest values for the quinolone core (1 FN/64) and quinolinone-like negatives (1 FP/69). **(H)** Representative structures of the eight ring systems, with the rings highlighted in the corresponding colour.

## Short caption

**Figure 2.** Model analysis at τ = 0.50: **(A)** global CReSS embedding with the external test set, FP and FN; **(B–D)** local neighbourhood and embedding-path interpolation around FN #12 (boundary crossed at α ≈ 0.20); **(E)** test-set map of the 90% confidence-interval width, with FP and FN marked; **(F)** per-molecule 90% intervals for the 64 positives and 4 FP; **(G)** accuracy per ring system (all ≥98.44%); **(H)** representative ring systems.
