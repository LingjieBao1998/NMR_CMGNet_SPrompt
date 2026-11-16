# Fig. S-w — C–F peak splitting / spectral resolution — caption (SI)

Files:
- `reports/fig_resolution_split.pdf` / `.png`
- analysis: `reports/resolution_split_analysis.py`
- numbers: `reports/resolution_split_metrics.json`

## Full caption

**Figure S-w.** Sensitivity of the XGBoost classifier to C–F peak splitting.
Two paired 48-spectrum sets contain the same fluorinated quinolones but represent
the C–F split components differently: in the *merged* set the split components
are collapsed to single peak positions (median 18.5 peaks per spectrum), while
in the *split* set they are retained separately (median 23 peaks). **(A)**
Paired predicted probabilities (merged vs split input; log–log axes); grey
points keep the same call at τ = 0.50, red points flip. **(B)** Recall at
τ = 0.50: 46/48 (95.8%) for merged input versus 41/48 (85.4%) for split input.
The median per-molecule change is |Δp| = 6.8×10⁻⁴, but 7/48 molecules flip
(6 lost, 1 recovered; maximum |Δp| = 0.95), so peak splitting is a real but
localized failure mode concentrated in a subset of spectra. This motivates
merging C–F doublets during peak-list curation before prediction. A literal
signal-to-noise sweep would require raw spectra with intensities; the current
input is a peak list without intensity information.

## Short caption

**Figure S-w.** C–F peak-splitting sensitivity for 48 paired fluorinated
quinolones: recall 46/48 (95.8%) with merged split components versus 41/48
(85.4%) with split components retained; 7/48 calls flip. Peak-list curation
(merging C–F doublets) is recommended.

## Key numbers (τ = 0.50)

| Input | n | Peaks (median) | Detected | Recall |
|---|---|---|---|---|
| Merged C–F components | 48 | 18.5 | 46 | 95.8% |
| Split C–F components | 48 | 23.0 | 41 | 85.4% |

Median |Δp| = 6.8×10⁻⁴; maximum |Δp| = 0.95; 7 flips (6 lost, 1 recovered).
