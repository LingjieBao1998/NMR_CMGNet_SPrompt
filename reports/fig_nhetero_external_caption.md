# Fig. S-v — N-heterocycle specificity of the XGBoost classifier — caption (SI)

Files:
- `reports/fig_nhetero_external.pdf` / `.png` (600 dpi)
- analysis/figure: `reports/nhetero_external.py`
- numbers: `reports/nhetero_external_analysis.json`

## Full caption

**Figure S-v.** Specificity of the XGBoost scaffold classifier against
non-quinolone nitrogen heterocycles. The 4,502-spectrum external test set was
processed with the same ring-system retention rule used to build the CMGNet
scaffold prompts (`RingSystemFinder` → removal of terminal ring substituents →
selection of the largest aromatic ring system; `main_cmgnet.py`), and the 4,438
negatives were grouped by ring chemistry. **(A)** Empirical survival curves of
the XGBoost score (fraction of spectra with score ≥ x) for the 64 quinolone
positives, 1,657 negatives containing at least one ring nitrogen, 2,238
carbocyclic negatives and 543 acyclic negatives; the dashed line is the
operating threshold τ_op = 0.40. N-heterocycles score systematically higher than
carbocycles, but the positive tail remains tiny. **(B)** False-positive rate at
τ_op with Wilson 95% confidence intervals: 3/1,311 (0.23%) when the nitrogen is
in the retained (prompt-relevant) ring system, 1/346 (0.29%) when it is only in
a side ring, 1/2,238 (0.045%) for carbocycles and 0/543 for acyclic negatives.
Overall, 4/1,657 N-heterocycle negatives are false positives (0.24%, 95% CI
0.09–0.62%). **(C)** Structures of the six highest-scoring non-quinolone
N-heterocycles: an azo-thiazole dye (p = 0.991), an N-phenyl quinolinone
(p = 0.933), a furoquinoline (p = 0.590), a pyridyl dihydropyridine (p = 0.418),
a nitro-furanone (p = 0.349) and a fused pyridinium zwitterion (p = 0.332).
All false positives are conjugated, quinolone-like N-heteroaromatics —
scaffold-boundary ambiguity rather than a general tendency to call arbitrary
N-heterocycles quinolones.

## Short caption

**Figure S-v.** Specificity of the XGBoost classifier against non-quinolone
N-heterocycles in the external test set (ring groups from the CMGNet scaffold
retention rule). (A) Score survival curves; (B) false-positive rate at
τ_op = 0.40 (4/1,657 N-heterocycles, 0.24%, 95% CI 0.09–0.62%; 1/2,238
carbocycles; 0/543 acyclic); (C) structures of the six top-scoring
N-heterocycles.

## Key numbers

| Group | n | FP at τ_op = 0.40 | Rate (95% CI) |
|---|---|---|---|
| Quinolone positives | 64 | 63 detected | 98.4% recall |
| N in retained ring | 1,311 | 3 | 0.23% (0.08–0.67%) |
| N only in side ring | 346 | 1 | 0.29% (0.05–1.62%) |
| All N-heterocycle negatives | 1,657 | 4 | 0.24% (0.09–0.62%) |
| Carbocycle negatives | 2,238 | 1 | 0.045% (0.008–0.253%) |
| Acyclic negatives | 543 | 0 | 0% (0–0.70%) |

The five false positives at τ_op = 0.40 are: an azo-thiazole dye (0.991), an
N-phenyl quinolinone (0.933), a furoquinoline (0.590), a benzopyranone
(0.540) and a pyridyl dihydropyridine (0.418) — four N-heterocycles and one
carbocycle.
