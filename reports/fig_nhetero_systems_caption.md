# Fig. S-v — class-specific accuracy + representative ring systems (SI caption)

Files:
- `reports/fig_nhetero_systems.pdf` / `.png` (600 dpi)
- analysis: `reports/nhetero_systems.py` (JSON) and
  `reports/make_nhetero_systems_figure.py` (figure)
- numbers: `reports/nhetero_systems.json`

**Threshold convention.** All numbers on this figure use the default decision
threshold τ = 0.50. At this threshold the external test set gives
TP 63 / FP 4 / FN 1 / TN 4,434 (F1 = 0.9618). The manuscript's F1 = 0.9545
corresponds to τ_op = 0.40 (TP 63 / FP 5 / FN 1); every threshold in
(0.349, 0.418] gives that matrix. If the manuscript is updated to τ = 0.50,
the single changed call is the negative at p = 0.4177.

## Full caption

**Figure S-v.** Class-specific accuracy of the XGBoost scaffold classifier on
the external test set (4,502 spectra; 64 quinolones), grouped by ring system.
Classes follow a priority reflecting similarity to the quinolone pharmacophore:
quinolone core → quinolinone-like → fused N-heteroarene → pyridine →
diazine/triazine → azole → saturated N-ring → carbocycle → acyclic. **(A)**
Accuracy at the default threshold τ = 0.50 shown as horizontal bars (axis
starts at 90%; the two classes with errors are highlighted, all other bars are
muted); black lines are Wilson 95% confidence intervals and parentheses give
the error count. For the
quinolone positives the error is the false-negative count (1/64 missed, 98.44%
accuracy); for the negatives it is the false-positive count. Accuracy is
≥98.44% for every ring system: quinolinone-like 1/69 (98.55%), fused
N-heteroarenes 1/488 (99.80%), pyridines 0/144 (100%), diazines/triazines 0/91
(100%), azoles 1/252 (99.60%), saturated N-rings 0/613 (100%), carbocycles
1/2,238 (99.96%) and acyclic negatives 0/543 (100%). In total, 3/1,657
N-heterocycle negatives are false positives (accuracy 99.82%). **(B)**
Representative molecule for each class (most frequent Murcko scaffold of the
class, highest-scoring member); the ring system is highlighted in the same
class colour as panel A. The three N-heterocycle false positives are an
N-phenyl quinolinone, a furoquinoline and an azo-thiazole dye — conjugated,
quinolone-like N-heteroaromatics rather than arbitrary N-heterocycles.
Ordinary indole, pyridine, pyrimidine, pyrazole, pyrrolidine and piperidine
systems remain at ≥99.3% accuracy.

## Short caption

**Figure S-v.** Accuracy of the XGBoost classifier per ring system at the
default threshold τ = 0.50 (A; Wilson 95% CI; axis 90–100%) and a
representative structure for each class with the ring system highlighted (B).
Accuracy is ≥98.44% for every system; errors concentrate in quinolinone-like
scaffolds.

## Key numbers (τ = 0.50)

| System | n | Error | Accuracy (95% CI) |
|---|---|---|---|
| Quinolone core (positives) | 64 | 1 FN | 98.44% (91.7–99.7%) |
| Quinolinone-like | 69 | 1 FP | 98.55% (92.2–99.7%) |
| Fused N-heteroarene | 488 | 1 FP | 99.80% (98.8–100.0%) |
| Pyridine | 144 | 0 FP | 100% (97.4–100%) |
| Diazine / triazine | 91 | 0 FP | 100% (95.9–100%) |
| Azole (5-ring) | 252 | 1 FP | 99.60% (97.8–99.9%) |
| Saturated N-ring | 613 | 0 FP | 100% (99.4–100%) |
| Carbocycle only | 2,238 | 1 FP | 99.96% (99.7–100%) |
| Acyclic | 543 | 0 FP | 100% (99.3–100%) |
| All N-heterocycle negatives | 1,657 | 3 FP | 99.82% (99.47–99.94%) |
