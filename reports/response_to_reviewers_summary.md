# Point-by-point response to reviewers

**Manuscript:** CMGNet-SPrompt — de novo detection of illegal quinolone additives in cosmetics from pure ¹³C NMR chemical shifts  
**Journal:** *Analytical Chemistry*

We thank the Editor and Reviewers for the constructive comments. Below we provide a point-by-point response. Reviewer comments are summarized in italics; changes in the revised manuscript/SI are indicated.

---

## Editor / Reviewer 1

### R1.0 — Application significance, real samples, matrix interference, comparison with MS
*Comment:* Limited evidence of quinolone abuse in cosmetics; no real-sample validation; ¹³C NMR susceptible to matrix interference; MS is the standard tool.

**Response.** We have reframed the method as a **scaffold-level screening tool that complements, not replaces, LC–MS/MS confirmation**. The revised Introduction/Discussion adds regulatory and market-surveillance context for (fluoro)quinolone adulteration in anti-acne/antibacterial cosmetic products and clarifies the intended workflow: **¹³C NMR screening → MS confirmation**. Because ¹³C shifts report the carbon skeleton directly and the model operates on peak lists, it is suited to flagging **undocumented analogues** absent from targeted MRM libraries. We explicitly state that the present validation uses pure-compound spectra (experimental + synthetic) and identify matrix spike-and-recovery validation as the next step; no claim of immediate field deployment is made.

### R1.1 — Detailed analysis of FP/FN cases
*Comment:* Dissect the 4 FP and 1 FN (structural similarity vs. hallucination; spectral quality).

**Response.** At the unified operating threshold **τ = 0.50** the external test set (n = 4,502; 64 positives) gives **TP 63 / FP 4 / FN 1 / TN 4,434** (Precision 0.940, Recall 0.984, F1 0.9618). A full error analysis is added (**Figure 2A–D** and SI):

- **FP (4):** All lie inside/at the edge of the training-positive manifold in the CReSS embedding (Fig. 2A) and share quinolone-like fused heteroaromatic/conjugated motifs but lack the 4-oxo-1,4-dihydroquinoline-3-carboxylic-acid core. These are **scaffold-boundary ambiguities**, not hallucinations; spectra above threshold are reported as “quinolone-like, requires MS confirmation”.
- **FN (#12, p = 0.012):** Not due to spectral quality (18 peaks, comparable to positives) and not due to split leakage (absent from all training sources). It represents a **coverage hole** (7-unsubstituted phenyl + 8-NO₂ + 6-F + N-cyclopropyl combination unseen; nearest positives cosine 0.857/0.819, p = 0.998/0.990) and **over-reliance on an aromatic peak pattern**: leave-one-peak-out ablation shows removing the 129.9/130.4/130.6 ppm triplet raises p from 0.012 to 0.910 (Fig. 2B–D). Embedding interpolation FN #12 → P1 crosses p = 0.5 at α ≈ 0.20, confirming a boundary case.

### R1.2 — Small positive class, single-split variance, scaffold leakage
*Comment:* Only 64 positives; SMILES deduplication may not prevent scaffold-level leakage; suggest scaffold hold-out/nested CV.

**Response.** We added (i) a **scaffold-level split audit** and blacklist re-verification (0 overlaps; FN #12 absent), (ii) **three-seed variance** for the classifier pipeline, and (iii) a **similarity-threshold leakage ablation** (remove training neighbours above cosine τ and retrain) to bound the contribution of near-duplicates. We clarify that the task targets a fixed pharmacophore under deployment-like conditions and that the FN itself is a novel scaffold combination the model fails on (i.e., memorization would not inflate performance). Generalization to entirely novel scaffold classes is stated as a limitation/follow-up.

### R1.3 — Terminology, boundary behavior, “simple”, compute
*Comment:* Use “conditional molecular generation under scaffold constraints”; define behavior near p ≈ 0.5; quantify simplicity/compute vs ensembles.

**Response.** Terminology revised throughout. We define an explicit **grey-zone/review policy** around the decision boundary and report compute (parameters, inference time, memory) versus ensemble baselines.

### R1.4 — Ablation wording and missing 3.5.2
*Comment:* Correlation vs causation; emotional wording; add ensemble-strategy ablation.

**Response.** Causal claims removed (“associated with”, not causal); wording neutralized. Section 3.5 split into **3.5.1 synthetic-data ablation** and **3.5.2 ensemble-strategy ablation** (unconstrained CMGNet, RF baseline, Voting/Union).

### R1.5 / R2.8 — Figures and references
**Response.** Figures redesigned at print width with vector/600 dpi exports and readable fonts (new **Figure 2 A–H**, PR/ROC Fig. Sx, calibration/CI figures). All references completed and reformatted to ACS style.

### R1.6, R1.7 — Reproducibility/server
**Response.** Code, weights, configs, evaluation scripts and inference examples prepared for release (GitHub/Zenodo/Docker); server/documentation updated.

---

## Reviewer 2

### R2.1 — Code, weights, data, examples
**Response.** Public release prepared: preprocessing, training/inference, evaluation/figure scripts, pretrained weights, examples, Docker; dataset released per source licences.

### R2.2 — Fair comparison with NMR-Solver / NMRMind
**Response.** Comparisons rerun/reported on a unified test set with input dimensionality, training data and settings stated; rows not directly comparable are explicitly marked. On the 64 external positives we also report **top-10 core-hit rate** (“any of top-10 contains the quinolone core”) and exact@10:

| Model | Core hit (≥1 in top-10) | Hit rate | Exact@10 |
|---|---|---|---|
| CMGNet-SPrompt (ex1209, 3 runs) | 63.33 ± 0.58 / 64 | **99.0% ± 0.9%** | 18.67 ± 0.58 / 64 (29.2%) |
| XGBoost-CReSS (classifier) | 63 / 64 | 98.4% | — |
| CMGNet-Frag (ex1209, 3 seeds) | 62.67 ± 0.58 / 64 | 97.9% ± 0.9% | 0 (fragments) |
| CMGNet-SMILES (ex1209, 3 seeds) | 62.33 ± 0.58 / 64 | 97.4% ± 0.9% | 19.33 ± 2.31 / 64 (**30.2%**) |
| CMGNet (original 0910) | 52 / 64 | 81.2% | 6 / 64 (9.4%) |
| NMRMind | 48 / 64 | 75.0% | 1 / 64 (1.6%) |
| NMR-Solver | 41 / 64 | 64.1% | 16 / 64 (25.0%) |

(Per-molecule counts for the last 10 positives are provided in `reports/table2_last10_hits.csv` / Table Sx.)

### R2.3 — FP/FN, other N-heterocycles, S/N & resolution sensitivity
**Response.** See R1.1 for FP/FN dissection (Fig. 2A–D). Ring-system analysis (τ = 0.50) shows high specificity across N-heterocycles (3/1,657 N-heterocycle negatives FP, 0.18%): quinolinone-like 1/69, fused N-heteroarene 1/488, pyridine 0/144, diazine/triazine 0/91, azole 1/252, saturated N-ring 0/613, carbocycle 1/2,238, acyclic 0/543 (**Fig. 2G–H**). C–F peak splitting sensitivity (merged vs split peak lists, 48 paired spectra): merged 46/48 (95.8%) vs split 41/48 (85.4%), 7/48 flips — supporting peak-list curation (merging C–F doublets) and noted as a localized failure mode.

### R2.4 — Uncertainty quantification (calibration, MC-Dropout, review workflow)
*Comment:* Provide calibration/intervals/uncertainty and integrate into workflow; cite suggested references.

**Response.** We treat discrimination, calibration and uncertainty-guided review separately (no external-test labels used for fitting/selection):

1. **Calibration (Fig. Sy).** Platt scaling fitted on the reserved validation split (selected over isotonic by cross-fitted NLL); prior-corrected for the enriched external set. Internal-test ECE reduced from 4.71×10⁻³ (raw) to 1.74×10⁻⁵; Brier skill score +0.61.
2. **Classifier CIs & prediction sets (Fig. 2E–F / Fig. S-u).** 90% bootstrap CIs of the calibrated probability and 90% class-conditional conformal prediction sets (external: 3.09% referred, 0 silent misses). **Fig. 2E** maps the 90% CI width over the full external test set in the CReSS embedding (colour = CI width, logarithmic grey→orange); FP are diamonds and the FN is a star, consistent with the embedding convention. **Fig. 2F** shows per-molecule 90% CIs for the 64 positives and 4 FP: two FP have intervals straddling the decision threshold, while **FN #12 is confidently wrong** (interval far below threshold), consistent with a genuine coverage gap rather than borderline noise. Three-seed retraining disagreement is concentrated at the decision boundary.
3. **Generator variability (MC-Dropout, Fig. Sz).** T = 20 stochastic passes on CMGNet (dropout p = 0.1); confident positives recover the quinolone core in every pass (mean pairwise CReSS cosine 0.80), borderline cases are dispersed (0.49). **FN #12 yields 0/20 quinolone-core candidates**, indicating a representation-level failure (not an inference artefact). MC-Dropout is not applied to XGBoost.
4. **Uncertainty-guided review.** Three-way rule (auto-positive / refer for MS confirmation / auto-negative): external test **0.47% referred, 0 silent misses** (21 referred: 5 positives incl. FN #12, 16 negatives); internal natural-prevalence set 5.52% referred, 1 residual silent miss (reported transparently). Referred spectra are never labelled negative.

Caveats stated: few positive calibration observations, population-specific calibration, CI/conformal ≠ out-of-domain guarantee, and review cannot conceal false negatives (residual silent-miss counts reported).

### R2.5 — Hyperparameters and sensitivity
**Response.** Selection protocol and search ranges documented; sensitivity analyses (bins/dropout/learning rate) added.

### R2.6 — Web server / API / Docker
**Response.** SI documents inputs/outputs, privacy, API, maintenance and Docker image for local deployment.

### R2.7 — Recent reviews
**Response.** Recent AI-for-NMR reviews (e.g. 10.1063/5.0277355, 10.1016/j.pnmrs.2022.101575) cited in Introduction.

---

## Summary of major revisions
- **Figure 2 (A–H):** unified model-analysis figure (embedding + FN #12 case, test-set CI map and per-molecule 90% CI, ring-system accuracy/representatives) at τ = 0.50.
- **New analyses:** FP/FN mechanistic dissection, PR/ROC (AUPRC 0.9891, AUROC 0.9998), calibration & prior correction, MC-Dropout generator variability, conformal/bootstrap/seed CIs, uncertainty-guided review, ring-system specificity, C–F splitting sensitivity, scaffold/leakage and seed-variance checks, fair baseline comparison.
- **Reproducibility & presentation:** code/weights release (planned), vector/600 dpi figures, completed references, clarified terminology and scope.

We believe these revisions substantially strengthen the manuscript and thank the reviewers for their time.
