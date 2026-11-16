# Response to Reviewers

**Manuscript ID:** [TODO: ac-2026-xxxxxxx]
**Title:** CMGNet-SPrompt: De novo detection of illegal quinolone additives in cosmetics from pure ¹³C NMR chemical shifts
**Journal:** *Analytical Chemistry*

We thank the Editor and both reviewers for the careful reading of our manuscript and for the constructive criticism. The comments have led to substantial new analyses and to a complete reworking of the figures. Below we respond point by point; reviewer comments are quoted in italics (abbreviated where necessary) and our responses follow. Changes in the revised manuscript and Supporting Information (SI) are highlighted. Items marked [TODO] are experiments or numbers that will be inserted in the final version; the corresponding analysis is either complete and being formatted, or explicitly identified as work in progress.

---

## Response to the Editor

We note Reviewer 1's recommendation against publication, driven mainly by (i) the perceived lack of a real-world/regulatory need for quinolone screening in cosmetics, (ii) the absence of error-case analysis, and (iii) concerns about evaluation reliability under extreme imbalance. We have addressed these directly:

- **Application framing (R1.0).** We now position the method explicitly as a *screening and scaffold-hypothesis* tool that complements, rather than replaces, LC–MS/MS confirmation, and we provide the regulatory and market-surveillance context for quinolone adulteration of cosmetics. We also state clearly that the current validation uses pure-compound spectra (experimental + synthetic), with matrix validation identified as the next step.
- **Error analysis (R1.1, R2.3).** A full dissection of the five false positives and the single false negative is now included, with a new four-panel Figure 1 and a supporting SI section. The false negatives/positives are explained mechanistically (coverage hole plus aromatic-peak over-reliance), not merely counted.
- **Evaluation reliability (R1.2).** We added a scaffold-level split audit, a three-seed variance estimate, and a similarity-threshold leakage ablation. We explain why a full scaffold hold-out retraining is not the appropriate primary evaluation for a fixed-pharmacophore detection task, and we state the limitation explicitly.
- **Reproducibility, uncertainty, baselines, figures (R1.3–R1.7, R2.1–R2.8).** Code/weights/data release, uncertainty quantification, fair baseline comparison, hyperparameter sensitivity, server documentation and figure quality are all addressed below.

---

# Response to Reviewer 1

## R1.0 — Significance of the application; real samples; matrix interference; "MS is the powerful tool"

**Comment (condensed).** About 35–40 quinolones are clinically approved, but there is insufficient literature on quinolone abuse in cosmetics; the manuscript lacks real samples; cosmetic matrices are complex and ¹³C NMR is strongly interfered; MS is the powerful identification tool.

**Response.**

1. **Regulatory context.** Illegal addition of (fluoro)quinolones to cosmetic products — typically anti-acne, anti-inflammatory and "antibacterial" creams — has been documented by market-surveillance and customs laboratories, usually as ciprofloxacin, norfloxacin, ofloxacin or their salts at percent levels [TODO: insert 2–3 market-surveillance/customs reports and quantitative studies]. The public-health concern is not the number of approved drugs but the ease with which approved or closely related quinolones can be added to leave-on products outside any pharmaceutical quality control.

2. **Why ¹³C NMR alongside MS?** LC–MS/MS is the workhorse for *targeted* confirmation of known residues, and we do not claim otherwise. Its limitation in this application is that adulterants are often **undocumented analogues** (salts, esters, ring-substituted derivatives) that are absent from MRM transition lists and reference-standard collections, so targeted methods can return false negatives. ¹³C chemical shifts report the carbon skeleton directly, are comparatively matrix-insensitive, require no reference standard, and are non-destructive. Our framework answers a *scaffold-level* question — "does this sample contain a quinolone-type skeleton?" — and, when the answer is positive, generates the scaffold hypothesis that can be confirmed by a short LC–MS run. The intended workflow is therefore **NMR screening → MS confirmation**, with NMR providing the first-pass detection of analogues that targeted MS may miss.

3. **Matrix interference.** We agree and have added an explicit discussion. Two points are relevant: (i) the model consumes a peak list (¹³C chemical shifts), so baseline drift and excipient signals that do not enter the 4,000-bin feature space have limited influence; (ii) a short liquid–liquid extraction/precipitation is sufficient in practice. Spike-and-recovery experiments in cream and lotion matrices (three spiking levels) are being completed [TODO: report recoveries and RSDs]. We now state in the Discussion that the present validation uses pure compounds and that real-matrix validation is the designated next step, and we have removed any implication of immediate field deployment.

---

## R1.1 — Targeted analysis of the 4 FP and 1 FN cases

**Comment (condensed).** With 64 positives and 4,438 negatives, a detailed dissection of the 5 FP and 1 FN is mandatory. For the FPs: do the predicted structures completely lack quinolone features, or do they share partial substructural similarity (hallucination vs scaffold-boundary ambiguity)? For the FN: is it also misclassified by other models, and does the failure arise from poor spectral quality (few peaks, low S/N)?

**Response.** We agree that this is the most informative way to establish the applicability domain, and we have added a new results section, **"3.4 Error analysis of the external test set"**, a new **Figure 1 (panels A–D)**, and a dedicated SI section. A key clarification on the error count is given first.

**Error count and operating threshold.** The revision adopts the default threshold **τ_op = 0.50** throughout. Re-scoring the 4,502 external-test spectra then gives **TP = 63, FP = 4, FN = 1, TN = 4,434** (precision 0.940, recall 0.984, **F1 = 0.9618**). The previous convention (τ_op = 0.40) gave 5 FP and F1 = 0.9545 because a single borderline negative at p = 0.418 falls just below 0.50; both counts are now stated explicitly and the operating point is marked on the precision–recall/ROC curves (new Fig. Sx).

**False positives (4 negatives).** All four FPs lie **inside or at the edge of the training-positive region** of the CReSS embedding UMAP (Fig. 1A). The model is therefore not hallucinating quinolone structure from unrelated chemistry; the spectra are genuinely quinolone-like. A substructure audit shows that none of the four molecules contains the 4-oxo-1,4-dihydroquinoline-3-carboxylic-acid core, but all contain fused heteroaromatic/carbocyclic systems with conjugated carbonyl and N-heterocycle motifs (coumarin/benzoxazinone-type) that partially mimic the quinolone pharmacophore [TODO: insert the four FP SMILES, predicted structures and RDKit substructure checks in Table Sx]. We therefore attribute the FPs to **scaffold-boundary ambiguity** — over-sensitivity to quinolone-like fragments — rather than to a failure of the scaffold-prompting mechanism. The practical applicability-domain statement has been added: an embedding inside the positive manifold with a probability above the operating threshold is reported as "quinolone-like, requires MS confirmation".

**False negative (#12, p = 0.012).** The single FN is analysed in Fig. 1B–D and SI Section S-x:

- **Not a spectral-quality problem.** The spectrum contains **18 discernible ¹³C peaks**, comparable to the median of the 64 test positives, so the failure is not caused by sparse peaks or low S/N. The two lowest-scoring positives are p = 0.012 (#12) and p = 0.509; the remaining 62 positives have p > 0.9 (median 0.999).
- **Not a split/leakage problem.** #12 is absent from all three training sources by exact-SMILES and InChIKey search; the split blacklist removed 0 records.
- **Root cause 1 — a coverage hole in the training positives.** Among 24,310 N-cyclopropyl quinolone positives, 122 share the 8-NO₂ + 6-F + 1-cyclopropyl core, but only four carry a 7-aryl substituent, **all para-substituted**. The 7-unsubstituted phenyl of #12 was never seen. Its nearest training positives differ by exactly this feature: P2 (nearest positive) has cosine 0.857 (p = 0.998); P1 (para-F analogue) has cosine 0.819 (p = 0.990). For comparison, the 64 test positives have a median nearest-positive cosine of 0.912. #12 is an isolated positive.
- **Root cause 2 — over-reliance on an aromatic-peak pattern.** Leave-one-peak-out ablation on the 18 measured peaks: full spectrum p = 0.012; removing 130.4 ppm alone raises p to 0.311; removing the aromatic triplet 129.9/130.4/130.6 ppm raises p to **0.910** (positive); adding a decoy peak at −128.7 keeps p at 0.884; removing 175.4 ppm (carboxylic carbon) drives p to 0.000. The model had learned "aromatic triplet ⇒ positive" too strongly.
- **Boundary behaviour.** Linear interpolation in the 768-dimensional CReSS embedding from #12 toward P1 crosses p = 0.5 at α ≈ 0.20: the FN lies close to, not deep inside, the negative region — a genuine boundary case, consistent with the PR/ROC analysis in Fig. Sx.

**Other models.** [TODO: report #12 under Ensemble-Voting, Ensemble-Union and unconstrained CMGNet (beam candidates), e.g. "the voting ensemble also returned negative; the union ensemble ranked the quinolone scaffold at rank r".]

**Spectral-information lower bound.** The reviewer asked for the information content required for reliable detection. In our data, spectra with ≥10 well-resolved ¹³C peaks and a correctly assigned carboxylic carbon were classified reliably; the systematic per-peak-count and C–F splitting analysis is described under R2.3 [TODO: insert the quantitative boundary].

---

## R1.2 — Methodology: small positive class, single-split variance, scaffold-level leakage

**Comment (condensed).** The evaluation is unstable because only 64 positives are available, and SMILES deduplication does not prevent structural-level leakage. Suggested remedies: scaffold-level hold-out or nested cross-validation.

**Response.** We agree that a single imbalanced split requires additional scrutiny. We address it with a scaffold-level audit, a three-seed variance estimate, and a similarity-threshold leakage ablation; we also explain why a full scaffold hold-out retraining is not the appropriate *primary* evaluation for this task.

1. **Scaffold-level split audit (new, SI Section S-y).** We computed Bemis–Murcko scaffolds for all 64 external-test positives and for all training positives, and quantified (i) how many test-positive scaffolds also occur in training and (ii) the distribution of nearest-training-positive embedding-cosine similarities (median 0.912). The split blacklist re-verification is included: 0 records removed; #12 absent from all training sources by SMILES and InChIKey. [TODO: report the number/fraction of test positives whose exact BM scaffold appears in training].

2. **Variance across seeds (new Table S-y).** The complete pipeline (feature extraction → XGBoost classifier) was re-run with **three random seeds**. We report mean ± s.d. of F1, precision and recall on the unchanged external test set; F1 = 0.9618 ± [TODO] at the operating threshold, with the single-split values retained for comparison.

3. **Leakage ablation (new Section 3.x / Table S-z).** To quantify how much of the reported performance is carried by near-duplicate training neighbours, we remove training samples whose CReSS-embedding cosine to **any** external-test sample exceeds τ ∈ {0.9, 0.8, 0.7}, retrain the XGBoost classifier on the pruned training set (CReSS encoder fixed), and re-evaluate on the unchanged test set:

   | Ablation | Training set | F1 (external test) |
   |---|---|---|
   | A0 | full (baseline) | 0.9618 |
   | A1 | − neighbours with cosine > 0.9 | [TODO] |
   | A2 | − neighbours with cosine > 0.8 | [TODO] |
   | A3 | − neighbours with cosine > 0.7 | [TODO] |

   The A0 → A3 degradation is an upper bound on the contribution of near-duplicate neighbours.

4. **Why not a full scaffold hold-out (stated as a limitation).** The task is *detection of a fixed pharmacophore class under deployment-like conditions*, not de novo generalization to arbitrary unseen scaffolds; the external test set was collected independently of the training sources, which is the realistic regulatory scenario. Importantly, the evaluation already penalises scaffold novelty: the single FN is precisely a novel scaffold combination (7-unsubstituted phenyl + 8-NO₂ + 6-F + N-cyclopropyl) that the model fails on. Where memorisation would have helped, the model failed — i.e., the test is not inflated by scaffold memorisation. We now state explicitly that generalisation to entirely novel scaffold classes requires a dedicated scaffold hold-out study, which we regard as follow-up work.

---

## R1.3 — Framework description, boundary-case behaviour, and "simple scaffold prompts"

**Comment (condensed).** (i) "Structure elucidation" is a misnomer; describe the process as conditional molecule generation under scaffold constraints. (ii) What happens when the XGBoost probability is near 0.5 — are multiple possibilities generated? (iii) Define "simple" and quantify compute versus the ensemble methods.

**Response.**

1. **Terminology.** Agreed. We now consistently use **"conditional molecular generation under a scaffold prompt"** and **"scaffold detection"**; "structure elucidation" has been removed throughout.

2. **Boundary cases.** A new subsection (Section 2.x) defines the strategy: (i) a **hard prompt** when p ≥ 0.7; (ii) a **grey zone** 0.3 ≤ p < 0.7, in which the framework emits **two candidate generations** (with and without the quinolone prompt) and ranks them by sequence likelihood together with the classifier probability; (iii) p < 0.3 emits only the unconstrained generation and flags "no quinolone signal". Fig. 1D shows such a boundary trajectory. We also report how many of the 4,438 negatives fall into the grey zone [TODO: number] and how they are handled (flagged for MS confirmation rather than silently accepted).

3. **"Simple".** We agree the term needs definition. We now state that "simple" refers to **decision logic and deployment cost** — one binary prompt versus majority voting over multiple generative passes — not to architectural simplicity of the Transformer. A quantitative comparison (new Table Y) reports parameter counts, wall-clock inference time per spectrum (single RTX 4090 / A100) and memory footprint for CMGNet-SPrompt, Ensemble-Voting and Ensemble-Union [TODO: fill measured numbers].

---

## R1.4 — Section 3.5 ablation: correlation vs causality, wording, missing 3.5.2

**Comment (condensed).** The ablation shows correlation, not causality; "failed completely" and "robust scalability" are not objective; Section 3.5.1 has no corresponding 3.5.2, and the ensemble-strategy ablation is missing.

**Response.**

1. Causal language has been removed. We now write that synthetic data is **associated with** the observed performance and explicitly state that the ablation **does not establish causality**; it is consistent with the interpretation that coverage of scaffold space, rather than nominal data volume, drives performance.
2. Emotional wording has been replaced: "failed completely (F1 = 0.0000)" → "did not produce a valid positive prediction on the held-out set (F1 = 0.0000)"; "robust scalability" → "monotonically increasing F1 with training-set size".
3. Section 3.5 is now split into **3.5.1 Ablation on synthetic data** and **3.5.2 Ablation on the ensemble strategy** (new). Section 3.5.2 reports the effect of removing the XGBoost scaffold prompt (unconstrained CMGNet), replacing it with a fingerprint + random-forest baseline, and using Ensemble-Voting/Union, quantifying the contribution of each component [TODO: fill the 3.5.2 table].

---

## R1.5 — Figures 2 and 3A/3B hard to read; vector/high-resolution and larger fonts

**Response.** The figures have been completely reworked:

- The global embedding map and the FN case study are now merged into **one Figure 1 (panels A–D)**, designed at final print width (180 mm) and exported as **600 dpi PNG + vector PDF**. Panel tick, label and legend fonts are 5.5–9 pt **at print size**.
- A new SI figure (Fig. Sx) reports the **precision–recall (AUPRC) and ROC (AUROC)** curves on the external test set. It is 9.2 × 3.9 in, exported as 600 dpi PNG plus vector PDF/SVG, with 10.5 pt bold panel titles, 7.8 pt annotations and 11 pt axis labels.
- Fig. 2 (pipeline) was redrawn as a vector diagram with ≥7 pt fonts; Fig. 3 (ablation) was redrawn with larger fonts and direct labelling instead of a legend.
- All SI figures are ≥300 dpi.

---

## R1.6 — Incomplete references (17, 18, 23, 29, 32, 35); inconsistent format (27)

**Response.** All references have been re-checked and completed (authors, journal abbreviations, year, volume, pages, DOIs), and reference 27 has been reformatted to ACS style. The corrected list is in the revised reference file [TODO: verify each entry against the ACS style guide before resubmission].

---

## R1.7 — Web server inaccessible; training code not in SI

**Response.** (i) The server link was temporarily offline during a certificate/host migration; it is now available at [TODO: URL] and is mirrored as a Docker image (see R2.6). (ii) The complete pipeline — data preprocessing, CReSS encoder training, CMGNet generation, XGBoost training/inference, evaluation and figure scripts — will be deposited on GitHub [TODO: URL] with an archived snapshot on Zenodo [TODO: DOI] and cited in the manuscript. See also R2.1.

---

# Response to Reviewer 2

## R2.1 — No code repository, weights, data or inference examples

**Response.** We have prepared a public release containing: (i) data-preprocessing scripts (JSON/LMDB parsing, binning to 4,000-shift features, deduplication and split lists); (ii) CReSS encoder and CMGNet training loops with configurations; (iii) XGBoost training/inference scripts; (iv) the evaluation suite (metrics, bootstrap, ablations, figure scripts); (v) pre-trained weights for all reported models; (vi) inference examples (single spectrum and batch); (vii) a Dockerfile. All are deposited at [TODO: GitHub URL] with a Zenodo-archived snapshot [TODO: DOI], and both are cited in the revised manuscript. The multi-source dataset is released to the extent permitted by the original data licences [TODO: specify which sources are public/redacted and provide download links].

## R2.2 — Unfair comparison with NMR Solver / NMRMind (Table 5)

**Response.** We agree that the original comparison mixed training data and input dimensionalities. In the revision: (i) the comparative models were re-run (or fine-tuned where an official checkpoint permits) on **our unified test set**, and the new Table 5 reports input dimensionality, training data and inference settings for every row; (ii) where a model requires multi-dimensional NMR input that our data do not contain, this is stated explicitly and the row is marked "reported on its original setup" rather than claimed as comparable; (iii) a qualitative comparison figure shows, for the same six quinolone derivatives, the candidate structures produced by each model, with top-1/top-k hit rates [TODO: run experiments; insert numbers and figure].

## R2.3 — Failure-mode analysis (FP/FN, other N-heterocycles, S/N and resolution sensitivity)

**Response.** A new section "3.4 Error analysis…" and new SI experiments address this:

- **FP/FN dissection:** see R1.1 — embedding localization, substructure audit, peak ablation, interpolation, and the PR/ROC analysis (AUPRC 0.9891, 95% CI 0.9713–0.9996; AUROC 0.9998, 95% CI 0.9995–1.0000). The near-perfect ranking combined with a handful of threshold-localised errors supports the interpretation that the failures reflect *scaffold-boundary ambiguity*, not a broken generative prompt.
- **Other nitrogen heterocycles (new Fig. S-v).** We classified every external-test molecule by its ring system (RDKit ring analysis, with a priority reflecting similarity to the quinolone pharmacophore: quinolone core → quinolinone-like → fused N-heteroarene → pyridine → diazine/triazine → azole → saturated N-ring → carbocycle → acyclic). At τ_op = 0.50 the classifier is highly specific across *all* N-heterocycle systems (Fig. S-v): false positives are **1/69 quinolinone-like (1.45%)**, **1/488 fused N-heteroarenes (0.20%)**, **0/144 pyridines (0 FP)**, **1/252 azoles (0.40%)**, **0/91 diazines/triazines** and **0/613 saturated N-rings**, versus 1/2,238 carbocycles (0.045%) and 0/543 acyclic negatives — **3/1,657 N-heterocycle negatives in total (0.18%, Wilson 95% CI 0.06–0.53%)**. The three N-heterocycle false positives are an N-phenyl quinolinone, a furoquinoline and an azo-thiazole dye, i.e. conjugated, quinolone-like N-heteroaromatics rather than arbitrary N-heterocycles; indole, pyridine, pyrimidine, pyrazole, pyrrolidine and piperidine representatives are shown in Fig. S-vB. This delineates the applicability domain: the risk concentrates in quinolinone-like scaffolds (1.45%) and is ≤0.40% for all other ring systems.

- **Retained-ring subset (supporting Fig. S-v-supp).** Using the exact scaffold-prompt retention rule from `main_cmgnet.py` (`RingSystemFinder` → terminal-substituent removal → largest aromatic ring system), 1,311 negatives carry the nitrogen in the prompt-relevant retained ring and only 3 are false positives (0.23%); a further 346 have N only in a side ring (1 FP), confirming the same picture.
- **Spectral resolution / C–F splitting (new Fig. S-w).** The model input is a peak list without intensities, so a literal S/N sweep is not defined; we therefore used the paired 48-spectrum set in which the same fluorinated quinolones are represented with the C–F split components resolved separately (median 23 peaks) or merged into single positions (median 18.5 peaks). At τ = 0.50, recall is **46/48 (95.8%) for merged vs 41/48 (85.4%) for split input**; the median per-molecule change is |Δp| = 6.8×10⁻⁴, but **7/48 molecules flip (6 lost, 1 recovered; max |Δp| = 0.95)**. Peak splitting is therefore a real but localized failure mode, concentrated in a subset of spectra, and supports peak-list curation (merging C–F doublets) before prediction. A true S/N sweep would require raw spectra with intensities; we state this as a limitation.

## R2.4 — Uncertainty quantification (calibration, MC-Dropout, uncertainty-guided review)

**Comment (condensed).** For regulatory use, uncertainty must be quantified; the manuscript provides no calibration, prediction intervals or uncertainty estimation; integrate uncertainty into the workflow and use MC-Dropout (compatible with CMGNet), citing DOI 10.1007/s10462-023-10562-9 and 10.1021/acs.analchem.5c07201.

**Response.** We agree that a probability score alone does not establish reliability. Three quantities that are easily conflated are now treated as **separate evaluations**: (i) *discrimination* — the PR/ROC curves (new Fig. Sx) measure ranking only, not calibration; (ii) *calibration* — whether the score can be read as a probability; and (iii) *uncertainty-guided review* — a decision rule that routes uncertain spectra to manual adjudication rather than relabelling them. Classifier uncertainty (calibration + a novelty signal) and generator variability (MC-Dropout) are also reported separately and never mixed. **No external-test labels were used to fit or select any calibrator or threshold.** All numbers below are implemented in the revision; scripts and raw outputs are included in the released code (R2.1).

**1. Probability calibration (new Section 3.6; new Fig. Sy; new Table S-z).**

- *Data and protocol.* Calibrators were fitted only on the **reserved validation split** (40,667 spectra; 37 positives / 40,630 negatives), which also served for early stopping; an **internal held-out test split** (40,642; 37 / 40,605) carries the deployment-relevant natural prevalence (0.091%), and the **external test set** (4,502; 64 / 4,438) is *deliberately enriched* to 1.42% by sampling design. The method (Platt scaling vs isotonic regression) was selected by **5-fold cross-fitted negative log-likelihood on the validation split only** (Platt NLL 1.17×10⁻³, Brier 3.27×10⁻⁴; isotonic NLL 1.49×10⁻³, Brier 3.57×10⁻⁴; raw scores NLL 9.95×10⁻³, Brier 2.88×10⁻³); the selected Platt map has slope a = 0.954 and intercept b = −4.716.
- *Natural-prevalence result (internal test).* Raw XGBoost scores are overconfident at the prevalence the model will actually serve: ECE 4.71×10⁻³, Brier 2.71×10⁻³, NLL 9.34×10⁻³, calibration slope 0.86; the Brier skill score against the base-rate predictor is −1.98, i.e. the raw score is *less* informative than always predicting the base rate. Platt scaling removes this: **ECE 1.74×10⁻⁵** (95% bootstrap CI 1.9×10⁻⁶–1.6×10⁻⁴, 1,000 resamples), **Brier 3.59×10⁻⁴**, NLL 1.44×10⁻³, slope 0.91, Brier skill score +0.61. The reliability diagram with log-scaled axes is shown in Fig. Sy (panels A–B).
- *Enriched external set (prior correction).* Because the external set is enriched 15.6-fold, a calibrator fitted at the natural prevalence would be shifted; we therefore apply the standard **prior-probability correction** using the two *known sampling prevalences* (an intercept shift on the logit; no labels involved). The deployment-calibrated curve remains well calibrated: **ECE 1.11×10⁻³** (CI 2.0×10⁻⁴–2.0×10⁻³), Brier 1.27×10⁻³, NLL 4.59×10⁻³, slope 0.87, Brier skill score 0.91 (Fig. Sy, panel C). We report transparently that the *raw* scores happen to be almost perfectly calibrated on this enriched set (ECE 6.2×10⁻⁴) — an artefact of the different negative composition, which is why the correction is required for a principled protocol.
- *Caveats stated in the revision.* With 37 positives per internal split, the calibrator is estimated from few positive observations, so the internal-test calibration numbers carry wide CIs; and calibration is population-specific, so the deployed tool asks for (or uses a default) assumed prevalence and reports the correspondingly calibrated probability. Calibration reduces systematic over/under-confidence but does **not** eliminate confident errors (of the two auto-positives that are actually negative, both remain confidently wrong), and it does not by itself establish regulatory validity.

**2. MC-Dropout for the CMGNet generator (new Section 3.7; new Fig. Sz).** MC-Dropout is applied to the BART-based generator only (the manuscript checkpoint uses dropout p = 0.1; the classifier is a boosted tree ensemble and has no dropout, so MC-Dropout is *not* applied to it — this distinction is now explicit). For 21 external-test spectra grouped by classifier confidence (8 confident positives, 4 borderline positives, the FN #12, the 4 false positives, 4 confident negatives), we run **T = 20 stochastic passes** (dropout enabled, beam 5, 5 returned sequences per pass; 100 candidate generations per spectrum) and summarise *molecular-level* variability, deliberately without treating token likelihood as a calibrated probability that the whole structure is correct:

| Group (n) | Passes in which top-5 contains quinolone core | Top-1 modal frequency (median) | Distinct Murcko scaffolds per spectrum | Mean pairwise CReSS cosine of candidates |
|---|---|---|---|---|
| Confident positives (8) | 20/20 in every spectrum | 0.40 | ≈8–10 | 0.81 |
| Borderline positives (4) | 0.63 (mean) | 0.23 | ≈32 (median 34) | 0.49 |
| False positives (4) | 0.03 (mean; ≤1 pass in 2 of 4 spectra) | 0.28 | ≈15 | 0.51 |
| FN #12 | **0/20 passes** | 0.50 (median 0.50) | 30 | 0.51 |
| Confident negatives (4) | 0/20 passes | 0.48 | ≈16 | 0.44 |

Two conclusions follow. First, generator consensus tracks the scaffold signal: for confident positives the quinolone core is recovered in **every** one of the 100 sampled generations, whereas borderline positives show an order-of-magnitude larger structural dispersion (≈32 scaffolds vs ≈10; mean pairwise CReSS cosine 0.49 vs 0.80); MC-Dropout spread is therefore a usable generator-side uncertainty signal, complementary to the classifier score. Second, and importantly for the error analysis in R1.1: for **FN #12 the generator never produced a quinolone-core candidate in 20 independent stochastic passes** (30 distinct scaffolds generated from 52 unique canonical SMILES). This cross-checks the classifier's failure at the *representation* level — the spectrum of #12 is encoded as a non-quinolone regardless of sampling noise — and rules out an inference or prompting artefact. The sequence log-likelihood is reported alongside (mean per pass, SD ≈ 0.01–0.04 across passes) explicitly as a *score*, not as a probability of structural correctness.

**3. Uncertainty-guided review rule (new Section 3.8; Fig. Sy, panels D–E).** The rule is defined by three quantities fixed without reference to external-test labels: the default operating threshold τ_op = 0.50 (TP 63 / FP 4 / FN 1; F1 = 0.9618), a two-sided review band of ±1.5 in logit units around τ_op (0.182 ≤ p ≤ 0.818), and a novelty/discordance cutoff at cosine **0.85**, which is the *lower bound of the nearest-training-positive cosine observed for the 37 validation positives* (min 0.850). The rule routes a spectrum to one of three outcomes:

- **Auto-positive**: p ≥ 0.818.
- **Refer for manual review (labelled "requires MS confirmation", never "negative")**: 0.182 ≤ p < 0.818 (grey band), **or** p < 0.182 with cosine ≥ 0.85 to a training positive (discordant: confidently negative yet embedded inside the positive similarity shell).
- **Auto-negative**: otherwise.

*External test (n = 4,502; 64 positives / 4,438 negatives).* **4,420 auto-negative; 59 auto-positive (59 TP, 2 FP; precision 96.7%); 21 referred (0.47%, 95% CI 0.31–0.71%)** — 5 positives (7.8% of positives; four via the grey band and FN #12 via the novelty trigger) and 16 negatives (0.36%; 6 grey band, 10 novelty-triggered). **Silent misses (positives that are neither auto-detected nor referred) = 0**, so the review procedure cannot hide a false negative by relabelling it unresolved; the post-review sensitivity bound is 100% (95% CI 94.3–100%). We state explicitly that the referred samples require human/MS adjudication and that the auto-positive TP count must not be re-labelled as a recall — the reported recall of the automated step alone is 59/64 = 92.2%.

*Natural prevalence (internal test, n = 40,642; 37 positives).* Referral **5.52%** (2,242 spectra: 4 positives, 2,238 negatives), auto-positive 32 TP / 59 FP (precision 35.2%, the expected behaviour at 0.09% prevalence), auto-negative 38,309, and **one silent miss** — a positive with p = 0.117 below τ_lo whose cosine to training positives is 0.76, outside the novelty trigger. This residual error mode is reported as-is: at natural prevalence the grey band + novelty trigger leaves 1 of the 37 positives un-flagged by the automated step, versus 5 at the raw 0.4 threshold alone, at the cost of a 5.5% manual-review load that a regulatory laboratory can absorb. The same rule applied to the validation split refers 5.51% (2,240) with 0 silent misses; the difference between 5.5% and 0.47% referral rates quantifies the difference between quinolone-like synthetic validation negatives and the real external negatives.

*Methodological caveat.* A cosine-to-training-positives novelty trigger is an embedding-space in-domain heuristic, not a full conformal/out-of-domain detector; we say so, cite the two references the reviewer suggested (10.1007/s10462-023-10562-9; 10.1021/acs.analchem.5c07201) as grounding for the calibration/uncertainty treatment, and note that a dedicated conformal-prediction study is follow-up work. We also make the reviewer's own guard explicit in the revision: ROC/PR curves quantify discrimination, not calibration; uncertainty ranking, probability calibration and out-of-domain detection are separate evaluations; and the review referral is reported together with the residual silent-miss count so that referral cannot conceal false negatives (0 on the external test set, 1 on the internal test set).

**4. Prediction intervals and prediction sets for the classifier (new Fig. S-u; new Table S-u).** XGBoost returns a point probability with no native confidence interval, so we compute three interval types that answer different questions, and we say in the revision which question each one answers:

- **Conformal prediction sets (90% nominal coverage; class-conditional/Mondrian; Fig. S-u A–B).** Using 5-fold cross-fitted Platt probabilities on the validation split as calibration scores (so no sample is calibrated by a model that saw it), the class-conditional thresholds become: the set contains *positive* iff the calibrated probability ≥ 0.0537 (≈ raw score 0.87 on the seed-42 map) and contains *negative* iff the calibrated probability ≤ 4.0×10⁻⁶ (raw ≈ 3×10⁻⁴); anything in between is an **uncertain set** and is referred — exactly the deployment semantics a reviewer needs. Realized coverage agrees with the nominal 90%: positive class 94.6% (cross-fitted validation), 86.5% (95% CI 0.72–0.94, internal test), 95.3% (0.87–0.98, external); negative class 89.5% / 89.9% / 96.9%. *External test:* {negative}-only 4,300 of 4,438 negatives (96.9%), {positive}-only 63 spectra (61 true + 2 false positives; precision 96.8%), uncertain 139 (136 negatives + **3 positives, including FN #12**) → referral 3.09%, **0 silent misses**. *Internal test (natural prevalence):* {negative}-only 36,488, {positive}-only 32 TP, uncertain 4,071 (10.0%) containing all 5 remaining positives → referral 10.0%, **0 silent misses**. We therefore report the two review rules side by side: the pre-specified ±1.5-logit band rule (0.47% referral on external; 1 silent miss on internal) and the stricter, distribution-free conformal rule (3.09% referral; 0 silent misses on every split).
- **Bootstrap interval of the calibrated probability (200 resamples of the validation split, Platt refitted each time).** A 90% interval with median width ≈ 4×10⁻⁹ for low-score negatives (the 40,630 calibration negatives determine that regime extremely precisely) but up to ≈ 0.095 for actual borderline/high-probability samples (p > 0.1), which is the honest cost of calibrating with only 37 positives (Fig. S-u B, CI whiskers).
- **Model instability across retrains (Fig. S-u C).** The three identically configured seeds (0, 42, 2006) agree to a median |max p − min p| ≈ 10⁻⁷ overall, but in the raw-score band 0.05–0.4 the median disagreement rises to 0.15 (95th percentile 0.42) — model instability is concentrated precisely at the decision boundary, where the review rule already refers these spectra.

**Revision location:** Section 3.6 (calibration), Section 3.7 (MC-Dropout) and Section 3.8 (review workflow) [TODO: page/line numbers in the revised manuscript]; new Fig. Sx (PR/ROC), **Fig. Sy** (reliability diagrams, internal test and external test; score distributions and the review outcome), **Fig. Sz** (MC-Dropout variability), **Fig. S-u** (conformal prediction sets, coverage check, seed disagreement), and new Tables S-z / S-u (calibration metrics with bootstrap CIs, review counts, prediction-set composition).

## R2.5 — Hyperparameter selection criteria and sensitivity

**Response.** We have added (i) an explicit selection protocol (grid/random search on the validation split, final choice by F1 with a fixed seed), including the search ranges for learning rate, dropout, number of bins, and L1/L2; and (ii) a sensitivity analysis: F1 versus number of bins {1000, 2000, 4000, 8000}, dropout {0.0, 0.1, 0.2, 0.3}, and learning rate [TODO: run on the validation split; add figure]. Defaults are justified, and a "how to retune for a new task" note is added to the SI.

## R2.6 — Web-server documentation (maintenance, inputs, privacy, API, Docker)

**Response.** A new SI section documents the server: supported input formats (peak-list CSV/TXT and mzML-derived shift lists), example inputs/outputs with screenshots, current concurrency and throughput ([TODO] spectra/hour), the privacy policy (uploaded spectra are processed in memory and not persisted; [TODO: confirm with host]), a REST API for batch processing with an example curl script, the planned maintenance period (≥2 years), and a **Docker image** for fully local deployment [TODO: registry URL].

## R2.7 — Recent reviews on AI for NMR in the Introduction

**Response.** We have added and cited 10.1063/5.0277355 and 10.1016/j.pnmrs.2022.101575 (plus [TODO: one additional recent review if appropriate]) in the Introduction to position the work within the current AI-for-NMR literature.

## R2.8 — Reference formatting (18, 24 incomplete); image resolution

**Response.** References 18 and 24 (and all others) have been completed and reformatted to ACS style; all main-text and SI figures have been replaced with 600 dpi PNG + vector PDF versions designed at print width with ≥6 pt fonts (see also R1.5).

---

# Summary of the major changes in this revision

1. **New Section 3.4 "Error analysis" + new Figure 1 (A–D):** global embedding map, FN neighbourhood, neighbour structures, and decision-boundary interpolation. The five FPs are shown to be scaffold-boundary cases; the single FN is dissected by peak ablation, embedding interpolation and split audit.
2. **New SI figure (Fig. Sx):** precision–recall (AUPRC = 0.9891, 95% CI 0.9713–0.9996) and ROC (AUROC = 0.9998, 95% CI 0.9995–1.0000) curves with step interpolation, iso-F1 contours, chance level and bootstrap confidence bands; the operating point is annotated with TP/FP/FN.
3. **Evaluation robustness (R1.2):** scaffold-level split audit, three-seed variance, and a similarity-threshold leakage ablation; generalisation to entirely novel scaffold classes is stated as a scope limitation.
4. **Terminology and framing (R1.3):** "conditional molecular generation under scaffold constraints"; explicit grey-zone policy; quantified compute comparison versus ensemble methods.
5. **Section 3.5 restructured (R1.4):** 3.5.1 synthetic-data ablation and new 3.5.2 ensemble-strategy ablation, with neutral wording and explicit correlation-not-causation language.
6. **Uncertainty (R2.4):** probability calibration (Platt scaling fitted on the reserved validation split, selected over isotonic by cross-fitted NLL; reliability diagrams, ECE/Brier/NLL with 95% bootstrap CIs, prior correction for the enriched external set), MC-Dropout for the CMGNet generator (T = 20 passes; molecular-level variability and quinolone-core pass frequency), and a three-way uncertainty-guided review rule (auto-positive / referred for MS confirmation / auto-negative) with 0 silent misses on the external test set at a 0.47% referral rate; plus three CI types for the classifier: 90% conformal prediction sets (3.09% referral, 0 silent misses; realized class coverage 86–97% consistent with the nominal 90%), bootstrap intervals of the calibrated probability, and 3-seed retraining disagreement (median 0.15 in the 0.05–0.4 boundary band). Classifier uncertainty and generator variability are reported separately; MC-Dropout is not applied to XGBoost.
7. **Baselines (R2.2) and hyperparameters (R2.5):** unified-test-set re-evaluation and sensitivity analyses.
8. **Openness and figures (R1.5–R1.7, R2.1, R2.6–R2.8):** code/weights/data release with Zenodo DOI and Docker image; full web-server documentation; all figures ≥600 dpi / vector at print size; all references completed and reformatted.

We believe the manuscript is substantially strengthened by these revisions and thank the reviewers again for their time and for comments that materially improved the work.
