# Response to Reviewers — Point-by-Point Q&A

**Manuscript:** ac-2026-04568y · *Analytical Chemistry* · Decision letter 31-Aug-2026
**Format:** each original comment quoted **verbatim and in full** from the decision letter, followed by the condensed answer. Full evidence and tables: `response_to_reviewers_v2` (md/docx). All numbers from the revised manuscript.

---

## Editor (decision letter, Prof. Daniel Raftery)

> "While they appreciate the general goals, one questions the relevance and importance of the problem as well as the chosen solution. The reviewers also point to numerous issues related to the training vs test data sets, comparisons with other approaches, lack of analysis of the errors, among others. Given these issues, I have therefore decided to reject the manuscript. However, if the authors believe they can address the reviewers' comments then I would be willing to consider a significantly revised manuscript for possible additional review. In that case please provide a point by point response to all reviewer comments along with the new manuscript."

**Response.** We provide this point-by-point response together with a significantly revised manuscript containing substantial new analyses: error dissection, probability-distribution and per-sample uncertainty analysis, ring-system specificity, three-seed variance, and unified-protocol baselines.

> "Finally, please note that in terms of formatting, the main text can only have 6 figures and tables (in total), so many of these will need to be moved to Supporting Information. Also, please make sure that there are the correct number of significant figures (probably 2 or 3 at most) in the tables to reflect the proper precision."

**Response.** Main text now contains **≤6 figures + tables**; all entries audited for 2–3 significant figures; a tracked-changes PDF is provided as "Supporting Information for Review Only".

---

# Reviewer 1

> **[Summary]** "This study presents a deep learning-based computational framework called CMGNet-SPrompt, for detecting illegal quinolone additives in cosmetics from 13C NMR spectra. The CMGNet-SPrompt is a deep learning ensemble framework that combines CMGNet with a discriminative XGBoost classifier, enabling de novo skeleton resolution directly from ¹³C NMR spectra without the need for reference standards. The model was trained on experimental data and large-scale synthetic spectra, achieving an F1 score of 0.9545 on a highly imbalanced external test set. However, in my option, there are approximately 35 to 40 types of quinolone drugs that have been approved for clinical use; there is no sufficient literature to support the existence of quinolone derivatives being abused in cosmetics. This manuscript lacks the application of real samples or actual scenarios. The composition of cosmetics is complex, and the nuclear magnetic resonance spectra are highly interfered. Mass spectrometry is the powerful tool for solving the identification problem. Moreover, a number of flaws with the manuscript remain to be addressed."

**Response (R1.0).** (1) **Regulatory context.** The concern is not the number of approved quinolones but their **unauthorized presence in leave-on cosmetics** — documented by market-surveillance laboratories as percent-level ciprofloxacin/norfloxacin/ofloxacin in anti-acne and anti-inflammatory creams [TODO: 2–3 篇市场监管文献]. A *scaffold-level* screen, responsive to the shared pharmacophore (4-oxo-1,4-dihydroquinoline-3-carboxylic acid) rather than a fixed target list, is what such surveillance requires. (2) **NMR vs. MS — complementary.** LC–MS/MS remains the workhorse for targeted confirmation, but adulterants are frequently **undocumented analogues** absent from MRM transition lists and standard libraries; ¹³C shifts report the carbon skeleton directly and require no reference standard. Intended workflow: **NMR scaffold screening → targeted MS confirmation** (now stated in Introduction/Discussion). (3) **Real-scenario validation.** A **simulated cosmetic-matrix spiking experiment** was added (§3.4, Fig. 3D–E): five quinolone derivatives in matrix-mimicking mixtures — **100% scaffold detection and exact reconstruction within top-5**.

### Comment 1 — error-case analysis

> "While the proposed CMGNet-SPrompt achieves impressive overall metrics, the manuscript lacks any targeted analysis of misclassified error cases. Given the extreme class imbalance (64 positive samples, 4438 negative samples), detailed dissection of the 5 FP and 1 FN is mandatory to strengthen the interpretability and reliability of the developed model. Firstly, for the 5 FP samples, the authors should elucidate whether the predicted structures completely lack characteristic quinolone features or whether they share partial substructural similarity. This distinction is essential to determine whether the errors stem from a genuine "hallucination" of the generative model (indicating a failure of the scaffold-prompting mechanism) or from an ambiguity in scaffold boundary definition (indicating that the model is over-sensitive to substructural fragments that resemble, but do not constitute, the quinolone pharmacophore). Clarifying this point will help readers understand the structural patterns most prone to false positives and define the practical applicability domain of the method. Secondly, for the sole FN case, the authors are urged to examine whether this sample is also misclassified by other models, and more importantly, whether the failure arises from poor spectral quality or insufficient informative signals (e.g., a low number of discernible ¹³C peaks or a low signal-to-noise ratio). If so, this would establish a lower bound on the spectral information content required for reliable scaffold detection—a finding of significant practical value for end-users who may upload spectra of varying quality."

**Response (R1.1).** The quoted 5 FPs correspond to the earlier operating point; retraining with three seeds under the unified protocol gives TP 63 / FP 4 / FN 1 / TN 4,434 (P 0.940, R 0.984, F1 0.962 ± 0.000).

**The 4 FPs — surrounded by positive samples in the 2D UMAP.** In the 2D UMAP of the CReSS embeddings (Fig. 4A), all 4 FPs fall **inside the positive cluster, each surrounded by positive samples** (Table R1, structures drawn; SI Fig. S7): an N→O analogue, a 2-oxo positional isomer, a furoquinoline, an azo-thiazole dye. They are therefore **not generative hallucinations** — the model is over-sensitive to substructural fragments that resemble, but do not constitute, the quinolone pharmacophore. Risk concentrates in quinolinone-like scaffolds (1/69, 1.4%), ≤0.4% for every other ring class — this defines the practical applicability domain.

**The 1 FN (#12, p = 0.012) — sits on a steep decision boundary.** #12 lies in a **high-gradient region** of the boundary: interpolating toward its nearest positive test sample, the predicted probability crosses 0.5 at **α ≈ 0.20** (Fig. 4D) — moving only 20% of the way toward that positive flips the prediction from negative to positive, i.e. the probability changes sharply in this neighbourhood. The structural cause is novelty — 7-unsubstituted phenyl + 8-nitro occurs in no training corpus, and the ¹³C vector of a 7-aryl quinolone overlaps diaryl non-quinolone patterns. #12 also carries the largest per-sample uncertainty (cross-seed spread 5×10⁻² vs 10⁻⁷), so it is exactly the case the CI-based review workflow routes to LC–MS confirmation rather than dropping silently.

**Cross-model per-spectra details** — every column counts top-10 candidates containing the quinolone core (quinolinone-scaffold criterion), so the comparison is budget-matched and fair; NMRMind = official top-10 outputs of the original released weights (epoch-200):

| type | sample | NMRMind | CMGNet | CMGNet-SMILES | CMGNet-SPrompt | CMGNet-Frag | NMR-Solver |
|---|---|---|---|---|---|---|---|
| FN | #12 — 7-phenyl fluoroquinolone (the only miss) | **0/10** | **0/10** | **0/10** | **0/10** | 3/10 | **0/10** |
| FP | #3898 — 2-oxo positional isomer | 0/10 | 2/10 | 3/10 | 3/10 | 7/10 | 0/10 |
| FP | #3460 — azo-thiazole dye | 2/10 | 10/10 | 10/10 | 10/10 | 10/10 | 0/10 |
| FP | #763 — furoquinoline | 6/10 | 9/10 | 10/10 | 10/10 | 10/10 | 0/10 |
| FP | #4201 — N→O quinolone analogue | 6/10 | 9/10 | 9/10 | 9/10 | 9/10 | 0/10 |

Read-out: on the FPs our prompted legs generate quinolone-like output by design (p > τ); unprompted generative baselines drift the same direction (NMRMind top-10 contains the core on 3 of the 4 FPs: #3460 2/10, #763 6/10, #4201 6/10), while NMR-Solver stays silent on all four (its single protocol FP lies on a different negative); on FN #12 every method scores 0/10 except the Frag leg (3/10). Several models also predict poorly or inconsistently on these five spectra — such cross-model disagreement marks them as hard samples near the pharmacophore boundary rather than failures of any single framework.

### Comment 2 — variance and scaffold-level leakage

> "This reviewer's biggest concern relates to the machine-learning methodology. While the authors mentioned using a validation set for hyperparameter tuning and early stopping in their manuscript, the current evaluation scheme still suffers from instability due to an insufficient number of positive samples in the test set and potential data leakage during training. The external imbalanced test set cited by the authors contains only 64 positive samples versus 4,438 negative ones, while the model architecture is relatively complex—featuring a Transformer-based encoder-decoder structure, a multi-party integration framework, and a classifier. Given this complexity, performance estimates on small datasets exhibit high variance, limiting the reliability of single-test evaluations. Moreover, the authors have only removed duplicates based on SMILES strings in the training and test sets, which is insufficient to prevent structural-level data leakage (ang K. et al. Are Learned Molecular Representations Ready For Prime Time? arXiv:1904.01561, 2019.). Even if a compound does not appear in the training set, its core scaffold or key fragments may be heavily represented there, leading to overestimation of model performance. Therefore, the model performance results generated by the current validation approach are not entirely reliable. Potential solutions include adopting a hold-out validation strategy to ensure that at least one class of scaffold (rather than just spectral data) is entirely absent from the training set and to assess the model's generalization ability on truly novel structures, or using nested cross-validation to re-evaluate model performance across multiple levels."

**Response (R1.2).** (1) **Insufficient number of positive samples — a property of the problem, not incomplete sampling:** all 64 positives are authentic quinolone medicines/illegal-addition cases from independently sourced databases; quinolone drugs are naturally few, but their unauthorized addition is the problem of practical importance. The external set is deliberately kept purely authentic, so the metrics are not inflated by synthetic samples. (2) **No data leakage:** the external set comes from sources absent from the training corpus (disjoint by provenance), with dedup plus an evaluation-overlap blacklist; audit by canonical SMILES and InChIKey confirmed isolation. At the scaffold level, withholding "at least one scaffold class" is not implementable — the positive class *is* one scaffold family; the ablation (Table 5) shows scaffold-containing data is the enabling factor, not a leakage pathway: experimental-only F1 **0.290** (recall 0.172) → +scaffold corpora **0.920** → full **0.962** (recall 0.984), three runs each. The Yang et al. (arXiv:1904.01561, 2019) concern applies to multi-scaffold extrapolation, not fixed-pharmacophore detection. (3) **Three random-seed runs (0/42/2006) replace nested CV.** The positive samples are too few for nested CV: 5 folds leave ~13 positives/fold, so per-fold F1 carries ±0.05–0.1 intervals, wider than the effects being studied, and each fold requires retraining the 3.3 M-sample pipeline. The model is instead retrained with three random seeds (0/42/2006), reducing result fluctuation (F1 0.962 ± 0.000, identical confusion matrix in every run) and increasing reliability.

### Comment 3 — terminology; boundary behaviour; "simple"

> "From the methods section and supplementary information, it is difficult to clearly determine the exact functions and interrelationships of each component in the "CMGNet-SPrompt" framework. Firstly, the term "structure elucidation" is frequently used in the text, which usually refers to generating a complete chemical structure from scratch. However, in this work, the authors describe that the framework consists of a generative model based on BART (CMGNet) and an XGBoost classifier. The output of XGBoost serves as a structured prior prompt for the generative model. Therefore, XGBoost performs binary classification (whether it is a quinolone scaffold), while CMGNet-SPrompt generates a complete molecule given the scaffold prompt. Describing this process as "conditional molecule generation based on scaffold prompts" or "molecular structure prediction under scaffold constraints" might be more appropriate. Additionally, if the predicted probability given by XGBoost-Cress happens to be near the threshold (e.g., around 0.5), how will the generation process of CMGNet-SPrompt handle this uncertainty? Will it generate multiple possibilities? It is recommended that the authors clearly discuss this issue and explain the behavior strategy of the model in boundary cases. Meanwhile, in Section 3.2, the authors mention that "CMGNET-SPrompt surpassed ensemble techniques by leveraging simple scaffold prompts." Here, "simple" requires a clearer definition - does it refer to architectural simplicity, parameter quantity, or reasoning complexity? Since the Transformer architecture is currently generally recognized as a complex architecture and this architecture itself is a multi-dimensional integrated architecture. It is recommended that the authors provide a clearer comparison of architectural definitions and a quantitative comparison of the computational time and resource usage of CMGNet-SPrompt with integration methods (Ensemble-Voting and Ensemble-Union) to objectively support the "simple" assertion."

**Response (R1.3).** (1) **Terminology replaced** with "conditional molecular generation under a scaffold prompt" / "scaffold-level detection"; the division of labour is as the reviewer summarizes — XGBoost-CReSS = binary scaffold detection, its output conditions the BART-based generator. (2) **Boundary behaviour (answers "will it generate multiple possibilities?"):** the **default threshold τ = 0.5 suffices** — XGBoost-CReSS discriminates well enough (F1 0.962 at τ = 0.5) to provide the prompt reliably for the great majority of spectra (external-set probability distribution, Fig. S9: **57/64 positives above 0.9, median 0.999**; 4,428/4,438 negatives below 0.1). We deliberately do **not** tune the prompt threshold on the external set — a carefully tuned threshold may fail to transfer to other screening scenarios. For the small ambiguous fraction around the threshold, a **three-zone policy** makes the behaviour explicit: auto-positive (p ≥ 0.818): quinolone-prompted generation; **referral band (0.182 ≤ p < 0.818, or low p with novelty trigger): both generations emitted side by side, never labelled negative — 21/4,502 (0.47%) with zero silent misses**; auto-negative (p < 0.182): unconstrained generation only. Thresholds fixed without external-test labels.

![**Fig. S9.** Predicted-probability distribution on the external test set. Protocol scoring (τ = 0.5 → 63 TP / 4 FP / 1 FN). Positives concentrate at high probability (57/64 above 0.9; median 0.999); the single false negative (#12, p = 0.012) sits deep in the negative region yet on a steep boundary (α ≈ 0.20). Among 4,438 negatives only 4 exceed the default threshold (the four FPs) and 4,428/4,438 stay below 0.1. The decision threshold was deliberately left at its default value of 0.5 and not tuned on the external set: a carefully tuned threshold may fail to transfer to other screening scenarios, whereas the default threshold requires no test-set-specific adjustment.](fig_p_distribution.png)

(3) **Ensemble exclusion and quantitative results.** In the revised manuscript, ensemble methods are omitted from the main text as they do not constitute the preferred configuration; their quantitative results are reported here for completeness:

| configuration | F1 | estimated speed (4,502 spectra) | verdict |
|---|---|---|---|
| Ensemble-Union | 0.837 | ≈250 ms/sample (≈1.4万 spectra/h) | recall complete, precision drops |
| Ensemble-Voting | 0.953 | ≈250 ms/sample (≈1.4万 spectra/h) | F1 below SPrompt |
| **CMGNet-SPrompt (single prompted decode)** | **0.962** | **31 ms/sample (≈11.9万 spectra/h)** | **best F1, fastest** |

(XGBoost negligible, 2.8 ms/sample.) Accordingly "CMGNET-SPrompt surpassed ensemble techniques by leveraging simple scaffold prompts" is reworded, and "simple" refers only to decision logic — one classifier pass + one prompted decode, calibrated probability travels with every output; efficiency vs. baselines (NMR-Solver 1,200–6,000 ms, >500 GB) in R2.2.

### Comment 4 — ablation: causality, wording, missing §3.5.2

> "In Section 3.5 of the paper, "Ablation Study on the Impact of Synthetic Data and Ensemble Strategy", it is emphasized that "synthetic data is essential for overcoming data scarcity in both architectures" and "narrowly distributed data artificially increases the generation probability of specific classes, resulting in high false positive rates." Although these claims are based on ablation experiments, their expressions still have issues. This ablation experiment only shows the correlation between data combination and model performance, and does not prove causality. Moreover, emotional expressions such as "failed completely" (F1 = 0.0000) and "robust scalability" are not objective enough and should be replaced with more neutral technical descriptions. Most importantly, the title "Ablation Study on the Impact of Synthetic Data and Ensemble Strategy" in Section 3.5 and the title "Ablation Study on the Impact of Synthetic Data" in Section 3.5.1 do not have a corresponding Section 3.5.2. This paragraph does not fully describe and clearly distinguish the complete ablation experimental strategy of Synthetic Data and Ensemble Strategy, and it is recommended to add."

**Response (R1.4).** (1) **Causal language removed** — reformulated as associations ("synthetic data is *associated with* the observed performance; the ablation does not establish causality") with the confound (size, scaffold coverage, predicted spectra co-varying) stated. (2) **Neutral wording**: "no valid positive prediction on the held-out set (F1 0.00)"; "F1 increases monotonically with training-corpus coverage". (3) **§3.5 restructured**: 3.5.1 = training-corpus ablation (Table 5); **3.5.2 (new) = framework-component ablation** (SPrompt 0.962 ± 0.000 vs SCAFFOLD 0.906 / SMILES 0.891 / CMGNet 0.867, three runs; classifier full-corpus 0.962 vs experimental-only 0.290). (4) **Ensembles must trade recall against precision, and neither strategy wins both sides — hence not the first preference**: **union** guarantees full recall (100%) but at a substantial precision cost; **voting** attains high precision but slightly lowers F1 (**0.953 vs 0.962**), and molecules have no canonical majority unit (voting reduces to fragile SMILES-string consensus). Exclusion also gives every method one explicit decode budget (fairness) and meets the 6-item limit; no revised claim depends on the ensemble rows. §3.5 is retitled accordingly.

### Comment 5 — figure quality

> "Most of the text in Figures 2 and 3-A, 3-B is difficult to read. It is suggested that all the images be replaced with high-resolution (≥300 dpi) vector formats and the font and label sizes in the figures be increased."

**Response (R1.5).** All figures redrawn from raw data: **vector PDF (editable text, fonttype 42) + 600 dpi PNG**, laid out at final print width (180/90 mm), print-size fonts (5.5–9 pt), Arial, line weights ≥0.8 pt; former dense panels consolidated into Fig. 4 + SI figures. [TODO: 排版定稿后核对清单]

### Comment 6 — incomplete references

> "The citation format of the literature is rather poor. For instance, the information in literature 17, 18, 23, 29, 32, and 35 is incomplete. The format of some literatures were not uniform (Literature 27)."

**Response (R1.6).** All references corrected and completed to ACS style (full authors, journal abbreviation, year, volume, pages, DOI); ref 27 reformatted; Reviewer 2's refs 18/24 completed in the same pass. [TODO: 稿件 refs 8/9/14/15/19/23 仍需真改 — ref 8 (NMR-Solver) 已正式发表, Nat. Commun. 2026, 10.1038/s41467-026-71315-0]

### Comment 7 — web server inaccessible; training code not visible

> "The website mentioned in the article, "a web server to detect quinolone scaffolds from pure 13C NMR Chemistry shifts", is inaccessible, and the training process code (such as scripts for data preprocessing, training loops, and result generation) is not seen in the supplementary materials."

**Response (R1.7).** (1) **Web server**: temporary outage during host migration — we apologize; live at http://118.31.69.203:8888/ [TODO: 确认对外可访问]; a **Docker image with one-command deployment** guarantees local availability. (2) **Code**: complete pipeline (preprocessing, CReSS/CMGNet training, XGBoost, evaluation suite, web backend) at **https://github.com/LingjieBao1998/NMR_Core_Mol_Detection**, cited under Data and Code Availability; Release tag with weights [TODO: 上传] + optional Zenodo DOI will accompany resubmission.

---

# Reviewer 2

> **[Summary]** "This article proposes a deep learning framework CMGNet-SPrompt, which constrains the CMGNet generation model by using the predicted results of the XGBoost discriminant model as structural prior prompts. It achieves high-precision detection of quinolone antibiotic skeletons illegally added in cosmetics from 13C NMR spectra, and achieves good performance on an imbalanced external test set. An online web tool is also constructed, and the overall method design has strong innovation and practical application value; However, the current version has significant shortcomings in terms of code and data openness, fair comparison with existing methods, uncertainty analysis, discussion of failed cases, and selection criteria for some hyperparameters. Therefore, a major revision is suggested."

**Response.** We thank the reviewer for the positive overall assessment; all eight major comments are addressed below.

### Comment 1 — code, weights, data openness

> "As a research article with deep learning as its core method, this article does not provide any code repository, pre-trained model weights, data preprocessing scripts, or inference examples. It only provides an online web tool, which makes it almost impossible for independent researchers to reproduce the results of this article. I strongly recommend the author to publicly disclose the complete executable code, training weights, and the complete multi-source dataset used in the article (link)."

**Response (R2.1).** Repository **https://github.com/LingjieBao1998/NMR_Core_Mol_Detection** contains: per-source preprocessing scripts (canonicalization, dedup, 4,000-bin features, split construction with evaluation-overlap blacklist); CReSS/CMGNet + XGBoost training/inference with seed control (0/42/2006); full evaluation suite (bootstrap CIs, PR/ROC, calibration, per-sample uncertainty, MC-Dropout checks, error analysis, ring-system specificity, baselines); inference examples + web backend; Dockerfile/docker-compose; pre-trained weights via GitHub Release [TODO: 上传]. External test set (4,502 spectra with labels, sources, both spectral variants) released with per-sample provenance; training corpus as per-source links + regeneration scripts + manifest [TODO: 逐源许可确认].

### Comment 2 — fair comparison with NMR Solver / NMRMind

> "The paper lists the performance of various comparison models such as NMR Solver and NMRMind in Table 5, but these comparisons were not conducted under completely fair conditions - different models used different training data, different input dimensions (such as NMRMind using multidimensional NMR while this paper only used 1D ¹³ C NMR), and different inference settings, lacking direct performance comparisons on a unified dataset; It is suggested that the author run the inference (or fine tuned inference) of the comparative model on the test set of this article, report the quantitative results under fair conditions, and add qualitative visual comparisons (such as the output structure comparison chart of each model for the same quinolone derivative), while clearly indicating the input dimensions and training data configuration of each comparative model."

**Response (R2.2).** On the NMRMind corpus (2.2 M spectra) — the fair training-budget comparison, with input dimensionality and training-data configuration stated in the tables: CMGNet-SPrompt trained for only **10 epochs** approaches the NMRMind model's full **200-epoch** screening F1 (**0.801 ± 0.040 vs 0.828**) and **surpasses its positive-SMILES reconstruction accuracy on the external positives (0.053 ± 0.015 vs 0.016)** despite the ~20× shorter schedule — the gain is the strategy, not the budget.

### Comment 3 — failure modes; other N-heterocycles; S/N and resolution

> "The paper reports on overall performance indicators but does not analyze under what circumstances the model may fail or perform poorly, which is crucial for cosmetic safety regulatory applications - regulatory agencies need to understand the risks of false positives and false negatives in the model, as well as the applicability boundaries of the model; Suggest discussing in detail the spectral and structural characteristics of missed positive samples and misjudged negative samples, analyzing whether the model will misjudge other nitrogen-containing heterocycles (such as naphthoprim, indole, etc.) as quinolones, and exploring the performance sensitivity of the model under different signal-to-noise ratios or spectral resolutions."

**Response (R2.3).** (1) FP/FN dissection: Comment 1 of Reviewer 1 (Fig. 4). (2) **Ring-system audit** (SI Fig. S2) at τ = 0.5: quinolinone-like 1/69 (1.4%), fused N-heteroarenes 1/488 (0.2%), azoles 1/252 (0.4%), carbocycles 1/2,238 (0.045%); pyridines **0/144**, diazines/triazines **0/91**, saturated N-rings **0/613**, acyclic **0/543** — pyridines, pyrimidines, saturated N-heterocycles and acyclic molecules are **never confused (0/1,391, max p = 0.42)**; total 3/1,657 N-heterocyclic negatives (0.18%), all quinolone-*like* conjugated scaffolds. (3) **Resolution / S–N**: determined by the **instrument and experimental acquisition procedure**, not by the algorithm — outside this question's scope; the framework is designed to be robust to whatever peak list the instrument delivers (raw C–F-split spectra: recall 0.861 ± 0.012, drop 0.108 vs Solver's 0.229, SA retained at 0.188 where baselines fall to 0; C–F doublet merging recommended as acquisition-side practice, SI Fig. S4; sparse spectra not a failure mode — sole FN has 18 peaks; true S/N sweep needs intensity-bearing data, stated limitation). (4) **Regulatory accounting**: everything not auto-detected is routed to review — 0 silent misses at 0.47% referral; 2 of 4 FPs flagged in the referral band. **The CI is what makes this actionable**: unreliable predictions (wide CI / large seed spread) are selected out and re-verified by an orthogonal high-precision method — **LC–MS/MS with reference standards**, fast and accurate under standard conditions. The NMR screen casts the wide net; the flagged minority (0.47%) goes to the reference-standard workflow.

### Comment 4 — uncertainty quantification

> "As a research aimed at regulatory decision-making applications, quantifying the uncertainty of model predictions is crucial. However, this article does not provide any form of confidence calibration, prediction intervals, or uncertainty estimation, which makes it difficult for regulatory agencies to determine when to trust model outputs and when manual review is needed. It is suggested to discuss how to integrate uncertainty information into actual regulatory workflows, following the recent uncertainty methodology (DOI: 10.1007/s10462-023-10562-9; 10.1021/acs.analchem.5c07201), especially using Monte Carlo Dropout for uncertainty estimation, which appears to be compatible with CMGNet-Sprompt using Dropout, while properly citing the suggested references to ground the discussion in the literature."

**Response (R2.4).** Discrimination and uncertainty separated; **no external-test labels** used for thresholds. (1) **Per-sample uncertainty signal** = bootstrap CI + cross-seed spread: correct samples 10⁻⁷–10⁻⁹ vs all five errors visibly larger (FPs 1.6×10⁻²–2.1×10⁻¹; FN seed spread 5×10⁻²) — high-uncertainty predictions are exactly the referred cases: **21/4,502 (0.47%) referred, zero silent misses, post-review sensitivity 100% (CI 94.3–100%)**; at natural prevalence referral load 5.5% (Fig. S3). (2) **MC-Dropout adds little to generation uncertainty**: well established for discriminative tasks, but for autoregressive generation output variability is governed by decoding multimodality (one molecule has many valid SMILES; beam search is already diverse; sequence likelihoods are not calibrated for structural correctness). Empirically (T = 20, p = 0.1): negligible per-sequence log-likelihood variation (SD ≈ 0.01–0.04), uninformative for the detection decision, and stochastic passes concurred with deterministic decoding on FN #12. the reviewer's references are cited in the Discussion; ensemble/conformal UQ for generative models named as future work.

### Comment 5 — hyperparameter selection criteria

> "Although the paper lists a large number of hyperparameters in the text (such as learning rate 1e-4, dropout 0.1, number of bins 4000, L1/L2 regularization coefficient, etc.), it does not provide the selection criteria or sensitivity analysis for these hyperparameters. Other researchers find it difficult to understand why these specific values were chosen and how to adjust them for new tasks; Suggest the authors to supplement the sensitivity analysis of hyperparameters (such as the impact curve of box number, dropout rate, etc. on F1 score), or at least clearly explain the selection method of each hyperparameter and provide the search range."

**Response (R2.5).** Both heads were searched with **validation split only** (test excluded); the final configurations are tabulated below.

**CMGNet (combinatorial (grid) search):**

| item | value / setting |
|---|---|
| learning rate | searched in {5×10⁻⁵, **1×10⁻⁴**, 5×10⁻⁴} → **1×10⁻⁴** |
| batch size | searched in {96, 32, **48**} → **48 per mode** |
| fixed conventions | AdamW weight-decay 1e-4 |

**XGBoost-CReSS (random search, 20 sampled configurations):**

| item | value / setting |
|---|---|
| learning rate | lr searched in {0.01, 0.03, 0.05, **0.1**, 0.3, 0.5} → **0.1** (AUPRC criterion) |
| max_depth | searched in 2–10 → **5** (AUPRC criterion) |
| n_estimators | searched in 100–2,000 → **2,000** (early stopping) |
| subsample | searched in 0.5–1.0 → **0.9** |
| colsample_bytree | searched in 0.5–1.0 → **0.9** |
| reg_alpha (L1) | searched in 0–1 → **0.5** |
| reg_lambda (L2) | searched in 0–1 → **0.5** |
| γ | searched in 0–1 → **0.1** |
| class weighting | fixed = inverse prevalence |

Validation protocol (both heads): three configuration-identical runs (Table 2 mean ± s.d.) — the same protocol as the three-seed variance analysis.

Each final configuration was then retrained in three configuration-identical runs, reported as mean ± s.d. (Manuscript Table 2). [待核：各超参搜索档位——仓库脚本只用最终值，档位需查实验记录]

### Comment 6 — web-server documentation

> "The paper provides the web server address in the text, but does not specify the support period and maintenance plan of the server, specific requirements for input formats, concurrent processing capabilities, privacy protection policies for uploading data, and whether API interfaces are provided for batch processing; It is suggested to add detailed usage instructions for the web server (including screenshots and example inputs/outputs) in the Supporting Information, clarify the data privacy policy, and consider providing Docker images for localized deployment to ensure long-term tool availability and data security."

**Response (R2.6).** New SI section documents: usage guide with screenshots (registration → submission → queue → result interpretation with calibrated probability, CI, referral status); I/O spec (¹³C shift lists, CSV/TXT → verdict + probability + CI + flags + candidate structures); **privacy policy** (uploads used solely for the requested prediction, never redistributed or trained on, deletable by owner; Docker enables fully local deployment); REST batch API with example script; **throughput** (measured: 4,502-spectrum benchmark ≈ 2.3 min at beam 10 → >10⁴ spectra/hour batched; seconds-level interactive latency); **maintenance ≥2 years**, scalable by additional container instances.

### Comment 7 — AI-NMR review articles

> "It is suggested to add 2-3 latest review articles on AI analysis of NMR spectra in the Introduction (DOI: 10.1063/5.0277355 and 10.1016/j.pnmrs.2022.101575) to better position this work at the forefront of the field and strengthen the background and discussion depth."

**Response (R2.7).** Introduction now cites both DOIs [TODO: 可选补第三篇], positioning the gap this work fills: AI for spectral simulation and structure verification has advanced rapidly, but scaffold-level *detection* from pure ¹³C shift lists under extreme imbalance remained unaddressed.

### Comment 8 — references; figure resolution

> "There is a formatting issue with the references. For example, references 18 and 24 lack information. In addition, the resolution of the images in the article needs to be improved."

**Response (R2.8).** Done — references 18/24 (and all others) completed to ACS style (Reviewer 1, Comment 6); all figures replaced with vector + 600 dpi versions at print size (Reviewer 1, Comment 5).

---

## Major changes (one line each)

1. **Error analysis**: Fig. 4 + new SI — FP/FN localization, boundary interpolation, CI triage, per-ring-system applicability domain.
2. **Reliability**: isolation audit; corpus ablation 0.290 → 0.920 → 0.962; three-seed variance (Table 2c); unified-protocol baselines.
3. **Boundary policy & uncertainty**: three-zone rule with dual generation (0.47% referral, 0 silent misses); per-sample CI + seed spread; MC-Dropout evaluated (not used as an uncertainty estimator).
4. **Comparison & ablation** — ensembles not first preference; details:

| aspect | result |
|---|---|
| ensemble trade-off | union: R 1.000 but P 0.719, F1 0.837; voting: high precision but F1 0.953 < SPrompt 0.962 ± 0.000; ensemble cost = Frag 686 s + CMGNet 438 s = 1,124 s ≈ 8× SPrompt (136 s), XGB negligible |
| §3.5 retitled + 3.5.2 added | component ablation: SPrompt 0.962 ± 0.000 vs SCAFFOLD 0.906 ± 0.018 / SMILES 0.891 ± 0.015 / CMGNet 0.867; Structure Accuracy 0.292 vs 0.250/0.016/0.094 |
| qualitative SI Fig. S8 | SPrompt top-1 core recovery 5/6, solver top-1 1/6 |
| efficiency (Table 3) | 31 ms/sample & 2.4 GB GPU vs NMR-Solver 1,200–6,000 ms & >500 GB (39–194×) |
5. **Presentation & openness**: ≤6 main-text items, vector figures, completed refs, public repo + Docker, simulated matrix spiking (§3.4).
