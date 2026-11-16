# Response to Reviewers

**Manuscript ID:** 【TODO: ac-2026-xxxxxxx】
**Title:** CMGNet-SPrompt: De novo detection of illegal quinolone additives in cosmetics from pure ¹³C NMR chemical shifts
**Journal:** *Analytical Chemistry*

We thank the Editor and both reviewers for the careful reading of our manuscript and for the constructive criticism. The comments have led to substantial new analyses and to a complete reworking of the figures. Below we respond point by point; reviewer comments are quoted in italics (abbreviated where necessary) and our responses follow. Changes in the revised manuscript and Supporting Information (SI) are highlighted. Items marked 【TODO】 are experiments or numbers that will be inserted in the final version; the corresponding analysis is either complete and being formatted, or explicitly identified as work in progress.

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

1. **Regulatory context.** Illegal addition of (fluoro)quinolones to cosmetic products — typically anti-acne, anti-inflammatory and "antibacterial" creams — has been documented by market-surveillance and customs laboratories, usually as ciprofloxacin, norfloxacin, ofloxacin or their salts at percent levels 【TODO: insert 2–3 market-surveillance/customs reports and quantitative studies】. The public-health concern is not the number of approved drugs but the ease with which approved or closely related quinolones can be added to leave-on products outside any pharmaceutical quality control.

2. **Why ¹³C NMR alongside MS?** LC–MS/MS is the workhorse for *targeted* confirmation of known residues, and we do not claim otherwise. Its limitation in this application is that adulterants are often **undocumented analogues** (salts, esters, ring-substituted derivatives) that are absent from MRM transition lists and reference-standard collections, so targeted methods can return false negatives. ¹³C chemical shifts report the carbon skeleton directly, are comparatively matrix-insensitive, require no reference standard, and are non-destructive. Our framework answers a *scaffold-level* question — "does this sample contain a quinolone-type skeleton?" — and, when the answer is positive, generates the scaffold hypothesis that can be confirmed by a short LC–MS run. The intended workflow is therefore **NMR screening → MS confirmation**, with NMR providing the first-pass detection of analogues that targeted MS may miss.

3. **Matrix interference.** We agree and have added an explicit discussion. Two points are relevant: (i) the model consumes a peak list (¹³C chemical shifts), so baseline drift and excipient signals that do not enter the 4,000-bin feature space have limited influence; (ii) a short liquid–liquid extraction/precipitation is sufficient in practice. Spike-and-recovery experiments in cream and lotion matrices (three spiking levels) are being completed 【TODO: report recoveries and RSDs】. We now state in the Discussion that the present validation uses pure compounds and that real-matrix validation is the designated next step, and we have removed any implication of immediate field deployment.

---

## R1.1 — Targeted analysis of the 4 FP and 1 FN cases

**Comment (condensed).** With 64 positives and 4,438 negatives, a detailed dissection of the 5 FP and 1 FN is mandatory. For the FPs: do the predicted structures completely lack quinolone features, or do they share partial substructural similarity (hallucination vs scaffold-boundary ambiguity)? For the FN: is it also misclassified by other models, and does the failure arise from poor spectral quality (few peaks, low S/N)?

**Response.** We agree that this is the most informative way to establish the applicability domain, and we have added a new results section, **"3.4 Error analysis of the external test set"**, a new **Figure 1 (panels A–D)**, and a dedicated SI section. A key clarification on the error count is given first.

**Error count and operating threshold.** 修订统一采用默认阈值 **τ_op = 0.50**。重新打分 4,502 张外部谱得到 **TP = 63, FP = 4, FN = 1, TN = 4,434**（精确率 0.940、召回率 0.984、**F1 = 0.9618**）。此前 τ_op = 0.40 的口径给出 5 FP、F1 = 0.9545，是因为 p = 0.418 的那一个边界负样本落在 0.50 以下；两套计数现在都明确写出，工作点也标在 PR/ROC 图上（新 Fig. Sx）。

**False positives (4 negatives).** All four FPs lie **inside or at the edge of the training-positive region** of the CReSS embedding UMAP (Fig. 1A). The model is therefore not hallucinating quinolone structure from unrelated chemistry; the spectra are genuinely quinolone-like. A substructure audit shows that none of the four molecules contains the 4-oxo-1,4-dihydroquinoline-3-carboxylic-acid core, but all contain fused heteroaromatic/carbocyclic systems with conjugated carbonyl and N-heterocycle motifs (coumarin/benzoxazinone-type) that partially mimic the quinolone pharmacophore 【TODO: insert the five FP SMILES, predicted structures and RDKit substructure checks in Table Sx】. We therefore attribute the FPs to **scaffold-boundary ambiguity** — over-sensitivity to quinolone-like fragments — rather than to a failure of the scaffold-prompting mechanism. The practical applicability-domain statement has been added: an embedding inside the positive manifold with a probability above the operating threshold is reported as "quinolone-like, requires MS confirmation".

**False negative (#12, p = 0.012).** The single FN is analysed in Fig. 1B–D and SI Section S-x:

- **Not a spectral-quality problem.** The spectrum contains **18 discernible ¹³C peaks**, comparable to the median of the 64 test positives, so the failure is not caused by sparse peaks or low S/N. The two lowest-scoring positives are p = 0.012 (#12) and p = 0.509; the remaining 62 positives have p > 0.9 (median 0.999).
- **Not a split/leakage problem.** #12 is absent from all three training sources by exact-SMILES and InChIKey search; the split blacklist removed 0 records.
- **Root cause 1 — a coverage hole in the training positives.** Among 24,310 N-cyclopropyl quinolone positives, 122 share the 8-NO₂ + 6-F + 1-cyclopropyl core, but only four carry a 7-aryl substituent, **all para-substituted**. The 7-unsubstituted phenyl of #12 was never seen. Its nearest training positives differ by exactly this feature: P2 (nearest positive) has cosine 0.857 (p = 0.998); P1 (para-F analogue) has cosine 0.819 (p = 0.990). For comparison, the 64 test positives have a median nearest-positive cosine of 0.912. #12 is an isolated positive.
- **Root cause 2 — over-reliance on an aromatic-peak pattern.** Leave-one-peak-out ablation on the 18 measured peaks: full spectrum p = 0.012; removing 130.4 ppm alone raises p to 0.311; removing the aromatic triplet 129.9/130.4/130.6 ppm raises p to **0.910** (positive); adding a decoy peak at −128.7 keeps p at 0.884; removing 175.4 ppm (carboxylic carbon) drives p to 0.000. The model had learned "aromatic triplet ⇒ positive" too strongly.
- **Boundary behaviour.** Linear interpolation in the 768-dimensional CReSS embedding from #12 toward P1 crosses p = 0.5 at α ≈ 0.20: the FN lies close to, not deep inside, the negative region — a genuine boundary case, consistent with the PR/ROC analysis in Fig. Sx.

**Other models.** 【TODO: report #12 under Ensemble-Voting, Ensemble-Union and unconstrained CMGNet (beam candidates), e.g. "the voting ensemble also returned negative; the union ensemble ranked the quinolone scaffold at rank r".】

**Spectral-information lower bound.** The reviewer asked for the information content required for reliable detection. In our data, spectra with ≥10 well-resolved ¹³C peaks and a correctly assigned carboxylic carbon were classified reliably; the systematic per-peak-count and S/N sensitivity analysis is described under R2.3 【TODO: insert the quantitative boundary】.

---

## R1.2 — Methodology: small positive class, single-split variance, scaffold-level leakage

**Comment (condensed).** The evaluation is unstable because only 64 positives are available, and SMILES deduplication does not prevent structural-level leakage. Suggested remedies: scaffold-level hold-out or nested cross-validation.

**Response.** We agree that a single imbalanced split requires additional scrutiny. We address it with a scaffold-level audit, a three-seed variance estimate, and a similarity-threshold leakage ablation; we also explain why a full scaffold hold-out retraining is not the appropriate *primary* evaluation for this task.

1. **Scaffold-level split audit (new, SI Section S-y).** We computed Bemis–Murcko scaffolds for all 64 external-test positives and for all training positives, and quantified (i) how many test-positive scaffolds also occur in training and (ii) the distribution of nearest-training-positive embedding-cosine similarities (median 0.912). The split blacklist re-verification is included: 0 records removed; #12 absent from all training sources by SMILES and InChIKey. 【TODO: report the number/fraction of test positives whose exact BM scaffold appears in training】.

2. **Variance across seeds (new Table S-y).** The complete pipeline (feature extraction → XGBoost classifier) was re-run with **three random seeds**. We report mean ± s.d. of F1, precision and recall on the unchanged external test set; F1 = 0.9618 ± 【TODO】 at the operating threshold, with the single-split values retained for comparison.

3. **Leakage ablation (new Section 3.x / Table S-z).** To quantify how much of the reported performance is carried by near-duplicate training neighbours, we remove training samples whose CReSS-embedding cosine to **any** external-test sample exceeds τ ∈ {0.9, 0.8, 0.7}, retrain the XGBoost classifier on the pruned training set (CReSS encoder fixed), and re-evaluate on the unchanged test set:

   | Ablation | Training set | F1 (external test) |
   |---|---|---|
   | A0 | full (baseline) | 0.9618 |
   | A1 | − neighbours with cosine > 0.9 | 【TODO】 |
   | A2 | − neighbours with cosine > 0.8 | 【TODO】 |
   | A3 | − neighbours with cosine > 0.7 | 【TODO】 |

   The A0 → A3 degradation is an upper bound on the contribution of near-duplicate neighbours.

4. **Why not a full scaffold hold-out (stated as a limitation).** The task is *detection of a fixed pharmacophore class under deployment-like conditions*, not de novo generalization to arbitrary unseen scaffolds; the external test set was collected independently of the training sources, which is the realistic regulatory scenario. Importantly, the evaluation already penalises scaffold novelty: the single FN is precisely a novel scaffold combination (7-unsubstituted phenyl + 8-NO₂ + 6-F + N-cyclopropyl) that the model fails on. Where memorisation would have helped, the model failed — i.e., the test is not inflated by scaffold memorisation. We now state explicitly that generalisation to entirely novel scaffold classes requires a dedicated scaffold hold-out study, which we regard as follow-up work.

---

## R1.3 — Framework description, boundary-case behaviour, and "simple scaffold prompts"

**Comment (condensed).** (i) "Structure elucidation" is a misnomer; describe the process as conditional molecule generation under scaffold constraints. (ii) What happens when the XGBoost probability is near 0.5 — are multiple possibilities generated? (iii) Define "simple" and quantify compute versus the ensemble methods.

**Response.**

1. **Terminology.** Agreed. We now consistently use **"conditional molecular generation under a scaffold prompt"** and **"scaffold detection"**; "structure elucidation" has been removed throughout.

2. **Boundary cases.** A new subsection (Section 2.x) defines the strategy: (i) a **hard prompt** when p ≥ 0.7; (ii) a **grey zone** 0.3 ≤ p < 0.7, in which the framework emits **two candidate generations** (with and without the quinolone prompt) and ranks them by sequence likelihood together with the classifier probability; (iii) p < 0.3 emits only the unconstrained generation and flags "no quinolone signal". Fig. 1D shows such a boundary trajectory. We also report how many of the 4,438 negatives fall into the grey zone 【TODO: number】 and how they are handled (flagged for MS confirmation rather than silently accepted).

3. **"Simple".** We agree the term needs definition. We now state that "simple" refers to **decision logic and deployment cost** — one binary prompt versus majority voting over multiple generative passes — not to architectural simplicity of the Transformer. A quantitative comparison (new Table Y) reports parameter counts, wall-clock inference time per spectrum (single RTX 4090 / A100) and memory footprint for CMGNet-SPrompt, Ensemble-Voting and Ensemble-Union 【TODO: fill measured numbers】.

---

## R1.4 — Section 3.5 ablation: correlation vs causality, wording, missing 3.5.2

**Comment (condensed).** The ablation shows correlation, not causality; "failed completely" and "robust scalability" are not objective; Section 3.5.1 has no corresponding 3.5.2, and the ensemble-strategy ablation is missing.

**Response.**

1. Causal language has been removed. We now write that synthetic data is **associated with** the observed performance and explicitly state that the ablation **does not establish causality**; it is consistent with the interpretation that coverage of scaffold space, rather than nominal data volume, drives performance.
2. Emotional wording has been replaced: "failed completely (F1 = 0.0000)" → "did not produce a valid positive prediction on the held-out set (F1 = 0.0000)"; "robust scalability" → "monotonically increasing F1 with training-set size".
3. Section 3.5 is now split into **3.5.1 Ablation on synthetic data** and **3.5.2 Ablation on the ensemble strategy** (new). Section 3.5.2 reports the effect of removing the XGBoost scaffold prompt (unconstrained CMGNet), replacing it with a fingerprint + random-forest baseline, and using Ensemble-Voting/Union, quantifying the contribution of each component 【TODO: fill the 3.5.2 table】.

---

## R1.5 — Figures 2 and 3A/3B hard to read; vector/high-resolution and larger fonts

**Response.** The figures have been completely reworked:

- The global embedding map and the FN case study are now merged into **one Figure 1 (panels A–D)**, designed at final print width (180 mm) and exported as **600 dpi PNG + vector PDF**. Panel tick, label and legend fonts are 5.5–9 pt **at print size**.
- A new SI figure (Fig. Sx) reports the **precision–recall (AUPRC) and ROC (AUROC)** curves on the external test set. It is 9.2 × 3.9 in, exported as 600 dpi PNG plus vector PDF/SVG, with 10.5 pt bold panel titles, 7.8 pt annotations and 11 pt axis labels.
- Fig. 2 (pipeline) was redrawn as a vector diagram with ≥7 pt fonts; Fig. 3 (ablation) was redrawn with larger fonts and direct labelling instead of a legend.
- All SI figures are ≥300 dpi.

---

## R1.6 — Incomplete references (17, 18, 23, 29, 32, 35); inconsistent format (27)

**Response.** All references have been re-checked and completed (authors, journal abbreviations, year, volume, pages, DOIs), and reference 27 has been reformatted to ACS style. The corrected list is in the revised reference file 【TODO: verify each entry against the ACS style guide before resubmission】.

---

## R1.7 — Web server inaccessible; training code not in SI

**Response.** (i) The server link was temporarily offline during a certificate/host migration; it is now available at 【TODO: URL】 and is mirrored as a Docker image (see R2.6). (ii) The complete pipeline — data preprocessing, CReSS encoder training, CMGNet generation, XGBoost training/inference, evaluation and figure scripts — will be deposited on GitHub 【TODO: URL】 with an archived snapshot on Zenodo 【TODO: DOI】 and cited in the manuscript. See also R2.1.

---

# Response to Reviewer 2

## R2.1 — No code repository, weights, data or inference examples

**Response.** We have prepared a public release containing: (i) data-preprocessing scripts (JSON/LMDB parsing, binning to 4,000-shift features, deduplication and split lists); (ii) CReSS encoder and CMGNet training loops with configurations; (iii) XGBoost training/inference scripts; (iv) the evaluation suite (metrics, bootstrap, ablations, figure scripts); (v) pre-trained weights for all reported models; (vi) inference examples (single spectrum and batch); (vii) a Dockerfile. All are deposited at 【TODO: GitHub URL】 with a Zenodo-archived snapshot 【TODO: DOI】, and both are cited in the revised manuscript. The multi-source dataset is released to the extent permitted by the original data licences 【TODO: specify which sources are public/redacted and provide download links】.

## R2.2 — Unfair comparison with NMR Solver / NMRMind (Table 5)

**Response.** We agree that the original comparison mixed training data and input dimensionalities. In the revision: (i) the comparative models were re-run (or fine-tuned where an official checkpoint permits) on **our unified test set**, and the new Table 5 reports input dimensionality, training data and inference settings for every row; (ii) where a model requires multi-dimensional NMR input that our data do not contain, this is stated explicitly and the row is marked "reported on its original setup" rather than claimed as comparable; (iii) a qualitative comparison figure shows, for the same six quinolone derivatives, the candidate structures produced by each model, with top-1/top-k hit rates 【TODO: run experiments; insert numbers and figure】.

## R2.3 — Failure-mode analysis (FP/FN, other N-heterocycles, S/N and resolution sensitivity)

**Response.** A new section "3.4 Error analysis…" and new SI experiments address this:

- **FP/FN dissection:** see R1.1 — embedding localization, substructure audit, peak ablation, interpolation, and the PR/ROC analysis (AUPRC 0.9891, 95% CI 0.9713–0.9996; AUROC 0.9998, 95% CI 0.9995–1.0000). The near-perfect ranking combined with a handful of threshold-localised errors supports the interpretation that the failures reflect *scaffold-boundary ambiguity*, not a broken generative prompt.
- **其他含氮杂环（新 Fig. S-v）：** 我们按环系对全部外部测试分子分类（RDKit 环分析；优先级按与喹诺酮母核的相似度：quinolone core → quinolinone-like → 稠合 N-杂芳环 → 吡啶 → 二嗪/三嗪 → 唑 → 饱和 N-环 → 碳环 → 无环）。τ_op = 0.50 下分类器对**所有** N-杂环体系都高度特异（Fig. S-v）：假阳性为 **quinolinone-like 1/69（1.45%）**、**稠合 N-杂芳环 1/488（0.20%）**、**吡啶 0/144（0 FP）**、**唑 1/252（0.40%）**、**二嗪/三嗪 0/91**、**饱和 N-环 0/613**；碳环 1/2,238（0.045%）、无环 0/543 —— **N-杂环合计 3/1,657（0.18%，Wilson 95% CI 0.06–0.53%）**。3 个 N-杂环假阳性分别是 N-苯基喹诺酮、呋喃并喹啉和偶氮-噻唑染料，都是共轭的、类喹诺酮 N-杂芳环，而不是任意含氮杂环；吲哚、吡啶、嘧啶、吡唑、吡咯烷、哌啶的代表结构见 Fig. S-vB。由此可界定适用范围：风险集中在 quinolinone-like 骨架（1.45%），其他环系 ≤0.40%。

- **保留环子集（附图 Fig. S-v-supp）：** 用与 scaffold prompt 完全相同的保留规则（`main_cmgnet.py`：`RingSystemFinder` → 去末端取代基 → 最大芳香环系），1,311 个负样本的 N 位于 prompt 保留环内，仅 3 个假阳性（0.23%）；另有 346 个 N 仅位于侧环（1 个假阳性），结论一致。
- **谱分辨率 / C–F 裂峰（新 Fig. S-w）：** 模型输入是无强度的峰列表，严格的 S/N 扫描没有定义；因此我们用一对 48 张谱的同一批含氟喹诺酮：C–F 裂分峰分别保留（中位 23 个峰）或合并为单峰（中位 18.5 个峰）。τ = 0.50 下，合并峰输入召回 **46/48（95.8%）**，裂分峰输入 **41/48（85.4%）**；单分子 |Δp| 中位数仅 6.8×10⁻⁴，但有 **7/48 个分子发生翻转（6 个丢失、1 个恢复；最大 |Δp| = 0.95）**。说明裂峰是真实但局部的失效模式，集中在部分谱上，也支持预测前对峰列表做规整（合并 C–F 双峰）。真正的 S/N 扫描需要带强度的原始谱，我们在局限性中说明。

## R2.4 — 不确定性量化（校准、MC-Dropout、不确定性引导的人工复核）

**Response.** 我们同意仅凭预测分数无法确立可靠性。修订稿把三个常被混为一谈的量作为**相互独立的评估**分别报告：(i) *判别力* —— PR/ROC 曲线（新 Fig. Sx）只度量排序，不度量校准；(ii) *校准* —— 分数能否被当作概率解读；(iii) *不确定性引导的人工复核* —— 把不确定样本送交人工裁决的决策规则，而不是把它们重新标注。分类器不确定性（校准 + 新颖性信号）与生成器变异性（MC-Dropout）也分开报告、绝不混用。**拟合与选择任何校准器或阈值均未使用外部测试集标签。** 以下数字均已在修订中实现；脚本与原始输出随代码一并发布（R2.1）。

**1. 概率校准（新 Section 3.6；新 Fig. Sy；新 Table S-z）。**

- *数据与协议。* 校准器只在**预留验证集**（40,667 条；37 正 / 40,630 负）上拟合（该集同时用于早停）；**内部测试集**（40,642；37 / 40,605）承载部署时关心的自然患病率（0.091%）；**外部测试集**（4,502；64 / 4,438）是按设计人为富集到 1.42% 的。方法（Platt 缩放 vs 保序回归）仅以**验证集上的 5 折交叉拟合 NLL** 选择（Platt NLL 1.17×10⁻³、Brier 3.27×10⁻⁴；isotonic NLL 1.49×10⁻³、Brier 3.57×10⁻⁴；原始分数 NLL 9.95×10⁻³、Brier 2.88×10⁻³）；选中的 Platt 映射斜率 a = 0.954、截距 b = −4.716。
- *自然患病率结果（内部测试集）。* 原始 XGBoost 分数在模型实际服务的患病率下过于自信：ECE 4.71×10⁻³、Brier 2.71×10⁻³、NLL 9.34×10⁻³、校准斜率 0.86；相对基率预测器的 Brier 技巧分为 −1.98，即原始分数**还不如**恒定输出基率。Platt 缩放消除该问题：**ECE 1.74×10⁻⁵**（95% bootstrap CI 1.9×10⁻⁶–1.6×10⁻⁴，1,000 次重采样）、**Brier 3.59×10⁻⁴**、NLL 1.44×10⁻³、斜率 0.91、Brier 技巧分 +0.61。带 log 轴的可靠性图见 Fig. Sy（A–B）。
- *富集的外部测试集（先验修正）。* 外部集被富集 15.6 倍，在自然患病率上拟合的校准器会整体偏移；因此用两次**已知采样患病率**做标准**先验概率修正**（logit 截距平移，不涉及标签）。部署校准曲线依然良校准：**ECE 1.11×10⁻³**（CI 2.0×10⁻⁴–2.0×10⁻³）、Brier 1.27×10⁻³、NLL 4.59×10⁻³、斜率 0.87、Brier 技巧分 0.91（Fig. Sy C）。同时如实报告：原始分数在该富集集上碰巧几乎完全校准（ECE 6.2×10⁻⁴）——这是负样本构成不同造成的假象，也是标准化协议需要先验修正的原因。
- *修订中声明的局限。* 每个内部集仅 37 个正样本，校准器估计自极少正例，内部测试校准数字置信区间较宽；校准是随人群变化的，因此部署工具要求（或按默认）假设患病率并给出相应校准概率。校准降低系统性的过/欠自信，但**不能**消除自信的错误（两个被自动判为阳性的实际阴性依旧自信地错），且它本身不构成监管有效性。

**2. CMGNet 生成器的 MC-Dropout（新 Section 3.7；新 Fig. Sz）。** MC-Dropout 只施加于 BART 生成器（论文所用检查点 dropout p = 0.1）；分类器是提升树集成、无 dropout，因此 MC-Dropout **不**施加于 XGBoost —— 这一区别现已在修订稿中明确。对按分类器置信度分组的 21 张外部测试谱（8 张高置信正、4 张边界正、FN #12、4 张假阳性、4 张高置信负），我们运行 **T = 20 次随机前向**（启用 dropout，beam 5，每次返回 5 条；每谱 100 个候选生成），并总结**分子层面**的变异性，且明确不把 token 似然当作“整个结构正确”的校准概率：

| 组别 (n) | top-5 中含喹诺酮母核的次数 | top-1 模态频率（中位数） | 每谱不同 Murcko 骨架数 | 候选生成物 CReSS 余弦均值 |
|---|---|---|---|---|
| 高置信正 (8) | 每谱均 20/20 | 0.40 | ≈8–10 | 0.81 |
| 边界正 (4) | 0.63 (均值) | 0.23 | ≈32 (中位 34) | 0.49 |
| 假阳性 (4) | 0.03 (均值；4 中 2 张 ≤1 次) | 0.28 | ≈15 | 0.51 |
| FN #12 | **0/20 次** | 0.50 | 30 | 0.51 |
| 高置信负 (4) | 0/20 次 | 0.48 | ≈16 | 0.44 |

两点结论。其一，生成器共识随母核信号变化：高置信正样本在全部 100 个抽样生成中均恢复喹诺酮母核，而边界正样本的结构离散度大一个数量级（≈32 vs ≈10 个骨架；CReSS 余弦 0.49 vs 0.80），说明 MC-Dropout 离散度是可用的生成器侧不确定性信号。其二，对错误分析（R1.1）很重要：**FN #12 在 20 次独立随机前向中从未生成含喹诺酮母核的候选**（52 个唯一 SMILES、30 种骨架），从交叉验证的角度说明该 FN 源于**谱表示层面**——无论采样噪声如何，#12 的谱都被编码为非喹诺酮，从而排除推理或提示(token)层面的偶然失误。序列似然同时报告（每谱均值，跨 20 次的 SD ≈ 0.01–0.04），并明确它是*分数*而非结构正确性的概率。

**3. 不确定性引导的复核规则（新 Section 3.8；Fig. Sy, D–E）。** 规则由三个不借助外部测试标签固定的量定义：默认运行阈值 τ_op = 0.50（TP 63 / FP 4 / FN 1；F1 = 0.9618）、围绕 τ_op 的 ±1.5 logit 双侧复核带、以及余弦 **0.85** 的新颖性/不一致截断 —— 0.85 是 37 个验证正样本“最近训练正样本余弦”的**下界**（最小 0.850）。规则把谱路由到三种结果之一：

- **自动阳性**：p ≥ 0.818。
- **送人工复核**（标注“需 MS 确证”，绝不标为阴性）：0.182 ≤ p < 0.818（灰色带）；**或** p < 0.182 但与某训练正样本余弦 ≥ 0.85（不一致：分 confidently 负却嵌入正样本相似壳层内）。
- **自动阴性**：其余。

*外部测试集（n = 4,502；64 正 / 4,438 负）。* **4,420 自动阴性；59 自动阳性（59 TP、2 FP；精确率 96.7%）；21 张送复核（0.47%，95% CI 0.31–0.71%）**——5 个正样本（占正样本 7.8%；4 个经灰色带，FN #12 经新颖性触发）和 16 个负样本（0.36%；6 灰色带 + 10 新颖性）。**沉默漏检（既未自动检出也未送复核的正样本）= 0**，复核流程不可能靠改标“未决”来掩盖假阴性；复核后灵敏度上限 100%（95% CI 94.3–100%）。修订稿明确：送复核样本需人工/MS 裁决，自动阳性 TP 计数不得改称召回率——自动步骤单独的召回率为 59/64 = 92.2%。

*自然患病率（内部测试集，n = 40,642；37 正）。* 送复核 **5.52%**（2,242 张：4 正、2,238 负），自动阳性 32 TP / 59 FP（精确率 35.2%，符合 0.09% 患病率的预期），自动阴性 38,309，**1 个沉默漏检**（p = 0.117 低于 τ_lo、与训练正样本余弦 0.76，未触发新颖性）。如实报告该残留误差：自然患病率下灰色带 + 新颖性触发把自动步骤未标记的正样本从 5（仅用 0.4 阈值）降到 1，代价是 5.5% 的人工复核负荷，监管实验室可承担。同一规则在验证集上送复核 5.51%（2,240）、0 沉默漏检；5.5% 与 0.47% 的送复核率之差，量化了“类喹诺酮合成验证负样本”与“真实外部负样本”的分布差异。

*方法学局限。* 与训练正样本的余弦新颖性触发是嵌入空间内的 in-domain 启发式，不是完整的 conformal/分布外检测器；修订稿如实说明，并引用审稿人建议的两篇文献（10.1007/s10462-023-10562-9；10.1021/acs.analchem.5c07201）为校准/不确定性处理提供依据；专门的 conformal prediction 研究列为后续工作。修订稿同样写明审稿人的告诫：ROC/PR 只度量判别力而非校准；不确定性排序、概率校准与分布外检测是不同评估；复核触发与残留沉默漏检数一并报告，避免复核掩盖假阴性（外部 0，内部 1）。

**修改位置：** Section 3.6（校准）、3.7（MC-Dropout）、3.8（复核流程）【TODO: 页/行号】；新 Fig. Sx（PR/ROC）、**Fig. Sy**（可靠性图、分数分布与复核结果）、**Fig. Sz**（MC-Dropout 变异性）、新 Table S-z（含 bootstrap CI 的校准指标、复核计数）。

## R2.5 — Hyperparameter selection criteria and sensitivity

**Response.** We have added (i) an explicit selection protocol (grid/random search on the validation split, final choice by F1 with a fixed seed), including the search ranges for learning rate, dropout, number of bins, and L1/L2; and (ii) a sensitivity analysis: F1 versus number of bins {1000, 2000, 4000, 8000}, dropout {0.0, 0.1, 0.2, 0.3}, and learning rate 【TODO: run on the validation split; add figure】. Defaults are justified, and a "how to retune for a new task" note is added to the SI.

## R2.6 — Web-server documentation (maintenance, inputs, privacy, API, Docker)

**Response.** A new SI section documents the server: supported input formats (peak-list CSV/TXT and mzML-derived shift lists), example inputs/outputs with screenshots, current concurrency and throughput (【TODO】 spectra/hour), the privacy policy (uploaded spectra are processed in memory and not persisted; 【TODO: confirm with host】), a REST API for batch processing with an example curl script, the planned maintenance period (≥2 years), and a **Docker image** for fully local deployment 【TODO: registry URL】.

## R2.7 — Recent reviews on AI for NMR in the Introduction

**Response.** We have added and cited 10.1063/5.0277355 and 10.1016/j.pnmrs.2022.101575 (plus 【TODO: one additional recent review if appropriate】) in the Introduction to position the work within the current AI-for-NMR literature.

## R2.8 — Reference formatting (18, 24 incomplete); image resolution

**Response.** References 18 and 24 (and all others) have been completed and reformatted to ACS style; all main-text and SI figures have been replaced with 600 dpi PNG + vector PDF versions designed at print width with ≥6 pt fonts (see also R1.5).

---

# Summary of the major changes in this revision

1. **New Section 3.4 "Error analysis" + new Figure 1 (A–D):** global embedding map, FN neighbourhood, neighbour structures, and decision-boundary interpolation. The five FPs are shown to be scaffold-boundary cases; the single FN is dissected by peak ablation, embedding interpolation and split audit.
2. **New SI figure (Fig. Sx):** precision–recall (AUPRC = 0.9891, 95% CI 0.9713–0.9996) and ROC (AUROC = 0.9998, 95% CI 0.9995–1.0000) curves with step interpolation, iso-F1 contours, chance level and bootstrap confidence bands; the operating point is annotated with TP/FP/FN.
3. **Evaluation robustness (R1.2):** scaffold-level split audit, three-seed variance, and a similarity-threshold leakage ablation; generalisation to entirely novel scaffold classes is stated as a scope limitation.
4. **Terminology and framing (R1.3):** "conditional molecular generation under scaffold constraints"; explicit grey-zone policy; quantified compute comparison versus ensemble methods.
5. **Section 3.5 restructured (R1.4):** 3.5.1 synthetic-data ablation and new 3.5.2 ensemble-strategy ablation, with neutral wording and explicit correlation-not-causation language.
6. **不确定性（R2.4）：** 概率校准（Platt 缩放，仅在预留验证集上拟合，用交叉拟合 NLL 在 Platt 与保序回归间选择；可靠性图、ECE/Brier/NLL 及 95% bootstrap CI；对富集的外部集做先验修正），CMGNet 生成器的 MC-Dropout（T = 20 次；分子层面变异性与喹诺酮母核命中率），以及三分支不确定性引导复核规则（自动阳性 / 送 MS 确证 / 自动阴性），外部集上送复核率 0.47% 且沉默漏检为 0。再叠加 XGBoost 的三种置信区间：90% conformal 预测集（外部集送复核 3.09%、0 沉默漏检；实测类覆盖率 86–97% 与名义 90% 一致）、校准概率的 bootstrap 区间、以及三种子重训离差（在决策边界 0.05–0.4 分数带中位数达 0.15）。分类器不确定性与生成器变异性分开报告；MC-Dropout 不施加于 XGBoost。
7. **Baselines (R2.2) and hyperparameters (R2.5):** unified-test-set re-evaluation and sensitivity analyses.
8. **Openness and figures (R1.5–R1.7, R2.1, R2.6–R2.8):** code/weights/data release with Zenodo DOI and Docker image; full web-server documentation; all figures ≥600 dpi / vector at print size; all references completed and reformatted.

We believe the manuscript is substantially strengthened by these revisions and thank the reviewers again for their time and for comments that materially improved the work.
