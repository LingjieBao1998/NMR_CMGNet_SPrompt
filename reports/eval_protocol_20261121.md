# 评测口径（外部集 4,502）— 判定规则、生效方式与可直接引用的文字

日期：2026-11-21 · 适用：CMGNet-0910 / SCAFFOLD / SMILES / SPrompt、NMRMind、NMR-Solver、XGBoost-CReSS

---

## 一、一段话口径（中文）

外部测试集共 4,502 条 ¹³C NMR 谱（64 正 / 4,438 负）。正负标签由**参考结构是否含喹诺酮母核**决定，母核用两个 SMILES 模板定义：4-喹诺酮-3-羧酸
`O=C1C(C(O)=O)=CNC2=CC=CC=C12` 及其氮杂（萘啶酸型）类似物
`O=C1C(C(O)=O)=CNC2=NC=CC=C12`，以 RDKit 子结构匹配判定（互变异构感知）。
模板只约束母核骨架：N-取代基、苯环上的其它取代基、以及母核外延的环系都不参与判定；
由于查询用 SMILES 而非 SMARTS，羧基的 O–H 不作为匹配条件，3-位羧酸酯同样算命中。
生成式模型在 top-k（k = 1, 3, 5, 10）下判阳的条件是：**排名前 k 的候选中至少有一个含母核**
（k 一律指排序列表的前缀，不是重新排序）。指标定义为
Precision = TP/(TP+FP)、Recall = TP/64、F1 = 2PR/(P+R)、
Specificity = TN/4,438、Accuracy = (TP+TN)/4,502，
分母固定，不随预测分布变化。SMILES 准确率 SA 只在 64 个正例上统计：
候选经 RDKit canonical 化后与参考结构字符串完全相等即算命中，逐 k 记为 exact@k。
判别式分类器（XGBoost-CReSS）不产生候选列表，在固定阈值 τ = 0.50 下做单次判决，因此没有 top-k 维度。

---

## 二、这段口径是怎么生效的（执行链）

1. **候选来源**：每个模型对每条谱输出 10 个候选，dump 里 `predictions[i].rank = i+1`。
   已核对所有 dump 均按 rank 1…10 排列（抽查 0 例乱序），所以 `predictions[:k]` 就是 top-k。
2. **单条候选判定**：候选 SMILES → `Chem.MolFromSmiles` → 与两个母核模板做
   `HasSubstructMatch`。解析失败（返回 `None`）的候选记“未命中”，**不剔除、不补位、不重排**，
   避免因丢弃无效候选而虚增命中。
3. **样本级判定**：`hit_k = any(母核命中 for 候选 in predictions[:k])`，
   即 top-k 内“任一命中”即判阳；这与“top-1 是否正确”是两件事，前者衡量筛选能力，后者衡量排序质量。
4. **混淆矩阵**：把 `hit_k` 与标签（参考结构是否含母核）比对，得 TP/FP/FN/TN；
   FN = 64 − TP，TN = 4,438 − FP（分母固定，故只需统计 TP/FP）。
5. **指标**：由四项计数直接套公式（见上），不引入任何平滑或阈值搜索。
6. **SA**：对 64 个正例，取 `canonical(参考)`，检查 top-k 内是否存在 `canonical(候选) == canonical(参考)`；
   逐 k 统计，记为 exact@1/3/5/10。
7. **分类器路径**：XGBoost 读 CReSS 嵌入输出概率，`p ≥ 0.50` 判阳，与 k 无关；
   因此 P/R/F1/Spec/Acc 只有一组，AUPRC/AUROC 另行报告。

代码位置（可复核）：

- 母核判定与指标公式：`analysis/metrics/topk_external_4502.py`（`CORE_MOL` / `has_core` / `metrics_from_counts`）
- 逐样本 dump 评测与聚合：`analysis/metrics/topk_cmgnet_modes_4502.py`（`eval_dump` / `aggregate` / `eval_xgb_seeds`）
- 官方参考实现（互变异构感知版）：`evaluate_smiles_beams10_samples50381.py::analyze_core_structure`

一致性核对：在 0910 的 45,020 个（4,502 × 10）候选上，普通 `HasSubstructMatch` 与官方
互变异构感知实现在**参考标签和候选命中上均 0 处不一致**，故本仓库统一用简单实现复算。

---

## 三、边界与容易被问到的点

| 情形 | 处理方式 | 说明 |
|---|---|---|
| 3-位羧酸被酯化 | 仍判含母核 | 查询是 SMILES，O–H 不参与匹配；本批 2,457 个命中候选中有 76 个是甲/乙酯 |
| N-烷基化（N-Me、N-环丙基） | 仍判含母核 | 模板 N 上不带 H 约束 |
| 母核外多并环 / 多取代 | 仍判含母核 | 子结构是“包含”关系，不惩罚额外结构 |
| 去掉 3-位羧基 | 不判含母核 | 模板含 `C(=O)O`，脱羧结构不命中 |
| 候选 SMILES 无法解析 | 记未命中 | 不剔除、不重排 |
| 片段模式（SCAFFOLD） | SA 恒为 0 | 片段/骨架解码不产出完整分子，属构造性结果，不是模型失败 |
| k 增大 | Recall 单调不减、FP 单调不减 | “任一命中”的 OR 语义决定，非实现细节 |

---

## 四、English methods paragraph（可直接用于正文 / SI）

**Evaluation protocol.** The external test set contained 4,502 ¹³C NMR spectra (64 positives and 4,438 negatives), labelled by whether the reference structure contains the quinolone core. The core was defined by two SMILES templates, 4-quinolone-3-carboxylic acid (O=C1C(C(=O)O)=CNC2=CC=CC=C12) and its aza analogue (the nalidixic-acid core, O=C1C(C(=O)O)=CNC2=NC=CC=C12), and detected by RDKit substructure matching with tautomer awareness. The templates constrain only the bicyclic core: N-substituents, ring substituents and any fused or appended rings are unconstrained, and because the queries are SMILES rather than SMARTS, the carboxyl O–H is not part of the match, so 3-carboxylate esters are also counted as core-containing. A generative model was scored positive at top-k (k = 1, 3, 5, 10) if at least one of its k highest-ranked candidates contained the core; k always denotes a prefix of the ranked candidate list, never a re-ranking. Precision = TP/(TP+FP), Recall = TP/64, F1 = 2PR/(P+R), Specificity = TN/4,438 and Accuracy = (TP+TN)/4,502, with fixed denominators. SMILES accuracy (SA) was evaluated on the 64 positives only: a candidate counted as correct when its RDKit-canonical SMILES was identical to that of the reference, reported per k as exact@k. Unparseable candidates were counted as misses and were neither removed nor re-ranked. The discriminative classifier was evaluated at a fixed threshold of τ = 0.50 and therefore has no top-k dimension.
