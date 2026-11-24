# 5 条模拟 13C 谱：多模型预测对比

数据：`data/sim5/fluoroquinolones_nmr.json`（5 条喹诺酮模拟谱，19–29 峰，覆盖左氧氟沙星、环丙沙星、莫西沙星、氧氟沙星、诺氟沙星）。

口径：`core@k` = 前 k 个候选含喹诺酮母核的样本比例；`exact@k` = 前 k 个候选与标准答案 canonical SMILES 完全一致的样本比例。多 seed 为 mean±sd（ddof=1），分母恒为 5。

## NMR-Solver

- decode_mode: `nmr_solver`；seed 数: 1；样本数: 5

| Top-k | Core Acc | Exact Match |
| --- | --- | --- |
| 1 | 0.800 | 0.000 |
| 3 | 0.800 | 0.000 |
| 5 | 0.800 | 0.000 |
| 10 | 0.800 | 0.000 |

## CMGNet-SPrompt (ex1209 5ep)

- decode_mode: `nmr+frag`；seed 数: 3；样本数: 5

| Top-k | Core Acc | Exact Match |
| --- | --- | --- |
| 1 | 1.000±0.000 | 0.267±0.115 |
| 3 | 1.000±0.000 | 0.267±0.115 |
| 5 | 1.000±0.000 | 0.267±0.115 |
| 10 | 1.000±0.000 | 0.267±0.115 |

<details><summary>逐 seed 明细 (exact@1 / exact@10)</summary>

- `output/sim5_sprompt_ex1209_s0/test_nmr+frag_beams10_samples5.json` → exact@1 0.400, exact@10 0.400
- `output/sim5_sprompt_ex1209_s42/test_nmr+frag_beams10_samples5.json` → exact@1 0.200, exact@10 0.200
- `output/sim5_sprompt_ex1209_s2006/test_nmr+frag_beams10_samples5.json` → exact@1 0.200, exact@10 0.200

</details>

## CMGNet-SMILES (ex1209 5ep)

- decode_mode: `smiles`；seed 数: 3；样本数: 5

| Top-k | Core Acc | Exact Match |
| --- | --- | --- |
| 1 | 1.000±0.000 | 0.333±0.115 |
| 3 | 1.000±0.000 | 0.333±0.115 |
| 5 | 1.000±0.000 | 0.333±0.115 |
| 10 | 1.000±0.000 | 0.400±0.000 |

<details><summary>逐 seed 明细 (exact@1 / exact@10)</summary>

- `output/sim5_ex1209_s0_smiles_beams10/test_smiles_beams10_samples5.json` → exact@1 0.400, exact@10 0.400
- `output/sim5_ex1209_s42_smiles_beams10/test_smiles_beams10_samples5.json` → exact@1 0.400, exact@10 0.400
- `output/sim5_ex1209_s2006_smiles_beams10/test_smiles_beams10_samples5.json` → exact@1 0.200, exact@10 0.400

</details>

## CMGNet-SCAFFOLD (ex1209 5ep)

- decode_mode: `frag`；seed 数: 3；样本数: 5

| Top-k | Core Acc | Exact Match |
| --- | --- | --- |
| 1 | 1.000±0.000 | 0.000±0.000 |
| 3 | 1.000±0.000 | 0.000±0.000 |
| 5 | 1.000±0.000 | 0.000±0.000 |
| 10 | 1.000±0.000 | 0.000±0.000 |

<details><summary>逐 seed 明细 (exact@1 / exact@10)</summary>

- `output/sim5_ex1209_s0_frag_beams10/test_frag_beams10_samples5.json` → exact@1 0.000, exact@10 0.000
- `output/sim5_ex1209_s42_frag_beams10/test_frag_beams10_samples5.json` → exact@1 0.000, exact@10 0.000
- `output/sim5_ex1209_s2006_frag_beams10/test_frag_beams10_samples5.json` → exact@1 0.000, exact@10 0.000

</details>

## CMGNet-SMILES (NMRMind m1010)

- decode_mode: `smiles`；seed 数: 3；样本数: 5

| Top-k | Core Acc | Exact Match |
| --- | --- | --- |
| 1 | 0.467±0.115 | 0.000±0.000 |
| 3 | 0.667±0.115 | 0.000±0.000 |
| 5 | 0.800±0.000 | 0.000±0.000 |
| 10 | 0.867±0.115 | 0.000±0.000 |

<details><summary>逐 seed 明细 (exact@1 / exact@10)</summary>

- `output/sim5_m1010_s0_smiles_beams10/test_smiles_beams10_samples5.json` → exact@1 0.000, exact@10 0.000
- `output/sim5_m1010_s42_smiles_beams10/test_smiles_beams10_samples5.json` → exact@1 0.000, exact@10 0.000
- `output/sim5_m1010_s2026_smiles_beams10/test_smiles_beams10_samples5.json` → exact@1 0.000, exact@10 0.000

</details>

## CMGNet-SCAFFOLD (NMRMind m1010)

- decode_mode: `frag`；seed 数: 3；样本数: 5

| Top-k | Core Acc | Exact Match |
| --- | --- | --- |
| 1 | 0.667±0.115 | 0.000±0.000 |
| 3 | 0.667±0.115 | 0.000±0.000 |
| 5 | 0.733±0.231 | 0.000±0.000 |
| 10 | 0.733±0.231 | 0.000±0.000 |

<details><summary>逐 seed 明细 (exact@1 / exact@10)</summary>

- `output/sim5_m1010_s0_frag_beams10/test_frag_beams10_samples5.json` → exact@1 0.000, exact@10 0.000
- `output/sim5_m1010_s42_frag_beams10/test_frag_beams10_samples5.json` → exact@1 0.000, exact@10 0.000
- `output/sim5_m1010_s2026_frag_beams10/test_frag_beams10_samples5.json` → exact@1 0.000, exact@10 0.000

</details>

## CMGNet-0910 (base, no retrain)

- decode_mode: `smiles`；seed 数: 1；样本数: 5

| Top-k | Core Acc | Exact Match |
| --- | --- | --- |
| 1 | 1.000 | 0.000 |
| 3 | 1.000 | 0.000 |
| 5 | 1.000 | 0.000 |
| 10 | 1.000 | 0.200 |

## 说明

- **CMGNet-SCAFFOLD 的 exact 恒为 0** 属预期：frag 模式只解码骨架片段，不产出完整分子 SMILES；该模式看 `core` 即可。
- **NMRMind (m1010) 变体在模拟谱上明显弱于 ex1209**：模拟谱由实验谱峰位合成，与 NMRMind 语料分布差异大，属分布外迁移，不代表其在真实测试集上的水平。
- **CMGNet-0910 (base) core=1.000 但 exact≈0**：原始 0910 权重生成的候选能命中母核，但整分子写法与标准答案差异大（无 ring-system 微调）。
- XGBoost-CReSS 为二分类判别器，只做「是否含喹诺酮母核」判定：5/5 判正、概率≈1.000，不产出 SMILES，故不进上表。
