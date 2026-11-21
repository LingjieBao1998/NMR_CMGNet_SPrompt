# CMGNet 三模式 top-k 指标 — 外部集 4,502(64 正 / 4,438 负)

阈值/判定: top-k 内任一候选含全母核(4-喹诺酮-3-羧酸及其氮杂类似物)即判阳;P=TP/(TP+FP), R=TP/64, F1=2PR/(P+R), Spec=TN/4,438, Acc=(TP+TN)/4,502。SA=参考结构 canonical 完全匹配 / 64(正例)。

多 seed 行写成 mean±sd(样本标准差 ddof=1,3 个种子)。

| 模型 | Top-k | TP | FP | FN | TN | Precision | Recall | F1 | Specificity | Accuracy | SA (exact@k) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CMGNet-0910 (base, SMILES) | 1 | 44 | 1 | 20 | 4437 | 0.9778 | 0.6875 | 0.8073 | 0.9998 | 0.9953 | 0.0469 |
| CMGNet-0910 (base, SMILES) | 3 | 50 | 1 | 14 | 4437 | 0.9804 | 0.7812 | 0.8696 | 0.9998 | 0.9967 | 0.0469 |
| CMGNet-0910 (base, SMILES) | 5 | 50 | 2 | 14 | 4436 | 0.9615 | 0.7812 | 0.8621 | 0.9995 | 0.9964 | 0.0625 |
| CMGNet-0910 (base, SMILES) | 10 | 52 | 4 | 12 | 4434 | 0.9286 | 0.8125 | 0.8667 | 0.9991 | 0.9964 | 0.0938 |
| CMGNet-SCAFFOLD (frag) | 1 | 61 | 6.00±2.00 | 3 | 4432.00±2.00 | 0.9110±0.0272 | 0.9531 | 0.9314±0.0142 | 0.9986±0.0005 | 0.9980±0.0004 | 0.0000 |
| CMGNet-SCAFFOLD (frag) | 3 | 62.67±0.58 | 8.00±2.00 | 1.33±0.58 | 4430.00±2.00 | 0.8872±0.0260 | 0.9792±0.0090 | 0.9308±0.0179 | 0.9982±0.0005 | 0.9979±0.0006 | 0.0000 |
| CMGNet-SCAFFOLD (frag) | 5 | 62.67±0.58 | 10.33±2.52 | 1.33±0.58 | 4427.67±2.52 | 0.8590±0.0305 | 0.9792±0.0090 | 0.9150±0.0210 | 0.9977±0.0006 | 0.9974±0.0007 | 0.0000 |
| CMGNet-SCAFFOLD (frag) | 10 | 62.67±0.58 | 11.67±2.31 | 1.33±0.58 | 4426.33±2.31 | 0.8435±0.0273 | 0.9792±0.0090 | 0.9062±0.0180 | 0.9974±0.0005 | 0.9971±0.0006 | 0.0000 |
| CMGNet-SMILES (smiles) | 1 | 61.00±1.00 | 6.00±2.65 | 3.00±1.00 | 4432.00±2.65 | 0.9117±0.0357 | 0.9531±0.0156 | 0.9315±0.0116 | 0.9986±0.0006 | 0.9980±0.0004 | 0.1771±0.0239 |
| CMGNet-SMILES (smiles) | 3 | 61.33±0.58 | 9.33±0.58 | 2.67±0.58 | 4428.67±0.58 | 0.8680±0.0060 | 0.9583±0.0090 | 0.9109±0.0008 | 0.9979±0.0001 | 0.9973 | 0.2344±0.0156 |
| CMGNet-SMILES (smiles) | 5 | 62.33±0.58 | 11.00±2.00 | 1.67±0.58 | 4427.00±2.00 | 0.8503±0.0242 | 0.9740±0.0090 | 0.9079±0.0174 | 0.9975±0.0005 | 0.9972±0.0006 | 0.2865±0.0090 |
| CMGNet-SMILES (smiles) | 10 | 62.33±0.58 | 13.67±2.08 | 1.67±0.58 | 4424.33±2.08 | 0.8206±0.0227 | 0.9740±0.0090 | 0.8906±0.0150 | 0.9969±0.0005 | 0.9966±0.0005 | 0.3021±0.0361 |

## 逐 seed 明细(可审计)

| 模型 | seed | Top-k | TP | FP | FN | TN | Exact |
|---|---|---|---|---|---|---|---|
| CMGNet-0910 (base, SMILES) | 0910 | 1 | 44 | 1 | 20 | 4437 | 3/64 |
| CMGNet-0910 (base, SMILES) | 0910 | 3 | 50 | 1 | 14 | 4437 | 3/64 |
| CMGNet-0910 (base, SMILES) | 0910 | 5 | 50 | 2 | 14 | 4436 | 4/64 |
| CMGNet-0910 (base, SMILES) | 0910 | 10 | 52 | 4 | 12 | 4434 | 6/64 |
| CMGNet-SCAFFOLD (frag) | seed0 | 1 | 61 | 8 | 3 | 4430 | 0/64 |
| CMGNet-SCAFFOLD (frag) | seed0 | 3 | 62 | 10 | 2 | 4428 | 0/64 |
| CMGNet-SCAFFOLD (frag) | seed0 | 5 | 62 | 13 | 2 | 4425 | 0/64 |
| CMGNet-SCAFFOLD (frag) | seed0 | 10 | 62 | 13 | 2 | 4425 | 0/64 |
| CMGNet-SCAFFOLD (frag) | seed42 | 1 | 61 | 4 | 3 | 4434 | 0/64 |
| CMGNet-SCAFFOLD (frag) | seed42 | 3 | 63 | 6 | 1 | 4432 | 0/64 |
| CMGNet-SCAFFOLD (frag) | seed42 | 5 | 63 | 8 | 1 | 4430 | 0/64 |
| CMGNet-SCAFFOLD (frag) | seed42 | 10 | 63 | 9 | 1 | 4429 | 0/64 |
| CMGNet-SCAFFOLD (frag) | seed2006 | 1 | 61 | 6 | 3 | 4432 | 0/64 |
| CMGNet-SCAFFOLD (frag) | seed2006 | 3 | 63 | 8 | 1 | 4430 | 0/64 |
| CMGNet-SCAFFOLD (frag) | seed2006 | 5 | 63 | 10 | 1 | 4428 | 0/64 |
| CMGNet-SCAFFOLD (frag) | seed2006 | 10 | 63 | 13 | 1 | 4425 | 0/64 |
| CMGNet-SMILES (smiles) | seed0 | 1 | 61 | 7 | 3 | 4431 | 13/64 |
| CMGNet-SMILES (smiles) | seed0 | 3 | 61 | 9 | 3 | 4429 | 16/64 |
| CMGNet-SMILES (smiles) | seed0 | 5 | 63 | 9 | 1 | 4429 | 18/64 |
| CMGNet-SMILES (smiles) | seed0 | 10 | 63 | 13 | 1 | 4425 | 18/64 |
| CMGNet-SMILES (smiles) | seed42 | 1 | 60 | 3 | 4 | 4435 | 11/64 |
| CMGNet-SMILES (smiles) | seed42 | 3 | 61 | 9 | 3 | 4429 | 15/64 |
| CMGNet-SMILES (smiles) | seed42 | 5 | 62 | 11 | 2 | 4427 | 18/64 |
| CMGNet-SMILES (smiles) | seed42 | 10 | 62 | 12 | 2 | 4426 | 18/64 |
| CMGNet-SMILES (smiles) | seed2006 | 1 | 62 | 8 | 2 | 4430 | 10/64 |
| CMGNet-SMILES (smiles) | seed2006 | 3 | 62 | 10 | 2 | 4428 | 14/64 |
| CMGNet-SMILES (smiles) | seed2006 | 5 | 62 | 13 | 2 | 4425 | 19/64 |
| CMGNet-SMILES (smiles) | seed2006 | 10 | 62 | 16 | 2 | 4422 | 22/64 |

## 数据来源(逐样本 dump,非表格)

| 模型 | seed | 文件 |
|---|---|---|
| CMGNet-0910 (base, SMILES) | 0910 | `/mnt/data2/NMR_Core_Mol_Detection/output/apr18_base0910_smiles_beams10/test_smiles_beams10_samples4502.json` |
| CMGNet-SCAFFOLD (frag) | seed0 | `/mnt/data2/NMR_Core_Mol_Detection/output/apr18_ex1209_s0_frag_beams10/test_frag_beams10_samples4502.json` |
| CMGNet-SCAFFOLD (frag) | seed42 | `/mnt/data2/NMR_Core_Mol_Detection/output/apr18_ex1209_s42_frag_beams10/test_frag_beams10_samples4502.json` |
| CMGNet-SCAFFOLD (frag) | seed2006 | `/mnt/data2/NMR_Core_Mol_Detection/output/apr18_ex1209_s2006_frag_beams10/test_frag_beams10_samples4502.json` |
| CMGNet-SMILES (smiles) | seed0 | `/mnt/data2/NMR_Core_Mol_Detection/output/apr18_ex1209_s0_smiles_beams10/test_smiles_beams10_samples4502.json` |
| CMGNet-SMILES (smiles) | seed42 | `/mnt/data2/NMR_Core_Mol_Detection/output/apr18_ex1209_s42_smiles_beams10/test_smiles_beams10_samples4502.json` |
| CMGNet-SMILES (smiles) | seed2006 | `/mnt/data2/NMR_Core_Mol_Detection/output/apr18_ex1209_s2006_smiles_beams10/test_smiles_beams10_samples4502.json` |

复算命令: `PYTHONPATH=. python analysis/metrics/topk_cmgnet_modes_4502.py`
