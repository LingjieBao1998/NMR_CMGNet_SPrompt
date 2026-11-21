# CMGNet 各模式 top-k 指标 — 外部集 4,502(64 正 / 4,438 负)

阈值/判定: top-k 内任一候选含全母核(4-喹诺酮-3-羧酸及其氮杂类似物)即判阳;P=TP/(TP+FP), R=TP/64, F1=2PR/(P+R), Spec=TN/4,438, Acc=(TP+TN)/4,502。SA=参考结构 canonical 完全匹配 / 64(正例)。

多 seed 行写成 mean±sd(样本标准差 ddof=1,3 个种子)。

> **SPrompt 行是生成侧**(模型自己吐的 10 个候选)。部署管线里的阳性/阴性判决由 CReSS 嵌入 + XGBoost 分类器门控,与 k 无关:TP 63 / FP 4 / FN 1 / TN 4,434 → P 0.9403、R 0.9844、F1 0.9618、Spec 0.9991、Acc 0.9989(3 种子完全一致,见下方 XGBoost 段)。

> SPrompt 的 3 次运行 = ex1209 5ep 的 seed0 / seed42 / seed2006(`train_newdb_1209_extrain_5ep_seed*`),即论文/SI 口径;其中 seed42 的输出目录被误命名为 `apr18_sprompt_ex1209_beams10`(以 hparams 的 ckpt 为准)。**以下两组不能与本表混用**:
>
> (a) `output/test_sprompt_seed0`(= `results/predictions/sprompt_seed0_external4502_beams10.json`,2026-11-16)用的是释放版 1-epoch 权重 `checkpoints/sprompt_ex1209_1ep_seed0.ckpt`,单次运行给出 TP@10 64 / FP@10 24、SA@10 21/64;
>
> (b) `output/infer_sprompt_external`(2026-11-16)用的是 1029v2 base 权重,TP@10 62 / FP@10 24。

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
| CMGNet-SPrompt (nmr+frag, 生成侧) | 1 | 62.67±0.58 | 9.33±2.08 | 1.33±0.58 | 4428.67±2.08 | 0.8708±0.0257 | 0.9792±0.0090 | 0.9217±0.0161 | 0.9979±0.0005 | 0.9976±0.0005 | 0.1458±0.0239 |
| CMGNet-SPrompt (nmr+frag, 生成侧) | 3 | 63 | 12.00±1.00 | 1 | 4426.00±1.00 | 0.8401±0.0112 | 0.9844 | 0.9065±0.0065 | 0.9973±0.0002 | 0.9971±0.0002 | 0.2188±0.0271 |
| CMGNet-SPrompt (nmr+frag, 生成侧) | 5 | 63 | 14.00±2.00 | 1 | 4424.00±2.00 | 0.8186±0.0213 | 0.9844 | 0.8937±0.0127 | 0.9968±0.0005 | 0.9967±0.0004 | 0.2396±0.0180 |
| CMGNet-SPrompt (nmr+frag, 生成侧) | 10 | 63.33±0.58 | 16.67±2.08 | 0.67±0.58 | 4421.33±2.08 | 0.7920±0.0209 | 0.9896±0.0090 | 0.8797±0.0145 | 0.9962±0.0005 | 0.9961±0.0005 | 0.2917±0.0090 |

## XGBoost-CReSS(分类器,与 k 无关,τ=0.50)

输入是 CReSS 嵌入,不生成候选,因此没有 top-k;3 个种子的混淆矩阵完全相同,只有 AUPRC/AUROC 有细微差别。

| seed | TP | FP | FN | TN | Precision | Recall | F1 | Specificity | Accuracy | AUPRC | AUROC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| seed0 | 63 | 4 | 1 | 4434 | 0.9403 | 0.9844 | 0.9618 | 0.9991 | 0.9989 | 0.9870 | 0.999813 |
| seed42 | 63 | 4 | 1 | 4434 | 0.9403 | 0.9844 | 0.9618 | 0.9991 | 0.9989 | 0.9891 | 0.999820 |
| seed2006 | 63 | 4 | 1 | 4434 | 0.9403 | 0.9844 | 0.9618 | 0.9991 | 0.9989 | 0.9875 | 0.999824 |
| **mean±sd** | 63 | 4 | 1 | 4434 | 0.9403 | 0.9844 | 0.9618 | 0.9991 | 0.9989 | 0.9879±0.0011 | 0.999819±0.000005 |

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
| CMGNet-SPrompt (nmr+frag, 生成侧) | seed0 | 1 | 62 | 10 | 2 | 4428 | 8/64 |
| CMGNet-SPrompt (nmr+frag, 生成侧) | seed0 | 3 | 63 | 11 | 1 | 4427 | 13/64 |
| CMGNet-SPrompt (nmr+frag, 生成侧) | seed0 | 5 | 63 | 12 | 1 | 4426 | 14/64 |
| CMGNet-SPrompt (nmr+frag, 生成侧) | seed0 | 10 | 64 | 16 | 0 | 4422 | 19/64 |
| CMGNet-SPrompt (nmr+frag, 生成侧) | seed42 | 1 | 63 | 7 | 1 | 4431 | 11/64 |
| CMGNet-SPrompt (nmr+frag, 生成侧) | seed42 | 3 | 63 | 12 | 1 | 4426 | 16/64 |
| CMGNet-SPrompt (nmr+frag, 生成侧) | seed42 | 5 | 63 | 14 | 1 | 4424 | 16/64 |
| CMGNet-SPrompt (nmr+frag, 生成侧) | seed42 | 10 | 63 | 15 | 1 | 4423 | 18/64 |
| CMGNet-SPrompt (nmr+frag, 生成侧) | seed2006 | 1 | 63 | 11 | 1 | 4427 | 9/64 |
| CMGNet-SPrompt (nmr+frag, 生成侧) | seed2006 | 3 | 63 | 13 | 1 | 4425 | 13/64 |
| CMGNet-SPrompt (nmr+frag, 生成侧) | seed2006 | 5 | 63 | 16 | 1 | 4422 | 16/64 |
| CMGNet-SPrompt (nmr+frag, 生成侧) | seed2006 | 10 | 63 | 19 | 1 | 4419 | 19/64 |

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
| CMGNet-SPrompt (nmr+frag, 生成侧) | seed0 | `/mnt/data2/NMR_Core_Mol_Detection/output/apr18_sprompt_ex1209_s0/test_nmr+frag_beams10_samples4502.json` |
| CMGNet-SPrompt (nmr+frag, 生成侧) | seed42 | `/mnt/data2/NMR_Core_Mol_Detection/output/apr18_sprompt_ex1209_beams10/test_nmr+frag_beams10_samples4502.json` |
| CMGNet-SPrompt (nmr+frag, 生成侧) | seed2006 | `/mnt/data2/NMR_Core_Mol_Detection/output/apr18_sprompt_ex1209_s2006/test_nmr+frag_beams10_samples4502.json` |
| XGBoost-CReSS | 0/42/2006 | `reports/calib_data/external_apr18peak.npz` (y) + `reports/calib_data/seed_probs.npz` (external_apr18peak_p) |

复算命令: `PYTHONPATH=. python analysis/metrics/topk_cmgnet_modes_4502.py`
