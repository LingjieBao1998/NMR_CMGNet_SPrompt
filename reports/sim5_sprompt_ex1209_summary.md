# 5 个模拟实验：CMGNet-SPrompt（ex1209 5ep，3 seed）预测结果

输入：`data/sim5/fluoroquinolones_nmr.json`（5 条模拟 13C 谱，19–29 个峰）。
模型：`train_newdb_1209_extrain_5ep_seed{0,42,2006}/best_models/last.ckpt`，`decode_mode=nmr+frag`、`num_beams=10`、SPrompt 门控全置 1（5 条均为已知阳性）。
评测：exact = RDKit canonical 完全一致；core = 喹诺酮母核子结构命中（与 4,502 外部集同口径）。


## 总体

| seed | core@1 | exact@1 | exact@10 | MRR(exact) | mean Tanimoto(top-1) |
|---|---|---|---|---|---|
| s0 | 1.00 | 0.40 | 0.40 | 0.40 | 0.759 |
| s42 | 1.00 | 0.20 | 0.20 | 0.20 | 0.617 |
| s2006 | 1.00 | 0.20 | 0.20 | 0.20 | 0.670 |

3 seed 并集：exact 至少命中一次 2/5；top-1 均含母核 5/5。


## 逐化合物

| 化合物 | 参考 SMILES | seed | top-1 预测 | exact rank | core in top-10 | Tanimoto |
|---|---|---|---|---|---|---|
| Ciprofloxacin | `OC(C1=CN(C2CC2)C3=CC(N4CCNCC4)=C(F)C=C3C1=O)=O` | s0 | `NCCNc1cc2c(cc1F)c(=O)c(C(=O)O)cn2C1CC1` | — | 10/10 | 0.559 |
| Ciprofloxacin | `OC(C1=CN(C2CC2)C3=CC(N4CCNCC4)=C(F)C=C3C1=O)=O` | s42 | `O=C(O)c1cn(C2CC2)c2cc(N3CCN(C(=O)c4ccco4)CC3)c(F)cc2c1=O` | — | 10/10 | 0.635 |
| Ciprofloxacin | `OC(C1=CN(C2CC2)C3=CC(N4CCNCC4)=C(F)C=C3C1=O)=O` | s2006 | `O=C(O)c1cn(C2CC2)c2cc(N3CCN(c4ccco4)CC3)c(F)cc2c1=O` | — | 10/10 | 0.678 |
| Ofloxacin | `FC1=C(N2CCN(C)CC2)C3=C4C(C(C(C(O)=O)=CN4C(C)CO3)=O)=C1` | s0 | `CC1COc2c(N3CCN(C)CC3)c(F)cc3c(=O)c(C(=O)O)cn1c23` | 1 | 10/10 | 1.000 |
| Ofloxacin | `FC1=C(N2CCN(C)CC2)C3=C4C(C(C(C(O)=O)=CN4C(C)CO3)=O)=C1` | s42 | `CC1CN(c2cc3c(cc2F)c(=O)c(C(=O)O)cn3-c2cc(N)c(F)cc2F)CCN1` | — | 10/10 | 0.342 |
| Ofloxacin | `FC1=C(N2CCN(C)CC2)C3=C4C(C(C(C(O)=O)=CN4C(C)CO3)=O)=C1` | s2006 | `CC1COc2c(N3CCN(C)CC3)c(F)cc3c(=O)c(C(=O)O)cn1c23` | 1 | 10/10 | 1.000 |
| Enoxacin | `CCN1C2=NC(N3CCNCC3)=C(F)C=C2C(C(C(O)=O)=C1)=O` | s0 | `CCn1cc(C(=O)O)c(=O)c2cc(F)c(N3CCNCC3)nc21` | 1 | 10/10 | 1.000 |
| Enoxacin | `CCN1C2=NC(N3CCNCC3)=C(F)C=C2C(C(C(O)=O)=C1)=O` | s42 | `CCn1cc(C(=O)O)c(=O)c2cc(F)c(N3CCNCC3)nc21` | 1 | 10/10 | 1.000 |
| Enoxacin | `CCN1C2=NC(N3CCNCC3)=C(F)C=C2C(C(C(O)=O)=C1)=O` | s2006 | `CCn1cc(C(=O)O)c(=O)c2cc(F)c(N3CCN(c4ncc(F)cn4)CC3)c(F)c21` | — | 10/10 | 0.485 |
| L3-3 | `O=C(O)c1cn(C2CC2)c2cc(N3CCN(Cc4ccc5nccnc5c4)CC3)c(F)cc2c1=O` | s0 | `O=C(O)c1cn(C2CC2)c2cc(N3CCN(C(=O)c4ccc[n+](Cc5ccccc5)c4)CC3)c(F)cc2c1=O` | — | 10/10 | 0.530 |
| L3-3 | `O=C(O)c1cn(C2CC2)c2cc(N3CCN(Cc4ccc5nccnc5c4)CC3)c(F)cc2c1=O` | s42 | `O=C(O)c1cn(C2CC2)c2cc(N3CCN(c4nc5c(cc4F)c(=O)c(C(=O)O)cn5Cc4ccc([N+](=O)[O-])cc4)CC3)c(F)cc2c1=O` | — | 10/10 | 0.560 |
| L3-3 | `O=C(O)c1cn(C2CC2)c2cc(N3CCN(Cc4ccc5nccnc5c4)CC3)c(F)cc2c1=O` | s2006 | `O=C(Cn1cncn1)Nc1c(F)c(N2CCN(c3cc4c(cc3F)c(=O)c(C(=O)O)cn4C3CC3)CC2)c(F)c2c1c(=O)c(C(=O)O)cn2C1CC1` | — | 9/10 | 0.484 |
| L4-1 | `OC(C1=CN(C2CC2)C3=CC(N4CCN(CC5=CC=C(N=CC=N6)C6=C5)CC4)=C(F)C=C3C1=O)=O` | s0 | `O=C(O)c1cn(C2CC2)c2cc(N3CCN(Cc4ccc([N+](=O)[O-])cc4)CC3)c(F)cc2c1=O` | — | 10/10 | 0.704 |
| L4-1 | `OC(C1=CN(C2CC2)C3=CC(N4CCN(CC5=CC=C(N=CC=N6)C6=C5)CC4)=C(F)C=C3C1=O)=O` | s42 | `O=C(O)c1cn(C2CC2)c2cc(N3CCN(CC(=NO)c4ccc([N+](=O)[O-])cc4)CC3)c(F)cc2c1=O` | — | 10/10 | 0.549 |
| L4-1 | `OC(C1=CN(C2CC2)C3=CC(N4CCN(CC5=CC=C(N=CC=N6)C6=C5)CC4)=C(F)C=C3C1=O)=O` | s2006 | `O=C(O)c1cn(C2CC2)c2cc(N3CCN(Cc4ccc([N+](=O)[O-])cc4)CC3)c(F)cc2c1=O` | — | 10/10 | 0.704 |

明细（每条样本 10 个候选的 SMILES/score/core/exact/Tanimoto）见 `reports/sim5_sprompt_ex1209_predictions.csv`。


> 注意：这 5 条谱为模拟实验（C_nmr 非实测），且 Ciprofloxacin/L3-3/L4-1 的 top-10 内未出现与参考完全一致的 SMILES，属于该模型在外部集上常见的 exact 漏检（4,502 外部集 SA@10 约 0.29）。
