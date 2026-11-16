#!/usr/bin/env python3
"""母核 Top-1/3/5/10 混淆矩阵 + SMILES exact-match 准确率 + 有效性 (valid)。
用法: python summarize_core_smiles.py <输出.md> <标题> <结果1.json> [结果2.json ...]

指标定义:
  混淆矩阵: 正类 = 参考分子含母核; Top-N 中任一预测含母核即算 TP
  ExactMatch@N : 前 N 个候选中任一个与参考 SMILES canonical 完全一致 的样本占比
  ExactMatch@1 : rank-1 预测与参考完全一致 的样本占比
  Valid@1      : rank-1 预测为合法 SMILES (RDKit 可解析) 的样本占比 —— 标准 validity
  ValidRate@N  : 每条样本前 N 个候选中合法 SMILES 平均占比
"""
import json
import sys

import numpy as np

out_md, title, paths = sys.argv[1], sys.argv[2], sys.argv[3:]

data = []
for p in paths:
    with open(p) as f:
        data += json.load(f)

# 与 summarize_confusion.py 一致: 按参考 SMILES 去重, 保留候选数最多的那条
uniq = {}
for x in data:
    r = x.get('reference_smiles', '')
    if r not in uniq or len(x.get('predictions', [])) > len(uniq[r].get('predictions', [])):
        uniq[r] = x
data = list(uniq.values())
n_pos = sum(1 for x in data if x['reference_analysis']['has_core_structure'])

lines = [f'# {title}', '',
         f'样本 {len(data)} 条, 含母核 {n_pos} 条', '',
         '## 一、母核混淆矩阵 (正类=含母核; Top-N 任一预测含母核即 TP)', '',
         '| Top-N | TP | FP | FN | TN | Recall | Precision | F1 | Specificity | Accuracy |',
         '|---|---|---|---|---|---|---|---|---|---|']

for n in [1, 3, 5, 10]:
    tp = fp = fn = tn = 0
    for x in data:
        rr = x['reference_analysis']['has_core_structure']
        pp = any(q.get('has_core_structure', False) for q in x.get('predictions', [])[:n])
        if rr and pp:
            tp += 1
        elif rr and not pp:
            fn += 1
        elif not rr and pp:
            fp += 1
        else:
            tn += 1
    rec = tp / (tp + fn) if (tp + fn) else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    spec = tn / (tn + fp) if (tn + fp) else 0
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    lines.append(f'| Top-{n} | {tp} | {fp} | {fn} | {tn} | {rec:.4f} | {prec:.4f} | '
                 f'{f1:.4f} | {spec:.4f} | {acc:.4f} |')

lines += ['', '## 二、SMILES 准确率与有效性', '',
          '| Top-N | ExactMatch@N | ExactMatch@1 | Valid@1 | ValidRate@N |',
          '|---|---|---|---|---|']

for n in [1, 3, 5, 10]:
    em_n = em_1 = v_1 = 0
    valid_frac = []
    for x in data:
        preds = x.get('predictions', [])[:n]
        if not preds:
            valid_frac.append(0.0)
            continue
        if any(q.get('is_exact_match', False) for q in preds):
            em_n += 1
        if preds[0].get('is_exact_match', False):
            em_1 += 1
        if preds[0].get('is_valid', False):
            v_1 += 1
        valid_frac.append(sum(1 for q in preds if q.get('is_valid', False)) / len(preds))
    tot = max(1, len(data))
    lines.append(f'| Top-{n} | {em_n / tot:.4f} | {em_1 / tot:.4f} | {v_1 / tot:.4f} | '
                 f'{float(np.mean(valid_frac)):.4f} |')

with open(out_md, 'w') as f:
    f.write('\n'.join(lines) + '\n')
print('\n'.join(lines))
print('saved:', out_md)
