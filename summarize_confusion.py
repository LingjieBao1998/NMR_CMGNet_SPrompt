#!/usr/bin/env python3
"""合并若干结果JSON/pkl并输出母核 Top-1/3/5/10 混淆矩阵 (markdown)。
用法: python summarize_confusion.py <输出.md> <标题> <结果1> [结果2 ...]"""
import json
import sys
import numpy as np

out_md, title, paths = sys.argv[1], sys.argv[2], sys.argv[3:]

data = []
for p in paths:
    if p.endswith('.pkl'):
        import pickle
        with open(p, 'rb') as f:
            data += pickle.load(f)
    else:
        with open(p) as f:
            data += json.load(f)

uniq = {}
for x in data:
    r = x.get('reference_smiles', '')
    if r not in uniq or len(x.get('predictions', [])) > len(uniq[r].get('predictions', [])):
        uniq[r] = x
data = list(uniq.values())
n_pos = sum(1 for x in data if x['reference_analysis']['has_core_structure'])

lines = [f'# {title}',
         f'',
         f'样本 {len(data)} 条, 含母核 {n_pos} 条',
         '',
         '| Top-N | TP | FP | FN | TN | Recall | Precision | F1 | Specificity | Accuracy | Validity |',
         '|---|---|---|---|---|---|---|---|---|---|---|']
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
    val = float(np.mean([any(q.get('is_valid', False) for q in x.get('predictions', [])[:n]) for x in data])) if data else 0
    lines.append(f'| Top-{n} | {tp} | {fp} | {fn} | {tn} | {rec:.4f} | {prec:.4f} | {f1:.4f} | {spec:.4f} | {acc:.4f} | {val:.4f} |')

with open(out_md, 'w') as f:
    f.write('\n'.join(lines) + '\n')
print('\n'.join(lines))
print('saved:', out_md)
