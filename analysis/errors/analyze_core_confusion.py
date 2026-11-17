#!/usr/bin/env python3
"""
从 --do_test 的结果 JSON 计算母核结构的 Top-N 混淆矩阵与指标。
语义与 evaluate_smiles_beams10_samples50381.py 一致:
  predicted_has_core(top-N) = Top-N 中任一预测分子含母核
  TP: 参考含母核 且 Top-N 命中 | FN: 参考含母核 但未命中
  FP: 参考不含母核 但预测命中 | TN: 参考不含母核 且未命中

用法: python analyze_core_confusion.py <结果.json> [更多.json ...]
"""
import json
import sys

TOP_NS = [1, 3, 5, 10]


def analyze(path, top_ns=TOP_NS):
    data = json.load(open(path))
    n_valid = len(data)
    n_pos = sum(1 for x in data if x['reference_analysis']['has_core_structure'])
    n_neg = n_valid - n_pos
    print(f"\n{'='*100}\n📄 {path}")
    print(f"   有效样本 {n_valid} (参考含母核 {n_pos} = {n_pos/max(1,n_valid)*100:.2f}%, 不含 {n_neg})")

    max_preds = max((len(x.get('predictions', [])) for x in data), default=0)
    print(f"   每条预测数: {max_preds}")

    rows = []
    for n in top_ns:
        tp = fp = fn = tn = 0
        for x in data:
            ref_pos = x['reference_analysis']['has_core_structure']
            preds = x.get('predictions', [])[:n]
            pred_pos = any(p.get('has_core_structure', False) for p in preds)
            if ref_pos and pred_pos:
                tp += 1
            elif ref_pos and not pred_pos:
                fn += 1
            elif not ref_pos and pred_pos:
                fp += 1
            else:
                tn += 1
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        acc = (tp + tn) / max(1, tp + tn + fp + fn)
        rows.append((n, tp, fp, fn, tn, prec, rec, f1, spec, acc))

    print(f"\n   {'Top-N':<7}|{'TP':>6} {'FP':>6} {'FN':>6} {'TN':>7} | {'Precision':>9} {'Recall':>8} {'F1':>8} {'Specificity':>11} {'Accuracy':>9}")
    print("   " + "-" * 92)
    for n, tp, fp, fn, tn, prec, rec, f1, spec, acc in rows:
        print(f"   Top-{n:<3}|{tp:>6} {fp:>6} {fn:>6} {tn:>7} | {prec:>9.4f} {rec:>8.4f} {f1:>8.4f} {spec:>11.4f} {acc:>9.4f}")
    return rows


if __name__ == '__main__':
    for p in sys.argv[1:]:
        analyze(p)
