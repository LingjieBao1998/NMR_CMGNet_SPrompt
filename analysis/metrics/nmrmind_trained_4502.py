#!/usr/bin/env python3
"""NMRMind 语料训练的模型在外部集 4,502 上的 top-k 指标汇总。

两类"用 NMRMind 训练"的结果:
  (1) CMGNet 用 NMRMind 语料训练(ring_system 三模式混合, m1010 / m1004),
      外部集 top-1/3/5/10 的母核命中与 exact 匹配;
  (2) XGBoost-CReSS 用 NMRMind 语料的 CReSS 嵌入训练(冻结编码器),
      读 output/xgb_cress_nmrmind/summary.json 的逐 seed 记录。
另附 NMRMind 原始权重(epoch-200)在外部集上的官方 top-10 输出,作为"模型本身"的对照。

判定口径与 results/table_cmgnet_modes_4502.md 完全一致(见 analysis/metrics/topk_external_4502.py)。

用法:
    PYTHONPATH=. python analysis/metrics/nmmind_trained_4502.py

作者:ZCode(Claude),2026-11-22
"""
import json
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

from analysis.metrics.topk_external_4502 import (
    N_NEG, N_POS, TOP_KS, has_core, metrics_from_counts)

RDLogger.DisableLog('rdApp.*')


def canon(smiles):
    """RDKit canonical SMILES;解析失败返回 None。"""
    if not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None

CORPUS = Path('/mnt/data2/NMR_Core_Mol_Detection')
METRIC_KEYS = ('precision', 'recall', 'f1', 'specificity', 'accuracy', 'sa')


def eval_dump(path):
    """读一份 4,502 dump,返回逐 k 的 TP/FP/exact 数组。"""
    data = json.load(open(path))
    if len(data) != N_POS + N_NEG:
        raise ValueError(f'{path}: {len(data)} 条,期望 {N_POS + N_NEG}')
    labels = [has_core(it['canonical_reference_smiles']) for it in data]
    refs = [canon(it['canonical_reference_smiles']) for it in data]
    if sum(labels) != N_POS:
        raise ValueError(f'{path}: 正例 {sum(labels)} != {N_POS}')
    out = {}
    for k in TOP_KS:
        tp = fp = ex = 0
        for it, lab, ref in zip(data, labels, refs):
            preds = it['predictions'][:k]
            hit = any(has_core(p.get('smiles')) for p in preds)
            if lab:
                tp += hit
                ex += bool(ref and any(canon(p.get('smiles')) == ref for p in preds))
            else:
                fp += hit
        out[k] = (tp, fp, ex)
    return out


def metrics(tp, fp, ex):
    """由 (TP, FP, exact) 算六项指标;exact 计数只对正例。"""
    m = metrics_from_counts(tp, fp, N_POS - tp, N_NEG - fp)
    m['sa'] = ex / N_POS
    return m


def agg(runs):
    """多 seed 聚合:逐 k、逐指标给 mean±sd(ddof=1)。"""
    res = {}
    for k in TOP_KS:
        vals = [metrics(*r[k]) for r in runs]
        cell = {}
        for key in METRIC_KEYS:
            arr = np.array([v[key] for v in vals], dtype=float)
            cell[key] = (arr.mean(), arr.std(ddof=1) if len(arr) > 1 else 0.0)
        cell['tp'] = (np.mean([r[k][0] for r in runs]),
                      np.std([r[k][0] for r in runs], ddof=1) if len(runs) > 1 else 0.0)
        cell['fp'] = (np.mean([r[k][1] for r in runs]),
                      np.std([r[k][1] for r in runs], ddof=1) if len(runs) > 1 else 0.0)
        res[k] = cell
    return res


def fmt(mean, sd, nd=4, count=False):
    if count:
        return f'{mean:.2f}±{sd:.2f}' if sd else f'{int(round(mean))}'
    return f'{mean:.{nd}f}±{sd:.{nd}f}' if sd else f'{mean:.{nd}f}'


def main():
    groups = [
        ('CMGNet-SCAFFOLD (frag)', 'frag',
         ['apr18_m1010_s0_frag_beams10/test_frag_beams10_samples4502.json',
          'apr18_m1010_s42_frag_beams10/test_frag_beams10_samples4502.json',
          'apr18_m1010_s2026_frag_beams10/test_frag_beams10_samples4502.json']),
        ('CMGNet-SMILES (smiles)', 'smiles',
         ['apr18_m1010_s0_smiles_beams10/test_smiles_beams10_samples4502.json',
          'apr18_m1010_s42_smiles_beams10/test_smiles_beams10_samples4502.json',
          'apr18_m1010_s2026_smiles_beams10/test_smiles_beams10_samples4502.json']),
        ('CMGNet-SPrompt (nmr+frag)', 'sprompt',
         ['apr18_m1010_s0_sprompt_beams10/test_nmr+frag_beams10_samples4502.json',
          'apr18_m1010_s42_sprompt_beams10/test_nmr+frag_beams10_samples4502.json',
          'apr18_m1010_s2026_sprompt_beams10/test_nmr+frag_beams10_samples4502.json']),
        ('CMGNet-SMILES (m1004, 单次)', 'smiles',
         ['apr18_m1004_smiles_beams10/test_smiles_beams10_samples4502.json']),
        ('CMGNet-SCAFFOLD (m1004, 单次)', 'frag',
         ['apr18_m1004_frag_beams10/test_frag_beams10_samples4502.json']),
    ]

    lines = ['# NMRMind 语料训练 → 外部集 4,502 汇总', '',
             f'外部集 {N_POS + N_NEG} 条({N_POS} 正 / {N_NEG} 负),判定与主表同口径:'
             'top-k 内任一候选含喹诺酮母核即判阳;SA = 参考 canonical 完全匹配 / 64。',
             '多 seed(m1010 组,seeds 0/42/2026)为 mean±sd(ddof=1)。', '',
             '| 模型 | Top-k | TP | FP | Precision | Recall | F1 | Specificity '
             '| Accuracy | SA |', '|---|---|---|---|---|---|---|---|---|---|']

    for name, mode, rels in groups:
        runs = [eval_dump(CORPUS / 'output' / r) for r in rels]
        a = agg(runs)
        for k in TOP_KS:
            c = a[k]
            lines.append(
                f'| {name} | {k} | {fmt(*c["tp"], count=True)} | {fmt(*c["fp"], count=True)} '
                f'| {fmt(*c["precision"])} | {fmt(*c["recall"])} | {fmt(*c["f1"])} '
                f'| {fmt(*c["specificity"])} | {fmt(*c["accuracy"])} | {fmt(*c["sa"])} |')

    # XGBoost-CReSS on NMRMind 语料(冻结 CReSS 嵌入)
    summ = json.load(open(CORPUS / 'output/xgb_cress_nmrmind/summary.json'))
    lines += ['', '## XGBoost-CReSS(NMRMind 语料训练,冻结 CReSS 嵌入,τ=0.50)', '',
              '| seed | ext AUPRC | ext AUROC | TP | FP | Precision | Recall | F1 |',
              '|---|---|---|---|---|---|---|---|']
    for seed, rec in sorted(summ['seeds'].items(), key=lambda kv: int(kv[0])):
        e = rec['external@0.5']
        lines.append(f"| {seed} | {e['auprc']:.4f} | {e['roc_auc']:.4f} | {e['tp']} | "
                     f"{e['fp']} | {e['precision']:.4f} | {e['recall']:.4f} | {e['f1']:.4f} |")

    # NMRMind 原始权重(官方 top-10 输出)作为"对方模型本身"的对照
    lines += ['', '## 对照:NMRMind 原始权重(epoch-200)官方 top-10 输出', '',
              '| 输出文件 | Top-k | TP | FP | Precision | Recall | F1 | SA |',
              '|---|---|---|---|---|---|---|---|']
    for f in ('april18peak_result10.json', 'external_4502_rerun_result10.json'):
        d = json.load(open(CORPUS / 'NMRMind_ref/result' / f))
        labels = [has_core(it['smiles'][0] if isinstance(it['smiles'], list) else it['smiles'])
                  for it in d]
        refs = [canon(it['smiles'][0] if isinstance(it['smiles'], list) else it['smiles'])
                for it in d]
        for k in TOP_KS:
            tp = fp = ex = 0
            for it, lab, ref in zip(d, labels, refs):
                res = it['result'][:k]
                hit = any(has_core(s) for s in res)
                if lab:
                    tp += hit
                    ex += bool(ref and any(canon(s) == ref for s in res))
                else:
                    fp += hit
            m = metrics(tp, fp, ex)
            lines.append(f'| {f} | {k} | {tp} | {fp} | {m["precision"]:.4f} '
                         f'| {m["recall"]:.4f} | {m["f1"]:.4f} | {m["sa"]:.4f} |')

    lines += [
        '', '## 备注', '',
        '1. m1010 = NMRMind 语料训练的 CMGNet(ring_system 三模式混合,seeds 0/42/2026),'
        '锚点与 SI Table S3 一致:SCAFFOLD R@10 0.682±0.048、SMILES R@10 0.583±0.063。',
        '2. `apr18_m1010_s2026_sprompt_beams10` 目录名用 2026,与论文的 seed 2006 差一位数字,'
        '内容按 seed 2026 处理;核对时以 hparams 为准。',
        '3. m1004 为 10-04 的首次单次运行,种子未在目录名中标注,只作参考,不参与 mean±sd。',
        '4. SPrompt 行是**生成侧**(模型自己出 10 个候选);正文 Table 5 的 NMRMind 语料 SPrompt '
        '行给的是分类器门控口径(P 0.970±0.026),两者不可直接对比。',
        '5. `external_4502_rerun_result10.json` 是另一次重跑(谱输入口径不同),'
        '比稿件口径低 8 个正例,引用时以 `april18peak_result10.json` 为准。',
    ]

    md = '\n'.join(lines) + '\n'
    out = Path('/mnt/data2/NMR_CMGNet_SPrompt/results/table_nmrmind_trained_4502.md')
    out.write_text(md, encoding='utf-8')
    print(md)
    print('written:', out)


if __name__ == '__main__':
    main()
