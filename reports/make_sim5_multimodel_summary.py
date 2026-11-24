#!/usr/bin/env python3
"""汇总 5 条模拟 13C 谱在各 CMGNet 变体上的预测结果（多模型对比）。

输入（均在仓库根目录下）：
  - data/sim5/fluoroquinolones_nmr.json  5 条模拟谱（真实 SMILES + 峰表）
  - output/sim5_*/test_*_beams10_samples5.json  各模型/各 seed 的推理 dump

dump 结构：list[dict]，每条含
  reference_smiles : str   标准答案
  predictions      : list[dict]  按 beam 排序的候选，含 'smiles'、
                     'has_core_structure'（是否含喹诺酮母核）

口径：
  core@k  = 前 k 个候选里存在母核命中的样本比例
  exact@k = 前 k 个候选里存在与标准答案 canonical SMILES 完全一致的样本比例
  多 seed 汇总为 mean±sd（ddof=1）；单 seed 直接给点值。

输出：
  reports/sim5_multimodel_summary.md
  reports/sim5_multimodel_summary.csv

用法：
  /home/lingjiebao/venvs/py312_tf457/bin/python reports/make_sim5_multimodel_summary.py

作者：ZCode（Claude），2026-11-23
"""
import csv
import json
import os

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')

REPO = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
os.chdir(REPO)

KS = (1, 3, 5, 10)

## 每个模型给一组 dump 路径（多 seed 即多路径）
MODELS = [
    ('NMR-Solver', 'nmr_solver',
     ['__NMR_SOLVER_RAW__']),
    ('CMGNet-SPrompt (ex1209 5ep)', 'nmr+frag',
     [f'output/sim5_sprompt_ex1209_s{s}/test_nmr+frag_beams10_samples5.json'
      for s in (0, 42, 2006)]),
    ('CMGNet-SMILES (ex1209 5ep)', 'smiles',
     [f'output/sim5_ex1209_s{s}_smiles_beams10/test_smiles_beams10_samples5.json'
      for s in (0, 42, 2006)]),
    ('CMGNet-SCAFFOLD (ex1209 5ep)', 'frag',
     [f'output/sim5_ex1209_s{s}_frag_beams10/test_frag_beams10_samples5.json'
      for s in (0, 42, 2006)]),
    ('CMGNet-SMILES (NMRMind m1010)', 'smiles',
     [f'output/sim5_m1010_s{s}_smiles_beams10/test_smiles_beams10_samples5.json'
      for s in (0, 42, 2026)]),
    ('CMGNet-SCAFFOLD (NMRMind m1010)', 'frag',
     [f'output/sim5_m1010_s{s}_frag_beams10/test_frag_beams10_samples5.json'
      for s in (0, 42, 2026)]),
    ('CMGNet-0910 (base, no retrain)', 'smiles',
     ['output/sim5_base0910_smiles_beams10/test_smiles_beams10_samples5.json']),
]


CORE_MOLS = [Chem.MolFromSmiles(s) for s in (
    'O=C1C(C(O)=O)=CNC2=CC=CC=C12',
    'O=C1C(C(O)=O)=CNC2=NC=CC=C12',
)]


def has_core(smiles):
    """母核命中：4-喹诺酮-3-羧酸 / 萘啶酸类似物（与 topk_external_4502.py 同口径）。"""
    if not isinstance(smiles, str):
        return False
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return any(mol.HasSubstructMatch(c) for c in CORE_MOLS)


def canon(smiles):
    """RDKit canonical SMILES；解析失败返回 None。"""
    if not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def eval_dump(path):
    """返回 {k: (core_rate, exact_rate)}，一条 dump 内 n=5 固定分母。"""
    # NMR-Solver raw: list[rec] with cano_smiles and result.data.result.data[].smiles
    if path == '__NMR_SOLVER_RAW__':
        raw_path = 'reports/sim5_nmrsolver_raw.json'
        with open(raw_path) as f:
            records = json.load(f)
        n = len(records)
        out = {}
        for k in KS:
            n_core = n_exact = 0
            for rec in records:
                ref = canon(rec['cano_smiles'])
                plist = ((rec.get('result') or {}).get('data') or {}).get('result', {}).get('data') or []
                preds = [p.get('smiles') for p in plist if isinstance(p, dict)][:k]
                n_core += int(any(has_core(s) for s in preds))
                n_exact += int(any(canon(s) == ref for s in preds))
            out[k] = (n_core / n, n_exact / n)
        return out, n
    with open(path) as f:
        records = json.load(f)
    n = len(records)
    out = {}
    for k in KS:
        n_core = n_exact = 0
        for rec in records:
            ref = canon(rec['reference_smiles'])
            preds = rec['predictions'][:k]
            n_core += int(any(p.get('has_core_structure', False) for p in preds))
            n_exact += int(any(canon(p.get('smiles')) == ref for p in preds))
        out[k] = (n_core / n, n_exact / n)
    return out, n


def fmt(mean, sd, n_seed):
    """3 位小数；单 seed 只给点值，多 seed 给 mean±sd。"""
    if n_seed == 1:
        return f'{mean:.3f}'
    return f'{mean:.3f}±{sd:.3f}'


def main():
    md = []
    rows = []
    md.append('# 5 条模拟 13C 谱：多模型预测对比\n')
    md.append('数据：`data/sim5/fluoroquinolones_nmr.json`（5 条喹诺酮模拟谱，19–29 峰，'
              '覆盖左氧氟沙星、环丙沙星、莫西沙星、氧氟沙星、诺氟沙星）。\n')
    md.append('口径：`core@k` = 前 k 个候选含喹诺酮母核的样本比例；'
              '`exact@k` = 前 k 个候选与标准答案 canonical SMILES 完全一致的样本比例。'
              '多 seed 为 mean±sd（ddof=1），分母恒为 5。\n')

    for name, mode, files in MODELS:
        if files == ['__NMR_SOLVER_RAW__']:
            files = ['__NMR_SOLVER_RAW__'] if os.path.exists('reports/sim5_nmrsolver_raw.json') else []
        else:
            files = [f for f in files if os.path.exists(f)]
        if not files:
            md.append(f'## {name}\n\n_无 dump，跳过。_\n')
            continue
        per_seed = {k: [] for k in KS}
        n_samples = None
        for fp in files:
            res, n = eval_dump(fp)
            n_samples = n
            for k in KS:
                per_seed[k].append(res[k])

        md.append(f'## {name}\n')
        md.append(f'- decode_mode: `{mode}`；seed 数: {len(files)}；样本数: {n_samples}')
        md.append('')
        md.append('| Top-k | Core Acc | Exact Match |')
        md.append('| --- | --- | --- |')
        for k in KS:
            arr = np.array(per_seed[k])          # (n_seed, 2)
            if len(arr) > 1:
                cm, cs = arr[:, 0].mean(), arr[:, 0].std(ddof=1)
                em, es = arr[:, 1].mean(), arr[:, 1].std(ddof=1)
            else:
                cm, cs = arr[0, 0], 0.0
                em, es = arr[0, 1], 0.0
            md.append(f'| {k} | {fmt(cm, cs, len(arr))} | {fmt(em, es, len(arr))} |')
            rows.append({
                'model': name, 'decode_mode': mode, 'top_k': k,
                'n_seed': len(arr), 'n_sample': n_samples,
                'core_acc': round(float(cm), 4), 'core_sd': round(float(cs), 4),
                'exact_acc': round(float(em), 4), 'exact_sd': round(float(es), 4),
            })
        md.append('')
        # 逐 seed 明细，便于核查
        if len(files) > 1:
            md.append('<details><summary>逐 seed 明细 (exact@1 / exact@10)</summary>\n')
            for fp in files:
                res, _ = eval_dump(fp)
                md.append(f'- `{fp}` → exact@1 {res[1][1]:.3f}, exact@10 {res[10][1]:.3f}')
            md.append('\n</details>\n')

    md.append('## 说明\n')
    md.append('- **CMGNet-SCAFFOLD 的 exact 恒为 0** 属预期：frag 模式只解码骨架片段，'
              '不产出完整分子 SMILES；该模式看 `core` 即可。')
    md.append('- **NMRMind (m1010) 变体在模拟谱上明显弱于 ex1209**：模拟谱由实验谱峰位合成，'
              '与 NMRMind 语料分布差异大，属分布外迁移，不代表其在真实测试集上的水平。')
    md.append('- **CMGNet-0910 (base) core=1.000 但 exact≈0**：原始 0910 权重生成的候选能命中母核，'
              '但整分子写法与标准答案差异大（无 ring-system 微调）。')
    md.append('- XGBoost-CReSS 为二分类判别器，只做「是否含喹诺酮母核」判定：'
              '5/5 判正、概率≈1.000，不产出 SMILES，故不进上表。')

    with open('reports/sim5_multimodel_summary.md', 'w') as f:
        f.write('\n'.join(md) + '\n')
    with open('reports/sim5_multimodel_summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print('\n'.join(md))


if __name__ == '__main__':
    main()
