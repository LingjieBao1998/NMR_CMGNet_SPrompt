#!/usr/bin/env python3
"""5 条模拟 13C 谱：不同 CMGNet 变体的母核命中率 / 完全匹配率柱状图。

数据来源：`reports/sim5_multimodel_summary.csv`（由
`reports/make_sim5_multimodel_summary.py` 生成）。
只画 Top-10：左柱 core@10，右柱 exact@10。

用法：
  /home/lingjiebao/venvs/py312_tf457/bin/python reports/make_sim5_multimodel_plot.py

作者：ZCode（Claude），2026-11-23
"""
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

REPO = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
os.chdir(REPO)

## 与前面 F1 / Recall 图同一套取色（自参考图采样），按模型顺序排
COLORS = ['#dcdcdc', '#c4c0dd', '#fb8c80', '#8cb8d7', '#fdbb71', '#bae178', '#fcd2e7']

rows = list(csv.DictReader(open('reports/sim5_multimodel_summary.csv')))
k10 = [r for r in rows if r['top_k'] == '10']

models = [r['model'] for r in k10]
# 简化显示名，避免横轴过长
SHORT = {
    'NMR-Solver': 'NMR-Solver',
    'CMGNet-SPrompt (ex1209 5ep)': 'CMGNet-SPrompt',
    'CMGNet-SMILES (ex1209 5ep)': 'CMGNet-SMILES',
    'CMGNet-SCAFFOLD (ex1209 5ep)': 'CMGNet-SCAFFOLD',
    'CMGNet-SMILES (NMRMind m1010)': 'CMGNet-SMILES\n(NMRMind)',
    'CMGNet-SCAFFOLD (NMRMind m1010)': 'CMGNet-SCAFFOLD\n(NMRMind)',
    'CMGNet-0910 (base, no retrain)': 'CMGNet-0910\n(base)',
}
labels = [SHORT.get(m, m) for m in models]
core = [float(r['core_acc']) for r in k10]
core_sd = [float(r['core_sd']) for r in k10]
exact = [float(r['exact_acc']) for r in k10]
exact_sd = [float(r['exact_sd']) for r in k10]

x = np.arange(len(models))
w = 0.36

fig, ax = plt.subplots(figsize=(12, 6.8), constrained_layout=True)
ax.bar(x - w / 2, core, w, yerr=core_sd, capsize=3, color=COLORS,
       edgecolor='black', linewidth=0.6, label='Core accuracy (Top-10)')
ax.bar(x + w / 2, exact, w, yerr=exact_sd, capsize=3, color=COLORS,
       edgecolor='black', linewidth=0.6, hatch='///', alpha=0.55,
       label='Exact SMILES match (Top-10)')

for xi, yi, sd in zip(x - w / 2, core, core_sd):
    ax.text(xi, yi + sd + 0.015, f'{yi:.3f}', ha='center', va='bottom', fontsize=8.5)
for xi, yi, sd in zip(x + w / 2, exact, exact_sd):
    ax.text(xi, yi + sd + 0.015, f'{yi:.3f}', ha='center', va='bottom', fontsize=8.5)

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=35, ha='right')
ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_ylim(0, 1.15)
ax.set_title('5 Simulated 13C Spectra: Core Hit and Exact Match across CMGNet Variants (Top-10)',
             fontsize=12.5, fontweight='bold')
ax.grid(axis='y', alpha=0.25, linewidth=0.6)
ax.legend(frameon=False, loc='upper right', ncol=2, fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.savefig('reports/sim5_multimodel_core_exact.png', dpi=200, bbox_inches='tight')
plt.savefig('reports/sim5_multimodel_core_exact.pdf', bbox_inches='tight')
print('ok')
