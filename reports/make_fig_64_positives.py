#!/usr/bin/env python3
"""All 64 external positives: raw model probability at tau=0.50, FN #12 highlighted.

Data: reports/calib_data/external_apr18peak.npz (y, p_raw, smiles)
Output: results/figures/fig_64_positives.png/.pdf
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

d = np.load('reports/calib_data/external_apr18peak.npz', allow_pickle=True)
y, p = d['y'].astype(int), d['p_raw'].astype(float)
pos_idx = np.where(y == 1)[0]
pos_p = p[pos_idx]
order = np.argsort(pos_p)
pos_p_sorted = pos_p[order]
pos_idx_sorted = pos_idx[order]

fig, ax = plt.subplots(figsize=(9.5, 4.2), constrained_layout=True)
x = np.arange(1, len(pos_p_sorted) + 1)

above = pos_p_sorted >= 0.5
below = ~above
ax.scatter(x[above], pos_p_sorted[above], s=28, color='#2C7FB8', edgecolors='white', linewidths=0.4, zorder=3, label='Predicted positive (p ≥ 0.50)')
ax.scatter(x[below], pos_p_sorted[below], s=42, color='#D55E00', edgecolors='black', linewidths=0.5, zorder=4, label='Predicted negative (p < 0.50)')

ax.axhline(0.5, color='#C0392B', linestyle='--', linewidth=1.2, zorder=2)
ax.text(64.5, 0.505, r'$\tau=0.50$', color='#C0392B', fontsize=9, va='bottom', ha='right')

fn_mask = pos_idx_sorted == 12
if fn_mask.any():
    xi = x[np.where(fn_mask)[0][0]]
    yi = pos_p_sorted[np.where(fn_mask)[0][0]]
    ax.annotate(f'FN #12\np={yi:.3f}', xy=(xi, yi), xytext=(xi + 4, 0.18),
                arrowprops=dict(arrowstyle='->', color='#333333', lw=0.8),
                fontsize=8.5, ha='left', va='center', color='#333333')

ax.set_xlabel('Positive samples ranked by predicted probability', fontsize=10)
ax.set_ylabel('Predicted probability', fontsize=10)
ax.set_ylim(-0.02, 1.02)
ax.set_xlim(0, 66)
ax.set_xticks([1, 16, 32, 48, 64])
ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
ax.grid(axis='y', color='#DDDDDD', linewidth=0.6, zorder=0)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(frameon=False, loc='upper left', fontsize=8.5, handletextpad=0.4)

fig.savefig('results/figures/fig_64_positives.png', dpi=300, bbox_inches='tight')
fig.savefig('results/figures/fig_64_positives.pdf', bbox_inches='tight')
print('saved')
