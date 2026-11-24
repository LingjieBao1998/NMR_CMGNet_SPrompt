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
# sort by probability for readability; keep track of original index for FN #12
order = np.argsort(pos_p)
pos_p_sorted = pos_p[order]
pos_idx_sorted = pos_idx[order]

fig, ax = plt.subplots(figsize=(10, 3.8), constrained_layout=True)
x = np.arange(len(pos_p_sorted))
colors = np.where(pos_p_sorted >= 0.5, '#4C72B0', '#D55E00')
ax.bar(x, pos_p_sorted, color=colors, edgecolor='none', width=0.8)

ax.axhline(0.5, color='red', linestyle='--', linewidth=1.0, label=r'$\tau=0.50$')

# highlight FN #12 (original index in 4502)
fn_mask = pos_idx_sorted == 12
if fn_mask.any():
    xi = np.where(fn_mask)[0][0]
    ax.bar(xi, pos_p_sorted[xi], color='#D55E00', edgecolor='black', linewidth=0.8, width=0.8, zorder=3)
    ax.annotate('FN #12\np=0.012', xy=(xi, pos_p_sorted[xi]), xytext=(xi+2.5, 0.18),
                arrowprops=dict(arrowstyle='->', color='black', lw=0.8),
                fontsize=8, ha='left', va='center')

ax.set_xlabel('64 positive samples (sorted by predicted probability)', fontsize=10)
ax.set_ylabel('Predicted probability', fontsize=10)
ax.set_ylim(0, 1.02)
ax.set_xlim(-1, len(x))
ax.set_title('Predicted Probabilities for 64 External Positive Samples', fontsize=11, fontweight='bold')
ax.legend(frameon=False, loc='upper left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.savefig('results/figures/fig_64_positives.png', dpi=300, bbox_inches='tight')
fig.savefig('results/figures/fig_64_positives.pdf', bbox_inches='tight')
print('saved')
