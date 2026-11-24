import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Patch
import numpy as np

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

models = ['NMR-Solver','NMRMind','XGBoost-CReSS','CMGNet','CMGNet-SPrompt']
clean = np.array([0.833, 0.791, 0.969, 0.854, 0.969])
raw   = np.array([0.604, 0.646, 0.861, 0.729, 0.861])

colors = ['#c4c0dd','#fb8c80','#8cb8d7','#fdbb71','#bae178']
x = np.arange(len(models))
w = 0.55

fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)

ax.bar(x, clean, width=w, color=colors, alpha=0.35, edgecolor='black',
       linewidth=0.6, hatch='///', zorder=1)
ax.bar(x, raw, width=w, color=colors, alpha=0.95, edgecolor='black',
       linewidth=0.6, zorder=2)

for xi, r, c in zip(x, raw, clean):
    ax.text(xi, r, f'{r:.3f}', ha='center', va='bottom', color='#1f4e79',
            fontsize=9, fontweight='bold', zorder=3,
            path_effects=[pe.withStroke(linewidth=2.2, foreground='white')])
    ax.text(xi, c, f'{c:.3f}', ha='center', va='bottom', color='#c00000',
            fontsize=9, fontweight='bold', zorder=3,
            path_effects=[pe.withStroke(linewidth=2.2, foreground='white')])

best_raw = raw.max()
best_clean = clean.max()
ax.axhline(best_raw, color='#d62728', linestyle='--', linewidth=1.2)
ax.text(0.015, best_raw+0.010, f'Raw Best = {best_raw:.3f}', color='#d62728',
        va='bottom', ha='left', fontsize=9, transform=ax.get_yaxis_transform())
ax.axhline(best_clean, color='#2ca02c', linestyle='--', linewidth=1.2)
ax.text(0.015, best_clean+0.010, f'Clean Best = {best_clean:.3f}', color='#2ca02c',
        va='bottom', ha='left', fontsize=9, transform=ax.get_yaxis_transform())

legend_elements = [
    Patch(facecolor='#b0b0b0', edgecolor='black', label='Raw Data'),
    Patch(facecolor='#b0b0b0', edgecolor='black', hatch='///', alpha=0.5, label='Clean Data'),
]

ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha='right')
ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_ylabel('Recall Score', fontsize=12, fontweight='bold')
ax.set_ylim(0, 1.16)
ax.set_title('Recall Scores: Raw Data vs Clean Data Performance', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.25, linewidth=0.6)
ax.legend(handles=legend_elements, frameon=False, ncol=2, loc='upper center')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig('/mnt/data2/NMR_Core_Mol_Detection/reports/recall_raw_vs_clean_overlay.png', dpi=200, bbox_inches='tight')
plt.savefig('/mnt/data2/NMR_Core_Mol_Detection/reports/recall_raw_vs_clean_overlay.pdf', bbox_inches='tight')
print('ok')
