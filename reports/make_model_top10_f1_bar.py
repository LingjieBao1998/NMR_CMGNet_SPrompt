import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

models = ['NMR-Solver','NMRMind','CMGNet','XGBoost-CReSS','CMGNet-SCAFFOLD','CMGNet-SMILES','CMGNet-SPrompt']
f1   = [0.917, 0.828, 0.867, 0.962, 0.906, 0.891, 0.962]
sd   = [0.000, 0.000, 0.000, 0.000, 0.018, 0.015, 0.000]
palette = ['#c4c0dd','#dcdcdc','#fb8c80','#8cb8d7','#fdbb71','#bae178','#fcd2e7']

x = np.arange(len(models))
fig, ax = plt.subplots(figsize=(12,7), constrained_layout=True)
ax.bar(x, f1, width=0.62, yerr=sd, capsize=3, color=palette, edgecolor='#3f3f3f', linewidth=0.6)
for xi, yi in zip(x, f1):
    ax.text(xi, yi+0.012, f'{yi:.3f}', ha='center', va='bottom', fontsize=9, color='#3f3f3f')

best = max(f1)
ax.axhline(best, color='red', linestyle='--', linewidth=1.3)
ax.text(0.02, best+0.012, f'Best F1 = {best:.3f}', color='red', ha='left', va='bottom',
        fontsize=10, transform=ax.get_yaxis_transform())

ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha='right')
ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
ax.set_ylim(0,1.08)
ax.set_title('F1 Score Comparison Across Models (Top-10)', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.25, linewidth=0.6)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig('/mnt/data2/NMR_Core_Mol_Detection/reports/model_top10_f1_bar.png', dpi=200, bbox_inches='tight')
plt.savefig('/mnt/data2/NMR_Core_Mol_Detection/reports/model_top10_f1_bar.pdf', bbox_inches='tight')
print('ok')
