import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

B = '/mnt/data2/NMR_Core_Mol_Detection/'
R = B + 'reports/'
os.makedirs(R, exist_ok=True)

smi = 'O=C(O)c1cn(C2CC2)c2c([N+](=O)[O-])c(-c3ccccc3)c(F)cc2c1=O'
mol = Chem.MolFromSmiles(smi)
d = rdMolDraw2D.MolDraw2DCairo(1400, 900)
o = d.drawOptions(); o.bondLineWidth = 3; o.minFontSize = 24; o.maxFontSize = 34
rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
d.FinishDrawing()
open(R + 'fn12_structure.png', 'wb').write(d.GetDrawingText())

labels = [
    'Full spectrum (18 peaks)',
    '- 130.4 ppm',
    '- 129.9, 130.4, 130.6 ppm',
    '- 129.9, 130.4, 130.6, 128.7 ppm',
    '- 175.4 ppm',
]
p = [0.0122, 0.3114, 0.9098, 0.8836, 0.0000]
colors = ['#c0392b' if v < 0.5 else '#1e8449' for v in p]

fig, ax = plt.subplots(figsize=(12, 6), dpi=160)
y = np.arange(len(labels))[::-1]
bars = ax.barh(y, p, color=colors, height=0.55)
ax.axvline(0.5, color='#555', ls='--', lw=1.5)
ax.text(0.505, y[0] + 0.35, 'decision threshold 0.5', fontsize=11, color='#555')
for yy, v in zip(y, p):
    ax.text(v + 0.012, yy, f'{v:.3f}', va='center', fontsize=13, fontweight='bold')
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=12)
ax.set_xlim(0, 1.08); ax.set_xlabel('Predicted probability of being positive', fontsize=12)
ax.set_title('FN #12 — leave-peaks-out ablation (CReSS embedding + XGBoost)', fontsize=14)
ax.grid(axis='x', alpha=0.25)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
plt.tight_layout()
plt.savefig(R + 'fn12_peak_ablation.png')
print('saved', R + 'fn12_structure.png', R + 'fn12_peak_ablation.png')
