"""Composite main-text figure for the FN #12 case study.

Panels:
  A  local UMAP around the FN
  B  neighbour molecules (positives / negatives)
  C  leave-peaks-out ablation
  D  embedding path #12 -> para-F analog (decision boundary)
"""
import io, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.ndimage import gaussian_filter
from PIL import Image
from rdkit import Chem, RDLogger
from rdkit.Chem.Draw import rdMolDraw2D
import joblib

RDLogger.DisableLog('rdApp.*')
import os as _os
B = _os.environ.get('NMR_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
B = B.rstrip('/') + '/'
R_OUT = B + 'reports/'
NV = 3_298_042

# ---------- data ----------
y = np.load(B + 'viz_tmp/train_labels.npy')[:NV]
C_fit = np.load(B + 'viz_tmp/coords_fit.npy')
fit_idx = np.load(B + 'viz_tmp/fit_idx.npy')
C_rest = np.load(B + 'viz_tmp/coords_rest.npy')
rest_idx = np.load(B + 'viz_tmp/rest_idx.npy')
C_te = C_fit[len(fit_idx):]
q = C_te[12]

XY = np.empty((NV, 2), dtype=np.float32)
m1 = fit_idx < NV; m2 = rest_idx < NV
XY[fit_idx[m1]] = C_fit[:len(fit_idx)][m1]
XY[rest_idx[m2]] = C_rest[m2]

R = 0.78
loc = np.where(np.linalg.norm(XY - q, axis=1) < R)[0]
P = XY[loc]; L = y[loc]

_Z = np.load(B + 'viz_tmp/train_emb_subset.npz')
_SUB_IDX, _SUB_EMB = _Z['idx'], _Z['emb']
def emb_rows(ids):
    ids = np.asarray(ids)
    pos = np.clip(np.searchsorted(_SUB_IDX, ids), 0, len(_SUB_IDX) - 1)
    return _SUB_EMB[pos].astype(np.float32)
E = np.asarray(emb_rows(loc), dtype=np.float32)
model = joblib.load(B + 'models/xgboost/model_xgb_cress_new_dataset_seed42.pkl')
p_loc = model.predict_proba(E)[:, 1]

H = 130
gx = np.linspace(q[0] - R, q[0] + R, H)
gy = np.linspace(q[1] - R, q[1] + R, H)
xi = ((P[:, 0] - gx[0]) / (2 * R) * (H - 1)).astype(int).clip(0, H - 1)
yi = ((P[:, 1] - gy[0]) / (2 * R) * (H - 1)).astype(int).clip(0, H - 1)
cnt = np.zeros((H, H)); sm = np.zeros((H, H))
np.add.at(cnt, (yi, xi), 1); np.add.at(sm, (yi, xi), p_loc)
cnt = gaussian_filter(cnt, 1.6); sm = gaussian_filter(sm, 1.6)
field = np.divide(sm, cnt, out=np.zeros_like(sm), where=cnt > 1e-6)

# ---------- molecules ----------
smi = dict(
    fn='O=C(O)c1cn(C2CC2)c2c([N+](=O)[O-])c(-c3ccccc3)c(F)cc2c1=O',
    p1='O=C(O)c1cn(C2CC2)c2c([N+](=O)[O-])c(-c3ccc(F)cc3)c(F)cc2c1=O',
    p2='O=C(O)c1cn(C2CC2)c2c(F)c(-c3ccccc3)c(F)cc2c1=O',
    p3='Cc1ccc(-c2c(F)cc3c(=O)c(C(=O)O)cn(C4CC4)c3c2[N+](=O)[O-])cc1',
    n1='Cc1c(OC(=O)Cc2ccccc2)ccc2c(=O)c(Cl)coc12',
    n2='Cc1nccn1-c1nc(NCc2ccc(Cl)c(Cl)c2)c2cc([N+](=O)[O-])sc2n1',
)


def mol_img(s, w=720, h=430):
    m = Chem.MolFromSmiles(s)
    d = rdMolDraw2D.MolDraw2DCairo(w, h)
    o = d.drawOptions(); o.bondLineWidth = 3; o.minFontSize = 22; o.maxFontSize = 30
    rdMolDraw2D.PrepareAndDrawMolecule(d, m); d.FinishDrawing()
    return np.asarray(Image.open(io.BytesIO(d.GetDrawingText())).convert('RGB'))


# ---------- embedding path ----------
emb_te = np.load(B + 'reports/calib_data/apr18_cress_emb.npy').astype(np.float32)
qv = emb_te[12]
p1_emb = np.asarray(emb_rows(np.array([3263827]))[0], dtype=np.float32)
p1_emb = p1_emb / np.linalg.norm(p1_emb)
alphas = np.linspace(0, 1, 101)
path = []
for a in alphas:
    x = (1 - a) * qv + a * p1_emb
    path.append(x / np.linalg.norm(x))
p_path = model.predict_proba(np.asarray(path, dtype=np.float32))[:, 1]
cross = np.interp(0.5, p_path, alphas)

# ---------- figure ----------
fig = plt.figure(figsize=(18, 11.6))
gs = fig.add_gridspec(2, 2, height_ratios=[1.45, 1.0], width_ratios=[1.10, 1.0],
                      hspace=0.28, wspace=0.10)

# --- A local UMAP ---
ax = fig.add_subplot(gs[0, 0])
ax.scatter(P[L == 0, 0], P[L == 0, 1], s=3, c='#c9ccd4', alpha=0.35, lw=0, label='Training negatives')
ax.scatter(P[L == 1, 0], P[L == 1, 1], s=7, c='#ff8c00', alpha=0.65, lw=0,
           marker='^', label='Training positives')
cs = ax.contour(gx, gy, field, levels=[0.5], colors='#1a1a1a', linewidths=1.4, linestyles='--')
ax.clabel(cs, fmt={0.5: 'model p = 0.5'}, fontsize=9)
ax.add_patch(Circle(q, 0.5, fill=False, ls=':', lw=1.2, ec='#444', alpha=0.8))
ax.text(q[0] + 0.03, q[1] - 0.62, 'r = 0.5', fontsize=9, color='#444')

ax.scatter(*q, marker='*', s=780, c='#1462c7', edgecolors='white', linewidths=2.2, zorder=12)
ax.annotate('FN #12  p = 0.012', xy=q, xytext=(q[0] + 0.44, q[1] + 0.46),
            fontsize=11, color='#0b3d91', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#0b3d91', alpha=0.95),
            arrowprops=dict(arrowstyle='->', color='#0b3d91', lw=1.6))

marks = [
    (3263827, 'P1 (para-F)  p=0.990', '#2ca02c', (-3.48, 9.72)),
    (3257602, 'P2 (nearest)  p=0.998', '#2ca02c', (-2.10, 8.80)),
    (386196, 'N1  p=0.010', '#444444', (-3.47, 9.00)),
]
for idx, txt, col, pos in marks:
    x, yy = XY[idx]
    ax.scatter(x, yy, s=210, facecolors='none', edgecolors=col, linewidths=2.0, zorder=11)
    ax.annotate(txt, xy=(x, yy), xytext=pos, fontsize=9.5, color=col,
                bbox=dict(boxstyle='round,pad=0.22', fc='white', ec=col, alpha=0.9),
                arrowprops=dict(arrowstyle='-', color=col, lw=1.0))

ax.set_xlim(q[0] - R, q[0] + R); ax.set_ylim(q[1] - R, q[1] + R)
ax.set_aspect('equal')
ax.set_xlabel('UMAP-1'); ax.set_ylabel('UMAP-2')
ax.set_title('A · Local UMAP around FN #12 (training points within r = 0.78)', fontsize=12, loc='left')
ax.legend(loc='upper left', frameon=True, fontsize=9, markerscale=1.6)
ax.grid(alpha=0.15)

# --- B gallery ---
gsub = gs[0, 1].subgridspec(3, 2, hspace=0.52, wspace=0.06)
gallery = [
    ('fn', 'FN #12', 'p = 0.012  (misclassified)', '#0b3d91'),
    ('p1', 'P1 · idx 3263827', 'para-F analog · cosine 0.819 · p = 0.990', '#1e8449'),
    ('p2', 'P2 · idx 3257602', 'nearest positive · cosine 0.857 · p = 0.998', '#1e8449'),
    ('p3', 'P3 · idx 3210578', 'positive · cosine 0.816 · p = 0.987', '#1e8449'),
    ('n1', 'N1 · idx 386196', 'negative · cosine 0.840 · p = 0.010', '#555555'),
    ('n2', 'N2 · idx 2728727', 'nearest negative · cosine 0.861 · p = 0.000', '#555555'),
]
for k, (key, name, sub, col) in enumerate(gallery):
    a = fig.add_subplot(gsub[k // 2, k % 2])
    a.imshow(mol_img(smi[key]))
    a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values():
        s.set_edgecolor(col); s.set_linewidth(2.0)
    a.set_title(f'{name}\n{sub}', fontsize=9, color=col, loc='left', pad=3)

# --- C ablation ---
axb = fig.add_subplot(gs[1, 0])
ab_labels = ['Full spectrum (18 peaks)', '− 130.4 ppm', '− 129.9, 130.4, 130.6 ppm',
             '− 129.9, 130.4, 130.6, 128.7 ppm', '− 175.4 ppm']
ab_p = [0.0122, 0.3114, 0.9098, 0.8836, 0.0000]
ab_col = ['#c0392b' if v < 0.5 else '#1e8449' for v in ab_p]
yy = np.arange(len(ab_labels))[::-1]
axb.barh(yy, ab_p, color=ab_col, height=0.55)
axb.axvline(0.5, color='#555', ls='--', lw=1.4)
axb.text(0.505, yy[0] + 0.34, 'threshold 0.5', fontsize=9, color='#555')
for y0, v in zip(yy, ab_p):
    axb.text(v + 0.012, y0, f'{v:.3f}', va='center', fontsize=10.5, fontweight='bold')
axb.set_yticks(yy); axb.set_yticklabels(ab_labels, fontsize=9.5)
axb.set_xlim(0, 1.08); axb.set_xlabel('Predicted probability', fontsize=10)
axb.set_title('C · Leave-peaks-out: removing 129.9 / 130.4 / 130.6 ppm flips the FN', fontsize=11.5, loc='left')
axb.grid(axis='x', alpha=0.2)
for s in ['top', 'right']:
    axb.spines[s].set_visible(False)

# --- D interpolation ---
axc = fig.add_subplot(gs[1, 1])
axc.axhspan(0, 0.5, color='#d6e4f0', alpha=0.45)
axc.axhspan(0.5, 1.05, color='#e3f2d9', alpha=0.45)
axc.plot(alphas, p_path, color='#0b3d91', lw=2.6)
axc.axhline(0.5, color='#333', ls='--', lw=1.4)
axc.axvline(cross, color='#c0392b', ls=':', lw=1.6)
axc.scatter([0, 1], [p_path[0], p_path[-1]], c=['#1462c7', '#1e8449'], s=90, zorder=5)
axc.annotate(f'p = 0.5 at α ≈ {cross:.2f}', xy=(cross, 0.5), xytext=(cross + 0.14, 0.28),
             fontsize=10.5, color='#c0392b',
             arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.5))
axc.text(0.02, p_path[0] + 0.05, 'FN #12  p = 0.012', fontsize=9.5, color='#0b3d91')
axc.text(0.63, 0.90, 'P1  p = 0.990', fontsize=9.5, color='#1e8449')
axc.set_xlim(0, 1); axc.set_ylim(0, 1.05)
axc.set_xlabel('Interpolation α:  FN #12  →  para-F analog P1', fontsize=10)
axc.set_ylabel('Predicted probability', fontsize=10)
axc.set_title('D · The FN sits on the decision boundary', fontsize=11.5, loc='left')
axc.grid(alpha=0.2)

fig.suptitle('FN #12: a boundary case driven by the aromatic peaks of the unsubstituted phenyl ring',
             fontsize=15, fontweight='bold', y=0.985)
fig.savefig(R_OUT + 'fig_fn12_case.png', dpi=170, bbox_inches='tight')
print('saved', R_OUT + 'fig_fn12_case.png', 'crossing', cross)
