"""Paper Figure: global embedding space + FN #12 case.

Layout (single Figure):
  Row 1:  A  global CReSS embedding space (training + test FP/FN) with ROI around FN #12
  Row 2:  B  local UMAP around FN #12
          C  neighbouring molecules (positives / negatives)
          D  decision boundary along the embedding path FN #12 -> para-F analog

The figure is laid out at final print width (180 mm); font sizes are real print
sizes. Exports 600 dpi PNG + PDF.
"""
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from PIL import Image
from scipy import ndimage
from rdkit import Chem, RDLogger
from rdkit.Chem.Draw import rdMolDraw2D
import joblib

RDLogger.DisableLog('rdApp.*')
import json
import os as _os
B = _os.environ.get('NMR_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
B = B.rstrip('/') + '/'
R_OUT = B + 'reports/'
NV = 3_298_042
ERR = [763, 3460, 3898, 4201, 12]
_z = np.load(B + 'reports/fig_xgb_pr_roc_curves.npz')
_m = json.load(open(B + 'reports/xgb_pr_roc_metrics.json'))
_cal = json.load(open(B + 'reports/calibration_review_metrics.json'))
_nh = json.load(open(B + 'reports/nhetero_systems.json'))

def _wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, ctr - half), min(1.0, ctr + half)

# ---------------------------------------------------------------- data ----
y_tr = np.load(B + 'viz_tmp/train_labels.npy')
C_fit = np.load(B + 'viz_tmp/coords_fit.npy')
fit_idx = np.load(B + 'viz_tmp/fit_idx.npy')
C_rest = np.load(B + 'viz_tmp/coords_rest.npy')
rest_idx = np.load(B + 'viz_tmp/rest_idx.npy')
C_te = C_fit[len(fit_idx):]
q = C_te[12].astype(float)

Cf_tr = C_fit[:len(fit_idx)]
fit_valid = fit_idx < NV
fit_lab = y_tr[fit_idx]
rest_valid = rest_idx < NV
rest_lab = y_tr[rest_idx]
neg_xy = np.concatenate([Cf_tr[fit_valid & (fit_lab == 0)],
                         C_rest[rest_valid & (rest_lab == 0)]], axis=0)
pos_xy = np.concatenate([Cf_tr[fit_valid & (fit_lab == 1)],
                         C_rest[rest_valid & (rest_lab == 1)]], axis=0)

# global axis ranges (must match embedding_umap_full.html)
x_all = np.concatenate([neg_xy[:, 0], pos_xy[:, 0], C_te[:, 0]])
y_all = np.concatenate([neg_xy[:, 1], pos_xy[:, 1], C_te[:, 1]])
x0g, x1g = float(x_all.min()), float(x_all.max())
y0g = 2.2
y_img_top = max(C_te[i][1] + 0.85 + 1.72 for i in ERR)
y1g = max(float(y_all.max()), y_img_top)
pxg, pyg = 0.01 * (x1g - x0g), 0.01 * (y1g - y0g)

# ------------------------------------------------------- local UMAP ----
XY = np.empty((NV, 2), dtype=np.float32)
m1 = fit_idx < NV
m2 = rest_idx < NV
XY[fit_idx[m1]] = Cf_tr[m1]
XY[rest_idx[m2]] = C_rest[m2]

R = 0.78
loc = np.where(np.linalg.norm(XY - q, axis=1) < R)[0]
P = XY[loc]
L = y_tr[loc]

_Z = np.load(B + 'viz_tmp/train_emb_subset.npz')
_SUB_IDX, _SUB_EMB = _Z['idx'], _Z['emb']
def emb_rows(ids):
    ids = np.asarray(ids)
    pos = np.clip(np.searchsorted(_SUB_IDX, ids), 0, len(_SUB_IDX) - 1)
    return _SUB_EMB[pos].astype(np.float32)
model = joblib.load(B + 'models/xgboost/model_xgb_cress_ex_train_seed42.pkl')

# ------------------------------------------------------------ molecules ----
smi = dict(
    fn='O=C(O)c1cn(C2CC2)c2c([N+](=O)[O-])c(-c3ccccc3)c(F)cc2c1=O',
    p1='O=C(O)c1cn(C2CC2)c2c([N+](=O)[O-])c(-c3ccc(F)cc3)c(F)cc2c1=O',
    p2='O=C(O)c1cn(C2CC2)c2c(F)c(-c3ccccc3)c(F)cc2c1=O',
    p3='Cc1ccc(-c2c(F)cc3c(=O)c(C(=O)O)cn(C4CC4)c3c2[N+](=O)[O-])cc1',
    n1='Cc1c(OC(=O)Cc2ccccc2)ccc2c(=O)c(Cl)coc12',
    n2='Cc1nccn1-c1nc(NCc2ccc(Cl)c(Cl)c2)c2cc([N+](=O)[O-])sc2n1',
)


def mol_img(s, w=900, h=540):
    m = Chem.MolFromSmiles(s)
    d = rdMolDraw2D.MolDraw2DCairo(w, h)
    o = d.drawOptions()
    o.bondLineWidth = 3
    o.minFontSize = 22
    o.maxFontSize = 30
    rdMolDraw2D.PrepareAndDrawMolecule(d, m)
    d.FinishDrawing()
    return np.asarray(Image.open(io.BytesIO(d.GetDrawingText())).convert('RGB'))


# -------------------------------------------------- embedding path ----
emb_te = np.load(B + 'reports/calib_data/apr18_cress_emb.npy').astype(np.float32)
qv = emb_te[12]
p1_emb = np.asarray(emb_rows(np.array([3263827]))[0], dtype=np.float32)
p1_emb = p1_emb / np.linalg.norm(p1_emb)
alphas = np.linspace(0, 1, 101)
path = []
for a in alphas:
    vec = (1 - a) * qv + a * p1_emb
    path.append(vec / np.linalg.norm(vec))
p_path = model.predict_proba(np.asarray(path, dtype=np.float32))[:, 1]
cross = float(np.interp(0.5, p_path, alphas))

# -------------------------------------------------- panel A image ----
gimg = np.asarray(Image.open(B + 'reports/assets/global_2x.png').convert('RGB'))
SC = 2
Lm, Rm, Tm, Bm = 80 * SC, 40 * SC, 60 * SC, 40 * SC
Wp, Hp = gimg.shape[1], gimg.shape[0]

# locate the FN star directly in the screenshot (plotly blue #0072B2);
# use a density peak (not the centroid) so the annotation arrow line cannot bias it
sb = np.abs(gimg.astype(int) - np.array([0, 114, 178])).sum(axis=2)
blue = (sb < 60).astype(float)
ys_s, xs_s = np.where(blue > 0)
sel = (xs_s < 0.5 * Wp) & (ys_s > 0.3 * Hp) & (ys_s < 0.8 * Hp)
if sel.sum() > 20:
    m = np.zeros_like(blue)
    m[ys_s[sel], xs_s[sel]] = 1.0
    dens = ndimage.uniform_filter(m, size=21)
    PY, PX = np.unravel_index(np.argmax(dens), dens.shape)
    PX, PY = float(PX), float(PY)
else:  # fallback: map from data coordinates
    fx = (q[0] - (x0g - pxg)) / ((x1g + pxg) - (x0g - pxg))
    fy = ((y1g + pyg) - q[1]) / ((y1g + pyg) - (y0g - pyg))
    PX = Lm + fx * (Wp - Lm - Rm)
    PY = Tm + fy * (Hp - Tm - Bm)

# crop the white border so panel A uses the space efficiently
mask = (gimg < 250).any(axis=2)
ys_, xs_ = np.where(mask)
pad = 20
x_off = max(int(xs_.min()) - pad, 0)
y_off = max(int(ys_.min()) - pad, 0)
x_end = min(int(xs_.max()) + pad, Wp)
y_end = min(int(ys_.max()) + pad, Hp)
gimg = gimg[y_off:y_end, x_off:x_end]
PX -= x_off
PY -= y_off

# ROI extent: same radius as the local map in panel B (data units)
XS = (x1g + pxg) - (x0g - pxg)
YS = (y1g + pyg) - (y0g - pyg)
sx = (Wp - Lm - Rm) / XS
sy = (Hp - Tm - Bm) / YS
rx, ry = R * sx, R * sy

AR = gimg.shape[1] / gimg.shape[0]          # width / height of panel-A raster

# ------------------------------------------------------------ figure ----
"""Row 1: A (full width).  Row 2: B | C | D.
All positions are derived from inches so the geometry is exact."""
FW = 7.10                                   # 180 mm
ML, MR = 0.14, 0.12                         # side margins
CW = FW - ML - MR                           # content width 6.84 in

AW = CW                                     # panel A spans the row
AH = AW / AR                                # its raster aspect

BW = 2.26                                   # B is square == row-2 height
ROW2_H = BW                                 # row-2 content height
GC_W = 2.30                                 # C gallery width
GAPX = 0.16                                 # column gap
DD_W = CW - BW - GC_W - 2 * GAPX            # D width

T_TOP = 0.10                                # top margin
TITLE_H = 0.17                              # title text height
GAP1 = 0.14                                 # vertical gap between rows
GAP2 = 0.03                                 # title-to-content gap
M_BOT = 0.46                                # bottom margin (B ticks + label)

row2_top = T_TOP + TITLE_H + GAP2 + AH + GAP1
GAP3 = 0.30
ROW3_H = 2.05
GAP4 = 0.44
ROW4_H = 2.32
row3_top = row2_top + TITLE_H + GAP2 + ROW2_H + GAP3
row4_top = row3_top + TITLE_H + GAP2 + ROW3_H + GAP4
FH = row4_top + TITLE_H + GAP2 + ROW4_H + 0.58

plt.rcParams.update({'font.size': 7, 'axes.linewidth': 0.6,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                     'figure.dpi': 600})
fig = plt.figure(figsize=(FW, FH))


def axes_in(x, y_top, w, h):
    """add_axes with top-left inch coordinates"""
    return fig.add_axes([x / FW, (FH - y_top - h) / FH, w / FW, h / FH])


# circular callout on the raster: mean of the two data-scale radii
rC = 0.5 * (rx + ry)

# ================= A: global embedding space with ROI =================
axA = axes_in(ML, T_TOP + TITLE_H + GAP2, AW, AH)
axA.imshow(gimg)
axA.set_xticks([])
axA.set_yticks([])
for s in axA.spines.values():
    s.set_visible(False)
axA.add_patch(Ellipse((PX, PY), 2 * rC, 2 * rC, fill=False,
                      ec='#0072B2', lw=1.4, ls=(0, (5, 3)), zorder=20))
axA.annotate('FN #12 → B', xy=(PX + 0.62 * rC, PY + 0.55 * rC),
             xytext=(PX + rC * 1.02 + 55, PY + rC * 0.62 + 90),
             fontsize=7.5, color='#0072B2', fontweight='bold', zorder=21,
             arrowprops=dict(arrowstyle='-|>', color='#0072B2', lw=0.9,
                             shrinkA=2, shrinkB=2),
             bbox=dict(boxstyle='round,pad=0.22', fc='white', ec='#0072B2',
                       lw=0.7, alpha=0.95))
fig.text(ML / FW, (FH - T_TOP - TITLE_H) / FH, 'A. Embedding Space',
         fontsize=9, va='bottom')

# ================= B: local UMAP =================
axB = axes_in(ML, row2_top + TITLE_H + GAP2, BW, BW)
axB.scatter(P[L == 0, 0], P[L == 0, 1], s=0.6, c='#c9ccd4', alpha=0.40, lw=0,
            rasterized=True)
axB.scatter(P[L == 1, 0], P[L == 1, 1], s=2.6, c='#E69F00', alpha=0.65, lw=0,
            marker='^', rasterized=True)

axB.scatter(*q, marker='*', s=170, c='#0072B2', edgecolors='white',
            linewidths=0.9, zorder=12)

# circle the neighbours and tag them directly (molecules shown in panel C)
NEIGH = [(3263827, 'P1', '#009E73', (-3.12, 9.47), 'left', 'bottom'),
         (3257602, 'P2', '#009E73', (-3.32, 9.03), 'left', 'bottom'),
         (386196, 'N1', '#444444', (-2.83, 9.25), 'left', 'top')]
for idx, tag, col, (tx, ty), ha, va in NEIGH:
    x, yy = XY[idx]
    axB.scatter(x, yy, s=50, facecolors='none', edgecolors=col,
                linewidths=1.1, zorder=11)
    axB.text(tx, ty, tag, fontsize=6.2, color=col, fontweight='bold',
             ha=ha, va=va, zorder=13,
             bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none',
                       alpha=0.85))

# label the FN to the left of the star
axB.annotate('FN #12\np = 0.012', xy=q, xytext=(q[0] - 0.66, q[1] + 0.24),
             fontsize=7, color='#0072B2', fontweight='bold', ha='left', va='bottom',
             bbox=dict(boxstyle='round,pad=0.22', fc='white', ec='#0072B2',
                       lw=0.6, alpha=0.95), zorder=13,
             arrowprops=dict(arrowstyle='-|>', color='#0072B2', lw=0.8,
                             shrinkA=2, shrinkB=3))

axB.set_xlim(q[0] - R, q[0] + R)
axB.set_ylim(q[1] - R, q[1] + R)
axB.set_aspect('equal')
axB.set_xlabel('UMAP-1', fontsize=7, labelpad=2)
axB.set_ylabel('UMAP-2', fontsize=7, labelpad=2)
axB.tick_params(labelsize=6, length=2.5, pad=2)
fig.text(ML / FW, (FH - row2_top - TITLE_H) / FH,
         'B. FN #12 neighbourhood', fontsize=9, va='bottom')

# ================= C: neighbouring structures =================
C_LEFT = ML + BW + GAPX
D_LEFT = C_LEFT + GC_W + GAPX
fig.text(C_LEFT / FW, (FH - row2_top - TITLE_H) / FH,
         'C. Neighbouring molecules', fontsize=9, va='bottom')
gallery = [
    ('fn', 'FN #12', 'p = 0.012  (misclassified)', '#0072B2'),
    ('p1', 'P1 · para-F analogue', 'cos 0.819 · p 0.990', '#009E73'),
    ('p2', 'P2 · nearest positive', 'cos 0.857 · p 0.998', '#009E73'),
    ('n1', 'N1 · negative', 'cos 0.840 · p 0.010', '#555555'),
]
GC_TOP = row2_top + TITLE_H + GAP2
cw = GC_W / 2
ch = ROW2_H / 2
IMG_H = min(ch - 0.22, (cw - 0.11) * 540 / 900)   # fit image exactly (900x540)
for k, (key, name, sub, col) in enumerate(gallery):
    cx0 = C_LEFT + (k % 2) * cw
    cy0 = GC_TOP + (k // 2) * ch
    a = axes_in(cx0 + 0.055, cy0 + ch - 0.05 - IMG_H, cw - 0.11, IMG_H)
    a.imshow(mol_img(smi[key]))
    a.set_xticks([])
    a.set_yticks([])
    for s in a.spines.values():
        s.set_edgecolor(col)
        s.set_linewidth(0.8)
    a.set_title(f'{name}\n{sub}', fontsize=5.6, color=col, loc='left', pad=2.0,
                linespacing=1.2)

# ================= D: decision boundary =================
axD = axes_in(D_LEFT, GC_TOP, DD_W, ROW2_H)
axD.plot(alphas, p_path, color='#0072B2', lw=1.5)
axD.axhline(0.5, color='#333', ls='--', lw=0.8)
axD.axvline(cross, color='#D55E00', ls=':', lw=0.9)
axD.scatter([0, 1], [p_path[0], p_path[-1]], c=['#0072B2', '#009E73'],
            s=24, zorder=5, edgecolors='white', linewidths=0.5)
axD.annotate(f'p = 0.5 at α ≈ {cross:.2f}', xy=(cross, 0.5),
             xytext=(cross + 0.10, 0.28), fontsize=6, color='#D55E00',
             arrowprops=dict(arrowstyle='-|>', color='#D55E00', lw=0.8))
axD.text(0.02, p_path[0] + 0.05, 'FN #12  p = 0.012', fontsize=5.8, color='#0072B2',
         bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.2), zorder=6)
axD.text(0.55, 0.90, 'P1  p = 0.990', fontsize=5.8, color='#009E73',
         bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.2), zorder=6)
axD.set_xlim(0, 1)
axD.set_ylim(0, 1.05)
axD.set_xlabel('Interpolation α:  FN #12  →  P1', fontsize=6.5, labelpad=2)
axD.set_ylabel('')                      # no side label: would collide with C
axD.text(0.02, 0.985, 'Predicted probability', fontsize=5.6, color='#333',
         ha='left', va='top')
axD.tick_params(labelsize=6, length=2.5, pad=2)
fig.text(D_LEFT / FW, (FH - row2_top - TITLE_H) / FH,
         'D. Interpolation towards P1', fontsize=9, va='bottom')
axD.grid(alpha=0.18, lw=0.4)


# ================= Row 3: N-heterocycle error rates (G) =================
Y3 = row3_top + TITLE_H + GAP2
fig.text(ML / FW, (FH - row3_top - TITLE_H) / FH,
         'E. N-heterocycle error rates', fontsize=9, va='bottom')
axG = axes_in(ML, Y3, CW, ROW3_H)
_order = ['quinolone core', 'quinolinone-like', 'fused N-heteroarene',
          'pyridine', 'diazine/triazine', 'azole', 'saturated N-ring',
          'carbocycle', 'acyclic']
_short = {'quinolone core': 'Quinolone', 'quinolinone-like': 'Quinolinone',
          'fused N-heteroarene': 'Fused N-het.', 'pyridine': 'Pyridine',
          'diazine/triazine': 'Diazine', 'azole': 'Azole',
          'saturated N-ring': 'Sat. N-ring', 'carbocycle': 'Carbocycle',
          'acyclic': 'Acyclic'}
_yy = np.arange(len(_order))[::-1]
for yv, c in zip(_yy, _order):
    ss = _nh['classes'][c]
    err = (ss['n'] - ss['fp_at_tau']) if c == 'quinolone core' else ss['fp_at_tau']
    rate = 100 * err / ss['n']
    lo, hi = _wilson(err, ss['n'])
    lo, hi = 100 * lo, 100 * hi
    col = '#0072B2' if c == 'quinolone core' else (
          '#D55E00' if c == 'quinolinone-like' else '#8C8C8C')
    axG.plot([lo, hi], [yv, yv], color=col, lw=1.1, zorder=2)
    for xx in (lo, hi):
        axG.plot([xx, xx], [yv - 0.10, yv + 0.10], color=col, lw=0.8, zorder=2)
    axG.plot([rate], [yv], 'o', ms=4.2, color=col, mec='white', mew=0.4,
             zorder=3)
    axG.text(2.12, yv, f"{err}/{ss['n']:,}", va='center', fontsize=5.6,
             color='#333')
axG.set_yticks(_yy, [_short[c] for c in _order], fontsize=6.0)
axG.get_yticklabels()[0].set_color('#0072B2')
axG.get_yticklabels()[0].set_fontweight('bold')
axG.get_yticklabels()[1].set_color('#D55E00')
axG.get_yticklabels()[1].set_fontweight('bold')
axG.set_xlim(-0.05, 2.65); axG.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
axG.tick_params(labelsize=6.0, length=2.4, pad=1.8)
axG.set_xlabel('Error rate (%) at $\\tau$ = 0.50   [FN for positives, FP for negatives]',
               fontsize=6.8, labelpad=2)
axG.grid(alpha=0.18, lw=0.4, axis='x')

# ================= Row 4: representative heterocycles (H) =================
Y4 = row4_top + TITLE_H + GAP2
fig.text(ML / FW, (FH - row4_top - TITLE_H) / FH,
         'F. Representative N-heterocycles and other ring systems', fontsize=9,
         va='bottom')
axH = axes_in(ML, Y4, CW, ROW4_H)
axH.axis('off')
_gal = [('quinolone core', 'Quinolone', '#0072B2'),
        ('quinolinone-like', 'Quinolinone-like', '#D55E00'),
        ('fused N-heteroarene', 'Fused N-het.', '#CC79A7'),
        ('pyridine', 'Pyridine', '#E69F00'),
        ('diazine/triazine', 'Diazine/triazine', '#56B4E9'),
        ('azole', 'Azole', '#009E73'),
        ('saturated N-ring', 'Saturated N-ring', '#8C6D31'),
        ('carbocycle', 'Carbocycle', '#6A3D9A')]
for k, (c, lab, col) in enumerate(_gal):
    ss = _nh['classes'][c]
    x0 = 0.005 + (k % 4) * 0.248
    y0 = 0.525 if k < 4 else 0.015
    iax = axH.inset_axes([x0, y0, 0.238, 0.385])
    iax.imshow(mol_img(ss['representative_smiles']))
    iax.set_xticks([]); iax.set_yticks([])
    for sp in iax.spines.values():
        sp.set_edgecolor(col); sp.set_linewidth(0.8)
    sub = (f"detected {ss['fp_at_tau']}/{ss['n']}" if c == 'quinolone core'
           else f"FP {ss['fp_at_tau']}/{ss['n']}")
    iax.set_title(f'{lab}\n{sub}', fontsize=5.6, color=col, pad=1.2,
                  linespacing=1.15)

fig.savefig(R_OUT + 'fig_model_analysis_v3.png', dpi=600, bbox_inches='tight',
            facecolor='white', pad_inches=0.06)
fig.savefig(R_OUT + 'fig_model_analysis_v3.pdf', bbox_inches='tight',
            facecolor='white', pad_inches=0.06)
print('saved', R_OUT + 'fig_model_analysis_v3.png/.pdf', 'crossing', cross)
