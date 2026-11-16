#!/usr/bin/env python3
"""Simple, publication-style figure (tau = 0.50): (A) accuracy bars per ring
system, (B) representative structures. Reads reports/nhetero_systems.json."""
import io
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from PIL import Image

import os as _os
B = _os.environ.get('NMR_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
B = B.rstrip('/') + '/'
RDLogger.DisableLog("rdApp.*")
r = json.load(open(B + "reports/nhetero_systems.json"))
S = r["classes"]

ORDER = ["quinolone core", "quinolinone-like", "fused N-heteroarene",
         "pyridine", "diazine/triazine", "azole", "saturated N-ring",
         "carbocycle", "acyclic"]
NICE = {"quinolone core": "Quinolone core", "quinolinone-like": "Quinolinone-like",
        "fused N-heteroarene": "Fused N-heteroarene",
        "pyridine": "Pyridine", "diazine/triazine": "Diazine / triazine",
        "azole": "Azole (5-ring)", "saturated N-ring": "Saturated N-ring",
        "carbocycle": "Carbocycle only", "acyclic": "Acyclic"}
COL = {"quinolone core": "#0072B2", "quinolinone-like": "#D55E00",
       "fused N-heteroarene": "#CC79A7", "pyridine": "#E69F00",
       "diazine/triazine": "#56B4E9", "azole": "#009E73",
       "saturated N-ring": "#8C6D31", "carbocycle": "#6A3D9A",
       "acyclic": "#9E9E9E"}
TEXT = "#262626"
GREY = "#777777"
BASE = "#BFD0DB"       # muted bars
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 7.5, "axes.labelsize": 8, "axes.titlesize": 8.8,
    "axes.labelcolor": TEXT, "text.color": TEXT, "xtick.color": TEXT,
    "ytick.color": TEXT, "axes.linewidth": 0.7, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, ctr - half), min(1.0, ctr + half)


def mol_image(smi, cls, w=960, h=600):
    from rdkit.Chem import rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit import Geometry
    m = Chem.MolFromSmiles(smi)
    sc = MurckoScaffold.GetScaffoldForMol(m)
    smatch = m.GetSubstructMatch(sc) if sc.GetNumAtoms() else ()
    atoms = set(smatch)
    rdDepictor.Compute2DCoords(m)
    d = rdMolDraw2D.MolDraw2DCairo(w, h)
    o = d.drawOptions()
    o.bondLineWidth = 4.5
    o.minFontSize = 38
    o.maxFontSize = 50
    o.padding = 0.04
    o.clearBackground = True
    col = hex_rgb(COL[cls])
    fill = tuple(0.88 + 0.12 * x for x in col)
    hi = {i: [col] for i in atoms}
    bonds = [b.GetIdx() for b in m.GetBonds()
             if b.GetBeginAtomIdx() in atoms and b.GetEndAtomIdx() in atoms]
    hb = {b: [col] for b in bonds}
    rads = {i: 0.22 for i in atoms}
    wid = {b: 1 for b in bonds}
    if atoms:
        d.DrawMoleculeWithHighlights(m, "", hi, hb, rads, wid)
        d.ClearDrawing()
        conf = m.GetConformer()
        for ring in m.GetRingInfo().AtomRings():
            if all(i in atoms for i in ring):
                ps = [Geometry.Point2D(conf.GetAtomPosition(i)) for i in ring]
                d.SetFillPolys(True)
                d.SetColour(fill)
                d.DrawPolygon(ps)
        d.drawOptions().clearBackground = False
        d.DrawMoleculeWithHighlights(m, "", hi, hb, rads, wid)
    else:
        rdMolDraw2D.PrepareAndDrawMolecule(d, m)
    d.FinishDrawing()
    return np.asarray(Image.open(io.BytesIO(d.GetDrawingText())).convert("RGB"))


def main():
    fig = plt.figure(figsize=(7.10, 4.85))
    fig.patch.set_facecolor("white")
    gs = GridSpec(2, 1, height_ratios=[1.0, 1.30], hspace=0.38,
                  left=0.175, right=0.975, top=0.935, bottom=0.045)
    axA = fig.add_subplot(gs[0])

    yy = np.arange(len(ORDER))[::-1]
    for y, c in zip(yy, ORDER):
        s = S[c]
        err = (s["n"] - s["fp_at_tau"]) if c == "quinolone core" else s["fp_at_tau"]
        acc = 100.0 * (1 - err / s["n"])
        lo_e, hi_e = wilson(err, s["n"])
        lo, hi = 100 * (1 - hi_e), 100 * (1 - lo_e)
        color = COL[c] if c in ("quinolone core", "quinolinone-like") else BASE
        axA.barh(y, acc - 90, left=90, height=0.56, color=color, zorder=2)
        axA.errorbar(acc, y, xerr=[[acc - lo], [hi - acc]], fmt="none",
                     ecolor="#333333", elinewidth=0.9, capsize=2.0,
                     capthick=0.8, zorder=3)
        axA.text(100.15, y, f"{acc:.2f}%", va="center", ha="left",
                 fontsize=5.9, color=TEXT)
        axA.text(101.15, y, f"({err}/{s['n']:,})", va="center", ha="left",
                 fontsize=5.6, color=GREY)
    axA.set_yticks(yy, [NICE[c] for c in ORDER], fontsize=6.3)
    axA.get_yticklabels()[0].set_fontweight("bold")
    axA.get_yticklabels()[0].set_color(COL["quinolone core"])
    axA.get_yticklabels()[1].set_fontweight("bold")
    axA.get_yticklabels()[1].set_color(COL["quinolinone-like"])
    axA.set_xlim(90, 103.1)
    axA.set_xticks([90, 92, 94, 96, 98, 100],
                   ["90", "92", "94", "96", "98", "100"], fontsize=6.1)
    axA.set_xlabel("Accuracy (%) at $\\tau$ = 0.50   [axis starts at 90%]",
                   fontsize=7.5)
    axA.set_title("A  Accuracy per ring system", loc="left", fontweight="bold",
                  fontsize=8.8, pad=5)
    axA.axvline(100, color="#CCCCCC", lw=0.7, ls=":", zorder=1)
    axA.grid(True, axis="x", lw=0.3, color="#EFEFEF", zorder=0)
    for sp in ("top", "right"):
        axA.spines[sp].set_visible(False)

    # ------------------------------------------------------------ panel B --
    inner = gs[1].subgridspec(2, 4, hspace=0.42, wspace=0.10)
    ax_first = None
    for k, c in enumerate(ORDER[:8]):
        ax = fig.add_subplot(inner[k // 4, k % 4])
        if ax_first is None:
            ax_first = ax
        s = S[c]
        ax.imshow(mol_image(s["representative_smiles"], c))
        ax.axis("off")
        stat = (f"detected {s['fp_at_tau']}/{s['n']:,}"
                if c == "quinolone core" else
                f"FP {s['fp_at_tau']}/{s['n']:,}")
        ax.set_title(f"{NICE[c]}\n{stat}", fontsize=6.2, color=TEXT,
                     linespacing=1.35, pad=2)

    pos = ax_first.get_position()
    fig.text(pos.x0, pos.y1 + 0.045, "B", fontsize=8.8,
             fontweight="bold", color=TEXT, ha="left")

    for ext in ("pdf", "png"):
        fig.savefig(B + f"reports/fig_nhetero_systems.{ext}",
                    dpi=600 if ext == "png" else None)
    print("saved classic figure")


if __name__ == "__main__":
    main()
