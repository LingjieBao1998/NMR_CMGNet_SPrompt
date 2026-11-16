#!/usr/bin/env python3
"""R2.3 -- N-heterocycle *system* breakdown of the external test set.

Each molecule is assigned to one heterocycle class by inspecting its rings
(RDKit ring info; aromatic/fused/exocyclic-carbonyl tests), with a priority that
reflects structural similarity to the quinolone pharmacophore:

  quinolone core > quinolinone-like > fused N-heteroarene > pyridine >
  diazine/triazine > azole > saturated N-ring > carbocycle > acyclic

For every class we report n, false positives at tau_op = 0.40 (Wilson 95% CI),
median / 95th-percentile score, and a representative molecule (most frequent
Murcko scaffold of that class, highest-scoring member).

Outputs
  reports/nhetero_systems.json
  reports/fig_nhetero_systems.pdf / .png
"""
import io
import json
import math
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw
from rdkit.Chem.Scaffolds import MurckoScaffold
from PIL import Image

B = "/mnt/data2/NMR_Core_Mol_Detection/"
sys.path.insert(0, B)
RDLogger.DisableLog("rdApp.*")

TAU = 0.50
QUERY = [Chem.MolFromSmiles("O=C1C(C(O)=O)=CNC2=CC=CC=C12"),
         Chem.MolFromSmiles("O=C1C(C(O)=O)=CNC2=NC=CC=C12")]
BLUE, ORANGE, GREY, GREEN, RED, LIGHT = ("#0072B2", "#E69F00", "#999999",
                                         "#009E73", "#D55E00", "#C9CCD4")
TEXT = "#262626"
ORDER = ["quinolone core", "quinolinone-like", "fused N-heteroarene",
         "pyridine", "diazine/triazine", "azole", "saturated N-ring",
         "carbocycle", "acyclic"]
COL = {"quinolone core": BLUE, "quinolinone-like": RED,
       "fused N-heteroarene": "#CC79A7", "pyridine": ORANGE,
       "diazine/triazine": "#56B4E9", "azole": GREEN,
       "saturated N-ring": "#8C6D31", "carbocycle": "#6A3D9A",
       "acyclic": LIGHT}
NICE = {"quinolone core": "Quinolone core", "quinolinone-like": "Quinolinone-like",
        "fused N-heteroarene": "Fused N-heteroarene",
        "pyridine": "Pyridine", "diazine/triazine": "Diazine / triazine",
        "azole": "Azole (5-ring)", "saturated N-ring": "Saturated N-ring",
        "carbocycle": "Carbocycle only", "acyclic": "Acyclic"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 7.5, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "axes.labelcolor": TEXT, "text.color": TEXT, "xtick.color": TEXT,
    "ytick.color": TEXT, "axes.linewidth": 0.7, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def rings(mol):
    return [list(r) for r in mol.GetRingInfo().AtomRings()]


def is_arom(mol, r):
    return all(mol.GetBondBetweenAtoms(r[i], r[(i + 1) % len(r)]).GetIsAromatic()
               for i in range(len(r)))


def fused(a, b):
    return len(set(a) & set(b)) >= 2


def ring_components(rings):
    n = len(rings)
    adj = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if fused(rings[i], rings[j]):
                adj[i].add(j)
                adj[j].add(i)
    comps = []
    seen = set()
    for i in range(n):
        if i in seen:
            continue
        stack, comp = [i], set()
        seen.add(i)
        while stack:
            k = stack.pop()
            comp.add(k)
            for j in adj[k]:
                if j not in seen:
                    seen.add(j)
                    stack.append(j)
        comps.append(comp)
    return comps


def class_ring_atoms(mol, cls):
    """Atoms of the ring system that defines the class (for highlighting)."""
    rings = [list(r) for r in mol.GetRingInfo().AtomRings()]
    if not rings:
        return set()
    arom = [is_arom(mol, r) for r in rings]
    nN = [sum(1 for i in r if mol.GetAtomWithIdx(i).GetSymbol() == "N")
          for r in rings]
    comps = ring_components(rings)
    comp_of = {i: c for c in comps for i in c}

    if cls == "quinolone core":
        for q in QUERY:
            match = mol.GetSubstructMatch(q)
            if match:
                return set(match)
    if cls == "quinolinone-like":
        for i, r in enumerate(rings):
            if len(r) != 6 or not arom[i] or nN[i] == 0:
                continue
            has_co = False
            for a_i in r:
                a = mol.GetAtomWithIdx(a_i)
                for b in a.GetBonds():
                    o = b.GetOtherAtom(a)
                    if (b.GetBondType() == Chem.BondType.DOUBLE and
                            o.GetSymbol() == "O" and not o.IsInRing()):
                        has_co = True
            if has_co:
                return set().union(*[set(rings[k]) for k in comp_of[i]])
    if cls == "fused N-heteroarene":
        for i, r in enumerate(rings):
            if arom[i] and nN[i] > 0 and len(comp_of[i]) > 1:
                return set().union(*[set(rings[k]) for k in comp_of[i]])
    if cls == "pyridine":
        for i, r in enumerate(rings):
            if len(r) == 6 and arom[i] and nN[i] == 1 and len(comp_of[i]) == 1:
                return set(r)
    if cls == "diazine/triazine":
        for i, r in enumerate(rings):
            if len(r) == 6 and arom[i] and nN[i] >= 2:
                return set(r)
    if cls == "azole":
        for i, r in enumerate(rings):
            if len(r) == 5 and arom[i] and nN[i] >= 1:
                return set(r)
    if cls == "saturated N-ring":
        for i, r in enumerate(rings):
            if not arom[i] and nN[i] >= 1:
                return set(r)
    if cls == "carbocycle":
        return set().union(*[set(r) for r in rings])
    return set()


def classify(mol):
    if mol is None:
        return "acyclic"
    if any(mol.HasSubstructMatch(q) for q in QUERY):
        return "quinolone core"
    rs = rings(mol)
    if not rs:
        return "acyclic"
    n_ring = [r for r in rs if any(mol.GetAtomWithIdx(i).GetSymbol() == "N"
                                   for i in r)]
    if not n_ring:
        return "carbocycle"
    # quinolinone-like: aromatic N 6-ring carrying an exocyclic C=O and fused
    # to a carbocyclic ring
    for r in n_ring:
        if len(r) != 6 or not is_arom(mol, r):
            continue
        has_co = False
        for i in r:
            a = mol.GetAtomWithIdx(i)
            for b in a.GetBonds():
                o = b.GetOtherAtom(a)
                if (b.GetBondType() == Chem.BondType.DOUBLE and
                        o.GetSymbol() == "O" and not o.IsInRing()):
                    has_co = True
        if not has_co:
            continue
        for r2 in rs:
            if r2 is r or not fused(r, r2):
                continue
            if all(mol.GetAtomWithIdx(i).GetSymbol() == "C" for i in r2):
                return "quinolinone-like"
    for r in n_ring:
        if is_arom(mol, r) and any(fused(r, r2) for r2 in rs if r2 is not r):
            return "fused N-heteroarene"
    for r in n_ring:
        if not is_arom(mol, r):
            continue
        nN = sum(1 for i in r if mol.GetAtomWithIdx(i).GetSymbol() == "N")
        if len(r) == 6:
            return "pyridine" if nN == 1 else "diazine/triazine"
        if len(r) == 5:
            return "azole"
    return "saturated N-ring"


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(max(0, ctr - half)), float(min(1, ctr + half))


def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def mol_image(smi, cls="", col=(0.95, 0.70, 0.15), w=760, h=470):
    m = Chem.MolFromSmiles(smi)
    atoms = class_ring_atoms(m, cls) if cls else set()
    from rdkit.Chem import rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit import Geometry
    rdDepictor.Compute2DCoords(m)
    d = rdMolDraw2D.MolDraw2DCairo(w, h)
    o = d.drawOptions()
    o.bondLineWidth = 4.5
    o.minFontSize = 34
    o.maxFontSize = 46
    o.padding = 0.12
    o.clearBackground = True
    fill = tuple(0.90 + 0.10 * x for x in col)   # light tint of the class colour
    if atoms:
        hi = {i: [col] for i in atoms}
        bonds = [b.GetIdx() for b in m.GetBonds()
                 if b.GetBeginAtomIdx() in atoms and b.GetEndAtomIdx() in atoms]
        hb = {b: [col] for b in bonds}
        rads = {i: 0.28 for i in atoms}
        wid = {b: 1 for b in bonds}
        # draw once to establish scale, fill the defined rings, then draw on top
        d.DrawMoleculeWithHighlights(m, "", hi, hb, rads, wid)
        d.ClearDrawing()
        conf = m.GetConformer()
        for r in m.GetRingInfo().AtomRings():
            if all(i in atoms for i in r):
                ps = [Geometry.Point2D(conf.GetAtomPosition(i)) for i in r]
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
    recs = json.load(open(B + "data/nmrmind_cmgnet_1004/"
                          "external_test_april18peak.json"))
    y = np.load(B + "viz_tmp/test_labels.npy").astype(int)
    p = np.load(B + "reports/calib_data/external_apr18peak.npz",
                allow_pickle=True)["p_raw"]

    rows = []
    for i, r in enumerate(recs):
        mol = Chem.MolFromSmiles(r["cano_smiles"])
        sc = MurckoScaffold.GetScaffoldForMol(mol) if mol is not None else None
        scaf = Chem.MolToSmiles(sc) if sc is not None and sc.GetNumAtoms() else ""
        rows.append(dict(idx=i, label=int(y[i]), p=float(p[i]),
                         smiles=r["cano_smiles"], mol=mol, cls=classify(mol),
                         scaffold=scaf))

    summary = {}
    for c in ORDER:
        sub = [r for r in rows if r["cls"] == c]
        if not sub:
            continue
        pp = np.array([r["p"] for r in sub])
        k = int((pp >= TAU).sum())
        lo, hi = wilson(k, len(sub))
        # representative: most frequent scaffold in the class, then top score
        from collections import Counter
        common = Counter(r["scaffold"] for r in sub).most_common(1)[0][0]
        cand = [r for r in sub if r["scaffold"] == common]
        rep = max(cand, key=lambda r: r["p"])
        summary[c] = {
            "n": len(sub), "fp_at_tau": k,
            "fp_rate": k / len(sub),
            "fp_rate_ci": [lo, hi],
            "median_p": float(np.median(pp)),
            "p95": float(np.quantile(pp, 0.95)),
            "max_p": float(pp.max()),
            "representative_smiles": rep["smiles"],
            "representative_scaffold": common,
            "representative_p": float(rep["p"]),
            "top_molecule": max(sub, key=lambda r: r["p"])["smiles"],
            "top_p": float(max(pp)),
        }

    json.dump({"tau": TAU, "classes": summary,
               "records": [{k: r[k] for k in ("idx", "label", "p", "smiles",
                                              "cls", "scaffold")}
                           for r in rows]},
              open(B + "reports/nhetero_systems.json", "w"), indent=1)

    # ------------------------------------------------------------- figure --
    fig = plt.figure(figsize=(7.10, 6.60))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.12, 1.33], hspace=0.42,
                             left=0.175, right=0.985, top=0.93, bottom=0.055)
    top = outer[0].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.62)
    axN = fig.add_subplot(top[0])
    axF = fig.add_subplot(top[1])

    plot_c = [c for c in ORDER if c in summary]
    yy = np.arange(len(plot_c))[::-1]
    for ypos, c in zip(yy, plot_c):
        s = summary[c]
        axN.barh(ypos, s["n"], height=0.62, color=COL[c], alpha=0.9, left=1)
        axN.text(s["n"] * 1.12, ypos, f"{s['n']:,}", va="center", fontsize=5.8)
    axN.set_yticks(yy, [NICE[c] for c in plot_c], fontsize=6.0)
    axN.set_xscale("log")
    axN.set_xlim(1, 6500)
    axN.set_xlabel("Number of external spectra (log scale)")
    axN.set_title("A  Composition by heterocycle system", loc="left",
                  fontweight="bold")
    axN.grid(True, axis="x", lw=0.35, color="#E5E5E5")

    for ypos, c in zip(yy, plot_c):
        s = summary[c]
        if c == "quinolone core":
            err = s["n"] - s["fp_at_tau"]      # missed positives (FN)
            rate = 100 * err / s["n"]
            lo, hi = wilson(err, s["n"])
            lab = f"{err}/{s['n']} FN"
        else:
            err = s["fp_at_tau"]
            rate = 100 * s["fp_rate"]
            lo, hi = s["fp_rate_ci"]
            lab = f"{err}/{s['n']}"
        lo, hi = 100 * lo, 100 * hi
        axF.plot([lo, hi], [ypos, ypos], color=COL[c], lw=1.2, zorder=2)
        axF.plot([rate], [ypos], "o", ms=4.6, color=COL[c], mec="white",
                 mew=0.5, zorder=3)
        axF.text(min(hi + 0.22, 9.55), ypos, lab, va="center", fontsize=5.6)
    axF.set_yticks(yy, [NICE[c] for c in plot_c], fontsize=5.8)
    axF.set_xlim(-0.10, 11.5)
    axF.set_xlabel("Error rate at $\\tau_{op}$ (%): FN (pos.) / FP (neg.)",
                   fontsize=7.0)
    axF.set_title("B  Error rate per system", loc="left",
                  fontweight="bold")
    axF.grid(True, axis="x", lw=0.35, color="#E5E5E5")
    axF.axvline(0, color="#BBBBBB", lw=0.6)

    # C: representative molecule for each of 8 systems
    gal = [c for c in ORDER if c in summary][:8]
    bottom = outer[1].subgridspec(2, 4, hspace=0.16, wspace=0.12)
    for k, c in enumerate(gal):
        ax = fig.add_subplot(bottom[k // 4, k % 4])
        s = summary[c]
        ax.imshow(mol_image(s["representative_smiles"], c, hex_rgb(COL[c])))
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#DDDDDD")
        lab = NICE[c]
        stat = (f"n={s['n']:,}   detected={s['fp_at_tau']}"
                if c == "quinolone core" else
                f"n={s['n']:,}   FP={s['fp_at_tau']}")
        ax.set_title(f"{lab}\n{stat}", fontsize=6.2,
                     color=COL[c] if c != "acyclic" else TEXT, pad=2)
        if k == 0:
            ax.text(-0.06, 1.42, "C  Representative system per class",
                    transform=ax.transAxes, fontsize=8.5, fontweight="bold",
                    color=TEXT, ha="left", va="bottom")

    for ext in ("pdf", "png"):
        fig.savefig(B + f"reports/fig_nhetero_systems.{ext}",
                    dpi=600 if ext == "png" else None)
    print(json.dumps({k: {kk: vv for kk, vv in v.items()
                          if kk != "representative_smiles"}
                      for k, v in summary.items()}, indent=1))


if __name__ == "__main__":
    main()
