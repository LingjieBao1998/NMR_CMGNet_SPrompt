#!/usr/bin/env python3
"""R2.3 -- map of ring systems in the external test set.

Each external molecule is reduced to its ring-only Bemis-Murcko scaffold; equal
scaffolds are grouped. Ring systems are embedded with the CReSS SMILES encoder
(cosine space, no fingerprints) and projected to 2D with UMAP. Points are sized
by group frequency and coloured by the highest XGBoost score observed in the
group, so the quinolone-like "risk island" can be seen against ordinary
carbocycles and heterocycles. Representative ring systems are drawn at their
coordinates.

Outputs
  reports/ring_system_map.json
  reports/fig_ring_system_map.pdf / .png
"""
import io
import json
import sys
from collections import Counter, defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.colors import LogNorm
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw
from rdkit.Chem.Scaffolds import MurckoScaffold
from PIL import Image

B = "/mnt/data2/NMR_Core_Mol_Detection/"
sys.path.insert(0, B)
RDLogger.DisableLog("rdApp.*")
from infer import ModelInference  # noqa: E402

TAU = 0.50
QUERY = [Chem.MolFromSmiles("O=C1C(C(O)=O)=CNC2=CC=CC=C12"),
         Chem.MolFromSmiles("O=C1C(C(O)=O)=CNC2=NC=CC=C12")]
TEXT = "#262626"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 7.5, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "axes.labelcolor": TEXT, "text.color": TEXT, "xtick.color": TEXT,
    "ytick.color": TEXT, "axes.linewidth": 0.7, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def ring_n(mol):
    return sum(1 for r in mol.GetRingInfo().AtomRings()
               if any(mol.GetAtomWithIdx(i).GetSymbol() == "N" for i in r))


def has_ring_hetero(mol):
    return any(a.GetSymbol() in ("N", "O", "S") and a.IsInRing()
               for a in mol.GetAtoms())


def mol_image(smi, w=420, h=280):
    m = Chem.MolFromSmiles(smi)
    d = Draw.MolDraw2DCairo(w, h)
    o = d.drawOptions()
    o.bondLineWidth = 3.0
    o.minFontSize = 16
    o.maxFontSize = 22
    o.clearBackground = True
    Draw.PrepareAndDrawMolecule(d, m)
    d.FinishDrawing()
    return np.asarray(Image.open(io.BytesIO(d.GetDrawingText())).convert("RGB"))


def main():
    recs = json.load(open(B + "data/nmrmind_cmgnet_1004/"
                          "external_test_april18peak.json"))
    y = np.load(B + "viz_tmp/test_labels.npy").astype(int)
    p = np.load(B + "reports/calib_data/external_apr18peak.npz",
                allow_pickle=True)["p_raw"]

    groups = defaultdict(lambda: {"n": 0, "n_pos": 0, "max_p": 0.0,
                                  "molecules": []})
    for i, r in enumerate(recs):
        mol = Chem.MolFromSmiles(r["cano_smiles"])
        sc = MurckoScaffold.GetScaffoldForMol(mol)
        if sc.GetNumAtoms() == 0:
            continue
        s = Chem.MolToSmiles(sc)
        g = groups[s]
        g["n"] += 1
        g["n_pos"] += int(y[i])
        g["max_p"] = max(g["max_p"], float(p[i]))
        g["molecules"].append(r["cano_smiles"])

    keep = [s for s, g in groups.items()
            if g["n"] >= 3 or g["max_p"] >= TAU or g["n_pos"] > 0]
    keep = sorted(keep, key=lambda s: -groups[s]["n"])
    print(f"groups: {len(groups)} total, {len(keep)} shown", flush=True)

    mi = ModelInference(B + "models/2_5_w_model/8.json",
                        B + "models/2_5_w_model/8.pth", "cuda")
    E = []
    for i in range(0, len(keep), 128):
        e = mi.smiles_encode(keep[i:i + 128])
        E.append(e.detach().cpu().numpy())
    E = np.concatenate(E, 0)
    E = E / np.linalg.norm(E, axis=1, keepdims=True)

    import umap
    XY = umap.UMAP(n_neighbors=15, min_dist=0.25, metric="cosine",
                   random_state=42).fit_transform(E)

    rows = []
    for k, s in enumerate(keep):
        mol = Chem.MolFromSmiles(s)
        g = groups[s]
        rows.append(dict(scaffold=s, n=g["n"], n_pos=g["n_pos"],
                         max_p=g["max_p"],
                         n_ring_N=ring_n(mol),
                         hetero=has_ring_hetero(mol),
                         quinolone=any(mol.HasSubstructMatch(q) for q in QUERY),
                         x=float(XY[k, 0]), y=float(XY[k, 1])))
    json.dump(rows, open(B + "reports/ring_system_map.json", "w"), indent=1)

    # ------------------------------------------------------------- figure --
    fig, ax = plt.subplots(figsize=(7.10, 5.60))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.055, top=0.91)
    X = np.array([r["x"] for r in rows])
    Y = np.array([r["y"] for r in rows])
    maxp = np.array([r["max_p"] for r in rows])
    n = np.array([r["n"] for r in rows])
    size = 6 + 46 * (np.sqrt(n) - np.sqrt(n.min())) / (np.sqrt(n.max()) - np.sqrt(n.min()))
    norm = LogNorm(vmin=1e-4, vmax=1.0)
    sca = ax.scatter(X, Y, s=size, c=np.clip(maxp, 1e-4, 1), cmap="RdYlBu_r",
                     norm=norm, edgecolor="white", linewidth=0.4, alpha=0.95,
                     zorder=3)
    cax = ax.inset_axes([0.015, 0.955, 0.26, 0.016])
    cb = fig.colorbar(sca, cax=cax, orientation="horizontal")
    cb.set_label("max XGBoost score", fontsize=5.6, labelpad=1)
    cb.ax.tick_params(labelsize=5.0, length=1.8, pad=1)

    # axis limits with side margins for structure callouts
    x0, x1 = X.min(), X.max()
    y0, y1 = Y.min(), Y.max()
    sx, sy = x1 - x0, y1 - y0
    ax.set_xlim(x0 - 0.60 * sx, x1 + 0.60 * sx)
    ax.set_ylim(y0 - 0.12 * sy, y1 + 0.12 * sy)

    # mark false-positive ring systems
    fp = [r for r in rows if r["max_p"] >= TAU and r["n_pos"] == 0]
    ax.scatter([r["x"] for r in fp], [r["y"] for r in fp], s=110,
               facecolors="none", edgecolors="#B2182B", linewidths=1.2,
               zorder=4)
    ax.text(0.01, 0.01, f"red circles: ring systems with a false positive "
            f"({len(fp)} of 210 shown groups)", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=6.0, color="#B2182B")

    # structure callouts in the side margins
    want = {
        "c1ccccc1": "benzene",
        "c1ccncc1": "pyridine",
        "c1ccc2[nH]ccc2c1": "indole",
        "C1CCNC1": "pyrrolidine",
        "c1cn[nH]c1": "pyrazole",
        "O=c1ccn(C2CC2)c2cc(N3CCNCC3)ccc12": "quinolone",
        "O=c1ccc2ccccc2n1-c1ccccc1": "N-Ph quinolinone",
        "c1ccc(N=Nc2nccs2)cc1": "azo-thiazole",
    }
    sel = [r for r in rows if r["scaffold"] in want]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left = sorted([r for r in sel if r["x"] < cx], key=lambda r: r["y"])
    right = sorted([r for r in sel if r["x"] >= cx], key=lambda r: r["y"])
    for side, group in (("L", left), ("R", right)):
        for k, r in enumerate(group):
            frac = (k + 0.5) / max(1, len(group))
            yy = y0 + 0.05 * sy + frac * 0.90 * sy
            xx = (x0 - 0.47 * sx) if side == "L" else (x1 + 0.47 * sx)
            ax.plot([xx, r["x"]], [yy, r["y"]], color="#AAAAAA", lw=0.45,
                    zorder=2)
            ab = AnnotationBbox(OffsetImage(mol_image(r["scaffold"]), zoom=0.105),
                                (xx, yy), frameon=True,
                                bboxprops=dict(boxstyle="round,pad=0.07",
                                               fc="white", ec="#CCCCCC", lw=0.4),
                                zorder=5)
            ax.add_artist(ab)
            ax.text(xx, yy + 0.048 * sy,
                    f"{want[r['scaffold']]} (n={r['n']})", ha="center",
                    va="bottom", fontsize=4.9, color=TEXT, zorder=8)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title("Ring-system space of the external test set "
                 "(CReSS SMILES embeddings, UMAP)", loc="left",
                 fontweight="bold")
    for sp in ax.spines.values():
        sp.set_edgecolor("#BBBBBB")
    ax.grid(True, lw=0.3, color="#EEEEEE", zorder=0)

    for ext in ("pdf", "png"):
        fig.savefig(B + f"reports/fig_ring_system_map.{ext}",
                    dpi=600 if ext == "png" else None)
    print("saved fig_ring_system_map; FP groups:",
          [(round(r["max_p"], 3), r["scaffold"]) for r in fp])


if __name__ == "__main__":
    main()
