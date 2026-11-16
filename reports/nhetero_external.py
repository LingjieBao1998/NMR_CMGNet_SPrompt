#!/usr/bin/env python3
"""R2.3 -- specificity of the XGBoost classifier on non-quinolone N-heterocycles.

The 4,502-spectrum external test set is filtered with the *same* ring-system
retention rule used to build the CMGNet scaffold prompts (main_cmgnet.py:
RingSystemFinder -> remove_specific_terminal_carbons -> find_largest_smiles,
which prefers aromatic ring systems and then the largest by atom count /
non-CH atom count).  Two views are recorded:

  * retained ring system (what the scaffold prompt would see), and
  * any ring nitrogen in the whole molecule (chemical definition).

Molecules are grouped as:
  quinolone_positive        label 1 (reference)
  n_hetero_retained         label 0, N in the retained (largest) ring system
  n_hetero_side             label 0, N only in a non-retained ring
  carbocycle_negative       label 0, rings but no ring N
  acyclic_negative          label 0, no ring system

Outputs
  reports/nhetero_external_analysis.json
  reports/fig_nhetero_external.pdf / .png
"""
import io
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw
from PIL import Image

import os as _os
B = _os.environ.get('NMR_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
B = B.rstrip('/') + '/'
sys.path.insert(0, B)
RDLogger.DisableLog("rdApp.*")
import main_cmgnet as M  # noqa: E402

TAU = 0.50
QUERY = [Chem.MolFromSmiles("O=C1C(C(O)=O)=CNC2=CC=CC=C12"),
         Chem.MolFromSmiles("O=C1C(C(O)=O)=CNC2=NC=CC=C12")]
BLUE, ORANGE, GREY, GREEN, RED = ("#0072B2", "#E69F00", "#999999",
                                  "#009E73", "#D55E00")
TEXT = "#262626"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 7.5, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "axes.labelcolor": TEXT, "text.color": TEXT, "xtick.color": TEXT,
    "ytick.color": TEXT, "axes.linewidth": 0.7, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def ring_stats(mol):
    if mol is None:
        return 0, 0
    n_rings = n_N = 0
    for ring in mol.GetRingInfo().AtomRings():
        n_rings += 1
        if any(mol.GetAtomWithIdx(i).GetSymbol() == "N" for i in ring):
            n_N += 1
    return n_rings, n_N


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(max(0, ctr - half)), float(min(1, ctr + half))


def mol_image(smi, w=760, h=470):
    m = Chem.MolFromSmiles(smi)
    d = Draw.MolDraw2DCairo(w, h)
    o = d.drawOptions()
    o.bondLineWidth = 3.2
    o.minFontSize = 20
    o.maxFontSize = 26
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

    rows = []
    for i, r in enumerate(recs):
        mol = Chem.MolFromSmiles(r["cano_smiles"])
        rs = M.get_ring_system(mol)
        big = M.find_largest_smiles(rs) if rs else ""
        bm = Chem.MolFromSmiles(big) if big else None
        b_rings, b_N = ring_stats(bm)
        a_rings, a_N = ring_stats(mol)
        q = bool(mol and any(mol.HasSubstructMatch(x) for x in QUERY))
        if y[i] == 1:
            cls = "quinolone_positive"
        elif a_N > 0 and b_N > 0:
            cls = "n_hetero_retained"
        elif a_N > 0:
            cls = "n_hetero_side"
        elif a_rings > 0:
            cls = "carbocycle_negative"
        else:
            cls = "acyclic_negative"
        rows.append(dict(idx=i, label=int(y[i]), p=float(p[i]),
                         smiles=r["cano_smiles"], retained_ring=big,
                         retained_rings=b_rings, retained_N=b_N,
                         mol_rings=a_rings, mol_ringN=a_N, quinolone=q,
                         cls=cls))

    classes = ["quinolone_positive", "n_hetero_retained", "n_hetero_side",
               "carbocycle_negative", "acyclic_negative"]
    summary = {}
    for c in classes:
        sub = [r for r in rows if r["cls"] == c]
        pp = np.array([r["p"] for r in sub])
        k = int((pp >= TAU).sum())
        lo, hi = wilson(k, len(sub))
        summary[c] = {
            "n": len(sub), "fp_at_tau": k,
            "fp_rate": k / len(sub) if sub else 0.0,
            "fp_rate_ci": [lo, hi],
            "median_p": float(np.median(pp)) if len(pp) else 0.0,
            "p95": float(np.quantile(pp, 0.95)) if len(pp) else 0.0,
            "max_p": float(pp.max()) if len(pp) else 0.0,
        }

    fps = sorted([r for r in rows if r["label"] == 0 and r["p"] >= TAU],
                 key=lambda r: -r["p"])
    out = {"tau": TAU,
           "rule": "RingSystemFinder -> remove_specific_terminal_carbons -> "
                   "find_largest_smiles (aromatic first, then atom count / non-CH count)",
           "summary": summary,
           "false_positives": [{k: r[k] for k in
                                ("idx", "p", "smiles", "retained_ring",
                                 "retained_N", "cls")} for r in fps],
           "top_n_heterocycle_negatives": [
               {k: r[k] for k in ("idx", "p", "smiles", "retained_ring")}
               for r in sorted([r for r in rows if r["cls"].startswith("n_hetero")],
                               key=lambda r: -r["p"])[:6]],
           "records": rows}
    json.dump(out, open(B + "reports/nhetero_external_analysis.json", "w"),
              indent=1)

    # ------------------------------------------------------------- figure --
    fig = plt.figure(figsize=(7.10, 6.20))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.30], hspace=0.42,
                             left=0.075, right=0.985, top=0.93, bottom=0.055)
    top = outer[0].subgridspec(1, 2, width_ratios=[2.15, 1.0], wspace=0.30)
    axA = fig.add_subplot(top[0])
    axB = fig.add_subplot(top[1])

    spec = [("quinolone_positive", BLUE, "Quinolone positive (n=64)"),
            ("n_hetero_retained", ORANGE,
             "N-heterocycle negative (n=%d)" %
             (summary["n_hetero_retained"]["n"] + summary["n_hetero_side"]["n"])),
            ("carbocycle_negative", GREY,
             "Carbocycle negative (n=%d)" % summary["carbocycle_negative"]["n"]),
            ("acyclic_negative", GREEN,
             "Acyclic negative (n=%d)" % summary["acyclic_negative"]["n"])]
    for c, col, lab in spec:
        if c == "n_hetero_retained":
            pp = np.array([r["p"] for r in rows if r["cls"].startswith("n_hetero")])
        else:
            pp = np.array([r["p"] for r in rows if r["cls"] == c])
        pp = np.sort(pp)
        x = np.clip(pp, 1e-6, 1.0)
        ccdf = 1.0 - np.arange(len(x)) / len(x)
        axA.step(x, ccdf, where="post", color=col, lw=1.5, label=lab)
    axA.axvline(TAU, color=RED, ls="--", lw=1.0)
    axA.text(TAU * 1.12, 0.42, r"$\tau_{op}$=0.50", color=RED, fontsize=6.0,
             rotation=90)
    axA.set_xscale("log")
    axA.set_yscale("log")
    axA.set_xlim(1e-6, 1.0)
    axA.set_ylim(1e-5, 1.2)
    axA.set_xlabel("XGBoost score")
    axA.set_ylabel("Fraction with score $\\geq x$")
    axA.set_title("A  Specificity against N-heterocycles", loc="left",
                  fontweight="bold")
    axA.legend(loc="lower left", fontsize=5.9, handlelength=1.2)
    axA.grid(True, lw=0.35, color="#E5E5E5")

    order = [("n_hetero_retained", "N in retained\nring (prompt)"),
             ("n_hetero_side", "N only in\nside ring"),
             ("carbocycle_negative", "Carbocycle"),
             ("acyclic_negative", "Acyclic")]
    cols = [ORANGE, "#F2C14E", GREY, GREEN]
    for i, (c, lab) in enumerate(order):
        s = summary[c]
        rate = 100 * s["fp_rate"]
        lo, hi = 100 * s["fp_rate_ci"][0], 100 * s["fp_rate_ci"][1]
        axB.barh(i, rate, height=0.58, color=cols[i], alpha=0.92)
        axB.plot([lo, hi], [i, i], color=TEXT, lw=0.9)
        axB.text(0.72, i, f"{s['fp_at_tau']}/{s['n']}", va="center",
                 fontsize=5.8)
    axB.set_yticks(range(4), [lab for _, lab in order], fontsize=5.9)
    axB.set_xlim(0, 0.85)
    axB.set_xlabel("False-positive rate at $\\tau_{op}$ (%)")
    axB.set_title("B  FP rate by ring class", loc="left", fontweight="bold")
    axB.grid(True, axis="x", lw=0.35, color="#E5E5E5")

    # C: 2 x 3 gallery of top-scoring non-quinolone N-heterocycles
    bottom = outer[1].subgridspec(2, 3, hspace=0.10, wspace=0.10)
    top6 = out["top_n_heterocycle_negatives"]
    for k, r in enumerate(top6):
        ax = fig.add_subplot(bottom[k // 3, k % 3])
        ax.imshow(mol_image(r["smiles"]))
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#DDDDDD")
        ax.set_title(f"p = {r['p']:.3f}", fontsize=7.0, color=TEXT, pad=1.5)
        if k == 0:
            ax.text(-0.02, 1.30, "C  Top-scoring non-quinolone N-heterocycles",
                    transform=ax.transAxes, fontsize=8.5, fontweight="bold",
                    color=TEXT, ha="left", va="bottom")

    for ext in ("pdf", "png"):
        fig.savefig(B + f"reports/fig_nhetero_external.{ext}",
                    dpi=600 if ext == "png" else None)
    print(json.dumps(summary, indent=2))
    print("FP:", [(r["idx"], round(r["p"], 4), r["cls"]) for r in fps])


if __name__ == "__main__":
    main()
