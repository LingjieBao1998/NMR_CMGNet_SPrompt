#!/usr/bin/env python3
"""R2.2 定性对比图:6 个跨取代基家族的外部阳性,
reference vs 各模型 top-1 输出结构,附母核命中/精确匹配标记.

数据源(全部已存在的产物):
  output/apr18_ex1209r_s42_smiles_beams10/...  CMGNet-SPrompt (最终模型)
  output/apr18_ex1209_s42_smiles_beams10/...   CMGNet-SMILES (无提示)
  output/apr18_base0910_smiles_beams10/...     CMGNet 0910 (原版)
  output/nmrsolver_top10/pos64_top10.json      NMR-Solver top-10
  (NMRMind 不入图: 定量对比 48/64, 见 NMRMind_ref/result/april18peak_result10.json;
   reports/last10_all_models.json 的 NMRMind 列不可引用——与权威文件矛盾)
输出: reports/fig_r2_qualitative.png (600dpi)
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from io import BytesIO
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D

RDLogger.DisableLog("rdApp.*")
ROOT = Path(os.environ.get('NMR_ROOT', str(Path(__file__).resolve().parent.parent)))
OUT = ROOT / "reports" / "fig_r2_qualitative.png"

BEAM = ROOT / "results" / "predictions"


def load_beam(rel):
    return json.load(open(BEAM / rel))


def top1(d, i):
    p = d[i]["predictions"][0]
    return p["smiles"], bool(p.get("has_core_structure"))


sprompt = load_beam("sprompt_seed42_smiles_beams10.json")
csmiles = load_beam("cmgnet_smiles_seed42_beams10.json")
c0910 = load_beam("cmgnet_0910_smiles_beams10.json")
# NMR-Solver: 64 阳性的真实逐样本 top-10 (task_pos64_<4438+idx> 与 april18peak 序对齐)
solver = json.load(open(BEAM / "nmrsolver_pos64_top10.json"))

# 跨家族 6 分子: N1-乙基+哌嗪 / N1-环丙+稠合手性吡咯烷+8-OMe / 乙酯+双F /
# 三环手性(氧氟沙星型)+N-甲基哌嗪 / 三环+双F无7-氨基 / 萘啶酸(aza)+7-甲基
PICK = [
    (54, "N-ethyl / 7-piperazine"),
    (55, "N1-cyclopropyl / fused pyrrolidine / 8-OMe"),
    (57, "3-carboxylic acid / 6,8-diF"),
    (58, "tricyclic (ofloxacin-type) / 7-NMe-piperazine"),
    (60, "tricyclic / diF, no 7-amino"),
    (63, "naphthyridine (aza) / 7-Me"),
]

# 互变异构感知母核匹配: 芳香式 4-oxo-1,4-dihydroquinoline / naphthyridine,
# NH 与 N-取代两种形式
CORE_SMARTS = [Chem.MolFromSmarts(s) for s in (
    "O=c1cc[nH]c2ccccc12", "O=c1cc[nH0]c2ccccc12",
    "c1ccc2c(=O)cc[nH]c2n1", "c1ccc2c(=O)cc[nH0]c2n1",
)]


def has_core(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None, False
    hit = any(m.HasSubstructMatch(q) for q in CORE_SMARTS if q is not None)
    return m, hit


def canon(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else ""


# 自检: 已知 top-10 命中的阳性参考分子必须能匹配到母核
_ref_test = canon(sprompt[54]["reference_smiles"])
assert has_core(_ref_test)[1], f"core SMARTS self-test failed on {_ref_test}"

CELL = 1.55  # inch


def draw_mol(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    d = rdMolDraw2D.MolDraw2DCairo(int(CELL * 300), int(CELL * 270))
    opts = d.drawOptions()
    opts.bondLineWidth = 1.6
    opts.minFontSize = 9
    opts.maxFontSize = 12
    rdMolDraw2D.PrepareAndDrawMolecule(d, m)
    d.FinishDrawing()
    return Image.open(BytesIO(d.GetDrawingText()))


cols = ["Reference", "CMGNet-SPrompt\n(ours)", "CMGNet-SMILES",
        "CMGNet 0910", "NMR-Solver"]
rows = []

for idx, fam in PICK:
    ref_smi = sprompt[idx]["reference_smiles"]
    ref_can = canon(ref_smi)
    cells = [("reference", ref_smi, None)]
    for name, d in (("SPrompt", sprompt), ("SMILES", csmiles), ("0910", c0910)):
        smi, core = top1(d, idx)
        cells.append((name, smi, core))
    smi = solver[f"task_pos64_{4438 + idx}"]["top10"][0]
    _, hit = has_core(smi)
    cells.append(("NMR-Solver", smi, hit))
    rows.append((idx, fam, ref_can, cells))

ncols = 1 + len(cols)  # 标签列 + 6 数据列
fig_w = 0.9 + len(cols) * CELL
fig_h = 0.75 + len(rows) * CELL * 0.95
fig, axes = plt.subplots(len(rows) + 1, ncols,
                         figsize=(fig_w, fig_h), dpi=600,
                         gridspec_kw={"width_ratios": [0.9] + [1] * len(cols)})
for j, c in enumerate(cols):
    axes[0][0].axis("off")
    axes[0][j + 1].axis("off")
    axes[0][j + 1].text(0.5, 0.6, c, ha="center", va="center", fontsize=7.5,
                        fontweight="bold", transform=axes[0][j + 1].transAxes)
for i, (idx, fam, ref_can, cells) in enumerate(rows):
    ax0 = axes[i + 1][0]
    ax0.axis("off")
    ax0.text(0.5, 0.62, f"#{idx}", ha="center", va="center", fontsize=7.5,
             fontweight="bold", transform=ax0.transAxes)
    ax0.text(0.5, 0.38, fam, ha="center", va="center", fontsize=5.6,
             transform=ax0.transAxes, color="#555555")
    for j, (name, smi, core) in enumerate(cells):
        ax = axes[i + 1][j + 1]
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        img = draw_mol(smi)
        if img is None:
            ax.text(0.5, 0.5, "invalid", ha="center", va="center",
                    fontsize=6, transform=ax.transAxes, color="red")
            continue
        ax.imshow(img)
        if name != "reference":
            exact = canon(smi) == ref_can
            if exact:
                col, mark = "#B8860B", "exact"
            elif core:
                col, mark = "#1a7f37", "core"
            else:
                col, mark = "#c62828", "miss"
            ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                   fill=False, ec=col, lw=1.6, clip_on=False))
            ax.text(0.5, 1.03, mark, ha="center", va="bottom", fontsize=5.8,
                    color=col, fontweight="bold", transform=ax.transAxes)
handles = [Rectangle((0, 0), 1, 1, fc="none", ec=c, lw=1.6)
           for c in ("#1a7f37", "#B8860B", "#c62828")]
fig.legend(handles, ["top-1 contains quinolone core", "top-1 exact match",
                     "top-1 lacks core"],
           loc="lower center", ncol=3, fontsize=6.2, frameon=False,
           bbox_to_anchor=(0.5, -0.005))
fig.suptitle("Top-1 output vs reference on six external positives "
             "(spanning substitution families)", fontsize=8.5, y=0.995)
fig.tight_layout(rect=(0, 0.03, 1, 0.985))
fig.savefig(OUT, dpi=600, bbox_inches="tight")
print("saved ->", OUT)
