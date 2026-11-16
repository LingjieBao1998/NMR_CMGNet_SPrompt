#!/usr/bin/env python3
"""R1.1 Table R1 配图:4 个 FP + 1 个边界样本的分子结构(600dpi).

p 值口径 = 2026-11-13 最新打分(与 response_to_reviewers_v2.md Table R1 一致):
  #3898 0.903 / #3460 0.995 / #763 0.734 / #4201 0.753; #1181 0.418(边界负样本)
输出:
  fig_fp_structures.png        5 分子一排(SI Fig. S7)
  fp_struct_<idx>.png          逐分子小图(嵌 Table R1 表格单元格)
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from rdkit import Chem
from rdkit.Chem import Draw

ROOT = Path(os.environ.get('NMR_ROOT', str(Path(__file__).resolve().parent.parent)))
OUT = ROOT / "reports" / "fig_fp_structures.png"

ITEMS = [
    ("#3898", "p = 0.903", "quinolinone-like (2-oxo isomer)",
     "COC(=O)c1c(O)c2ccccc2n(-c2ccccc2)c1=O"),
    ("#3460", "p = 0.995", "azole (azo-thiazole dye)",
     "Oc1ccc(N=Nc2nccs2)c(O)c1"),
    ("#763", "p = 0.734", "fused N-heteroarene (furoquinoline)",
     "COC(=O)C(C)CCOc1ccc2c(OC)c3ccoc3nc2c1"),
    ("#4201", "p = 0.753", "carbocycle (N-to-O quinolone analogue)",
     "Cc1cc(=O)c2cc(C#N)ccc2o1"),
    ("#1181", "p = 0.418", "borderline below τ (dihydropyridone)",
     "COC(=O)C1=C(C)NC(Cl)=C(C=O)C1c1cccnc1"),
]

mols, legends = [], []
for idx, p, cls, smi in ITEMS:
    m = Chem.MolFromSmiles(smi)
    assert m is not None, smi
    mols.append(m)
    legends.append(f"{idx}   {p}\n{cls}")
    # 逐分子小图(表格嵌入用): 稍大画布, 300dpi 即可满足 docx 缩放
    single = Draw.MolsToGridImage([m], molsPerRow=1, subImgSize=(360, 300),
                                  legends=[f"{idx}  {p}"], useSVG=False)
    single.save(ROOT / "reports" / f"fp_struct_{idx.lstrip('#')}.png")

img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(360, 320),
                           legends=legends, useSVG=False)
img.save(OUT)
print("saved ->", OUT, "+ 5 per-molecule fp_struct_*.png")
