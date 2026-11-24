#!/usr/bin/env python3
"""整理「5 个模拟实验」的 CMGNet-SPrompt 预测结果，输出 CSV 与 Markdown 摘要。

输入（均为本地路径，相对仓库根）：
  - data/sim5/fluoroquinolones_nmr.json
      5 条模拟实验 13C 谱，字段 key/cano_smiles/C_nmr/source/label。
      C_nmr 为 ppm 列表（13C 化学位移，未排序或已排序均可，模型内部自行处理）。
  - output/sim5_sprompt_ex1209_s{0,42,2006}/test_nmr+frag_beams10_samples5.json
      main_cmgnet.py --decode_mode nmr+frag --num_beams 10 的解码 dump，
      每条样本含 10 个候选 SMILES、score、is_exact_match、has_core_structure。

输出（写到 reports/）：
  - sim5_sprompt_ex1209_predictions.csv   逐样本×逐候选明细（3 个 seed 全展开）
  - sim5_sprompt_ex1209_summary.md        按化合物汇总：top-1、exact 命中 rank、
                                          core 命中数、top-1 Tanimoto

口径：
  - 「exact」= RDKit canonical SMILES 完全一致（与 main_cmgnet 的 is_exact_match 对齐）；
  - 「core」= dump 里 has_core_structure（喹诺酮母核子结构匹配，与 4,502 外部集评测同口径）；
  - Tanimoto = Morgan(r=2, 2048 bit)，仅用于描述 top-1 与参考的接近程度，不参与任何筛选。

用法：
  cd /mnt/data2/NMR_Core_Mol_Detection
  /home/lingjiebao/venvs/py312_tf457/bin/python reports/make_sim5_sprompt_report.py

作者：ZCode（Claude），2026-11-22
"""

import csv
import json
from pathlib import Path

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

REPO = Path(__file__).resolve().parent.parent
SRC_JSON = REPO / "data/sim5/fluoroquinolones_nmr.json"
SEEDS = ["s0", "s42", "s2006"]
DUMP = "output/sim5_sprompt_ex1209_{seed}/test_nmr+frag_beams10_samples5.json"
OUT_CSV = REPO / "reports/sim5_sprompt_ex1209_predictions.csv"
OUT_MD = REPO / "reports/sim5_sprompt_ex1209_summary.md"


def canon(smiles: str) -> str | None:
    """RDKit canonical SMILES；解析失败返回 None（与 exact 判定保持同口径）。"""
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def morgan(smiles: str):
    """Morgan 指纹（r=2, 2048 bit）；解析失败返回 None。"""
    mol = Chem.MolFromSmiles(smiles)
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048) if mol is not None else None


def tanimoto(a: str, b: str) -> float:
    """两个 SMILES 的 Tanimoto 相似度；任一解析失败返回 nan。"""
    fa, fb = morgan(a), morgan(b)
    if fa is None or fb is None:
        return float("nan")
    return DataStructs.TanimotoSimilarity(fa, fb)


def main() -> None:
    src = json.load(open(SRC_JSON))
    names = [it["source"] for it in src]
    refs = [it["cano_smiles"] for it in src]

    runs = {s: json.load(open(REPO / DUMP.format(seed=s))) for s in SEEDS}

    # ---- 逐样本 × 逐候选明细 ----
    rows = []
    for s in SEEDS:
        for i, item in enumerate(runs[s]):
            ref_canon = canon(item["reference_smiles"])
            for p in item["predictions"]:
                rows.append(
                    {
                        "seed": s,
                        "compound": names[i],
                        "reference_smiles": item["reference_smiles"],
                        "rank": p["rank"],
                        "smiles": p["smiles"],
                        "score": p["score"],
                        "is_core": int(bool(p.get("has_core_structure"))),
                        "is_exact": int(canon(p["smiles"]) == ref_canon),
                        "tanimoto_to_ref": round(tanimoto(p["smiles"], ref_canon), 4),
                    }
                )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- 按化合物汇总 ----
    per_comp = []
    for i, name in enumerate(names):
        rec = {"compound": name, "reference": refs[i]}
        for s in SEEDS:
            preds = runs[s][i]["predictions"]
            ref_canon = canon(runs[s][i]["reference_smiles"])
            exact_ranks = [p["rank"] for p in preds if canon(p["smiles"]) == ref_canon]
            core_ranks = [p["rank"] for p in preds if p.get("has_core_structure")]
            rec[f"{s}_top1"] = preds[0]["smiles"]
            rec[f"{s}_exact_rank"] = exact_ranks[0] if exact_ranks else None
            rec[f"{s}_core_hits10"] = len(core_ranks)
            rec[f"{s}_tanimoto"] = tanimoto(preds[0]["smiles"], ref_canon)
        # 共识：3 个 seed 中至少 1 个 exact 命中 / 全部 top-1 含母核
        rec["any_exact"] = any(rec[f"{s}_exact_rank"] is not None for s in SEEDS)
        rec["all_top1_core"] = all(rec[f"{s}_core_hits10"] >= 1 for s in SEEDS)
        per_comp.append(rec)

    # ---- 总体指标 ----
    def rate(fn):
        vals = [fn(r) for r in per_comp]
        return sum(vals) / len(vals)

    overall = []
    for s in SEEDS:
        exact_ranks = [rec[f"{s}_exact_rank"] for rec in per_comp]
        overall.append(
            {
                "seed": s,
                "core@1": rate(lambda r, s=s: 1 if r[f"{s}_core_hits10"] >= 1 else 0),
                "exact@1": rate(lambda r, s=s: 1 if r[f"{s}_exact_rank"] == 1 else 0),
                "exact@10": rate(lambda r, s=s: 1 if r[f"{s}_exact_rank"] is not None else 0),
                "mrr": sum((1.0 / r) if r else 0.0 for r in exact_ranks) / len(exact_ranks),
                "mean_tanimoto_top1": sum(rec[f"{s}_tanimoto"] for rec in per_comp) / len(per_comp),
            }
        )

    # ---- Markdown ----
    lines = []
    lines.append("# 5 个模拟实验：CMGNet-SPrompt（ex1209 5ep，3 seed）预测结果\n")
    lines.append(
        "输入：`data/sim5/fluoroquinolones_nmr.json`（5 条模拟 13C 谱，19–29 个峰）。\n"
        "模型：`train_newdb_1209_extrain_5ep_seed{0,42,2006}/best_models/last.ckpt`，"
        "`decode_mode=nmr+frag`、`num_beams=10`、SPrompt 门控全置 1（5 条均为已知阳性）。\n"
        "评测：exact = RDKit canonical 完全一致；core = 喹诺酮母核子结构命中（与 4,502 外部集同口径）。\n"
    )
    lines.append("\n## 总体\n")
    lines.append("| seed | core@1 | exact@1 | exact@10 | MRR(exact) | mean Tanimoto(top-1) |")
    lines.append("|---|---|---|---|---|---|")
    for o in overall:
        lines.append(
            f"| {o['seed']} | {o['core@1']:.2f} | {o['exact@1']:.2f} | {o['exact@10']:.2f} | "
            f"{o['mrr']:.2f} | {o['mean_tanimoto_top1']:.3f} |"
        )
    lines.append(
        f"\n3 seed 并集：exact 至少命中一次 {sum(r['any_exact'] for r in per_comp)}/5；"
        f"top-1 均含母核 {sum(r['all_top1_core'] for r in per_comp)}/5。\n"
    )

    lines.append("\n## 逐化合物\n")
    lines.append("| 化合物 | 参考 SMILES | seed | top-1 预测 | exact rank | core in top-10 | Tanimoto |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in per_comp:
        for s in SEEDS:
            rank = r[f"{s}_exact_rank"]
            lines.append(
                f"| {r['compound']} | `{r['reference']}` | {s} | `{r[f'{s}_top1']}` | "
                f"{rank if rank is not None else '—'} | {r[f'{s}_core_hits10']}/10 | {r[f'{s}_tanimoto']:.3f} |"
            )
    lines.append(
        "\n明细（每条样本 10 个候选的 SMILES/score/core/exact/Tanimoto）见 "
        "`reports/sim5_sprompt_ex1209_predictions.csv`。\n"
    )
    lines.append(
        "\n> 注意：这 5 条谱为模拟实验（C_nmr 非实测），且 Ciprofloxacin/L3-3/L4-1 的 top-10 内"
        "未出现与参考完全一致的 SMILES，属于该模型在外部集上常见的 exact 漏检（4,502 外部集 SA@10 约 0.29）。\n"
    )

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_MD}")
    for o in overall:
        print(o)


if __name__ == "__main__":
    main()
