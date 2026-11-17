#!/usr/bin/env python3
"""cmgnet 生成式模型在外部测试集(4502, 正64)的 top-N P/R/F1:三个数据配置 × 3 seed × frag/smiles.

判定逻辑与 calculate_cmgnet_metrics.py 一致:top-N 内任一 prediction 含母核结构
即判为阳性,与 reference_analysis.has_core_structure 对比出混淆矩阵。
"""
import json
import math
from pathlib import Path

ROOT = Path("/mnt/data2/NMR_Core_Mol_Detection/output")
TOP_NS = [1, 3, 5, 10]

# 组名 -> {seed: 目录模板};exp30ep 取 best,1209extrain 的 seed42 用 1ep 重训(retrain)
GROUPS = {
    "exp30ep(纯实验语料,best)": {
        "42": "external_4502_exp30ep_best_seed42_{rep}_beams10",
        "0": "external_4502_exp30ep_best_seed0_{rep}_beams10",
        "2006": "external_4502_exp30ep_best_seed2006_{rep}_beams10",
    },
    "extrain5ep(last)": {
        "42": "external_4502_extrain5ep_seed42_last_{rep}_beams10",
        "0": "external_4502_extrain5ep_seed0_last_{rep}_beams10",
        "2006": "external_4502_extrain5ep_seed2006_last_{rep}_beams10",
    },
    "1209extrain(seed42=1ep重训)": {
        "42": "external_4502_1ep_1209extrain_retrain_{rep}_beams10",
        "0": "external_4502_1209extrain_seed0_last_{rep}_beams10",
        "2006": "external_4502_1209extrain_seed2006_last_{rep}_beams10",
    },
}
SEEDS = ["42", "0", "2006"]
REPS = ["frag", "smiles"]


def topn_metrics(data, top_ns=TOP_NS):
    ref_pos = sum(1 for x in data if x["reference_analysis"]["has_core_structure"])
    out = {"n": len(data), "pos": ref_pos}
    for n in top_ns:
        tp = fp = fn = 0
        for item in data:
            ref = item["reference_analysis"]["has_core_structure"]
            pred = any(p.get("has_core_structure", False) for p in item["predictions"][:n])
            if ref and pred:
                tp += 1
            elif not ref and pred:
                fp += 1
            elif ref:
                fn += 1
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        out[n] = {"p": p, "r": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
    return out


def mean_std(vals):
    m = sum(vals) / len(vals)
    s = 0.0 if len(vals) < 2 else math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))
    return m, s


def fmt(vals):
    m, s = mean_std(vals)
    return f"{m:.4f}±{s:.4f}"


def main():
    lines = []
    ap = lines.append
    ap("# CMGNet(生成式)外部测试集 top-N 精度:三数据配置 × 3 seed × frag/smiles\n")
    ap("**日期:2026-10-26**")
    ap("")
    ap("外部测试集 4,502 条(母核正例 64,与 xgboost 检测头同一外部集)。")
    ap("判定:top-N 候选中任一含母核结构即判阳;P/R/F1 相对该二分类。mean±std 为 seed 间(ddof=1)。")
    ap("ckpt 选择:exp30ep 取 best,extrain5ep 取 last;1209extrain 的 seed0/2006 取 last,seed42 用 1ep 重训版(seed42_retrain,见 run train_newdb_1209_extrain_5ep_seed42_retrain)。")
    ap("")

    results = {}
    for gname, tpls in GROUPS.items():
        results[gname] = {}
        for rep in REPS:
            rows = {}
            for sd in SEEDS:
                d = ROOT / tpls[sd].format(seed=sd, rep=rep)
                f = d / f"test_{rep}_beams10_samples4502.json"
                data = json.load(open(f))
                rows[sd] = topn_metrics(data)
            results[gname][rep] = rows

    # ---- 汇总表:每组×rep, top-N × (P/R/F1) mean±std ----
    ap("## 汇总(seed 间 mean±std)\n")
    ap("| 配置 | 表示 | 指标 | top-1 | top-3 | top-5 | top-10 |")
    ap("|---|---|---|---|---|---|---|")
    for gname in GROUPS:
        for rep in REPS:
            rows = results[gname][rep]
            for key, label in [("p", "Precision"), ("r", "Recall"), ("f1", "F1")]:
                cells = [fmt([rows[sd][n][key] for sd in SEEDS]) for n in TOP_NS]
                ap(f"| {gname} | {rep} | {label} | " + " | ".join(cells) + " |")
    ap("")

    # ---- 逐 seed 明细 ----
    ap("## 逐 seed 明细\n")
    for gname in GROUPS:
        ap(f"### {gname}\n")
        for rep in REPS:
            ap(f"**{rep}**")
            ap("")
            ap("| seed | top-N | Precision | Recall | F1 | (tp, fp, fn) |")
            ap("|---|---|---|---|---|---|")
            rows = results[gname][rep]
            for sd in SEEDS:
                for n in TOP_NS:
                    m = rows[sd][n]
                    ap(f"| {sd} | {n} | {m['p']:.4f} | {m['r']:.4f} | {m['f1']:.4f} "
                       f"| ({m['tp']}, {m['fp']}, {m['fn']}) |")
            ap("")

    out = ROOT / "cmgnet_topn_external_summary_20261026.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
