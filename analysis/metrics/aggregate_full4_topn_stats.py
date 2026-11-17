#!/usr/bin/env python3
"""full4 1ep 重训(4来源语料)外部测试 top-N 指标, 与最近几轮同表对比.

口径与 aggregate_cmgnet_topn_stats.py 完全一致(top-N 内任一 prediction
含母核即判阳), 复用其 topn_metrics。
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_cmgnet_topn_stats import topn_metrics, ROOT, TOP_NS, REPS

GROUPS = {
    "full4_nmr0(4来源,1ep,无NMR噪声)": {
        "0": "external_4502_full4_nmr0_1ep_seed0_{rep}_beams10",
        "2006": "external_4502_full4_nmr0_1ep_seed2006_{rep}_beams10",
    },
    "full4(4来源,1ep,有噪对照)": {
        "0": "external_4502_full4_1ep_seed0_{rep}_beams10",
        "2006": "external_4502_full4_1ep_seed2006_{rep}_beams10",
    },
    "nopubchem_cdk(1ep,1027)": {
        "0": "external_4502_nopubchem_cdk1ep_seed0_{rep}_beams10",
        "2006": "external_4502_nopubchem_cdk1ep_seed2006_{rep}_beams10",
    },
    "nopubchem_14mol(1ep,1027)": {
        "0": "external_4502_nopubchem_14mol1ep_seed0_{rep}_beams10",
        "2006": "external_4502_nopubchem_14mol1ep_seed2006_{rep}_beams10",
    },
    "1209extrain(对照,seed42=1ep重训)": {
        "42": "external_4502_1ep_1209extrain_retrain_{rep}_beams10",
        "0": "external_4502_1209extrain_seed0_last_{rep}_beams10",
        "2006": "external_4502_1209extrain_seed2006_last_{rep}_beams10",
    },
}


def mean_std(vals):
    m = sum(vals) / len(vals)
    s = 0.0 if len(vals) < 2 else math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))
    return m, s


def fmt(vals):
    m, s = mean_std(vals)
    return f"{m:.4f}±{s:.4f}"


def main():
    results = {}
    for gname, tpls in GROUPS.items():
        results[gname] = {}
        for rep in REPS:
            rows = {}
            for sd, tpl in tpls.items():
                f = ROOT / tpl.format(seed=sd, rep=rep) / f"test_{rep}_beams10_samples4502.json"
                data = json.load(open(f))
                rows[sd] = topn_metrics(data)
            results[gname][rep] = rows

    lines = []
    ap = lines.append
    ap("# CMGNet 外部测试集 top-N:full4(4来源语料) vs 最近几轮\n")
    ap("**日期:2026-10-28**")
    ap("")
    ap("外部测试集 4,502 条(母核正例 64)。判定:top-N 候选中任一含母核结构即判阳;P/R/F1 相对该二分类。")
    ap("mean±std 为 seed 间(ddof=1);所有评测均在 5090(CUDA id 1), beams=10。")
    ap("full4 训练集 = new_db(1029,含 14/64 外部正例氟喹诺酮) + 1209 lmdb + extrain evalclean + CDK 类似物;")
    ap("因此其外部召回较 1209/nopubchem 轮更多受语料污染影响, 解读需打折。")
    ap("")

    ap("## 汇总(seed 间 mean±std)\n")
    ap("| 配置 | 表示 | 指标 | top-1 | top-3 | top-5 | top-10 |")
    ap("|---|---|---|---|---|---|---|")
    for gname in GROUPS:
        for rep in REPS:
            rows = results[gname][rep]
            for key, label in [("p", "Precision"), ("r", "Recall"), ("f1", "F1")]:
                cells = [fmt([rows[sd][n][key] for sd in rows]) for n in TOP_NS]
                ap(f"| {gname} | {rep} | {label} | " + " | ".join(cells) + " |")
    ap("")

    ap("## 逐 seed 明细\n")
    for gname in GROUPS:
        ap(f"### {gname}\n")
        for rep in REPS:
            ap(f"**{rep}**")
            ap("")
            ap("| seed | top-N | Precision | Recall | F1 | (tp, fp, fn) |")
            ap("|---|---|---|---|---|---|")
            rows = results[gname][rep]
            for sd in rows:
                for n in TOP_NS:
                    m = rows[sd][n]
                    ap(f"| {sd} | {n} | {m['p']:.4f} | {m['r']:.4f} | {m['f1']:.4f} "
                       f"| ({m['tp']}, {m['fp']}, {m['fn']}) |")
            ap("")

    out = ROOT / "cmgnet_topn_external_summary_20261028.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
